import cudf
import numpy as np
import torch
from cuml import DBSCAN
from cuml import KMeans
from cuml.neighbors import KNeighborsClassifier
from cuml.neighbors import NearestNeighbors
from kitman.field_calibration import REPRESENTATION_HEIGHT
from kitman.field_calibration import REPRESENTATION_WIDTH
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.soccernetv2.match import PlayerPatchesDataset as SNV2Dataset
from IO.soccernetv3.sequences import PlayerPatchesDataset as SNV3Dataset
from models.backbone import Backbone
from util.data import CollateFrames


def join_close_embeddings(embeddings, indices1, indices2, threshold):
    dist_matrix = cosine_distances(embeddings[indices1], embeddings[indices2])
    min_distance = dist_matrix.min(axis=1)

    indices1 = np.copy(indices1)
    indices1[indices1] = min_distance < threshold

    min_distance = dist_matrix.min(axis=0)
    indices2 = np.copy(indices2)
    indices2[indices2] = min_distance < threshold

    return np.logical_or(indices1, indices2)


def extract_embeddings(model, loader):
    model.eval()

    load_labels = loader.dataset.load_labels

    indices, embeddings, coords, labels = [], [], [], []
    with tqdm(enumerate(loader), total=len(loader)) as t:
        for i, batch in t:
            if load_labels:
                players, coords_, labels_, lengths = batch
                labels_ = labels_.cpu().detach().numpy()
            else:
                players, coords_, lengths = batch

            if torch.cuda.is_available():
                players = players.cuda()

            coords_ = coords_.cpu().detach().numpy()

            for j, k in enumerate(lengths):
                indices.append(np.repeat(i * loader.batch_size + j, k))

                embeddings_ = model(players[j, :k, ...])
                embeddings_ = embeddings_.cpu().detach().numpy()
                embeddings.append(embeddings_)

                coords.append(coords_[j, :k, ...])

                if load_labels:
                    start = j * players.shape[1]
                    labels.extend(labels_[start:start + k])

    indices = np.concatenate(indices)
    embeddings = np.concatenate(embeddings)
    coords = np.concatenate(coords)
    labels = np.asarray(labels) if load_labels else None
    return indices, embeddings, coords, labels


def build_affinity_matrix(embeddings, conf):
    if conf.affinity == 'rbf':
        if conf.distance == 'cosine':
            dist_matrix = cosine_distances(embeddings)
        else:
            raise Exception("Other distances not implemented yet")

        K = np.exp(-conf.gamma * dist_matrix)
        D = np.sqrt(1 / K.sum(axis=1))
        return np.multiply(D[np.newaxis, :], np.multiply(K, D[np.newaxis, :]))

    elif conf.affinity == 'nearest_neighbors':
        X = cudf.DataFrame(embeddings)
        nn = NearestNeighbors(n_neighbors=conf.optimization.num_neighbors,
                              metric=conf.distance,
                              metric_params=None).fit(X)
        connectivity = nn.kneighbors_graph(X,
                                           n_neighbors=conf.optimization.num_neighbors)

        connectivity = 0.5 * (connectivity + connectivity.T)
        return csr_matrix(connectivity.get())
    else:
        raise Exception("Other affinities not implemented yet")


def cluster_players(embeddings, conf):
    if conf.algorithm == 'SpectralClustering':
        affinity_matrix_ = build_affinity_matrix(embeddings, conf)

        if conf.affinity == 'nearest_neighbors':
            spectral = SpectralClustering(n_clusters=conf.num_clusters,
                                          affinity='precomputed_nearest_neighbors',
                                          random_state=0)
            spectral.fit(affinity_matrix_)
            return spectral
        elif conf.affinity == 'rbf':
            svd = TruncatedSVD(n_components=conf.truncated_svd_num_components)
            U = svd.fit_transform(affinity_matrix_)

            kmeans = KMeans(n_clusters=conf.num_clusters)
            kmeans.fit(normalize(U[:, 0:conf.num_clusters]))
            return kmeans

    raise Exception("Other algorithms not implemented yet")


def fit_predict_players(embeddings, conf):
    num_embeddings = embeddings.shape[0]
    if num_embeddings > conf.optimization.max_num_embeddings:
        indices = np.random.choice(num_embeddings,
                                   conf.optimization.max_num_embeddings, replace=False)
        remaining_indices = list(set(range(num_embeddings)) - set(indices))

        algorithm = cluster_players(embeddings[indices], conf)

        knn = KNeighborsClassifier(n_neighbors=conf.optimization.num_neighbors,
                                   metric=conf.distance)

        # FIXME: It is assumed that the clustering algorithm has a labels_ attribute
        knn.fit(embeddings[indices], algorithm.labels_)
        remaining_predictions = knn.predict(embeddings[remaining_indices])

        predictions = np.asarray(algorithm.labels_.tolist() + remaining_predictions.tolist())
        indices = indices.tolist() + remaining_indices
        return predictions[np.argsort(indices)]
    else:
        algorithm = cluster_players(embeddings, conf)

        # FIXME: It is assumed that the clustering algorithm has a labels_ attribute
        return algorithm.labels_


def separate_goalkeepers(embeddings, coords, conf):
    half_width = REPRESENTATION_WIDTH // 2
    half_height = REPRESENTATION_HEIGHT // 2

    # Filtering people outside pitch
    indices_1 = coords[:, 1] > -(half_height - 3)
    indices_2 = coords[:, 1] < (half_height - 3)
    indices_3 = coords[:, 0] > - half_width
    indices_4 = coords[:, 0] < half_width
    indices = indices_1 & indices_2 & indices_3 & indices_4

    # GOALKEEPER 1
    indices_1 = np.linalg.norm(coords - [-half_width, 0], axis=1) < conf.max_goalkeeper_distance
    gk1_indices = indices & indices_1

    # GOALKEEPER 2
    indices_1 = np.linalg.norm(coords - [half_width, 0], axis=1) < conf.max_goalkeeper_distance
    gk2_indices = indices & indices_1

    # NO GOALKEEPERS & NO SIDES
    indices_1 = np.linalg.norm(coords - [-half_width, 0], axis=1) > conf.min_distance_goal
    indices_2 = np.linalg.norm(coords - [half_width, 0], axis=1) > conf.min_distance_goal
    mf_indices = indices & indices_1 & indices_2

    non_gk_indices_1 = join_close_embeddings(embeddings, mf_indices, gk1_indices, 0.1)
    non_gk_indices_2 = join_close_embeddings(embeddings, mf_indices, gk2_indices, 0.1)

    non_gk_indices = join_close_embeddings(embeddings, non_gk_indices_1, non_gk_indices_2, 0.05)
    gk1_indices[non_gk_indices] = False
    gk2_indices[non_gk_indices] = False

    return mf_indices, gk1_indices, gk2_indices


def estimate_goalkeepers(embeddings, indices, conf):
    gk_cluster = DBSCAN(eps=0.1, min_samples=150, metric=conf.distance).fit(embeddings[indices])
    gk_predictions = gk_cluster.labels_.astype(int)

    clusters, counts = np.unique(gk_predictions, return_counts=True)

    most_densed_cluster = -1
    sorted_indices = np.argsort(-counts)
    for idx in sorted_indices:
        if clusters[idx] >= 0:
            most_densed_cluster = clusters[idx]
            break
    indices = np.copy(indices)
    indices[indices] = gk_predictions == most_densed_cluster
    return indices


def cluster(conf, args, device):

    if args.is_soccernet_v3:
        data_paths = args.sequences
        soccernet_dataset = SNV3Dataset
    else:
        data_paths = args.matches
        soccernet_dataset = SNV2Dataset

    cluster_method = 'gk_separation' if conf.clustering.separate_goalkeepers else 'spectral'

    model = Backbone(conf.backbone.model.name, conf.backbone.model.dim, conf.backbone.weights).to(device)

    for data_path in data_paths:
        dataset = soccernet_dataset(data_path,
                                    num_processes=args.num_loading_threads,
                                    use_background=conf.backbone.background,
                                    cluster_method=cluster_method)

        num_classes = conf.clustering.num_clusters if dataset.load_labels else None
        loader = DataLoader(dataset=dataset,
                            collate_fn=CollateFrames(num_data_outputs=dataset.num_data_outputs,
                                                     num_classes=num_classes,
                                                     padding_token=-1),
                            shuffle=False,
                            batch_size=conf.clustering.optimization.batch_size,
                            num_workers=args.num_workers)

        data_indices, embeddings, coords = extract_embeddings(model, loader)[:3]

        if conf.clustering.separate_goalkeepers:
            non_gk_indices, gk1_indices, gk2_indices = separate_goalkeepers(embeddings, coords, conf.clustering)

            conf.clustering.num_clusters = 3
            non_gk_predictions = fit_predict_players(embeddings[non_gk_indices], conf.clustering)

            gk1_indices = estimate_goalkeepers(embeddings, gk1_indices, conf.clustering)
            gk2_indices = estimate_goalkeepers(embeddings, gk2_indices, conf.clustering)

            predictions = -np.ones(embeddings.shape[0], dtype=int)
            predictions[non_gk_indices] = non_gk_predictions
            predictions[gk1_indices] = 3
            predictions[gk2_indices] = 4

            indices = np.copy(non_gk_indices)
            indices[gk1_indices] = True
            indices[gk2_indices] = True

            knn = KNeighborsClassifier(n_neighbors=conf.clustering.optimization.num_neighbors,
                                       metric=conf.clustering.distance)
            knn.fit(embeddings[indices], predictions[indices])

            remaining_indices = np.logical_not(indices)
            predictions[remaining_indices] = knn.predict(embeddings[remaining_indices])
        else:
            predictions = fit_predict_players(embeddings, conf.clustering)

        # Assign the cluster labels to the correct player category according to the cluster size
        clusters, counts = np.unique(predictions, return_counts=True)
        assignments_map = {clusters[i]: j for i, j in zip(np.argsort(counts), [3, 4, 0, 1, 2])}
        assign = np.vectorize(lambda x: assignments_map.get(x))
        predictions = assign(predictions)

        means = [np.mean(embeddings[(predictions == i)], axis=0) for i in range(5)]
        dataset.save_means(means)

        dataset.save_predictions(predictions)
