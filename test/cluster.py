import wandb
from torch.utils.data import DataLoader
from cuml.neighbors import KNeighborsClassifier
from IO.soccernetv2.match import PlayerPatchesDataset as SNV2Dataset
from IO.soccernetv3.sequences import PlayerPatchesDataset as SNV3Dataset
from clustering import extract_embeddings
from models.backbone import Backbone
from util.data import CollateFrames
from sklearn.metrics import accuracy_score

__all__ = ['test_cluster']


def test_cluster(conf, args, device):
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
                                    load_predictions=True,
                                    use_background=conf.backbone.background,
                                    cluster_method=cluster_method)

        loader = DataLoader(dataset=dataset,
                            collate_fn=CollateFrames(num_data_outputs=dataset.num_data_outputs,
                                                     num_classes=conf.clustering.num_clusters,
                                                     padding_token=-1),
                            shuffle=False,
                            batch_size=conf.clustering.optimization.batch_size,
                            num_workers=args.num_workers)

        embeddings, predictions = extract_embeddings(model, loader)[1::2]

        knn = KNeighborsClassifier(n_neighbors=conf.clustering.optimization.num_neighbors,
                                   metric=conf.clustering.distance)
        knn.fit(embeddings, predictions)

        test_dataset = soccernet_dataset(data_path,
                                         load_gt=True,
                                         use_background=conf.backbone.background,
                                         num_processes=args.num_loading_threads)

        test_loader = DataLoader(dataset=test_dataset,
                                 collate_fn=CollateFrames(num_data_outputs=test_dataset.num_data_outputs,
                                                          num_classes=conf.data.num_classes,
                                                          padding_token=conf.data.padding_token),
                                 shuffle=False,
                                 batch_size=conf.optimization.batch_size,
                                 num_workers=args.num_workers)

        test_embeddings, labels = extract_embeddings(model, test_loader)[1::2]
        predictions = knn.predict(test_embeddings)

        accuracy = accuracy_score(labels, predictions)
        if wandb.run is not None:
            wandb.log({'Test/test_accuracy': accuracy})
        print(f"Test accuracy = {accuracy}")

        if args.save_predictions:
            test_dataset.save_predictions(predictions, filename=args.save_predictions)
