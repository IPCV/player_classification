from __future__ import annotations

import copy
import itertools
import random
from functools import partial
from itertools import permutations, product
from multiprocessing import Pool
from pathlib import Path
from typing import List, final, Union

import numpy as np
import torch
from addict import Dict
from kitman.data import PlayerPatches, BuildFilename, NonZeroBasedIndex
from kitman.field_calibration import calculate_player_position, CENTER_COORDS
from kitman.regions import area, \
    get_masked_patch, get_patch
from kitman.segmentation import POINTREND_PERSON_ID
from kitman.segmentation import SegmentedPlayer
from kitman.snv2 import FrameIndex, \
    load_homographies, \
    load_scaling_matrix, \
    MatchPaths
from kitman.snv2.field_calibration import meter2radar
from skimage import io
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def filter_players_off_field(segmented_players, homography, touchline_dist: float):
    filtered_players = []
    for p in segmented_players:
        xy = meter2radar(calculate_player_position(p.bb, homography)) - CENTER_COORDS
        if any(abs(xy) > CENTER_COORDS - touchline_dist):
            continue
        filtered_players.append(p)
    return filtered_players


REFEREE = 0
TEAM_1, TEAM_2 = 1, 2
GOALKEEPER_A, GOALKEEPER_B = 3, 4

VALID_CLASSES = [REFEREE, TEAM_1, TEAM_2, GOALKEEPER_A, GOALKEEPER_B]


class MatchDataset:

    def __init__(self, path: Path, min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000,
                 touchline_dist=3, load_gt=False, use_background=False, load_predictions=False,
                 only_players_on_field=False, cluster_method=None):

        self.idx = None
        self.paths = MatchPaths(path)

        self.segmented_players = {0: {}, 1: {}}
        self.num_patches = [0, 0]
        self.total_num_frames = 0
        self.total_num_patches = 0

        self.num_original_segmented_frames = [0, 0]
        self.valid_frames_ids = {0: [], 1: []}
        self.homographies = {0: {}, 1: {}}
        self.calibration_confidence = {0: {}, 1: {}}
        self.use_background = use_background
        self.load_labels = load_gt or load_predictions
        self.cluster_method = cluster_method

        if load_gt:
            self._load_groundtruth()
        else:
            self._load_segmentations(min_segmentation_score, min_calibration_confidence, max_bb_area, touchline_dist,
                                     use_background, load_predictions, only_players_on_field)

    @final
    def _load_groundtruth(self):
        groundtruth = np.load(self.paths.groundtruth, allow_pickle=True).item()
        self.num_original_segmented_frames = groundtruth['num_original_segmented_frames']
        for half in range(2):
            for idx in groundtruth[half].keys():
                gt = groundtruth[half][idx]

                segmented_players = SegmentedPlayer.to_segmented_players(gt, filtered_class_ids=VALID_CLASSES)
                num_segmented_players = len(segmented_players)

                if num_segmented_players == 0:
                    continue

                self.homographies[half][idx] = gt['homography']
                self.calibration_confidence[half][idx] = gt['calibration_confidence']
                self.segmented_players[half][idx] = segmented_players
                self.num_patches[half] += num_segmented_players

                self.valid_frames_ids[half].append(idx)
                self.total_num_frames += 1
            self.total_num_patches += self.num_patches[half]

    @final
    def _load_segmentations(self, min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000,
                            touchline_dist=3, use_background=False, load_predictions=False, only_players_on_field=True):

        scaling_matrix = load_scaling_matrix(self.paths.sampling_aspect_ratio)
        filtered_class_ids = POINTREND_PERSON_ID if not load_predictions else VALID_CLASSES
        for half in range(2):
            if not load_predictions:
                segmentations_path = self.paths.segmentations[half]
            elif use_background:
                segmentations_path = self.paths.clustered_segmentations_bkg[self.cluster_method, half]
            else:
                segmentations_path = self.paths.clustered_segmentations[self.cluster_method, half]
            segmented_players = SegmentedPlayer.load(segmentations_path, min_segmentation_score, filtered_class_ids)
            homographies = load_homographies(self.paths.calibrations[half], scaling_matrix)

            self.num_original_segmented_frames[half] = len(segmented_players)
            num_frames = min(len(segmented_players), len(homographies))
            for idx in range(num_frames):
                if homographies[idx]['confidence'] < min_calibration_confidence:
                    continue

                if any(area(s.bb) > max_bb_area for s in segmented_players[idx]):
                    continue

                if only_players_on_field:
                    segmented_players_ = filter_players_off_field(segmented_players[idx],
                                                                  homographies[idx]['matrix'],
                                                                  touchline_dist)
                else:
                    segmented_players_ = segmented_players[idx]

                if len(segmented_players_) == 0:
                    continue

                self.valid_frames_ids[half].append(idx)
                self.homographies[half][idx] = homographies[idx]['matrix']
                self.calibration_confidence[half][idx] = homographies[idx]['confidence']
                self.segmented_players[half][idx] = segmented_players_
                self.num_patches[half] += len(segmented_players_)

            self.total_num_patches += self.num_patches[half]
            self.total_num_frames += len(self.valid_frames_ids[half])

    @staticmethod
    def load_labels(match_path: Path, load_gt=False, use_background=False, filename=None):
        labels = []

        paths = MatchPaths(match_path)
        if load_gt:
            groundtruth = np.load(paths.groundtruth, allow_pickle=True).item()
            for half in range(2):
                for idx in sorted(groundtruth[half].keys()):
                    gt = groundtruth[half][idx]
                    segmented_players = SegmentedPlayer.to_segmented_players(gt, filtered_class_ids=VALID_CLASSES)
                    for p in segmented_players:
                        if p.class_id < 5:
                            labels.append(p.class_id)
        else:
            if filename is not None:
                segmentations_path = BuildFilename.create(paths.match, filename, NonZeroBasedIndex())
            elif use_background:
                segmentations_path = paths.clustered_segmentations_bkg
            else:
                segmentations_path = paths.clustered_segmentations

            for half in range(2):
                segmented_players = SegmentedPlayer.load(segmentations_path[half])
                for frame_idx in range(len(segmented_players)):
                    for p in segmented_players[frame_idx]:
                        if p.class_id < 5:
                            labels.append(p.class_id)
        return labels

    @final
    def split_by(self, frames_ids: List[FrameIndex] = None) -> MatchDataset:
        other = copy.deepcopy(self)
        other.segmented_players = {0: {}, 1: {}}
        other.valid_frames_ids = {0: [], 1: []}
        other.total_num_frames = 0
        other.total_num_patches = 0
        other.num_patches = [0, 0]

        for id_ in frames_ids:
            if id_.frame_idx in self.valid_frames_ids[id_.half]:
                other.valid_frames_ids[id_.half].append(id_.frame_idx)
                other.segmented_players[id_.half][id_.frame_idx] = self.segmented_players[id_.half][id_.frame_idx]
                num_patches = len(self.segmented_players[id_.half][id_.frame_idx])
                other.total_num_patches += num_patches
                other.num_patches[id_.half] += num_patches
                other.total_num_frames += 1
        return other

    def __len__(self):
        return self.total_num_frames


def _load_player_patches_from_frame(args):
    idx, frame_path, segmented_players, homography, load_background, center_coords, load_labels = args
    frame = io.imread(frame_path)

    half_side = 1. if idx.half == 0 else -1.
    patches, coords, labels = [], [], []
    for p in segmented_players:

        if load_background:
            patches.append(get_patch(frame, p.bb))
        else:
            patches.append(get_masked_patch(frame, p.bb, p.mask_cnts))

        xy = meter2radar(calculate_player_position(p.bb, homography))

        if center_coords:
            xy -= CENTER_COORDS
        xy[0] *= half_side
        coords.append(xy)

        if load_labels:
            labels.append(p.class_id)

    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64) if load_labels else None
    return PlayerPatches(idx, patches, coords, labels)


class PlayerPatchesDataset(MatchDataset, Dataset):
    def __init__(self, dataset: Union[Path, MatchDataset], patch_transform=None, coord_transform=None,
                 label_transform=None, min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000,
                 touchline_dist=3, center_coords=True, num_processes=12, load_gt=False, use_background=False,
                 load_predictions=False, only_players_on_field=False, cluster_method=None):

        if isinstance(dataset, Path):
            super().__init__(dataset, min_segmentation_score, min_calibration_confidence, max_bb_area, touchline_dist,
                             load_gt, use_background, load_predictions, only_players_on_field, cluster_method)
        else:
            self.__dict__ = dataset.__dict__.copy()

        self._load_player_patches(num_processes, center_coords)

        if patch_transform is None:
            self.patch_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.patch_transform = patch_transform

        self.coord_transform = coord_transform

        if self.load_labels:
            self.num_data_outputs = 3
            self.label_transform = label_transform
        else:
            self.num_data_outputs = 2
            self.label_transform = None

    def __len__(self):
        return len(self.data)

    def _load_player_patches(self, num_processes, center_coords):
        self.data = []

        with tqdm(total=self.total_num_patches, desc='Loading patches progress', leave=True, position=0) as pbar:
            for half in range(2):
                indices, frame_paths, segmentations, homographies = [], [], [], []
                for idx in self.valid_frames_ids[half]:
                    indices.append(FrameIndex(half, idx))
                    frame_paths.append(self.paths.frames[half, idx])
                    segmentations.append(self.segmented_players[half][idx])
                    homographies.append(self.homographies[half][idx])
                load_background = [self.use_background] * len(self.valid_frames_ids[half])
                center_coords_ = [center_coords] * len(self.valid_frames_ids[half])
                load_labels_ = [self.load_labels] * len(self.valid_frames_ids[half])

                with Pool(processes=num_processes) as pool:
                    for player_patches in pool.imap_unordered(_load_player_patches_from_frame,
                                                              zip(indices, frame_paths, segmentations, homographies,
                                                                  load_background, center_coords_, load_labels_)):
                        self.data.append(player_patches)
                        pbar.update(len(player_patches))

        for p in self.data:
            p.coords = torch.as_tensor(p.coords)
            if self.load_labels:
                p.labels = torch.as_tensor(p.labels)
        self.data = sorted(self.data, key=lambda x: (x.half, x.frame_idx))

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.data[idx].patches])

        coords = self.data[idx].coords
        if self.coord_transform:
            coords = self.coord_transform(coords)

        if self.num_data_outputs == 3:
            labels = self.data[idx].labels
            if self.label_transform:
                labels = self.label_transform(labels)
            return patches, coords, labels
        else:
            return patches, coords

    def save_means(self, means, filename=None):
        if filename is not None:
            means_path = self.paths.match.joinpath(filename)
        elif self.use_background:
            means_path = self.paths.clustered_bkg_means[self.cluster_method]
        else:
            means_path = self.paths.clustered_means[self.cluster_method]
        np.save(means_path, means)

    def save_predictions(self, predictions, filename=None):

        if filename is not None:
            filename = BuildFilename.create(self.paths.base_path, filename, NonZeroBasedIndex())

        idx = 0
        for half in range(2):
            segmentations = []
            for frame_idx in range(self.num_original_segmented_frames[half]):
                frame_data = {'boxes': [], 'masks': [], 'class_ids': [], 'scores': []}
                if frame_idx in self.valid_frames_ids[half]:
                    for p in self.segmented_players[half][frame_idx]:
                        if p.class_id < 5:
                            frame_data['boxes'].append(p.bb)
                            frame_data['masks'].append(p.mask_cnts)
                            frame_data['scores'].append(p.score)
                            frame_data['class_ids'].append(predictions[idx])
                            idx += 1
                segmentations.append(frame_data)

            if filename is not None:
                predictions_path = filename[half]
            elif self.use_background:
                predictions_path = self.paths.clustered_segmentations_bkg[self.cluster_method, half]
            else:
                predictions_path = self.paths.clustered_segmentations[self.cluster_method, half]
            np.save(predictions_path, segmentations)


class DataIndex(FrameIndex):
    def __init__(self, match_idx, half, frame_idx):
        super().__init__(half, frame_idx)
        self.match_idx = match_idx


def add_kit(kits, match_idx1, match_idx2, cat_idx1, cat_idx2):
    if match_idx2 not in kits[match_idx1][cat_idx1]:
        kits[match_idx1][cat_idx1][match_idx2] = {cat_idx2}
    else:
        kits[match_idx1][cat_idx1][match_idx2].add(cat_idx2)


def check_similarity(similarity, means, i, j, idx1, idx2, threshold=0.25):
    dist = cosine_distances(means[i][idx1].reshape(1, -1),
                            means[j][idx2].reshape(1, -1)).item()
    if dist < threshold:
        add_kit(similarity, i, j, idx1, idx2)
        add_kit(similarity, j, i, idx2, idx1)


def are_similar(similarity, match_idx1, cat_idx1, match_idx2, cat_idx2):
    return match_idx2 in similarity[match_idx1][cat_idx1] and cat_idx2 in similarity[match_idx1][cat_idx1][match_idx2]


def clashes_goalkeepers_colors(similarity, match_idx1, match_idx2, cat_idx):
    if are_similar(similarity, match_idx2, cat_idx, match_idx1, GOALKEEPER_A):
        return True

    if are_similar(similarity, match_idx2, cat_idx, match_idx1, GOALKEEPER_A):
        return True

    return False


def clashes_team_colors(similarity, match_idx1, match_idx2, cat_idx):
    if are_similar(similarity, match_idx2, cat_idx, match_idx1, TEAM_1):
        return True

    if are_similar(similarity, match_idx2, cat_idx, match_idx1, TEAM_2):
        return True

    return False


def referee_clashes_team_colors(similarity, match_idx1, match_idx2):
    if are_similar(similarity, match_idx2, REFEREE, match_idx1, REFEREE):
        return True
    return clashes_team_colors(similarity, match_idx1, match_idx2, REFEREE)


def _load_player_patches_from_frame_by_class(args):
    frame_path, segmented_players, load_background = args
    frame = io.imread(frame_path)

    patches = {}
    for p in segmented_players:
        patch = get_patch(frame, p.bb) if load_background else get_masked_patch(frame, p.bb, p.mask_cnts)
        if p.class_id not in patches:
            patches[p.class_id] = [patch]
        else:
            patches[p.class_id].append(patch)
    return patches


class MatchesDataset(Dataset):
    def __init__(self, match_paths: List[Path], patch_transform=None, coord_transform=None, label_transform=None,
                 min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000, touchline_dist=3,
                 center_coords=True, num_processes=12, load_gt=False, use_background=False, load_predictions=False,
                 only_players_on_field=False, cluster_method=None, num_frames_per_half=5, similarity_threshold=0.1,
                 num_changes_per_class=5, clustering_distance='cosine'):

        self._load_matches(match_paths, min_segmentation_score, min_calibration_confidence, max_bb_area, touchline_dist,
                           num_processes, load_gt, use_background, load_predictions, only_players_on_field,
                           cluster_method)

        self._calculate_valid_combinations(cluster_method, similarity_threshold)

        if patch_transform is None:
            self.patch_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.patch_transform = patch_transform

        self.coord_transform = coord_transform
        self.use_background = use_background
        self.load_labels = load_gt or load_predictions

        if self.load_labels:
            self.num_data_outputs = 4
            self.label_transform = label_transform
        else:
            self.num_data_outputs = 3
            self.label_transform = None

        self.data = []
        self.center_coords = center_coords
        self.num_processes = num_processes
        self.num_frames_per_half = num_frames_per_half
        self.num_changes_per_class = num_changes_per_class

    def _load_matches(self, match_paths: List[Path], min_segmentation_score, min_calibration_confidence, max_bb_area,
                      touchline_dist, num_processes, load_gt, use_background, load_predictions, only_players_on_field,
                      cluster_method):

        match_dataset = partial(MatchDataset,
                                min_segmentation_score=min_segmentation_score,
                                min_calibration_confidence=min_calibration_confidence,
                                max_bb_area=max_bb_area,
                                touchline_dist=touchline_dist,
                                load_gt=load_gt,
                                use_background=use_background,
                                load_predictions=load_predictions,
                                only_players_on_field=only_players_on_field,
                                cluster_method=cluster_method)

        self.num_matches = len(match_paths)
        self.num_classes = 5
        self.matches = []
        with tqdm(total=self.num_matches, desc='Loading matches progress', leave=True, position=0) as pbar:
            with Pool(processes=num_processes) as pool:
                for match in pool.imap_unordered(match_dataset, match_paths):
                    self.matches.append(match)
                    pbar.update(1)
        self.matches = sorted(self.matches, key=lambda x: x.paths.match)

        for match_idx, m in enumerate(self.matches):
            m.idx = match_idx
            m.frames_by_class = {i: [] for i in range(self.num_classes)}
            m.total_category_count = np.zeros(self.num_classes, dtype=int)
            m.categories_counts = {half: {} for half in range(2)}

            for half in range(2):
                for idx in m.valid_frames_ids[half]:
                    player_classes = [p.class_id for p in m.segmented_players[half][idx]]
                    categories, counts = np.unique(player_classes, return_counts=True)
                    m.categories_counts[half][idx] = np.zeros(self.num_classes, dtype=int)
                    for category, count in zip(categories, counts):
                        m.categories_counts[half][idx][category] = count
                        m.total_category_count[category] += count

                    for c in categories:
                        m.frames_by_class[c].append((half, idx))

    def _calculate_valid_combinations(self, cluster_method, similarity_threshold):
        means = []
        for m in self.matches:
            if self.use_background:
            	clustered_path = m.paths.clustered_bkg_means[cluster_method]
            else:
            	clustered_path = m.paths.clustered_means[cluster_method]
            means.append(np.load(clustered_path, allow_pickle=True))

        similarity = [{j: {} for j in range(self.num_classes)} for _ in range(self.num_matches)]

        kits = [REFEREE, TEAM_1, TEAM_2, GOALKEEPER_A, GOALKEEPER_B]
        for i in range(self.num_matches):
            for j in range(i + 1, self.num_matches):
                for (idx1, idx2) in itertools.product(kits, repeat=2):
                    check_similarity(similarity, means, i, j, idx1, idx2, similarity_threshold)

        self.valid_combinations = [{j: {} for j in range(self.num_classes)} for _ in range(self.num_matches)]
        for i, j in permutations(range(self.num_matches), 2):
            if not referee_clashes_team_colors(similarity, i, j):
                add_kit(self.valid_combinations, i, j, REFEREE, REFEREE)

            for team_idx in [TEAM_1, TEAM_2]:

                if clashes_team_colors(similarity, i, j, team_idx):
                    continue

                if clashes_goalkeepers_colors(similarity, i, j, team_idx):
                    continue
                add_kit(self.valid_combinations, i, j, TEAM_1, team_idx)
                add_kit(self.valid_combinations, i, j, TEAM_2, team_idx)

            for gk_idx in [GOALKEEPER_A, GOALKEEPER_B]:
                if clashes_team_colors(similarity, i, j, gk_idx):
                    continue

                if not are_similar(similarity, j, gk_idx, i, GOALKEEPER_A):
                    add_kit(self.valid_combinations, i, j, GOALKEEPER_A, gk_idx)

                if not are_similar(similarity, j, gk_idx, i, GOALKEEPER_B):
                    add_kit(self.valid_combinations, i, j, GOALKEEPER_B, gk_idx)

    def _choose_random_match(self, match_idx=None):

        # Randomly choosing a match and its frames
        if match_idx is None:
            match_idx = np.random.randint(self.num_matches)
        match = self.matches[match_idx]

        sampled_frame_ids = {}
        player_counts = np.zeros(self.num_classes, dtype=int)
        for half in range(2):
            num_frames_to_sample = min(self.num_frames_per_half, len(match.valid_frames_ids[half]))
            sampled_frame_ids[half] = sorted(np.random.choice(match.valid_frames_ids[half],
                                                              num_frames_to_sample,
                                                              replace=False))
            for idx in sampled_frame_ids[half]:
                player_counts += match.categories_counts[half][idx]
        return match, sampled_frame_ids, player_counts

    def _calculate_mix_matches(self, match_idx, player_counts):

        # Randomly determining the set of possible matches to mix
        mix_matches_indices = set()
        num_changes_per_class = np.zeros(self.num_classes, dtype=int)

        for cat_idx, player_count in enumerate(player_counts):
            candidate_matches = []
            for candidate_idx, mix_indices in self.valid_combinations[match_idx][cat_idx].items():
                for mix_idx in mix_indices:
                    if player_count <= self.matches[candidate_idx].total_category_count[mix_idx]:
                        candidate_matches.append(candidate_idx)

            if len(candidate_matches) == 0:
                continue

            num_changes_per_class[cat_idx] = min(self.num_changes_per_class, len(candidate_matches))

            indices = np.random.choice(np.arange(len(candidate_matches)), num_changes_per_class[cat_idx], replace=False)

            for idx in indices:
                mix_matches_indices.add(candidate_matches[idx])

        mix_matches_indices = list(mix_matches_indices)
        np.random.shuffle(mix_matches_indices)
        return mix_matches_indices, num_changes_per_class

    def _reduce_mix_matches(self, match_idx, player_counts, mix_matches_ids, num_changes_per_class):

        # Reducing the set of possible matches to mix
        mix_matches = {i: [] for i in range(self.num_classes)}
        sampled_matches_ids = set()

        num_sampled_matches = np.zeros(self.num_classes, dtype=int)
        for mix_match_idx in mix_matches_ids:
            for cat_idx in range(self.num_classes):
                if mix_match_idx not in self.valid_combinations[match_idx][cat_idx]:
                    continue

                for mix_cat_idx in self.valid_combinations[match_idx][cat_idx][mix_match_idx]:
                    if num_sampled_matches[cat_idx] == num_changes_per_class[cat_idx]:
                        break

                    if player_counts[cat_idx] > self.matches[mix_match_idx].total_category_count[mix_cat_idx]:
                        continue

                    num_sampled_matches[cat_idx] += 1
                    mix_matches[cat_idx].append((mix_match_idx, mix_cat_idx))
                    sampled_matches_ids.add(mix_match_idx)

            if all([num_sampled_matches[i] == num_changes_per_class[i] for i in range(5)]):
                break
        return sampled_matches_ids, mix_matches

    def _choose_random_mix_matches(self, match_idx, player_counts):
        mix_matches_ids, num_changes_per_class = self._calculate_mix_matches(match_idx, player_counts)

        sampled_matches_ids, mix_matches = self._reduce_mix_matches(match_idx,
                                                                    player_counts,
                                                                    mix_matches_ids,
                                                                    num_changes_per_class)
        return sampled_matches_ids, mix_matches, num_changes_per_class

    def _sampling_mix_matches_frames(self, player_counts, mix_matches_indices, mix_matches):

        # Randomly choosing frames from the sampled matches
        num_patches_to_load = {idx: np.zeros(self.num_classes, dtype=int) for idx in mix_matches_indices}
        mix_frames_ids = {idx: [] for idx in mix_matches_indices}

        for cat_idx in [GOALKEEPER_A, GOALKEEPER_B, REFEREE, TEAM_1, TEAM_2]:
            for mix_match_idx, mix_cat_idx in mix_matches[cat_idx]:
                sampled_match = self.matches[mix_match_idx]
                random.shuffle(sampled_match.frames_by_class[mix_cat_idx])
                frames_indices = sampled_match.frames_by_class[mix_cat_idx]
                idx = 0
                while num_patches_to_load[mix_match_idx][mix_cat_idx] < player_counts[cat_idx]:
                    mix_frames_ids[mix_match_idx].append(frames_indices[idx])
                    half, frame_idx = frames_indices[idx]
                    num_patches_to_load[mix_match_idx] += sampled_match.categories_counts[half][frame_idx]
                    idx += 1
        return mix_frames_ids, num_patches_to_load

    def load_epoch_data(self, match_idxs=None):
        self.data = []

        match, sampled_frame_ids, player_counts = self._choose_random_match(match_idxs)

        mix_matches_ids, mix_matches, num_changes_per_class = self._choose_random_mix_matches(match.idx,
                                                                                              player_counts)

        mix_frames_ids, num_patches_to_load = self._sampling_mix_matches_frames(player_counts,
                                                                                mix_matches_ids,
                                                                                mix_matches)

        # Loading patches from base match
        total_num_patches = player_counts.sum()
        with tqdm(total=total_num_patches, desc='Loading patches progress', leave=True, position=0) as pbar:
            for half in range(2):
                indices, frame_paths, segmentations, homographies = [], [], [], []
                for idx in sampled_frame_ids[half]:
                    indices.append(DataIndex(match.idx, half, idx))
                    frame_paths.append(match.paths.frames[half, idx])
                    segmentations.append(match.segmented_players[half][idx])
                    homographies.append(match.homographies[half][idx])

                load_background = [self.use_background] * len(indices)
                center_coords_ = [self.center_coords] * len(indices)
                load_labels_ = [self.load_labels] * len(indices)

                with Pool(processes=self.num_processes) as pool:
                    for player_patches in pool.imap_unordered(_load_player_patches_from_frame,
                                                              zip(indices, frame_paths, segmentations, homographies,
                                                                  load_background, center_coords_, load_labels_)):
                        self.data.append(player_patches)
                        pbar.update(len(player_patches))
        self.data = sorted(self.data, key=lambda x: (x.match_idx, x.half, x.frame_idx))

        new_label = {REFEREE: REFEREE}
        if self.team_clusters[match.idx][TEAM_1] > self.team_clusters[match.idx][TEAM_2]:
            new_label[TEAM_1] = TEAM_2
            new_label[TEAM_2] = TEAM_1
        else:
            new_label[TEAM_1] = TEAM_1
            new_label[TEAM_2] = TEAM_2

        if self.gk_clusters[match.idx][GOALKEEPER_A] > self.gk_clusters[match.idx][GOALKEEPER_B]:
            new_label[GOALKEEPER_A] = GOALKEEPER_B
            new_label[GOALKEEPER_B] = GOALKEEPER_A
        else:
            new_label[GOALKEEPER_A] = GOALKEEPER_A
            new_label[GOALKEEPER_B] = GOALKEEPER_B

        original_labels = []
        for player_patches in self.data:
            original_labels.append(player_patches.labels)

            labels = [new_label[l] for l in player_patches.labels]
            player_patches.labels = np.asarray(labels, dtype=np.int64)

        # Loading patches from matches to mix
        mixing_patches = {}
        for mix_match_idx in mix_matches_ids:
            mixing_patches[mix_match_idx] = {i: [] for i in range(self.num_classes)}
            total_num_patches = num_patches_to_load[mix_match_idx].sum()
            with tqdm(total=total_num_patches, desc='Loading patches progress', leave=True, position=0) as pbar:
                frame_paths, segmentations = [], []
                for (half, frame_idx) in mix_frames_ids[mix_match_idx]:
                    frame_paths.append(self.matches[mix_match_idx].paths.frames[half, frame_idx])
                    segmentations.append(self.matches[mix_match_idx].segmented_players[half][frame_idx])
                load_background = [self.use_background] * len(frame_paths)

                with Pool(processes=self.num_processes) as pool:
                    for player_patches in pool.imap_unordered(_load_player_patches_from_frame_by_class,
                                                              zip(frame_paths, segmentations, load_background)):
                        for mix_cat_idx in range(self.num_classes):
                            if mix_cat_idx in player_patches:
                                mixing_patches[mix_match_idx][mix_cat_idx].extend(player_patches[mix_cat_idx])
                        pbar.update(len(player_patches))

        frames_by_cat = {i: [] for i in range(5)}
        for cat_idx, num_changes in enumerate(num_changes_per_class):
            if num_changes > 0:
                for data_idx, d in enumerate(self.data):
                    if match.categories_counts[d.half][d.frame_idx][cat_idx] > 0:
                        frames_by_cat[cat_idx].append(data_idx)

        match_counter = 1
        for cat_idx, mixing_indices in mix_matches.items():
            for mix_match_idx, mix_cat_idx in mixing_indices:
                cat_counter = 0
                category_patches = mixing_patches[mix_match_idx][mix_cat_idx]

                new_label = {REFEREE: REFEREE,
                             TEAM_1: TEAM_1,
                             TEAM_2: TEAM_2,
                             GOALKEEPER_A: GOALKEEPER_A,
                             GOALKEEPER_B: GOALKEEPER_B}

                if cat_idx == TEAM_1:
                    if self.team_clusters[match.idx][TEAM_2] < self.team_clusters[mix_match_idx][mix_cat_idx]:
                        new_label[TEAM_1] = TEAM_2
                        new_label[TEAM_2] = TEAM_1

                elif cat_idx == TEAM_2:
                    if self.team_clusters[match.idx][TEAM_1] > self.team_clusters[mix_match_idx][mix_cat_idx]:
                        new_label[TEAM_1] = TEAM_2
                        new_label[TEAM_2] = TEAM_1

                elif cat_idx == GOALKEEPER_A:
                    if self.gk_clusters[match.idx][GOALKEEPER_B] < self.gk_clusters[mix_match_idx][mix_cat_idx]:
                        new_label[GOALKEEPER_A] = GOALKEEPER_B
                        new_label[GOALKEEPER_B] = GOALKEEPER_A

                elif cat_idx == GOALKEEPER_B:
                    if self.gk_clusters[match.idx][GOALKEEPER_A] > self.gk_clusters[mix_match_idx][mix_cat_idx]:
                        new_label[GOALKEEPER_A] = GOALKEEPER_B
                        new_label[GOALKEEPER_B] = GOALKEEPER_A

                for data_idx in frames_by_cat[cat_idx]:
                    patches, new_labels = [], []
                    for i, label in enumerate(original_labels[data_idx]):
                        if label == cat_idx:
                            patches.append(category_patches[cat_counter])
                            cat_counter += 1
                        else:
                            patches.append(self.data[data_idx].patches[i])
                        new_labels.append(new_label[label])

                    index = DataIndex(self.data[data_idx].match_idx + match_counter,
                                      self.data[data_idx].half,
                                      self.data[data_idx].frame_idx)

                    player_patches = PlayerPatches(index,
                                                   patches,
                                                   self.data[data_idx].coords,
                                                   np.asarray(new_labels, dtype=np.int64))

                    self.data.append(player_patches)
                match_counter += 1

        for p in self.data:
            p.coords = torch.as_tensor(p.coords)
            if self.load_labels:
                p.labels = torch.as_tensor(p.labels)

    def clear_epoch_data(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.data[idx].patches])

        coords = self.data[idx].coords
        if self.coord_transform:
            coords = self.coord_transform(coords)

        if self.num_data_outputs == 4:
            labels = self.data[idx].labels
            if self.label_transform:
                labels = self.label_transform(labels)
            return patches, coords, labels, self.data[idx].match_idx
        else:
            return patches, coords, self.data[idx].match_idx


def hash_pair(pair):
    return hash(tuple(sorted(pair)))


def sample_from_set(set_):
    return random.choices(list(set_), k=1)[0]


class CombineMatchesDataset(Dataset):
    def __init__(self, match_paths: List[Path], patch_transform=None, coord_transform=None, label_transform=None,
                 min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000, touchline_dist=3,
                 center_coords=True, num_processes=12, load_gt=False, use_background=False, load_predictions=False,
                 only_players_on_field=False, cluster_method=None, num_frames_per_half=5, similarity_threshold=0.09,
                 num_changes_per_class=5, clustering_distance='cosine', num_random_comb_per_step=1, mix=True,
                 kit_clusters_csv=None):

        self._load_matches(match_paths, min_segmentation_score, min_calibration_confidence, max_bb_area, touchline_dist,
                           num_processes, load_gt, use_background, load_predictions, only_players_on_field,
                           cluster_method)

        self.use_background = use_background

        self._calculate_clusters(cluster_method, clustering_distance, similarity_threshold, kit_clusters_csv)

        if patch_transform is None:
            self.patch_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.patch_transform = patch_transform

        self.coord_transform = coord_transform
        self.load_labels = load_gt or load_predictions

        if self.load_labels:
            self.num_data_outputs = 4
            self.label_transform = label_transform
        else:
            self.num_data_outputs = 3
            self.label_transform = None

        self.data = []
        self.center_coords = center_coords
        self.num_processes = num_processes
        self.num_frames_per_half = num_frames_per_half
        self.num_changes_per_class = num_changes_per_class
        self.num_random_comb_per_step = num_random_comb_per_step
        self.mix = mix

    def _load_matches(self, match_paths: List[Path], min_segmentation_score, min_calibration_confidence, max_bb_area,
                      touchline_dist, num_processes, load_gt, use_background, load_predictions, only_players_on_field,
                      cluster_method):

        match_dataset = partial(MatchDataset,
                                min_segmentation_score=min_segmentation_score,
                                min_calibration_confidence=min_calibration_confidence,
                                max_bb_area=max_bb_area,
                                touchline_dist=touchline_dist,
                                load_gt=load_gt,
                                use_background=use_background,
                                load_predictions=load_predictions,
                                only_players_on_field=only_players_on_field,
                                cluster_method=cluster_method)

        self.num_matches = len(match_paths)
        self.num_classes = 5
        self.matches = []
        with tqdm(total=self.num_matches, desc='Loading matches progress', leave=True, position=0) as pbar:
            with Pool(processes=num_processes) as pool:
                for match in pool.imap_unordered(match_dataset, match_paths):
                    self.matches.append(match)
                    pbar.update(1)
        self.matches = sorted(self.matches, key=lambda x: x.paths.match)

        for match_idx, m in enumerate(self.matches):
            m.idx = match_idx
            m.frames_by_class = {i: [] for i in range(self.num_classes)}
            m.total_category_count = np.zeros(self.num_classes, dtype=int)
            m.categories_counts = {half: {} for half in range(2)}

            for half in range(2):
                for idx in m.valid_frames_ids[half]:
                    player_classes = [p.class_id for p in m.segmented_players[half][idx]]
                    categories, counts = np.unique(player_classes, return_counts=True)
                    m.categories_counts[half][idx] = np.zeros(self.num_classes, dtype=int)
                    for category, count in zip(categories, counts):
                        m.categories_counts[half][idx][category] = count
                        m.total_category_count[category] += count

                    for c in categories:
                        m.frames_by_class[c].append((half, idx))

    @staticmethod
    def _calculate_centroids(means, labels, num_labels):
        means_by_label = [[] for _ in range(num_labels)]
        for i, label in enumerate(labels):
            means_by_label[label].append(means[i])
        return [np.mean(means_by_label[i], axis=0) for i in range(num_labels)]

    @staticmethod
    def _calculate_valid_clusters(cluster_means, team1_mean, team2_mean, similarity_threshold):
        valid_clusters = set()
        for i in range(len(cluster_means)):
            cluster_mean = cluster_means[i].reshape(1, -1)

            team1_dist = cosine_distances(cluster_mean, team1_mean).item()
            if team1_dist < similarity_threshold:
                continue

            team2_dist = cosine_distances(cluster_mean, team2_mean).item()
            if team2_dist < similarity_threshold:
                continue
            valid_clusters.add(i)
        return valid_clusters

    @staticmethod
    def read_clusters_csv(csv_filepath):

        dtype = [('Match', 'U200'),
                 ('Referee', int),
                 ('Team 1', int),
                 ('Team 2', int),
                 ('Goalkeeper A', int),
                 ('Goalkeeper B', int)]

        data = np.genfromtxt(csv_filepath, delimiter=',', dtype=dtype, names=True, encoding='utf-8')

        matches, referees, teams, goalkeepers = [], [], [], []
        for row in data:
            matches.append(row[0])
            referees.append(row[1])
            teams.append([row[2], row[3]])
            goalkeepers.append([row[4], row[5]])
        referees = np.asarray(referees)
        teams = np.asarray(teams)
        goalkeepers = np.asarray(goalkeepers)

        order = np.argsort(matches)
        return referees[order], teams[order], goalkeepers[order]

    def _calculate_clusters(self, cluster_method, clustering_distance, similarity_threshold, kit_clusters_csv=None):
        means = []
        for m in self.matches:
            if self.use_background:
                clustered_path = m.paths.clustered_bkg_means[cluster_method]
            else:
                clustered_path = m.paths.clustered_means[cluster_method]
            means.append(np.load(clustered_path, allow_pickle=True))

        referee_means, team_means, gk_means = [], [], []
        for i in range(self.num_matches):
            referee_means.append(means[i][REFEREE, :])
            for j in [TEAM_1, TEAM_2]:
                team_means.append(means[i][j, :])
            for j in [GOALKEEPER_A, GOALKEEPER_B]:
                gk_means.append(means[i][j, :])

        referee_means = np.asarray(referee_means)
        team_means = np.asarray(team_means)
        gk_means = np.asarray(gk_means)

        if kit_clusters_csv is None:
            referee_clusters = DBSCAN(eps=similarity_threshold,
                                      min_samples=1,
                                      metric=clustering_distance).fit(referee_means)
            referee_clusters = referee_clusters.labels_.astype(int)

            teams_clusters = DBSCAN(eps=similarity_threshold,
                                    min_samples=1,
                                    metric=clustering_distance).fit(team_means)
            team_clusters = teams_clusters.labels_.astype(int)

            gk_clusters = DBSCAN(eps=similarity_threshold,
                                 min_samples=1,
                                 metric=clustering_distance).fit(gk_means)
            gk_clusters = gk_clusters.labels_.astype(int)
        else:
            referee_clusters, team_clusters, gk_clusters = CombineMatchesDataset.read_clusters_csv(kit_clusters_csv)
            team_clusters = team_clusters.flatten()
            gk_clusters = gk_clusters.flatten()

        self.referee_clusters = referee_clusters
        self.num_referee_clusters = len(np.unique(self.referee_clusters))
        referee_means_by_cluster = CombineMatchesDataset._calculate_centroids(referee_means, self.referee_clusters,
                                                                              self.num_referee_clusters)

        self.num_team_clusters = len(np.unique(team_clusters))
        team_means_by_cluster = CombineMatchesDataset._calculate_centroids(team_means, team_clusters,
                                                                           self.num_team_clusters)
        team_clusters = team_clusters.reshape(self.num_matches, 2)
        self.team_clusters = [{TEAM_1: team_clusters[i, 0],
                               TEAM_2: team_clusters[i, 1]}
                              for i in range(self.num_matches)]

        self.num_gk_clusters = len(np.unique(gk_clusters))
        gk_means_by_cluster = CombineMatchesDataset._calculate_centroids(gk_means, gk_clusters, self.num_gk_clusters)
        gk_clusters = gk_clusters.reshape(self.num_matches, 2)
        self.gk_clusters = [{GOALKEEPER_A: gk_clusters[i, 0],
                             GOALKEEPER_B: gk_clusters[i, 1]}
                            for i in range(self.num_matches)]

        self.real_matches = {}
        for referee_cluster, team_cluster in zip(referee_clusters, team_clusters):
            # TODO: Check if matches colors repeat in the dataset split
            self.real_matches[hash((referee_cluster, hash_pair(team_cluster)))] = (referee_cluster, team_cluster)

        self.augmented_matches = {}
        match_counter = self.num_matches
        for team_comb in itertools.combinations(range(self.num_team_clusters), 2):
            team1_mean = team_means_by_cluster[team_comb[0]].reshape(1, -1)
            team2_mean = team_means_by_cluster[team_comb[1]].reshape(1, -1)

            valid_gk_clusters = CombineMatchesDataset._calculate_valid_clusters(gk_means_by_cluster, team1_mean,
                                                                                team2_mean, similarity_threshold)
            if len(valid_gk_clusters) == 0:
                continue

            valid_referee_clusters = CombineMatchesDataset._calculate_valid_clusters(referee_means_by_cluster,
                                                                                     team1_mean, team2_mean,
                                                                                     similarity_threshold)

            for referee_cluster in valid_referee_clusters:
                hashed_id = hash((referee_cluster, hash_pair(team_comb)))
                if hashed_id in self.real_matches:
                    continue

                self.augmented_matches[hashed_id] = {'team_comb': team_comb,
                                                     'referee': referee_cluster,
                                                     'match_id': match_counter,
                                                     'valid_gk_clusters': valid_gk_clusters}
                match_counter += 1

        self.matches_by_team_cluster = {c: set() for c in range(self.num_team_clusters)}
        self.team_combinations = []
        for match_id, team_cluster in enumerate(team_clusters):
            self.matches_by_team_cluster[team_cluster[0]].add(match_id)
            self.matches_by_team_cluster[team_cluster[1]].add(match_id)
            self.team_combinations.append(hash_pair(team_cluster))

        self.matches_by_gk_cluster = {c: set() for c in range(self.num_gk_clusters)}
        for match_id, gk_cluster in enumerate(gk_clusters):
            self.matches_by_gk_cluster[gk_cluster[0]].add(match_id)
            self.matches_by_gk_cluster[gk_cluster[1]].add(match_id)

        self.matches_by_referee_cluster = {c: set() for c in range(self.num_referee_clusters)}
        for match_id, referee_cluster in enumerate(self.referee_clusters):
            self.matches_by_referee_cluster[referee_cluster].add(match_id)

        self.permute_invariance_labels = [self.order_labels_by_cluster(m) for m in self.matches]

    def order_labels_by_cluster(self, match):
        ordered_labels = {REFEREE: REFEREE}
        if self.team_clusters[match.idx][TEAM_1] > self.team_clusters[match.idx][TEAM_2]:
            ordered_labels[TEAM_1] = TEAM_2
            ordered_labels[TEAM_2] = TEAM_1
        else:
            ordered_labels[TEAM_1] = TEAM_1
            ordered_labels[TEAM_2] = TEAM_2

        if self.gk_clusters[match.idx][GOALKEEPER_A] > self.gk_clusters[match.idx][GOALKEEPER_B]:
            ordered_labels[GOALKEEPER_A] = GOALKEEPER_B
            ordered_labels[GOALKEEPER_B] = GOALKEEPER_A
        else:
            ordered_labels[GOALKEEPER_A] = GOALKEEPER_A
            ordered_labels[GOALKEEPER_B] = GOALKEEPER_B
        return ordered_labels

    def make_permutation_invariant(self, match, data):
        order_label = self.permute_invariance_labels[match.idx]
        for player_patches in data:
            labels = [order_label[l] for l in player_patches.labels]
            player_patches.labels = np.asarray(labels, dtype=np.int64)

    def _sample_frames(self, match):
        sampled_frame_ids = {}
        for half in range(2):
            num_frames_to_sample = min(self.num_frames_per_half, len(match.valid_frames_ids[half]))
            sampled_frame_ids[half] = sorted(np.random.choice(match.valid_frames_ids[half],
                                                              num_frames_to_sample,
                                                              replace=False))
        return sampled_frame_ids

    def _load_frames(self, match, sampled_frame_ids):
        data = []

        total_num_patches = 0
        for half in range(2):
            for idx in sampled_frame_ids[half]:
                total_num_patches += match.categories_counts[half][idx].sum()

        # Loading patches from base match
        with tqdm(total=total_num_patches, desc='Loading patches progress', leave=True, position=0) as pbar:
            for half in range(2):
                indices, frame_paths, segmentations, homographies = [], [], [], []
                for idx in sampled_frame_ids[half]:
                    indices.append(DataIndex(match.idx, half, idx))
                    frame_paths.append(match.paths.frames[half, idx])
                    segmentations.append(match.segmented_players[half][idx])
                    homographies.append(match.homographies[half][idx])

                load_background = [self.use_background] * len(indices)
                center_coords_ = [self.center_coords] * len(indices)
                load_labels_ = [self.load_labels] * len(indices)

                with Pool(processes=self.num_processes) as pool:
                    for player_patches in pool.imap_unordered(_load_player_patches_from_frame,
                                                              zip(indices, frame_paths, segmentations, homographies,
                                                                  load_background, center_coords_, load_labels_)):
                        data.append(player_patches)
                        pbar.update(len(player_patches))
        data = sorted(data, key=lambda x: (x.match_idx, x.half, x.frame_idx))
        self.make_permutation_invariant(match, data)
        return data

    def _extract_patches_by_category(self, data):
        patches_by_category = {i: [] for i in range(self.num_classes)}
        for player_patches in data:
            for label, patch in zip(player_patches.labels, player_patches.patches):
                patches_by_category[label].append(patch)
        for i in range(self.num_classes):
            np.random.shuffle(patches_by_category[i])
        return patches_by_category

    def _remove_seen_match(self, match_idx):
        for cat_idx in [REFEREE, TEAM_1, TEAM_2, GOALKEEPER_A, GOALKEEPER_B]:
            if cat_idx == REFEREE:
                cluster = self.referee_clusters[match_idx]
                remaining_matches_by_cluster = self.remaining_matches_by_referee_cluster
            elif cat_idx in {TEAM_1, TEAM_2}:
                cluster = self.team_clusters[match_idx][cat_idx]
                remaining_matches_by_cluster = self.remaining_matches_by_team_cluster
            else:
                cluster = self.gk_clusters[match_idx][cat_idx]
                remaining_matches_by_cluster = self.remaining_matches_by_gk_cluster

            matches = remaining_matches_by_cluster.get(cluster, None)
            if matches is not None:
                matches.remove(match_idx)
                if len(matches) == 0:
                    remaining_matches_by_cluster.pop(cluster)

    def _sample_remaining_matches(self, remaining_matches_by_cluster, all_matches, cat_idx):

        if cat_idx in remaining_matches_by_cluster:
            matches_to_sample = remaining_matches_by_cluster[cat_idx]
            match_id = sample_from_set(matches_to_sample)
            self._remove_seen_match(match_id)
            not_seen = True
        else:
            matches_to_sample = all_matches[cat_idx]
            match_id = sample_from_set(matches_to_sample)
            not_seen = False

        return self.matches[match_id], not_seen

    def _sample_remaining_matches_by_class(self, valid_clusters, matches_by_cluster):
        valid_matches_ids = set()
        for c in valid_clusters:
            valid_matches_ids = valid_matches_ids.union(matches_by_cluster[c])
        valid_remaining_matches_ids = valid_matches_ids.intersection(self.remaining_matches)
        if len(valid_remaining_matches_ids) > 0:
            match_id = sample_from_set(valid_remaining_matches_ids)
            self._remove_seen_match(match_id)
            not_seen = True
        else:
            match_id = sample_from_set(valid_matches_ids)
            not_seen = False
        return self.matches[match_id], not_seen

    def start_epoch(self):
        if self.mix:
            self.remaining_augmented_matches = set(self.augmented_matches.keys())
        else:
            self.remaining_augmented_matches = set()

        self.remaining_matches_by_referee_cluster = copy.deepcopy(self.matches_by_referee_cluster)
        self.remaining_matches_by_team_cluster = copy.deepcopy(self.matches_by_team_cluster)
        self.remaining_matches_by_gk_cluster = copy.deepcopy(self.matches_by_gk_cluster)
        self.remaining_matches = set([i for i in range(self.num_matches)])

    def _calculate_augmented_matches_combinations(self, matches):
        referees = set(self.referee_clusters[m.idx] for m in matches)

        teams_by_match = [[self.team_clusters[m.idx][TEAM_1], self.team_clusters[m.idx][TEAM_2]] for m in matches]
        teams = set(t for teams in teams_by_match for t in teams)

        gks_by_match = [[self.gk_clusters[m.idx][GOALKEEPER_A], self.gk_clusters[m.idx][GOALKEEPER_B]] for m in matches]
        goalkeepers = set(g for goalkeepers in gks_by_match for g in goalkeepers)

        matches_by_referees_cluster = {r: set() for r in referees}
        matches_by_team_cluster = {t: set() for t in teams}
        matches_by_gk_cluster = {g: set() for g in goalkeepers}

        for idx, m in enumerate(matches):
            matches_by_referees_cluster[self.referee_clusters[m.idx]].add(idx)

            matches_by_team_cluster[self.team_clusters[m.idx][TEAM_1]].add(idx)
            matches_by_team_cluster[self.team_clusters[m.idx][TEAM_2]].add(idx)

            matches_by_gk_cluster[self.gk_clusters[m.idx][GOALKEEPER_A]].add(idx)
            matches_by_gk_cluster[self.gk_clusters[m.idx][GOALKEEPER_B]].add(idx)

        augmented_matches = []
        for team_comb in itertools.combinations(teams, 2):
            for referee_cluster in referees:

                hashed_id = hash((referee_cluster, hash_pair(team_comb)))
                if hashed_id not in self.remaining_augmented_matches:
                    continue

                augmented_match = self.augmented_matches.get(hashed_id, None)
                if augmented_match is None:
                    continue

                cluster_by_category = np.zeros(self.num_classes, dtype=int)
                matches_by_category = np.zeros(self.num_classes, dtype=int)

                for team_cluster, team_category in zip(team_comb, [TEAM_1, TEAM_2]):
                    cluster_by_category[team_category] = team_cluster
                    matches_by_category[team_category] = sample_from_set(matches_by_team_cluster[team_cluster])

                cluster_by_category[REFEREE] = referee_cluster
                matches_by_category[REFEREE] = sample_from_set(matches_by_referees_cluster[referee_cluster])

                valid_gk_clusters = goalkeepers.intersection(augmented_match['valid_gk_clusters'])
                if len(valid_gk_clusters) < 2:
                    continue

                for gk_category in [GOALKEEPER_A, GOALKEEPER_B]:
                    gk_cluster = sample_from_set(valid_gk_clusters)
                    valid_gk_clusters.remove(gk_cluster)

                    cluster_by_category[gk_category] = gk_cluster
                    matches_by_category[gk_category] = sample_from_set(matches_by_gk_cluster[gk_cluster])

                # Permutation Invariant
                if cluster_by_category[GOALKEEPER_A] > cluster_by_category[GOALKEEPER_B]:
                    cluster_by_category[GOALKEEPER_A], cluster_by_category[GOALKEEPER_B] = (
                        cluster_by_category[GOALKEEPER_B],
                        cluster_by_category[GOALKEEPER_A])
                    matches_by_category[GOALKEEPER_A], matches_by_category[GOALKEEPER_B] = (
                        matches_by_category[GOALKEEPER_B],
                        matches_by_category[GOALKEEPER_A])

                augmented_matches.append((augmented_match, cluster_by_category, matches_by_category))
                self.remaining_augmented_matches.remove(hashed_id)
        return augmented_matches

    def _find_match_labels(self, matches, matches_by_category, cluster_by_category, categories, clusters,
                           match_label_by_aug_label):

        for cat_idx in categories:
            match = matches[matches_by_category[cat_idx]]

            idx = 0 if cluster_by_category[cat_idx] == clusters[match.idx][categories[0]] else 1
            original_label = categories[idx]

            invariant_permutated = self.permute_invariance_labels[match.idx]
            match_label_by_aug_label[cat_idx] = invariant_permutated[original_label]

    def _calculate_match_labels(self, matches, matches_by_category, cluster_by_category):
        match_label_by_aug_label = np.zeros(self.num_classes, dtype=int)

        self._find_match_labels(matches,
                                matches_by_category,
                                cluster_by_category,
                                [TEAM_1, TEAM_2],
                                self.team_clusters,
                                match_label_by_aug_label)

        self._find_match_labels(matches,
                                matches_by_category,
                                cluster_by_category,
                                [GOALKEEPER_A, GOALKEEPER_B],
                                self.gk_clusters,
                                match_label_by_aug_label)
        return match_label_by_aug_label

    def _sample_by_augmented_match(self, augmented_match):
        cluster_by_category = np.zeros(self.num_classes, dtype=int)
        matches_by_category = np.zeros(self.num_classes, dtype=int)
        matches, not_seen = [], []

        # Sampling matches by team cluster
        for i, (team_cluster, cat_idx) in enumerate(zip(augmented_match['team_comb'], (TEAM_1, TEAM_2))):
            match, not_seen_ = self._sample_remaining_matches(self.remaining_matches_by_team_cluster,
                                                              self.matches_by_team_cluster,
                                                              team_cluster)
            matches.append(match)
            not_seen.append(not_seen_)

            cluster_by_category[cat_idx] = team_cluster
            matches_by_category[cat_idx] = i

        # Checking if it is necessary to sample another match for the REFEREE class
        referee_cluster = augmented_match['referee']
        for match_idx, match in enumerate(matches):
            if self.referee_clusters[match.idx] == referee_cluster:
                matches_by_category[REFEREE] = match_idx
                cluster_by_category[REFEREE] = self.referee_clusters[match.idx]
                break
        else:
            match, not_seen_ = self._sample_remaining_matches(self.remaining_matches_by_referee_cluster,
                                                              self.matches_by_referee_cluster,
                                                              referee_cluster)
            matches.append(match)
            not_seen.append(not_seen_)
            last_match_idx = len(matches) - 1
            matches_by_category[REFEREE] = last_match_idx
            cluster_by_category[REFEREE] = self.referee_clusters[match.idx]

        # Checking if it is necessary to sample another match for the GOALKEEPER classes
        valid_gk_clusters = copy.deepcopy(augmented_match['valid_gk_clusters'])
        augmented_gk = GOALKEEPER_A
        for (match_id, match), gk_category_idx in product(enumerate(matches), [GOALKEEPER_A, GOALKEEPER_B]):
            gk_cluster = self.gk_clusters[match.idx][gk_category_idx]
            if gk_cluster in valid_gk_clusters:
                matches_by_category[augmented_gk] = match_id
                cluster_by_category[augmented_gk] = gk_cluster
                valid_gk_clusters.remove(gk_cluster)

                if augmented_gk == GOALKEEPER_A:
                    augmented_gk = GOALKEEPER_B
                else:
                    break
        else:
            missing_gks = [GOALKEEPER_A, GOALKEEPER_B] if augmented_gk == GOALKEEPER_A else [GOALKEEPER_B]
            for missing_gk in missing_gks:
                match, not_seen_ = self._sample_remaining_matches_by_class(valid_gk_clusters,
                                                                           self.matches_by_gk_cluster)
                matches.append(match)
                not_seen.append(not_seen_)
                last_match_idx = len(matches) - 1
                matches_by_category[missing_gk] = last_match_idx

                gk_clusters = self.gk_clusters[match.idx]
                if gk_clusters[GOALKEEPER_A] in valid_gk_clusters:
                    cluster_by_category[missing_gk] = gk_clusters[GOALKEEPER_A]
                else:
                    cluster_by_category[missing_gk] = gk_clusters[GOALKEEPER_B]

        if cluster_by_category[GOALKEEPER_A] > cluster_by_category[GOALKEEPER_B]:
            cluster_by_category[GOALKEEPER_A], cluster_by_category[GOALKEEPER_B] = (cluster_by_category[GOALKEEPER_B],
                                                                                    cluster_by_category[GOALKEEPER_A])
            matches_by_category[GOALKEEPER_A], matches_by_category[GOALKEEPER_B] = (matches_by_category[GOALKEEPER_B],
                                                                                    matches_by_category[GOALKEEPER_A])
        return cluster_by_category, matches_by_category, matches, not_seen

    def _load_random_team_combination(self):
        if len(self.remaining_augmented_matches) > 0:

            random_augmented_match_id = sample_from_set(self.remaining_augmented_matches)
            aug_match = self.augmented_matches[random_augmented_match_id]
            self.remaining_augmented_matches.remove(random_augmented_match_id)

            cluster_by_category, matches_by_category, matches, not_seen = self._sample_by_augmented_match(aug_match)

            # Loading frames and player patches
            data, patches_by_category = [], []
            for match, not_seen_ in zip(matches, not_seen):
                sample_frame_ids = self._sample_frames(match)
                match_data = self._load_frames(match, sample_frame_ids)
                data.append(match_data)
                patches_by_category.append(self._extract_patches_by_category(match_data))
                if not_seen_:
                    self.data.extend(match_data)

            augmented_matches = [(aug_match, cluster_by_category, matches_by_category)]
            augmented_matches.extend(self._calculate_augmented_matches_combinations(matches))

            for aug_match, cluster_by_category, matches_by_category in augmented_matches:
                match_label_by_aug_label = self._calculate_match_labels(matches, matches_by_category,
                                                                        cluster_by_category)

                base_idx = random.choices(np.arange(len(matches)), k=1)[0]
                cat_counters = np.zeros(self.num_classes, dtype=int)
                for player_patches in data[base_idx]:
                    patches, coords, labels = [], [], []
                    for i, label in enumerate(player_patches.labels):
                        match_idx = matches_by_category[label]
                        match_label = match_label_by_aug_label[label]
                        if cat_counters[label] >= len(patches_by_category[match_idx][match_label]):
                            continue
                        patches.append(patches_by_category[match_idx][match_label][cat_counters[label]])
                        coords.append(player_patches.coords[i])
                        labels.append(label)
                        cat_counters[label] += 1
                    if len(patches) == 0:
                        continue
                    index = DataIndex(aug_match['match_id'],
                                      player_patches.half,
                                      player_patches.frame_idx)
                    self.data.append(PlayerPatches(index, patches, coords, labels))
        elif len(self.remaining_matches) > 0:
            for _ in range(5):
                if len(self.remaining_matches) == 0:
                    break
                match_id = sample_from_set(self.remaining_matches)
                self.remaining_matches.remove(match_id)
                match = self.matches[match_id]
                sample_frame_ids = self._sample_frames(match)
                self.data.extend(self._load_frames(match, sample_frame_ids))
        else:
            return

    def load_step_data(self):
        self.data = []
        for _ in range(self.num_random_comb_per_step):
            if self.num_remaining_matches() == 0:
                break
            self._load_random_team_combination()

        for p in self.data:
            p.coords = torch.as_tensor(p.coords)
            if self.load_labels:
                p.labels = torch.as_tensor(p.labels)
        return len(self.data)

    def load_all_data(self):
        self.data = []
        while self.num_remaining_matches() > 0:
            self._load_random_team_combination()

        for p in self.data:
            p.coords = torch.as_tensor(p.coords)
            if self.load_labels:
                p.labels = torch.as_tensor(p.labels)
        return len(self.data)

    def clear_epoch_data(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def num_remaining_matches(self):
        return len(self.remaining_augmented_matches) + len(self.remaining_matches)

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.data[idx].patches])

        coords = self.data[idx].coords
        if self.coord_transform:
            coords = self.coord_transform(coords)

        if self.num_data_outputs == 4:
            labels = self.data[idx].labels
            if self.label_transform:
                labels = self.label_transform(labels)
            return patches, coords, labels, self.data[idx].match_idx
        else:
            return patches, coords, self.data[idx].match_idx


def _load_player_patches_from_match(args):
    match_idx, match_path, args = args

    match = MatchDataset(match_path,
                         min_segmentation_score=args.min_segmentation_score,
                         min_calibration_confidence=args.min_calibration_confidence,
                         max_bb_area=args.max_bb_area,
                         touchline_dist=args.touchline_dist,
                         load_gt=args.load_gt,
                         use_background=args.use_background,
                         load_predictions=args.load_predictions,
                         only_players_on_field=args.only_players_on_field,
                         cluster_method=args.cluster_method)

    patches = []
    for half in range(2):
        num_frames_to_sample = min(args.num_frames_per_half, len(match.valid_frames_ids[half]))
        sampling_ids = np.random.choice(match.valid_frames_ids[half], num_frames_to_sample, replace=False)
        for idx in sampling_ids:
            index = DataIndex(match_idx, half, idx)
            frame_path = match.paths.frames[half, idx]
            segmentation = match.segmented_players[half][idx]
            homography = match.homographies[half][idx]

            patches.append(_load_player_patches_from_frame((index,
                                                            frame_path,
                                                            segmentation,
                                                            homography,
                                                            args.use_background,
                                                            args.center_coords,
                                                            args.load_labels)))
    return patches


class ValidationMatchesDataset(Dataset):
    def __init__(self, match_paths: List[Path], patch_transform=None, coord_transform=None, label_transform=None,
                 min_segmentation_score=0.65, min_calibration_confidence=0.75, max_bb_area=40_000, touchline_dist=3,
                 center_coords=True, num_processes=12, load_gt=False, use_background=False, load_predictions=False,
                 only_players_on_field=False, cluster_method=None, num_frames_per_half=5):

        if patch_transform is None:
            self.patch_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.patch_transform = patch_transform

        self.coord_transform = coord_transform
        self.use_background = use_background
        self.load_labels = load_gt or load_predictions

        if self.load_labels:
            self.num_data_outputs = 4
            self.label_transform = label_transform
        else:
            self.num_data_outputs = 3
            self.label_transform = None

        self.data = []
        self.center_coords = center_coords
        self.num_processes = num_processes
        self.num_frames_per_half = num_frames_per_half

        arguments = Dict({'min_segmentation_score': min_segmentation_score,
                          'min_calibration_confidence': min_calibration_confidence,
                          'max_bb_area': max_bb_area,
                          'touchline_dist': touchline_dist,
                          'load_gt': load_gt,
                          'use_background': use_background,
                          'load_predictions': load_predictions,
                          'only_players_on_field': only_players_on_field,
                          'cluster_method': cluster_method,
                          'center_coords': center_coords,
                          'load_labels': self.load_labels,
                          'num_frames_per_half': num_frames_per_half
                          })

        self.num_matches = len(match_paths)
        self.data = []

        match_paths = sorted(match_paths)
        matches_to_process = [(match_idx, match_path, arguments) for match_idx, match_path in enumerate(match_paths)]
        with tqdm(total=self.num_matches, desc="Loading all validation patches progress", leave=True,
                  position=0) as pbar:
            with Pool(processes=num_processes) as pool:
                for patches in pool.imap_unordered(_load_player_patches_from_match, matches_to_process):
                    self.data.extend(patches)
                    pbar.update(1)

        for p in self.data:
            p.coords = torch.as_tensor(p.coords)
            if self.load_labels:
                p.labels = torch.as_tensor(p.labels)
        self.data = sorted(self.data, key=lambda x: (x.match_idx, x.half, x.frame_idx))

    def clear_epoch_data(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.data[idx].patches])

        coords = self.data[idx].coords
        if self.coord_transform:
            coords = self.coord_transform(coords)

        if self.num_data_outputs == 4:
            labels = self.data[idx].labels
            if self.label_transform:
                labels = self.label_transform(labels)
            return patches, coords, labels, self.data[idx].match_idx
        else:
            return patches, coords, self.data[idx].match_idx
