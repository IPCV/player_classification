from abc import ABC
from multiprocessing import Pool
from pathlib import Path
from typing import List

import numpy as np
import torch
from kitman.data import PlayerPatches, Players
from kitman.field_calibration import DIM_IMAGE
from kitman.field_calibration import calculate_player_position, CENTER_COORDS
from kitman.field_calibration import to_coords
from kitman.regions import get_masked_patch, get_patch
from kitman.segmentation import SegmentedPlayer
from kitman.snv3 import SequenceIndex, SequencePaths
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class SequencesDataset(ABC):
    def __init__(self, sequence_paths: List[Path], load_gt=True, use_background=True, segmentations_fname=None,
                 cluster_method=None):
        self.num_sequences, self.total_num_frames = len(sequence_paths), 0
        self.paths, self.homographies, self.segmented_players, self.num_original_segmented_frames = [], [], [], []
        self.valid_frames_ids = {}
        self.use_background = use_background
        self.cluster_method = cluster_method

        for seq_idx, sequence_path in enumerate(sequence_paths):
            path = SequencePaths(sequence_path)
            self.paths.append(path)

            homography = np.load(path.homographies, allow_pickle=True)
            self.homographies.append(homography)

            if segmentations_fname is None:
                segmentations_path = path.segmentations if load_gt else path.clustered_segmentations[cluster_method]
            else:
                segmentations_path = path.sequence.joinpath(segmentations_fname)

            segmented_players = SegmentedPlayer.load(segmentations_path)

            self.segmented_players.append(segmented_players)
            self.valid_frames_ids[seq_idx] = [idx for idx, segs in enumerate(segmented_players) if len(segs) > 0]
            self.total_num_frames += len(self.valid_frames_ids[seq_idx])
            self.num_original_segmented_frames.append(len(segmented_players))

    def __len__(self):
        return self.total_num_frames

    @staticmethod
    def load_labels(sequence_paths: List[Path], load_gt=True, use_background=True, filename=None):
        labels = []
        for seq_idx, sequence_path in enumerate(sequence_paths):
            path = SequencePaths(sequence_path)

            if filename is not None:
                segmentations_path = path.sequence.joinpath(filename)
            elif load_gt:
                segmentations_path = path.segmentations
            elif use_background:
                segmentations_path = path.clustered_segmentations_bkg
            else:
                segmentations_path = path.clustered_segmentations

            segmented_players = SegmentedPlayer.load(segmentations_path)
            for frame_idx in range(len(segmented_players)):
                for p in segmented_players[frame_idx]:
                    if p.class_id < 5:
                        labels.append(p.class_id)
        return labels


def _load_player_patches_from_frame(args):
    idx, frame_path, segmented_players, homography, load_background = args
    frame = io.imread(frame_path)

    patches, coords, labels = [], [], []
    for p in segmented_players:
        if p.class_id < 5:
            if load_background:
                patches.append(get_patch(frame, p.bb))
            else:
                patches.append(get_masked_patch(frame, p.bb, p.mask_cnts))
            xy = calculate_player_position(p.bb, homography, frame.shape)
            coords.append(to_coords(xy, DIM_IMAGE))
            labels.append(p.class_id)
    coords = np.asarray(coords, dtype=np.float32)
    return PlayerPatches(idx, patches, coords, labels)


class PlayerPatchesDataset(SequencesDataset, Dataset):
    def __init__(self, sequence_paths: List[Path], patch_transform=None, coord_transform=None, label_transform=None,
                 center_coords=True, num_processes=12, load_gt=True, use_background=True, segmentations_fname=None,
                 cluster_method=None):

        super().__init__(sequence_paths, load_gt, use_background, segmentations_fname, cluster_method)

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

        self.label_transform = label_transform

        self.num_data_outputs = 3

        self.load_labels = True

    def _load_player_patches(self, num_processes, center_coords):
        self.data = []

        indices, frame_paths, segmented_players, homographies = [], [], [], []
        for seq_idx, path in enumerate(self.paths):
            for frame_idx in self.valid_frames_ids[seq_idx]:
                indices.append(SequenceIndex(seq_idx, frame_idx))
                frame_paths.append(path.frames[frame_idx])
                segmented_players.append(self.segmented_players[seq_idx][frame_idx])
                homographies.append(self.homographies[seq_idx][frame_idx])
        load_background = [self.use_background] * self.total_num_frames

        with tqdm(total=self.total_num_frames, desc='Loading player patches progress', leave=True, position=0) as pbar:
            with Pool(processes=num_processes) as pool:
                for player_patches in pool.imap_unordered(_load_player_patches_from_frame,
                                                          zip(indices, frame_paths, segmented_players, homographies,
                                                              load_background)):
                    self.data.append(player_patches)
                    pbar.update(1)

        for p in self.data:
            if center_coords:
                p.coords = torch.as_tensor(p.coords - CENTER_COORDS, dtype=torch.float32)
            else:
                p.coords = torch.as_tensor(np.asarray(p.coords, dtype=np.float32))
            p.labels = torch.as_tensor(np.asarray(p.labels, dtype=np.int64))
        self.data = sorted(self.data, key=lambda x: (x.seq_idx, x.frame_idx))

    def __len__(self):
        return self.total_num_frames

    def __getitem__(self, idx):
        patches = torch.stack([self.patch_transform(p) for p in self.data[idx].patches])

        coords = self.data[idx].coords
        if self.coord_transform:
            coords = self.coord_transform(coords)

        labels = self.data[idx].labels
        if self.label_transform:
            labels = self.label_transform(labels)

        return patches, coords, labels

    def save_means(self, means, filename=None):
        for seq_idx, path in enumerate(self.paths):
            if filename is not None:
                means_path = path.sequence.joinpath(filename)
            elif self.use_background:
                means_path = path.clustered_bkg_means[self.cluster_method]
            else:
                means_path = path.clustered_means[self.cluster_method]
            np.save(means_path, means)

    def save_predictions(self, predictions, filename=None):

        idx = 0
        for seq_idx, path in enumerate(self.paths):
            segmentations = []
            for frame_idx in range(self.num_original_segmented_frames[seq_idx]):
                frame_data = {'boxes': [], 'masks': [], 'class_ids': [], 'scores': []}
                if frame_idx in self.valid_frames_ids[seq_idx]:
                    for p in self.segmented_players[seq_idx][frame_idx]:
                        if p.class_id < 5:
                            frame_data['boxes'].append(p.bb)
                            frame_data['masks'].append(p.mask_cnts)
                            frame_data['scores'].append(p.score)
                            frame_data['class_ids'].append(predictions[idx])
                            idx += 1
                segmentations.append(frame_data)

            if filename is not None:
                predictions_path = path.sequence.joinpath(filename)
            elif self.use_background:
                predictions_path = path.clustered_segmentations_bkg[self.cluster_method]
            else:
                predictions_path = path.clustered_segmentations[self.cluster_method]
            np.save(predictions_path, segmentations)


class PlayersDataset(SequencesDataset):
    def __init__(self, sequence_paths: List[Path], center_coords=True, load_gt=True, use_background=True,
                 segmentations_fname=None, cluster_method=None):

        super().__init__(sequence_paths, load_gt, use_background, segmentations_fname, cluster_method)

        self.data = []
        for seq_idx, path in enumerate(self.paths):
            frame = None

            for frame_idx in self.valid_frames_ids[seq_idx]:
                coords, labels = [], []
                if frame is None:
                    frame = io.imread(path.frames[frame_idx])

                homography = self.homographies[seq_idx][frame_idx]

                for p in self.segmented_players[seq_idx][frame_idx]:
                    if p.class_id < 5:
                        xy = calculate_player_position(p.bb, homography, frame.shape)
                        coords.append(to_coords(xy, DIM_IMAGE))
                        labels.append(p.class_id)
                coords = np.asarray(coords, dtype=np.float32)
                if center_coords:
                    coords -= CENTER_COORDS

                idx = SequenceIndex(seq_idx, frame_idx)
                self.data.append(Players(idx, coords, labels))

        self.data = sorted(self.data, key=lambda x: (x.seq_idx, x.frame_idx))

    def __len__(self):
        return self.total_num_frames

    def __getitem__(self, idx):
        return self.data[idx]
