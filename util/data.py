from math import ceil

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.datasets import ImageFolder


class ImageFolders(Dataset):
    def __init__(self, match_paths, transform=None):
        self.match_folders = [ImageFolder(root=m, transform=transform) for m in match_paths]
        self.num_images = [len(d) for d in self.match_folders]
        self.total_num_images = np.sum(self.num_images)

        self.cdf = np.cumsum(self.num_images, dtype=np.int64)

    def __len__(self):
        return self.total_num_images

    def __getitem__(self, idx):
        folder_idx = np.searchsorted(self.cdf, idx)
        if folder_idx > 0:
            idx = int(idx - self.cdf[folder_idx - 1]) - 1
        return self.match_folders[folder_idx][idx]


class RandomSeparateFolderSampler(Sampler):

    def __init__(self, image_folders, batch_size, drop_last=True):
        super().__init__(image_folders)
        self.num_folders = len(image_folders.match_folders)
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            self.num_batches = [n // batch_size for n in image_folders.num_images]
        else:
            self.num_batches = [ceil(n / batch_size) for n in image_folders.num_images]
        self.total_num_batches = sum(self.num_batches)

        self.batch_count = np.zeros(self.num_folders, dtype=np.int32)
        self.remaining_folders = set([i for i in range(self.num_folders)])

        self.indices = []
        for i in range(self.num_folders):
            indices = np.arange(image_folders.num_images[i])
            if i > 0:
                indices += image_folders.cdf[i - 1] + 1
            np.random.shuffle(indices)
            self.indices.append(indices)

    def __iter__(self):
        while len(self.remaining_folders) > 0:
            folder_indices = sorted(self.remaining_folders)
            histogram = np.asarray([self.num_batches[idx] - self.batch_count[idx] for idx in folder_indices])

            cdf = np.cumsum(histogram, dtype=np.float64)
            cdf /= cdf[-1]

            sampled_idx = np.searchsorted(cdf, np.random.rand(1)[0])
            idx = folder_indices[sampled_idx]

            self.batch_count[idx] += 1
            if self.batch_count[idx] == self.num_batches[idx]:
                self.remaining_folders.remove(idx)

            start_idx = self.batch_size * (self.batch_count[idx] - 1)
            end_idx = min(self.batch_size * self.batch_count[idx], len(self.indices[idx]))
            yield self.indices[idx][start_idx:end_idx]

    def __len__(self):
        return self.total_num_batches


def _last_dim_padding(num_dim: int, padding_length: int):
    pad = 2 * num_dim * [0]
    pad[-1] = padding_length
    return tuple(pad)


def _pad_last_dim(tensor: Tensor, padding_length: int, value: int = 0):
    pad = _last_dim_padding(tensor.dim(), padding_length)
    return F.pad(input=tensor, pad=pad, mode='constant', value=value)


class CollateFrames:
    def __init__(self, num_data_outputs: int, num_classes: int = None, padding_token: int = -1):
        self.num_data_outputs = num_data_outputs
        self.num_classes = num_classes
        self.load_labels = num_classes is not None
        if self.load_labels:
            self.matches_idx = 3 if num_data_outputs == 4 else None
        else:
            self.matches_idx = 2 if num_data_outputs == 3 else None
        self.load_matches_indices = self.matches_idx is not None
        self.padding_token = padding_token

    def __call__(self, data):
        lengths = np.asarray([len(d[0]) for d in data])
        padding_lengths = np.max(lengths) - lengths

        players, coords, labels, match_ids = [], [], [], []
        for datum, pad_length in zip(data, padding_lengths):
            players_batch, coords_batch = datum[:2]
            players.append(_pad_last_dim(players_batch, pad_length))
            coords.append(_pad_last_dim(coords_batch, pad_length))

            if self.load_labels:
                labels_batch = datum[2]
                labels.append(_pad_last_dim(labels_batch, pad_length, self.padding_token))

            if self.load_matches_indices:
                match_ids.append(datum[self.matches_idx])

        players = torch.stack(players)
        coords = torch.stack(coords)

        batch = [players, coords]
        if self.load_labels:
            labels = torch.cat(labels, 0)
            batch.append(labels)

        if self.load_matches_indices:
            match_ids = np.asarray(match_ids)
            batch.append(match_ids)

        batch.append(lengths)

        return tuple(batch)


def make_seq_mask(lengths):
    batch_size, sequence_size = len(lengths), np.max(lengths)
    seq_mask = torch.zeros((batch_size, sequence_size), dtype=torch.bool)
    for i, l in enumerate(lengths):
        seq_mask[i, l:] = True
    return seq_mask


def unmask_flat_sequences(sequences, lengths):
    sequence_size = np.max(lengths)
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.cpu().numpy()
    return [sequences[idx:idx + lengths[i]] for i, idx in enumerate(range(0, len(sequences), sequence_size))]
