from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy


class SequenceCountPenalty(nn.Module):
    def __init__(self, gamma: float, max_count_per_class: Dict[int, int], ignore_index=None):
        super(SequenceCountPenalty, self).__init__()
        self.gamma = gamma
        self.max_count_per_class = max_count_per_class
        self.ignore_index = ignore_index

    def forward(self, predictions, mask=None):
        predicted_labels = predictions.argmax(axis=2)

        if mask is not None:
            predicted_labels[mask] = self.ignore_index

        result = 0
        for penalized_class, max_count in self.max_count_per_class.items():
            occurrence_counts = torch.sum(predicted_labels == penalized_class, dim=1) - max_count
            mask = occurrence_counts > 0
            result += torch.sum(occurrence_counts[mask])

        return self.gamma * result


def swap(tensor, value1, value2, clone=True):
    indices1 = (tensor == value1)
    indices2 = (tensor == value2)
    if clone:
        tensor = tensor.clone()
    tensor[indices1], tensor[indices2] = value2, value1
    return tensor


class PermutationInvariant:
    def __init__(self, num_classes: int = 5, ignore_index=None, device=None):
        self.ignore_index = ignore_index
        if ignore_index is not None:
            num_classes += 1
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        if device is not None:
            self.accuracy = self.accuracy.to(device)

    def __call__(self, groundtruth, predictions, match_ids, mask=None):
        groundtruth = groundtruth.reshape(predictions.shape[:2])

        predictions = predictions.argmax(axis=2)
        if mask is not None:
            predictions[mask] = self.ignore_index

        team_swap = swap(groundtruth, 1, 2)
        gk_swap = swap(groundtruth, 3, 4)

        for match_id in np.unique(match_ids):
            idx = (match_id == match_ids)
            pred = predictions[idx, ...]

            normal_acc = self.accuracy(groundtruth[idx, ...], pred)
            team_swap_acc = self.accuracy(team_swap[idx, ...], pred)
            gk_swap_acc = self.accuracy(gk_swap[idx, ...], pred)

            if team_swap_acc > normal_acc:
                if gk_swap_acc > normal_acc:
                    groundtruth[idx, ...] = swap(team_swap[idx, ...], 3, 4, False)
                else:
                    groundtruth[idx, ...] = team_swap[idx, ...]
            elif gk_swap_acc > normal_acc:
                groundtruth[idx, ...] = gk_swap[idx, ...]
        return groundtruth.view(-1)
