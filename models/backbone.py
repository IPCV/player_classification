import os
from typing import BinaryIO, Union, IO

import torch
import torchvision.models as models
from torch import Tensor
from torch import nn
from typing_extensions import TypeAlias

from .position_encoding import PositionalEncoding1D

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]


class Backbone(nn.Module):
    def __init__(self, name: str, dim: int = 128, weights: FILE_LIKE = None):
        super().__init__()
        if name == 'mobilenet_v3_small':
            backbone = models.mobilenet_v3_small(weights=None)
            classifier = list(backbone.classifier.children())[:3]
            classifier.append(nn.Linear(1024, dim))
            backbone.classifier = nn.Sequential(*classifier)
        elif name == 'mobilenet_v3_large':
            backbone = models.mobilenet_v3_large(weights=None)
            classifier = list(backbone.classifier.children())[:3]
            classifier.append(nn.Linear(1280, dim))
            backbone.classifier = nn.Sequential(*classifier)
        else:
            backbone = models.resnet50(weights=None)
            backbone.fc = nn.Linear(2048, dim)

        self.backbone = backbone
        self.dim = dim

        self.load_weights(weights)

    def load_weights(self, weights: FILE_LIKE = None):
        if weights:
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = self.backbone(x)
        return x


class Joiner(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.backbone = backbone

        if args.use_player_coords and args.use_positional_encoding:
            self.position_encoding = PositionalEncoding1D(2)
        else:
            self.position_encoding = None

        self.patch_encoding = PositionalEncoding1D(args.dim) if args.use_patch_encoding else None

        if args.use_modality_encoding:
            self.modality_embedding_patch = nn.Parameter(torch.randn(args.dim), requires_grad=True)
            if args.use_player_coords:
                self.modality_embedding_coords = nn.Parameter(torch.randn(2), requires_grad=True)

        self.use_position_encoding = self.position_encoding is not None
        self.use_patch_encoding = self.patch_encoding is not None
        self.use_modality_encoding = args.use_modality_encoding
        self.use_player_coords = args.use_player_coords
        self.batch_first = args.batch_first

    def apply_positional_encoding(self, embeddings: Tensor, encoding: PositionalEncoding1D):
        positional_encodings = encoding(embeddings)
        if not self.batch_first:
            positional_encodings = positional_encodings.transpose(0, 1)
        return embeddings + positional_encodings

    def forward(self, players: Tensor, coords: Tensor):

        embeddings = torch.stack([self.backbone(p) for p in players])
        if self.use_patch_encoding:
            embeddings = self.apply_positional_encoding(embeddings, self.patch_encoding)

        if self.use_modality_encoding:
            embeddings += self.modality_embedding_patch

        if self.use_player_coords:
            if self.use_position_encoding:
                coords = self.apply_positional_encoding(coords, self.position_encoding)

            if self.use_modality_encoding:
                coords += self.modality_embedding_coords
            embeddings = torch.stack([torch.cat((e, c), 1) for e, c in zip(embeddings, coords)])
        return embeddings


def build_backbone(args):
    backbone = Backbone(args.model.name, args.model.dim, args.weights)
    model = Joiner(backbone, args.model)
    return model
