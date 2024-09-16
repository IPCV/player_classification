import os
from typing import Optional, BinaryIO, Union, IO

from torch import nn, Tensor
import torch
from .backbone import build_backbone
from .transformer import build_transformer
from typing_extensions import TypeAlias

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]


class TransKit(nn.Module):
    def __init__(self, backbone, transformer, weights: FILE_LIKE = None):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.load_weights(weights)

    def forward(self, players: Tensor, coords: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        features = self.backbone(players, coords)
        outputs = self.transformer(features, src_mask, src_key_padding_mask)
        return outputs

    def load_weights(self, weights: FILE_LIKE = None):
        if weights:
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['model_state_dict'])


def build(args):
    backbone = build_backbone(args.backbone)
    transformer = build_transformer(args.transformer.model)
    return TransKit(backbone, transformer)
