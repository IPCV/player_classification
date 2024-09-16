from typing import Optional, Union, Callable

import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules import Module
from torch.nn.modules.normalization import LayerNorm


class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_classes: int = 5,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.decoder = Linear(d_model, num_classes)

        self._reset_parameters()

        self.d_model = d_model
        self.num_classes = num_classes
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(memory)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device='cpu') -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


def build_transformer(args):
    return Transformer(d_model=args.dim,
                       nhead=args.num_heads,
                       num_encoder_layers=args.num_encoder_layers,
                       num_classes=args.num_classes,
                       dim_feedforward=args.dim_feedforward,
                       dropout=args.dropout,
                       norm_first=args.norm_first,
                       batch_first=args.batch_first)
