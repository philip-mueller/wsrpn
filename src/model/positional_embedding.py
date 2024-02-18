from math import ceil
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SinCosEmbedding2D(nn.Module):
    """Source: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py"""
    def __init__(self, channels: int, **kwargs):
        """
        :param channels: Channel dimension of the tensor.
        """
        super(SinCosEmbedding2D, self).__init__()
        channels = ceil(channels / 4) * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.channels = channels
        self.cached_enc = None

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Add positional encoding to the tensor.
        :param tensor: A 4d tensor of size (b, w, h, ch)
        """
        assert tensor.ndim == 4, "Tensor must be 4d"

        device = tensor.device

        if self.cached_enc is None or self.cached_enc.shape != tensor.shape:
            self.cached_enc = None
            b, w, h, c = tensor.shape
            pos_x = torch.arange(w, device=device).float()
            pos_y = torch.arange(h, device=device).float()
            sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
            sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
            emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
            emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
            emb = torch.zeros((w, h, self.channels * 2), device=device).type(tensor.type())
            emb[:, :, : self.channels] = emb_x
            emb[:, :, self.channels: 2 * self.channels] = emb_y

            self.cached_enc = emb[None, :, :, :c].repeat(b, 1, 1, 1)
        return self.cached_enc.to(device)


class LearnableEmbedding2D(nn.Module):
    def __init__(self, size: Tuple[int, int, int], **kwargs):
        """
        :param size: Tuple of (width, height, channels)
        """
        super(LearnableEmbedding2D, self).__init__()
        self.register_parameter("emb", nn.Parameter(torch.randn(size)))

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Add positional encoding to the tensor.
        :param tensor: A 4d tensor of size (b, w, h, ch)
        """
        assert tensor.ndim == 4, "Tensor must be 4d"
        return self.emb.repeat(tensor.shape[0], 1, 1, 1)


POSITIONAL_EMBEDDINGS_2D = {
    "sin_cos": SinCosEmbedding2D,
    "learnable": LearnableEmbedding2D,
}


def positional_embedding_2d(embed_type: str, **kwargs) -> nn.Module:
    return POSITIONAL_EMBEDDINGS_2D[embed_type](**kwargs)
