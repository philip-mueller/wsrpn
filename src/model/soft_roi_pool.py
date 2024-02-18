from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
#import torch_scatter

from src.model.model_components import MLP


class BBoxMLP(nn.Module):
    def __init__(self, num_hidden_layers, d_in: int, d_hidden: int,
                 scale_range: Tuple[float, float] = (0.0, 1.0),
                 use_bn=False, use_ratios=True, dropout=0.0) -> None:
        super().__init__()
        num_params = 4
        self.mlp = MLP(
            num_hidden_layers,
            d_in=d_in, d_hidden=d_hidden, d_out=num_params, use_bn=use_bn, dropout=dropout, dropout_last_layer=False)
        self.min_scale, self.max_scale = scale_range
        self.use_ratios = use_ratios

    def forward(self, x: Tensor, reference_positions=None, pos_emb=None) -> Tensor:
        """
        # we want value in ranges [0, 1]
        # - for x, y 0.5 is the center of the image
        # - for w, h 1.0 means covering the whole image
        :param x: ROI features (N x K x d)
        :param reference_positions: (N x K x 2)
        :return: roi_params (N x K x 4 or 5)
        """
        if pos_emb is not None:
            x = x + pos_emb
        roi_params = self.mlp(x)  # (N x K x 4)

        if self.use_ratios:
            area = self.min_scale + torch.sigmoid(roi_params[:, :, 2]) * (self.max_scale - self.min_scale)
            sqrt_ratio = torch.exp(roi_params[:, :, 3]).sqrt()
            sqrt_area = area.sqrt()
            w = sqrt_area * sqrt_ratio
            h = sqrt_area / sqrt_ratio
            sigma = torch.stack([w, h], dim=-1)
        else:
            sigma = self.min_scale + torch.sigmoid(roi_params[:, :, 2:]) * (self.max_scale - self.min_scale)

        if reference_positions is None:
            mu = torch.sigmoid(roi_params[:, :, :2])
        else:
            offsets = torch.sigmoid(roi_params[:, :, :2]) - 0.5
            offsets = offsets * self.max_scale
            mu = reference_positions + offsets

        return torch.cat([mu, sigma], dim=-1)


def get_sample_grid(H: int, W: int, device, dtype) -> Tensor:
    # (H x W)
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype),
                          indexing='ij')
    # (H x W x 2)
    sampled_grid = torch.stack([x, y], dim=-1)
    # consider pixel centers instead of left-upper position
    sampled_grid += 0.5
    # normalize positions into range [0, 1]
    sampled_grid[:, :, 0] /= W
    sampled_grid[:, :, 1] /= H
    return sampled_grid


def generalized_gauss_1d_log_pdf(mu: Tensor, sigma: Tensor, sampled_grid: Tensor,
                                 beta: float = 2) -> Tensor:
    """
    :param mu: (N x K)
    :param sigma: (N x K)
    :param sampled_grid: Sampled points (P) where P is the number of sampled points of the Gaussian pdf
    :return (N x K x P)
    """
    assert len(sampled_grid.shape) == 1
    assert len(mu.shape) == 2
    assert len(sigma.shape) == 2
    # (unnormalized) log pdf = -0.5*((x-mu)/sigma)^2
    # log_pdf = - (1 / beta) * (
    #     (sampled_grid[None, None] - mu[:, :, None]) / sigma[:, :, None]
    # ).pow(beta)
    log_pdf = -(
        (sampled_grid[None, None] - mu[:, :, None]).abs() / sigma[:, :, None]
    ).pow(beta)
    return log_pdf


def separable_generalized_gaussian_pdf(roi_params: Tensor, sampled_grid: Tensor,
                                       beta: float = 2) -> Tensor:
    """
    :param roi_params: (N x K x 4)
    :param sampled_grid: (... x 2)
    :return: (N x K x ...)
    """
    N, K, _ = roi_params.shape
    *dims, _ = sampled_grid.shape
    sampled_grid = sampled_grid.view(-1, 2)  # (... x 2)
    mu = roi_params[:, :, :2]  # (N x K x 2)
    sigma = roi_params[:, :, 2:]  # (N x K x 2)
    # compute x and y Gaussian pdf's independently (in log-space and non-normalized)
    log_scores_x = generalized_gauss_1d_log_pdf(mu[..., 0], sigma[..., 0],
                                                sampled_grid[..., 0], beta)  # (N x K x ...)
    log_scores_y = generalized_gauss_1d_log_pdf(mu[..., 1], sigma[..., 1],
                                                sampled_grid[..., 1], beta)  # (N x K x ...)
    # combine them in log space (multiplication in prob space)
    log_scores = log_scores_x + log_scores_y  # (N x K x ...)

    # Normalize to max value = 1
    scores = torch.exp(log_scores)
    probs = scores / (scores.max(-1, keepdim=True).values + 1e-12)

    return probs.view(N, K, *dims)


class SoftRoiPool(nn.Module):
    def __init__(self, beta: float = 2, kernel=separable_generalized_gaussian_pdf, sp_hard_threshold=0.5):
        super().__init__()
        self.kernel = partial(kernel, beta=beta)
        self.sp_hard_threshold = sp_hard_threshold

    def forward(self, x: Tensor, roi_params: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: Featue map of image (N x H x W x d) or the superpixels (N x n_sup x d)
        :param roi_params: Parameters of bounding boxes (N x n_roi_tokens x 4) for n_roi_tokens boxes.
             For Gaussian this is (\mu_x, \mu_y, \sigma_x, \sigma_y) each in the range of [0, 1]
        :return roi_features, roi_maps
            - roi_features: Pooled features of each roi (N x n_roi_tokens x d)
            - roi_maps: Attention maps of each roi (N x n_roi_tokens x H x W) or (N x n_roi_tokens x n_sup)
        """
        N, H, W, d = x.shape
        norm_factor = H * W
        N, n_roi_tokens, _ = roi_params.shape

        # Compute kernel on sampling grid
        sampled_grid = get_sample_grid(H, W, device=x.device, dtype=x.dtype)  # (H x W x 2)
        roi_map = self.kernel(roi_params, sampled_grid)  # (N x n_roi_tokens x H x W)

        # Batched matrix multiplication and normalize
        flat_roi_map = roi_map.view(N, n_roi_tokens, -1)  # (N x n_roi_tokens x (H*W)) or (N x n_roi_tokens x n_sup)
        flat_features = x.view(N, -1, d)  # (N x (H*W) x d) or (N x n_sup x d)
        roi_features = torch.einsum('nrf,nfd->nrd', flat_roi_map, flat_features)  # (N x n_roi_tokens x d)
        roi_features /= norm_factor

        return roi_features, roi_params, roi_map
