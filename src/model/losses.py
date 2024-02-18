from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SupConPerClassLoss(nn.Module):
    # see https://github.com/ivanpanshin/SupCon-Framework/blob/5981e39837b1964f9bedfc30ee0da689d8fbc1b4/tools/losses.py
    def __init__(
        self,
        temperature: float = 0.07,
        pos_alignment_weight: float = 1.0,
        neg_alignment_weight: float = 1.0,
        normalize_weights: bool = True,
        ignore_no_label: bool = False,
        neg_from_all: bool = False
    ) -> None:
        """
        :param temperature: Temperature scaling parameter
        :param pos_alignment_weight: Weights for samples with positive targets
                                     (for the specific class)
        :param neg_alignment_weight: Weights for samples with negative targets
                                     (for the specific class)
        :param normalize_weights: True -> weights are interpreted as relative
                                  weights, False -> weights are absolute weights
        :param ignore_no_label: If True, do not compute this loss for the
                                no-finding label (last index in C dimension)
        :param neg_from_all: If True -> also use other class features as negatives
        """
        super(SupConPerClassLoss, self).__init__()
        self.temperature = temperature
        self.neg_alignment_weight = neg_alignment_weight
        self.pos_alignment_weight = pos_alignment_weight
        self.normalize_weights = normalize_weights
        self.ignore_no_label = ignore_no_label
        self.neg_from_all = neg_from_all

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """
        :param embeddings: (N x C x d) where C includes no-finding
        :param labels: (N x C) where C includes no-finding
        :return:
        """
        if self.ignore_no_label:
            embeddings = embeddings[:, :-1, :]
            labels = labels[:, :-1]
        N, C, d = embeddings.shape

        embeddings = F.normalize(embeddings, dim=-1)  # (N x C x d)
        embeddings = einops.rearrange(embeddings, 'n c d -> c n d')  # (C x N x d)
        scores = torch.bmm(embeddings, embeddings.transpose(-1, -2))  # (C x N x N)
        scores = scores / self.temperature

        pos_mask: Tensor = labels[:, :, None] == labels.T[None, :, :]  # (N x C x N)
        # pos_mask: Tensor = (labels[:, :, None] * labels.T[None, :, :]).bool()  # (N x C x N)
        pos_mask = einops.rearrange(pos_mask, 'i c j -> c i j')  # (C x N x N)

        pos_mask.diagonal(dim1=-2, dim2=-1)[...] = False  # mask i=j (same samples)
        pos_mask = F.normalize(pos_mask.to(dtype=scores.dtype), p=1.0, dim=-1)
        pos_scores = (pos_mask * scores).sum(-1)  # (C x N)
        if self.neg_from_all:
            emb_from_classes = einops.rearrange(embeddings, 'c n d -> (c n) d')  # ((C*N) x d)
            neg_scores = torch.mm(emb_from_classes, emb_from_classes.T)  # ((C*N) x (C*N))
            # (C x N x (C*N))
            neg_scores = einops.rearrange(neg_scores, '(c1 n1) (c2 n2) -> c1 n1 (c2 n2)', c1=C, c2=C)
        else:
            neg_scores = scores  # (C x N x N)
            neg_scores.diagonal(dim1=-2, dim2=-1)[:, :] = float('-inf')  # mask i=j for softmax

        sample_class_loss = - pos_scores + torch.logsumexp(neg_scores, dim=-1)  # (C x N)
        if self.pos_alignment_weight != 1.0 or self.neg_alignment_weight != 1.0:
            sample_class_weights = sample_class_loss.new_full(
                sample_class_loss.shape, self.neg_alignment_weight)  # (C x N)
            sample_class_weights[labels.T] = self.pos_alignment_weight  # (C x N)
            if self.normalize_weights:
                sample_loss = ((sample_class_weights * sample_class_loss).sum(0)
                               / sample_class_weights.sum(0))  # (N)
            else:
                sample_loss = (sample_class_weights * sample_class_loss).mean(0)  # (N)
            loss = sample_loss.mean()  # (1)
        else:
            loss = sample_class_loss.mean()  # (1)

        return loss


class RoiPatchClassConsistencyLoss(nn.Module):
    """
    Teacher-Student Loss between Patch classes and ROI classes
    using KL divergence losses
    """
    def __init__(self,
                 cls_aggregate_mode='MIL',
                 sg_patch_features=True,
                 sg_roi_features=False,
                 exclusive_classes=True,
                 pos_class_only=False,
                 ignore_nofind=False):
        """
        :param cls_aggregate_mode: How to aggregate cls probs for each patch
                                   from multiple ROIs assigned to that patch
                                   MIL -> noisy OR aggregation
        :param sg_patch_features: If True -> stop-gradient to patch_cls_probs
        :param sg_roi_features: If True -> stop-gradient to roi_cls_probs
        """

        super(RoiPatchClassConsistencyLoss, self).__init__()
        assert cls_aggregate_mode in ('MIL', 'cls_tokens')
        self.cls_aggregate_mode = cls_aggregate_mode
        self.sg_patch_features = sg_patch_features
        self.sg_roi_features = sg_roi_features
        self.exclusive_classes = exclusive_classes
        self.pos_class_only = pos_class_only
        self.ignore_nofind = ignore_nofind

    def kl_div(self, patch_roi_cls_probs, patch_cls_probs):
        """
        :param patch_roi_cls_probs: (N x [n_roi] x H x W x C)
        :param patch_cls_probs: (N x [1] x H x W x C)
        :return (N x [n_roi] x H x W)
        """
        eps = 1e-3
        if self.exclusive_classes:
            # renormalizer -> exclusive classes
            patch_roi_cls_probs = patch_roi_cls_probs / patch_roi_cls_probs.sum(dim=-1, keepdim=True)
            patch_cls_probs = patch_cls_probs / patch_cls_probs.sum(dim=-1, keepdim=True)
            # multiclass KLD
            return F.kl_div(
                patch_roi_cls_probs.clamp_min(eps).log(),
                patch_cls_probs,
                reduction='none'
            ).sum(-1)
        elif self.pos_class_only:
            # only positive components of binary KLD, summed over all classes
            return F.kl_div(
                patch_roi_cls_probs.clamp_min(eps).log(),
                patch_cls_probs,
                reduction='none'
            ).mean(-1)
        else:
            patch_roi_cls_probs = patch_roi_cls_probs.clamp(min=eps, max=1. - eps)
            # binary KLD per class, averaged over all classes
            loss = (F.kl_div(
                patch_roi_cls_probs.log(),
                patch_cls_probs,
                reduction='none'
            ) + F.kl_div(
                (1. - patch_roi_cls_probs).log(),
                1. - patch_cls_probs,
                reduction='none'
            ))
            return loss.mean(-1)

    def forward(self, patch_cls_probs, roi_cls_probs, roi_patch_map):
        """

        :param patch_cls_probs: (N x H x W x C)
        :param roi_cls_probs: (N x n_roi_tokens x C)
        :param roi_patch_map: (N x n_roi_tokens x H_map x W_map)
        :return:
        """
        eps = 1e-3
        if self.sg_patch_features:
            patch_cls_probs = patch_cls_probs.detach()
        patch_cls_probs = patch_cls_probs / (patch_cls_probs.sum(-1, keepdim=True) + eps)
        if self.sg_roi_features:
            roi_cls_probs = roi_cls_probs.detach()

        N, H, W, C = patch_cls_probs.shape
        N, n_roi, H_map, W_map = roi_patch_map.shape
        if (H, W) != (H_map, W_map):
            # in the case of superpixels the patch map might be on a finer
            # scale than the patch features
            # (N x n_roi_tokens x H x W)
            roi_patch_map = F.interpolate(roi_patch_map, size=(H, W), mode='bilinear')

        if self.cls_aggregate_mode == 'MIL':
            patch_roi_cls_probs = roi_patch_map[..., None] * roi_cls_probs[:, :, None, None, :]  # (N x n_roi x H x W x C)
            # irrelevant patches (based on roi_patch_map) get no-finding probs
            patch_roi_cls_probs[..., -1] += (1 - roi_patch_map)
            # noisy-OR (with special treatment of no-finding)
            patch_roi_other_cls_probs, patch_roi_no_finding_probs = patch_roi_cls_probs[..., :-1], patch_roi_cls_probs[..., -1]
            patch_roi_other_cls_probs = 1. - (1. - patch_roi_other_cls_probs).prod(dim=1)  # (N x H x W x (C-1))

            if self.ignore_nofind:
                patch_roi_cls_probs = patch_roi_other_cls_probs
                patch_cls_probs = patch_cls_probs[..., :-1]
            else:
                patch_roi_no_finding_probs = patch_roi_no_finding_probs.prod(dim=1)  # (N x H x W)
                patch_roi_cls_probs = torch.cat(
                    [patch_roi_other_cls_probs, patch_roi_no_finding_probs[..., None]],
                    dim=-1
                )  # (N x H x W x C)

            # (N x H x W)
            cls_prob_loss = self.kl_div(patch_roi_cls_probs, patch_cls_probs)
            return cls_prob_loss.mean()
        elif self.cls_aggregate_mode == 'cls_tokens':
            aggregated_cls_probs = roi_cls_probs[:, :, :-1].diagonal(dim1=-2, dim2=-1)  # (N x C-1)
            patch_roi_cls_probs = aggregated_cls_probs[:, :, None, None] * roi_patch_map  # (N x C-1 x H x W)
            patch_roi_cls_probs = einops.rearrange(patch_roi_cls_probs, 'n c h w -> n h w c')
            if self.ignore_nofind:
                patch_cls_probs = patch_cls_probs[..., :-1]  # (N x H x W x C-1)
            else:
                patch_roi_no_finding_probs = roi_patch_map * roi_cls_probs[:, :, -1, None, None]  # (N x n_roi x H x W)
                patch_roi_no_finding_probs += (1 - roi_patch_map)  # (N x n_roi x H x W)
                patch_roi_no_finding_probs = patch_roi_no_finding_probs.prod(dim=1)  # (N x H x W)
                patch_roi_cls_probs = torch.cat(
                    [patch_roi_cls_probs, patch_roi_no_finding_probs[..., None]],
                    dim=-1
                )  # (N x H x W x C)

            loss = self.kl_div(patch_roi_cls_probs, patch_cls_probs)
            return loss.mean()
        else:
            raise ValueError(self.cls_aggregate_mode)


def compute_weights_per_batch(global_labels: Tensor) -> Tensor:
    C = global_labels.shape[-1]
    global_labels = global_labels.view(-1, C)
    N, C = global_labels.shape
    N_pos = global_labels.sum(0)  # (C)
    N_neg = N - N_pos  # (C)

    weight_pos = (N + 1) / (N_pos + 1)  # (C)
    weight_neg = (N + 1) / (N_neg + 1)  # (C)

    return weight_pos, weight_neg


def weighted_binary_cross_entropy(cls_probs, global_label, clamp_min=None,
                                  ignore_no_finding=False):
    if cls_probs.ndim > 2:
        *dims, C = cls_probs.shape
        cls_probs = cls_probs.reshape(-1, C)
        global_label = global_label.reshape(-1, C)
    if ignore_no_finding:
        cls_probs = cls_probs[..., :-1]
        global_label = global_label[..., :-1]

    weight_pos, weight_neg = compute_weights_per_batch(global_label)  # (C)

    cls_probs = cls_probs.float()
    if clamp_min is not None:
        cls_probs = cls_probs.clamp(min=clamp_min, max=1. - clamp_min)
    global_label = global_label.float()

    loss = - weight_pos * global_label * cls_probs.log() - weight_neg * (1. - global_label) * (1. - cls_probs).log()
    return loss.mean()


def weighted_binary_cross_entropy_wsrpn(
    cls_probs: Tensor,
    or_probs,
    and_probs,
    global_label,
    clamp_min=None,
    use_or_nofind=False,
    use_and_nofind=False,
    cls_bbox_weights=None
):
    and_no_find_label = global_label[:, -1]
    cls_labels = global_label[:, :-1]
    C = global_label.shape[1] - 1

    probs = cls_probs
    labels = cls_labels

    if use_or_nofind:
        # (N)
        or_no_find_label = torch.ones_like(and_no_find_label)
        probs = torch.cat([probs, or_probs[:, None]], dim=-1)
        labels = torch.cat([labels, or_no_find_label[:, None]], dim=-1)

    if use_and_nofind:
        probs = torch.cat([probs, and_probs[:, None]], dim=-1)
        labels = torch.cat([labels, and_no_find_label[:, None]], dim=-1)

    weight_pos, weight_neg = compute_weights_per_batch(labels)  # (C)

    probs = probs.float()
    if clamp_min is not None:
        probs = probs.clamp(min=clamp_min, max=1. - clamp_min)
    labels = labels.float()
    # (N x C+...)
    loss = - weight_pos * labels * probs.log() - weight_neg * (1. - labels) * (1. - probs).log()
    if cls_bbox_weights is not None:
        assert len(cls_bbox_weights) == C
        C_total = loss.shape[1]
        cls_bbox_weights = 1. / cls_bbox_weights
        cls_bbox_weights = cls_bbox_weights / cls_bbox_weights.sum(0, keepdim=True)  # (C)
        cls_bbox_weights = cls_bbox_weights * C / C_total  # (C)
        if C_total > C:
            cls_bbox_weights = torch.cat([cls_bbox_weights, cls_bbox_weights.new_full((C_total - C,), 1. / C_total)])  # (C_total)
        loss = (cls_bbox_weights[None, :] * loss).sum(-1)  # (N)
        loss = loss.mean()
    else:
        loss = loss.mean()
    return loss

