from functools import partial
import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from einops import einops, repeat
from scipy import ndimage
from skimage import measure
from torch import Tensor, arange, einsum, nn
from torch.nn.init import xavier_uniform_
from torchvision.ops import batched_nms


class MLP(nn.Module):
    def __init__(self, num_hidden_layers: int, d_in: int, d_hidden: int,
                 d_out: int, use_bn=False, dropout=0.0, dropout_last_layer=True):
        """
        Note: supports multi-dim input but assumed "channels last", i.e. always
        the last dimension is d_in or d_out.

        :param num_layers: If num_hidden_layers == 0, then only use a single linear layer
        :param d_in:
        :param d_hidden:
        :param d_out:
        :param use_bn
        """
        super(MLP, self).__init__()
        current_dim_in = d_in
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim_in, d_hidden, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(d_hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim_in = d_hidden
        layers.append(nn.Linear(d_hidden, d_out))
        if dropout_last_layer and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        *dims, d = x.shape
        if len(dims) > 1:
            x = x.reshape(-1, d)

        x = self.layers(x)
        return x.view(*dims, -1)


def softmax(logits: Tensor, tau: float = 1, dim: int = -1, hard=None) -> Tensor:
    y_soft = (logits / tau).softmax(dim)
    if hard is False:
        return y_soft

    # Straight through estimator.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    if hard is True:
        return y_hard

    return y_hard, y_soft


def gumbel_softmax(logits: Tensor, tau: float = 1, dim: int = -1, hard=None) -> Tensor:
    # see https://github.com/NVlabs/GroupViT/blob/b4ef51b8ae997f4741811025ac2290df3423a27a/models/group_vit.py#L63

    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    if hard is False:
        return y_soft

    # Straight through estimator.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    if hard is True:
        return y_hard

    return y_hard, y_soft


class RoiTokenAttentionLayer(nn.Module):
    def __init__(self, d_embedding: int, d_hidden: int, num_cross_att_heads: int,
                 use_cross_att=True, use_mlp_out=False, qkv_bias=True, tau=1.,
                 gumbel=False, hard=False, intra_temporal_attention=False,
                 skip_class_emb=False, dropout=.0):
        super(RoiTokenAttentionLayer, self).__init__()
        self.skip_class_emb = skip_class_emb
        self.gumbel = gumbel
        self.tau = tau
        self.hard = hard
        self.intra_temporal_attention = intra_temporal_attention
        self.scale = math.sqrt(d_embedding)

        # ----- Layer -----
        # Input norms
        self.norm_tokens = nn.LayerNorm(d_embedding)
        self.norm_feature_map = nn.LayerNorm(d_embedding)
        # Cross attention
        self.use_cross_att = use_cross_att
        if use_cross_att:
            self.cross_att_layer = nn.MultiheadAttention(d_embedding,
                                                         num_heads=num_cross_att_heads,
                                                         batch_first=True,
                                                         dropout=dropout)
            self.norm_mlp_cross_att = nn.LayerNorm(d_embedding)
            self.mlp_cross_att = MLP(num_hidden_layers=1, d_in=d_embedding,
                                     d_hidden=d_hidden, d_out=d_embedding,
                                     use_bn=False, dropout=dropout, dropout_last_layer=True)
        self.norm_cross_att = nn.LayerNorm(d_embedding)

        # Assign attention
        self.q_proj = nn.Linear(d_embedding, d_embedding, bias=qkv_bias)
        self.k_proj = nn.Linear(d_embedding, d_embedding, bias=qkv_bias)
        self.v_proj = nn.Linear(d_embedding, d_embedding, bias=qkv_bias)
        self.proj = nn.Linear(d_embedding, d_embedding)
        self.proj_drop = nn.Dropout(dropout)

        # MLP out
        self.use_mlp_out = use_mlp_out
        if use_mlp_out:
            self.mlp_out = MLP(num_hidden_layers=1, d_in=d_embedding,
                               d_hidden=d_hidden, d_out=d_embedding,
                               use_bn=False, dropout=dropout, dropout_last_layer=True)
            self.norm_mlp_out = nn.LayerNorm(d_embedding)

    def get_att_probs(self, att_scores: Tensor) -> Tensor:
        """
        :param att_scores: (N x n_roi_tokens x P) P is pixels or patches
        :return:
        """
        if self.intra_temporal_attention:
            # intra-temporal attention such that each token focuses on different regions
            # related to https://arxiv.org/pdf/1705.04304.pdf
            cum_sums = att_scores.cumsum(dim=1)[:, :-1, :]
            att_scores = att_scores.clone()
            att_scores[:, 1:, :] -= cum_sums

        if self.gumbel and self.training:
            probs = gumbel_softmax(att_scores, dim=-1, tau=self.tau, hard=self.hard)  # (N x n_roi_tokens x P)
        else:
            probs = softmax(att_scores, dim=-1, tau=self.tau, hard=self.hard)  # (N x n_roi_tokens x P)

        return probs

    def assign_att(self, roi_tokens, x, return_att=True):
        q = self.q_proj(roi_tokens)  # (N x n_roi_tokens x d)
        k = self.k_proj(x)  # (N x P x d)
        v = self.v_proj(x)  # (N x P x d)
        att_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (N x n_roi_tokens x P)
        att_probs = self.get_att_probs(att_scores)  # (N x n_roi_tokens x P)

        out = torch.bmm(att_probs, v)  # (N x n_roi_tokens x d)
        out = self.proj(out)  # (N x n_roi_tokens x d)
        out = self.proj_drop(out)  # (N x n_roi_tokens x d)
        return out, att_probs

    def forward(self, roi_tokens: Tensor, x: Tensor, return_att=True) \
            -> Tuple[Tensor, Dict[str, Tensor]]:
        """

        :param roi_tokens: (N x n_roi_tokens x d)
        :param x: (N x P x d)
        :return:
        """
        x = self.norm_feature_map(x)
        if self.use_cross_att:
            roi_tokens = roi_tokens + self.cross_att_layer(self.norm_tokens(roi_tokens), x, x, need_weights=False)[0]
            roi_tokens = roi_tokens + self.mlp_cross_att(self.norm_mlp_cross_att(roi_tokens))
        roi_tokens_assigned, token_map = self.assign_att(self.norm_cross_att(roi_tokens), x, return_att=return_att)
        if self.skip_class_emb:
            roi_tokens_assigned = roi_tokens_assigned + roi_tokens
        if self.use_mlp_out:
            roi_tokens_assigned = roi_tokens_assigned + self.mlp_out(self.norm_mlp_out(roi_tokens_assigned))

        return roi_tokens_assigned, token_map


def lse_pool(x: Tensor, r: float = 5.0):
    """
    LogSumExp Pooling.
    Similar to avg-pool if r -> 0
    Similar to max-pool if r -> inf

    :param x: (N, C, H, W) input tensor
    :param r: Smoothing factor. default = 5.0
    :returns: (N, C) tensor pooled along spatial dimension
    """
    x_max = x.amax((-2, -1), keepdim=True)  # (N, C, 1, 1)
    x = x - x_max
    x = torch.div((r * x).exp().mean((-2, -1)).log(), r)
    x = x + x_max[..., 0, 0]
    return x


def max_min_pool(x: Tensor, k: int, m: int, alpha: Optional[float] = 1.0) -> Tensor:
    """
    max-min-pooling per feature map

    :param x: Input feature-map (N, C, H, W)
    :param k: Number of top highest scores to consider
    :param m: Number of top lowest scores to consider
    :param alpha: Weighting factor for negative scores
    :return: pooled feature vector (N, C)
    """
    N, C = x.shape[:2]
    top_k = torch.topk(x.view(N, C, -1), k=k, dim=-1, largest=True).values  # (N, C, k)
    low_m = torch.topk(x.view(N, C, -1), k=m, dim=-1, largest=False).values  # (N, C, m)
    return top_k.sum(-1) + alpha * low_m.sum(-1)  # (N, C)


def get_bboxes_from_heatmaps_(
    hms: Tensor,
    obj_score_hm: Tensor,
    threshold: float,
    cls: int,
    apply_sigmoid: bool
) -> Tensor:
    """Get bounding boxes, cls_probs and obj_probs for each connected component
    in an image. cls_probs and obj_probs will be pooled from hms and
    obj_score_hm respecively.

    :param hms: (CL - 1, W, H) Class heatmap
    :param obj_score_hm: (W, H) Heatmap of objectness
    :param threshold: Threshold to binarize the heatmap
    :param cls: Prediced class
    :param apply_sigmoid: If True, heatmaps are scores and will be converted
                          to probs with sigmoid

    :return bboxes: (M, 6 + CL - 1) Bounding boxes,
                    each (x, y, w, h, pred_cls, obj_prob, cls_probs)
    """
    hm = hms[cls]  # (H, W)
    hm_norm = (hm - hm.min()) / (hm.max() - hm.min())  # (H, W)
    hm_bin = torch.where(hm_norm > threshold, 1, 0)  # (H, W)
    # Get connected components
    labels, M = measure.label(hm_bin, return_num=True)  # (H, W), M
    # Get bounding boxes and object scores of connected components
    bboxes = ndimage.find_objects(labels)  # (m, 4)
    bboxes = torch.tensor([
        [
            bbox[1].start,
            bbox[0].start,
            bbox[1].stop - bbox[1].start,
            bbox[0].stop - bbox[0].start,
            cls,
            obj_score_hm[labels == i + 1].mean(),  # average of obj_score_hm at connected component
            *hms[:, labels == i + 1].mean(1)
        ] for i, bbox in enumerate(bboxes)
    ])  # (m, 6 + CL - 1)
    if apply_sigmoid:
        bboxes[:, 5] = torch.sigmoid(bboxes[:, 5])
        bboxes[:, 6:] = torch.sigmoid(bboxes[:, 6:])
    return bboxes


def create_box_predictions_from_heatmaps(
    heatmap: Tensor,
    global_prediction_hard: Tensor,
    thresholds: List[float],
    apply_sigmoid: bool,
    use_nms: Optional[bool] = False,
    nms_threshold: Optional[float] = None,
    filter_top1_box_per_class: Optional[bool] = False
) -> Tuple[List[Tensor], List[Tensor]]:
    """Threshold the heatmap and compute a bounding box for every isolated
    region and every class.
    :param heatmap: (N, CL, H, W) Object prediction heatmaps per sample and
                    class. Should be same size as the image.
    :param global_prediction_hard: (N, CL-1) See ObjectDetectorPrediction
    :param global_prediction_prob: (N, CL-1) See ObjectDetectorPrediction
    :param thresholds: List of thresholds for binarizing the heatmaps
    :param apply_sigmoid: If True heatmap values are scores and are converted
                          to probabilities after aggregation via sigmoid
    :param use_nms: If True, non-maximum suppression is used
    :param nms_threshold: IoU Threshold for non-maximum suppression
    :param filter_top1_box_per_class: If True, only the most confident predicted
                                      box per class is kept.

    :return box_prediction_hard: (N, M_i, 6) See ObjectDetectorPrediction
    :return box_prediction_probs: (N, M_i, 5+CL-1) See ObjectDetectorPrediction
    """
    assert len(thresholds) > 0
    N, CL = global_prediction_hard.shape

    # Separate objectness score map
    if apply_sigmoid:  # no_obj_scores
        obj_score_heatmap = -heatmap[:, -1]
    else:  # no_obj_probs
        obj_score_heatmap = 1 - heatmap[:, -1]
    heatmap = heatmap[:, :-1]

    box_prediction_hard = []
    box_prediction_probs = []
    # Iterate over batch
    for hms, obj_score_hm, global_preds in zip(heatmap,
                                               obj_score_heatmap,
                                               global_prediction_hard):
        pred_classes = torch.arange(global_preds.shape[0])[global_preds == 1]  # (cl)

        if len(pred_classes) == 0:
            # No classes predicted -> dummy bounding boxes
            box_pred_hard = torch.tensor(
                [[0, 0, 1, 1, random.randint(0, CL - 1), 0.]])
            box_pred_probs = torch.tensor(
                [[0, 0, 1, 1, 0., *torch.zeros(CL)]])
        else:
            bboxes = []
            # Get bounding boxes for each threshold
            for threshold in thresholds:
                bbox_fn = partial(
                    get_bboxes_from_heatmaps_,
                    hms=hms,
                    obj_score_hm=obj_score_hm,
                    threshold=threshold,
                    apply_sigmoid=apply_sigmoid
                )
                # Get bbox (x,y,w,h,pred_cls,obj_prob,cls_probs) of each connected component and class
                if len(bboxes) == 0:  # First threshold
                    bboxes = [bbox_fn(cls=cls) for cls in pred_classes]  # (cl, M, 6 + CL - 1)
                else:  # Other thresholds
                    bboxes = [
                        torch.cat([bbox, bbox_fn(cls=cls)])
                        for bbox, cls in zip(bboxes, pred_classes)
                    ]  # (cl, M, 6 + CL - 1)

            # Keep only the top scoring box per class
            if filter_top1_box_per_class:
                filtered_bboxes = []
                for bbox in bboxes:
                    filtered_bboxes.append(bbox[bbox[:, 5].argmax(), None])
                bboxes = filtered_bboxes

            bboxes = torch.cat(bboxes)  # (M, 6 + CL - 1)

            # Apply non-maximum suppression
            if use_nms:
                obj_probs = bboxes[:, 5]  # (m)
                boxes = bboxes[:, :4].clone()
                # Convert from (x,y,w,h) to (x1,y1,x2,y2)
                boxes[:, 2:4] += boxes[:, :2]  # (m, 4)
                cls_idxs = bboxes[:, 4]
                box_keep_inds = batched_nms(boxes, obj_probs, cls_idxs, nms_threshold)  # (m)
                bboxes = bboxes[box_keep_inds]  # (m, 4)

            # Keep only the top scoring box per class
            # if filter_top1_box_per_class:
            #     filtered_bboxes = []
            #     unique_classes = torch.unique(bboxes[:, 4])
            #     for c in unique_classes:
            #         bboxes_c = bboxes[bboxes[:, 4] == c]
            #         filtered_bboxes.append(bboxes_c[bboxes_c[:, 5].argmax()])
            #     bboxes = torch.stack(filtered_bboxes)  # (cl, 4)

            # Get box_prediction_hard (bbox with class_id and confidence)
            box_pred_hard = bboxes[:, :6]  # (m, 6)
            # Get box_prediction_probs (bbox with obj_score and class probs)
            box_pred_probs = torch.cat([bboxes[:, :4], bboxes[:, 5:]], dim=1)  # (m, 5 + CL - 1)

        box_prediction_hard.append(box_pred_hard)
        box_prediction_probs.append(box_pred_probs)

    return box_prediction_hard, box_prediction_probs
