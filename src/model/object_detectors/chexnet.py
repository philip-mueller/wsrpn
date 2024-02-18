"""
Model as in "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays
with Deep Learning"
https://arxiv.org/abs/1711.05225
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import MISSING
from torch import Tensor

from src.model.backbone import BackboneConfig
from src.model.backbone.backbone_loader import load_backbone
from src.model.losses import (SupConPerClassLoss,
                              weighted_binary_cross_entropy)
from src.model.model_components import (create_box_predictions_from_heatmaps,
                                        lse_pool)
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)
from src.model.positional_embedding import positional_embedding_2d


@dataclass
class CheXNetConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    backbone: BackboneConfig = MISSING

    use_pos_emb: bool = MISSING
    pool_mode: str = MISSING  # mean, max, lse, noisyOR

    # Post-processing
    filter_top1_per_class: bool = MISSING
    use_nms: bool = MISSING

    # Losses
    use_bce_loss: bool = MISSING
    use_supcon_loss: Optional[bool] = MISSING
    supcon_temperature: Optional[float] = MISSING
    supcon_d_hidden: Optional[int] = MISSING
    supcon_d_emb: Optional[int] = MISSING

    # List of values to threshold the heatmaps to get connected components
    heatmap_thresholds: List[float] = MISSING


def avg_pool(x: Tensor) -> Tensor:
    N, cl = x.shape[:2]
    return x.view(N, cl, -1).mean(dim=2)  # (N, CL)


def max_pool(x: Tensor) -> Tensor:
    N, cl = x.shape[:2]
    return x.view(N, cl, -1).amax(dim=2)  # (N, CL)


def noisyOR_pool(x: Tensor) -> Tensor:
    N, cl = x.shape[:2]
    x = x.view(N, cl, -1)
    cls_scores = x[:, :-1].amax(dim=2)  # OR logic for classes (max)
    no_finding_scores = x[:, -1].amin(dim=1, keepdim=True)  # AND logic for no_finding (min)
    return torch.cat([cls_scores, no_finding_scores], dim=1)  # (N, CL)


class CheXNet(ObjectDetectorModelInterface):
    CONFIG_CLS = CheXNetConfig

    def __init__(self, config: CheXNetConfig) -> None:
        super(CheXNet, self).__init__(config)
        self.config = config

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_feats = self.backbone.d[self.backbone_layer]

        # Position embeddings for patches
        if config.use_pos_emb:
            self.pos_emb = positional_embedding_2d('sin_cos', channels=num_feats)

        self.classifier_conv = nn.Conv2d(num_feats, config.num_classes, 1)

        if config.use_supcon_loss:
            self.projector = nn.Sequential(
                nn.Linear(num_feats, config.supcon_d_hidden),
                nn.ReLU(),
                nn.Linear(config.supcon_d_hidden, config.supcon_d_emb)
            )
            self.supcon_loss = SupConPerClassLoss(
                temperature=config.supcon_temperature,
                pos_alignment_weight=1.0,
                neg_alignment_weight=1.0,
                normalize_weights=True,
                ignore_no_label=True
            )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: Input batch of images (N, C, H, W)

        :return feature_scores: (N, CL, h, w) Spatial map of class logits
        :return cls_features: Class logits (N, CL)
        :return cls_probs: Class probabilities (N, CL)
        """
        features = self.encode_features(x)  # (N, d, h, w)
        feature_scores = self.classifier_conv(features)  # (N, CL, h, w)
        cls_scores = self.pool(feature_scores)  # (N, CL)
        cls_probs = torch.sigmoid(cls_scores)  # (N, CL)

        if self.config.use_supcon_loss:
            features = rearrange(features, 'n d h w -> n (h w) d')
            feature_probs = rearrange(
                torch.sigmoid(feature_scores),
                'n c h w -> n c (h w)'
            )  # (N, CL, P)
            eps = 1.0  # 1.0 seems to work better than small eps
            normalized_feature_probs = (feature_probs
                                        / (feature_probs.sum(2, keepdim=True) + eps))
            # Apply obj_probs to reweight the individual contributions of
            # different boxes (obj_probs normalized to sum=1 per sample)
            cls_features = torch.bmm(normalized_feature_probs, features)  # (N, CL, d)
        else:
            cls_features = None

        return feature_scores, cls_features, cls_probs

    def encode_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encodes the input image using the backbone.

        :param x: input image (N, C, H, W)
        :return feature_map: Encoded feature map (N, d, h, w)
        """
        feature_map = self.backbone(x)[self.backbone_layer]  # (N, d, h, w)

        if self.config.use_pos_emb:
            feature_map = rearrange(feature_map, 'n d h w -> n h w d')  # (N, h, w, d)
            pos_emb = self.pos_emb(feature_map)  # (N, h, w, 2)
            feature_map = feature_map + pos_emb  # (N, h, w, d)
            feature_map = rearrange(feature_map, 'n h w d -> n d h w')  # (N, d, h, w)

        return feature_map

    def pool(self, feature_scores: Tensor) -> Tensor:
        if self.config.pool_mode == 'mean':
            return avg_pool(feature_scores)
        elif self.config.pool_mode == 'max':
            return max_pool(feature_scores)
        elif self.config.pool_mode == 'lse':
            return lse_pool(feature_scores)
        elif self.config.pool_mode == 'noisyOR':
            return noisyOR_pool(feature_scores)
        else:
            raise NotImplementedError

    def train_step(
        self,
        x: Tensor,
        global_label: Tensor,
        target_boxes: Optional[List[Tensor]] = None,
        return_predictions: bool = False,
        **kwargs
    ) -> Tuple[float, Dict[str, Tensor], ObjectDetectorPrediction]:
        """Training step, compute losses.
        :param x: Input batch (N, C, H, W)
        :param global_label: One-hot target labels (N, CL)
        :param target_boxes: List of N boxes (x, y, w, h, cls_id) if provided
                             else None
        """
        _, _, H, W = x.shape
        feature_scores, cls_features, cls_probs = self(x)

        losses = dict()

        if self.config.use_bce_loss:
            losses['bce'] = weighted_binary_cross_entropy(
                cls_probs,
                global_label,
                clamp_min=1e-7,
                ignore_no_finding=False
            )
        if self.config.use_supcon_loss:
            projected_cls_features = self.projector(cls_features)  # (N, CL)
            losses['supcon'] = self.supcon_loss(
                projected_cls_features, global_label)
        
        assert len(losses) > 0
        loss = sum(losses.values()) / len(losses)

        if return_predictions:
            predictions = self.inference_from_scores(x, feature_scores, cls_probs)
        else:
            predictions = None

        return loss, losses, predictions

    def inference_from_scores(
        self,
        x: Tensor,
        heatmaps: Tensor,
        cls_probs: Tensor
    ) -> ObjectDetectorPrediction:
        """Get bounding box predictions for a batch of images
        :param x: Batch of input images (N, C, H, W)
        :param heatmaps: Per class localization heatmaps (N, d, h, w)
        :param cls_probs: Predicted class probabilites (N, CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        N, CL = cls_probs.shape

        cls_probs = cls_probs.float()
        global_prediction_probs = cls_probs[..., :-1]  # (N, CL - 1)
        global_obj_probs = 1. - cls_probs[..., -1]  # (N)
        global_prediction_hard = (global_prediction_probs > 0.5).int()

        # Resize heatmap
        heatmaps = F.interpolate(
            heatmaps,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # (N, CL, H, W)
        segmentation = heatmaps.argmax(1).to(torch.uint8)  # (N, H, W)

        # Get everything but the segmentation to CPU
        global_prediction_probs = global_prediction_probs.cpu()
        global_prediction_hard = global_prediction_hard.cpu()
        global_obj_probs = global_obj_probs.cpu()
        heatmaps = heatmaps.cpu()

        # Create per-class bounding boxes
        (box_prediction_hard,
         box_prediction_probs) = create_box_predictions_from_heatmaps(
            heatmaps,
            global_prediction_hard,
            self.config.heatmap_thresholds,
            apply_sigmoid=True,
            use_nms=self.config.use_nms,
            nms_threshold=0.5,
            filter_top1_box_per_class=self.config.filter_top1_per_class
        )

        return ObjectDetectorPrediction(
            global_prediction_hard=global_prediction_hard,
            global_prediction_probs=global_prediction_probs,
            global_obj_probs=global_obj_probs,
            box_prediction_hard=box_prediction_hard,
            box_prediction_probs=box_prediction_probs,
            seg_masks_from_patches=segmentation,
        )

    def inference(self, x: Tensor, **kwargs) -> ObjectDetectorPrediction:
        feature_scores, cls_features, cls_probs = self(x)
        return self.inference_from_scores(x, feature_scores, cls_probs)
