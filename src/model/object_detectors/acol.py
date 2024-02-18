"""
Model as in "Weakly Supervised Deep Learning for Thoracic Disease
Classification and Localization on Chest X-rays"
"https://arxiv.org/abs/1807.06067"
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch import Tensor

from src.model.backbone import BackboneConfig
from src.model.backbone.backbone_loader import load_backbone
from src.model.losses import weighted_binary_cross_entropy
from src.model.model_components import create_box_predictions_from_heatmaps
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)


@dataclass
class ACoLConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    backbone: BackboneConfig = MISSING

    # Threshold to detect most discriminative regions in heatmap1
    erase_threshold: float = MISSING

    # List of values to threshold the heatmaps to get connected components
    heatmap_thresholds: List[float] = MISSING

    # Post-processing
    filter_top1_per_class: bool = MISSING
    use_nms: bool = MISSING

    # Losses
    use_bce_loss: bool = MISSING

    # Only needed for train
    use_superpixels: bool = False


class ACoL(ObjectDetectorModelInterface):
    CONFIG_CLS = ACoLConfig

    def __init__(self, config: ACoLConfig) -> None:
        super(ACoL, self).__init__(config)
        self.config = config

        # Assert erase_threshold > 0
        assert config.erase_threshold > 0, 'Erase_threshold must be > 0'

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_feats = self.backbone.d[self.backbone_layer]

        self.classifier1 = nn.Conv2d(num_feats, config.num_classes, 1)
        self.classifier2 = nn.Conv2d(num_feats, config.num_classes, 1)

    @staticmethod
    def avg_pool(x: Tensor):
        return x.mean((-2, -1))  # (N, C, H, W) -> (N, C)

    def forward(
        self,
        x: Tensor,
        targets: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Input batch of images (N x C x H x W)
        :param targets: Target label (only needed for training) (N, CL)

        :return heatmaps: Localization heatmaps per class (N x CL x h x w)
        :return cls_probs: Class probabilities from classification branch
                           (N x CL)
        :return localizer_probs: Class probabilities from localizer branch
                                 (N x CL)
        """
        features = self.backbone(x)[self.backbone_layer]  # (N, d, h, w)

        # Get heatmaps and scores from branch 1
        heatmaps1 = self.classifier1(features)  # (N, CL, h, w)
        scores1 = self.avg_pool(heatmaps1)  # (N, CL)
        cls_probs1 = torch.sigmoid(scores1)  # (N, CL)

        # During test time, targets are inferenced from predictions
        if targets is None:
            targets = (cls_probs1 > 0.5).int()  # (N, CL)

        # Catch case that all predictions are < 0.5
        no_targets = targets.sum(1) == 0
        if torch.any(no_targets):
            targets[no_targets, -1] = 1

        # Erase most discriminative regions from features
        keep_map = torch.where(heatmaps1 > self.config.erase_threshold, 1, 0)  # (N, CL, h, w)
        keep_map *= targets[..., None, None]  # (N, CL, h, w)
        keep_map = keep_map.max(1, keepdim=True).values  # (N, 1, h, w)
        features_thresh = features * (1 - keep_map)

        # Get heatmaps and scores from branch 2
        heatmaps2 = self.classifier2(features_thresh)
        scores2 = self.avg_pool(heatmaps2)
        cls_probs2 = torch.sigmoid(scores2)

        # Fuse heatmaps
        heatmaps = torch.max(heatmaps1, heatmaps2)

        return heatmaps, cls_probs1, cls_probs2

    def train_step(
        self,
        x: Tensor,
        global_label: Tensor,
        return_predictions: bool = False,
        **kwargs
    ) -> Tuple[float, Dict[str, Tensor], ObjectDetectorPrediction]:
        """Training step, compute losses.
        :param x: Input batch (N x C x H x W)
        :param global_label: One-hot target labels (N, CL)
        """
        if not return_predictions:
            # Training. Use global_label to erase
            heatmaps, cls_probs1, cls_probs2 = self(x, global_label)
        else:
            # Test or validation. Do not use global_label to erase
            heatmaps, cls_probs1, cls_probs2 = self(x)

        losses = dict()

        if self.config.use_bce_loss:
            bce1 = weighted_binary_cross_entropy(
                cls_probs1,
                global_label,
                clamp_min=1e-7,
                ignore_no_finding=False
            )
            bce2 = weighted_binary_cross_entropy(
                cls_probs2,
                global_label,
                clamp_min=1e-7,
                ignore_no_finding=False
            )
            losses['bce'] = 0.5 * (bce1 + bce2)
        assert len(losses) > 0
        loss = sum(losses.values()) / len(losses)

        if return_predictions:
            predictions = self.inference_from_scores(x, heatmaps, cls_probs1)
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
        :param x: Batch of input images (N x C X H x W)
        :param heatmaps: Heatmaps with class scores (N x CL x h x w)
        :param cls_probs: Predicted class probabilites (N x CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        cls_probs = cls_probs.float()
        global_prediction_probs = cls_probs[..., :-1]  # (N, CL - 1)
        global_obj_probs = cls_probs[..., -1]  # (N)
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
        heatmaps, cls_probs1, cls_probs2 = self(x)
        return self.inference_from_scores(x, heatmaps, cls_probs1)
