"""
Model as in "Self-Transfer Learning for Fully Weakly Supervised Object
Localization"
https://arxiv.org/abs/1602.01625
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
class STLModelConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    backbone: BackboneConfig = MISSING

    # List of values to threshold the heatmaps to get connected components
    heatmap_thresholds: List[float] = MISSING

    # Post-processing
    filter_top1_per_class: bool = MISSING
    use_nms: bool = MISSING

    # Losses
    use_bce_loss: bool = MISSING

    # Only needed for train
    use_superpixels: bool = False


class AvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean((-2, -1))  # (N, C, H, W) -> (N, C)


class STLModel(ObjectDetectorModelInterface):
    CONFIG_CLS = STLModelConfig

    def __init__(self, config: STLModelConfig) -> None:
        super(STLModel, self).__init__(config)
        self.config = config

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_feats = self.backbone.d[self.backbone_layer]

        self.classifier = nn.Sequential(
            AvgPool(),
            nn.Linear(num_feats, config.num_classes),
            nn.Sigmoid()
        )
        self.localizer_conv = nn.Conv2d(num_feats, config.num_classes, 1)
        self.localizer = nn.Sequential(
            AvgPool(),
            nn.Sigmoid()
        )

    @staticmethod
    def alpha_scheduler(step: int) -> float:
        """Linear schedule from 0.1 at step 0 to 0.9 at step 10000"""
        return min(0.1 + 0.8 * step / 10000, 0.9)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Input batch of images (N x C x H x W)

        :return heatmaps: Localization heatmaps per class (N x CL x h x w)
        :return cls_probs: Class probabilities from classification branch
                           (N x CL)
        :return localizer_probs: Class probabilities from localizer branch
                                 (N x CL)
        """
        features = self.backbone(x)[self.backbone_layer]  # (N, d, h, w)
        classification_probs = self.classifier(features)  # (N, CL)
        heatmaps = self.localizer_conv(features)  # (N, CL, h, w)
        localization_probs = self.localizer(heatmaps)  # (N, CL)

        return heatmaps, classification_probs, localization_probs

    def train_step(
        self,
        x: Tensor,
        global_label: Tensor,
        step: int,
        return_predictions: bool = False,
        **kwargs
    ) -> Tuple[float, Dict[str, Tensor], ObjectDetectorPrediction]:
        """Training step, compute losses.
        :param x: Input batch (N x C x H x W)
        :param global_label: One-hot target labels (N, CL)
        """
        heatmaps, classification_probs, localization_probs = self(x)

        losses = dict()

        classification_loss = weighted_binary_cross_entropy(
            classification_probs,
            global_label,
            clamp_min=1e-7,
            ignore_no_finding=False
        )
        localization_loss = weighted_binary_cross_entropy(
            localization_probs,
            global_label,
            clamp_min=1e-7,
            ignore_no_finding=False
        )
        losses['classification_bce'] = classification_loss
        losses['localization_bce'] = localization_loss

        alpha = self.alpha_scheduler(step)
        loss = (1 - alpha) * classification_loss + alpha * localization_loss

        if return_predictions:
            predictions = self.inference_from_scores(
                x, heatmaps, localization_probs)
        else:
            predictions = None

        return loss, losses, predictions

    def inference_from_scores(
        self,
        x: Tensor,
        heatmaps: Tensor,
        localization_probs: Tensor
    ) -> ObjectDetectorPrediction:
        """Get bounding box predictions for a batch of images
        :param x: Batch of input images (N x C X H x W)
        :param heatmaps: Localization heatmaps per class (N x CL x h x w)
        :param localization_probs: Predicted class probabilites (N x CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        localization_probs = localization_probs.float()
        global_prediction_probs = localization_probs[..., :-1]  # (N, CL - 1)
        global_obj_probs = localization_probs[..., -1]  # (N)
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
        heatmaps, _, localization_probs = self(x)
        return self.inference_from_scores(x, heatmaps, localization_probs)
