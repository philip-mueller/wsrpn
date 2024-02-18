"""
Model as in "From Image-level to Pixel-level Labeling with Convolutional
Networks"
https://arxiv.org/abs/1411.6228
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
from src.model.model_components import (create_box_predictions_from_heatmaps,
                                        lse_pool)
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)
from src.model.losses import weighted_binary_cross_entropy


@dataclass
class LSEModelConfig(ModelConfig):
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

class LSEModel(ObjectDetectorModelInterface):
    CONFIG_CLS = LSEModelConfig

    def __init__(self, config: LSEModelConfig) -> None:
        super(LSEModel, self).__init__(config)
        self.config = config

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_ftrs = self.backbone.d[self.backbone_layer]

        self.classifier_conv = nn.Conv2d(num_ftrs, config.num_classes, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: Input batch of images (N x C x H x W)

        :return cls_probs: Class probabilities (N x CL)
        """
        features = self.backbone(x)[self.backbone_layer]  # (N, d, h, w)
        feature_scores = self.classifier_conv(features)  # (N, CL, h, w)
        cls_scores = lse_pool(feature_scores, r=5.0)  # (N, CL)
        cls_probs = torch.sigmoid(cls_scores)  # (N, CL)

        return feature_scores, cls_probs

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
        features, cls_probs = self(x)

        losses = dict()

        if self.config.use_bce_loss:
            losses['bce'] = weighted_binary_cross_entropy(
                cls_probs,
                global_label,
                clamp_min=1e-7,
                ignore_no_finding=False
            )

        assert len(losses) > 0
        loss = sum(losses.values()) / len(losses)

        if return_predictions:
            predictions = self.inference_from_scores(x, features, cls_probs)
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
        :param heatmaps: Batch of per-class feature maps (N x CL x h x w)
        :param cls_probs: Predicted class probabilites (N x CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        device = x.device

        cls_probs = cls_probs.float()
        global_prediction_probs = cls_probs[..., :-1]
        global_obj_probs = cls_probs[..., -1]
        global_prediction_hard = (global_prediction_probs > 0.5).int()

        # Softmax because every location should have only one prediction
        heatmaps = F.softmax(heatmaps, dim=1)  # (N, CL, h, w)

        # Get everything but the segmentation to CPU
        global_prediction_probs = global_prediction_probs.cpu()
        global_prediction_hard = global_prediction_hard.cpu()
        global_obj_probs = global_obj_probs.cpu()

        # Resize heatmaps
        heatmaps = F.interpolate(
            heatmaps,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # (N, CL, H, W)

        segmentation = heatmaps.argmax(1).to(torch.uint8)  # (N, H, W)
        heatmaps = heatmaps.cpu()

        # Create per-class bounding boxes
        (box_prediction_hard,
            box_prediction_probs) = create_box_predictions_from_heatmaps(
            heatmaps,
            global_prediction_hard,
            self.config.heatmap_thresholds,
            apply_sigmoid=False,
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

    def inference(self, x: Tensor, **kwargs) \
            -> ObjectDetectorPrediction:
        features, cls_probs = self(x)
        return self.inference_from_scores(x, features, cls_probs)
