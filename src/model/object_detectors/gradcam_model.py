from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch import Tensor, autocast

from src.model.backbone import BackboneConfig
from src.model.backbone.backbone_loader import load_backbone
from src.model.grad_cam import GradCAM
from src.model.losses import weighted_binary_cross_entropy
from src.model.model_components import (create_box_predictions_from_heatmaps,
                                        lse_pool)
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)


@dataclass
class GradCAMModelConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    backbone: BackboneConfig = MISSING

    # Pixel attribution
    attribution_method: str = MISSING

    # Feature aggregation
    feature_aggregation: str = MISSING

    # List of values to threshold the heatmaps to get connected components
    heatmap_thresholds: List[float] = MISSING

    # Post-processing
    filter_top1_per_class: bool = MISSING
    use_nms: bool = MISSING

    # Losses
    use_bce_loss: bool = MISSING

    # Only needed for train
    use_superpixels: bool = False


class GradCAMModel(ObjectDetectorModelInterface):
    CONFIG_CLS = GradCAMModelConfig

    def __init__(self, config: GradCAMModelConfig) -> None:
        super(GradCAMModel, self).__init__(config)
        self.config = config

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_ftrs = self.backbone.d[self.backbone_layer]

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, config.num_classes),
            nn.Sigmoid()
        )

        attribution_kwargs = dict(
            target_layer=list(
                self.backbone.backbone_layers[self.backbone_layer].children()
            )[-1]
        )
        if config.attribution_method == "GradCAM":
            self.att_fn = GradCAM(**attribution_kwargs)
        else:
            raise NotImplementedError("Attribution method",
                                      config.attribution_method,
                                      "not implemented yet.")

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: Input batch of images (N x C x H x W)

        :return cls_probs: Class probabilities (N x CL)
        """
        features = self.backbone(x)[self.backbone_layer]
        features = F.relu(features, inplace=True)

        if self.config.feature_aggregation == 'mean':
            features = features.mean((-2, -1))
        elif self.config.feature_aggregation == 'max':
            features = features.amax((-2, -1))
        elif self.config.feature_aggregation == 'lse':
            features = lse_pool(features, r=5.0)

        cls_probs = self.classifier(features)

        return cls_probs

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
        cls_probs = self(x)

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
            predictions = self.inference_from_scores(x, cls_probs)
        else:
            predictions = None

        return loss, losses, predictions

    def inference_from_scores(
        self,
        x: Tensor,
        cls_probs: Tensor,
    ) -> ObjectDetectorPrediction:
        """Get bounding box predictions for a batch of images
        :param x: Batch of input images (N x C X H x W)
        :param cls_probs: (Optional) Predicted class probabilites (N x CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        N, CL = cls_probs.shape
        H, W = x.shape[2:]

        cls_probs = cls_probs.detach().float()

        global_prediction_probs = cls_probs[..., :-1]  # (N, CL - 1)
        global_obj_probs = cls_probs[..., -1]  # (N)
        global_prediction_hard = (global_prediction_probs > 0.5).int()

        # Get class activation map for every image and class
        with torch.enable_grad():
            heatmaps = torch.zeros(N, CL, H, W, device=x.device)  # (N, CL, H, W)
            for i in range(CL):
                with autocast(device_type='cuda', enabled=False):
                    cls_probs = self(x)
                    cls_probs[:, i].mean().backward()
                    heatmaps_i = self.att_fn(x)
                heatmaps[:, i] = heatmaps_i

        # Create segmentation: Take the maximum class at every spatial location
        segmentation = heatmaps.argmax(1).to(torch.uint8)  # (N, H, W)

        # Normalize heatmaps from 0 to 1
        hm_min = heatmaps.amin((1, 2, 3), keepdim=True)
        hm_max = heatmaps.amax((1, 2, 3), keepdim=True)
        heatmaps = (heatmaps - hm_min) / (hm_max - hm_min)

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

    def inference(self, x: Tensor, **kwargs) -> ObjectDetectorPrediction:
        cls_probs = self(x)
        return self.inference_from_scores(x, cls_probs)
