"""
Model as in "Weakly Supervised Deep Detection Networks"
https://arxiv.org/abs/1511.02853
"""
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch import Tensor
from torchvision.ops import roi_pool, nms

from src.model.backbone import BackboneConfig
from src.model.backbone.backbone_loader import load_backbone
from src.model.image_box_proposals import (edge_boxes, random_boxes,
                                           selective_search)
from src.model.losses import weighted_binary_cross_entropy
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)


@dataclass
class WSDDNConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    backbone: BackboneConfig = MISSING

    proposal_method: str = MISSING
    n_random_proposals: int = MISSING
    nms_threshold: float = MISSING

    # Hidden dimension of fc6 and fc7
    hidden_dim: int = MISSING

    # Losses
    use_bce_loss: bool = MISSING

    # Only needed for train.py
    use_superpixels: bool = False


class WSDDN(ObjectDetectorModelInterface):
    CONFIG_CLS = WSDDNConfig

    def __init__(self, config: WSDDNConfig) -> None:
        super(WSDDN, self).__init__(config)
        self.config = config

        # Init Backbone model
        self.backbone = load_backbone(self.config.backbone)
        self.backbone_layer = self.backbone.DEFAULT_LAYER
        self.backbone.set_extracted_feature_layers([self.backbone_layer])
        num_feats = self.backbone.d[self.backbone_layer]

        # self.fc67 = nn.Sequential(
        #     nn.Linear(num_feats, config.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_dim, config.hidden_dim),
        #     nn.ReLU()
        # )
        # self.fcc = nn.Linear(config.hidden_dim, config.num_classes)
        # self.fcd = nn.Linear(config.hidden_dim, config.num_classes)
        self.fcc = nn.Linear(num_feats, config.num_classes)
        self.fcd = nn.Linear(num_feats, config.num_classes)

    def get_box_proposals(self, x: Tensor) -> Tensor:
        """Get region proposals from the input image.

        :param x: (1, C, H, W) Input image
        :return box_proposals: (N, 4) Bounding box proposals, each (x, y, h, w)
                               and relative to the image size
        """
        if self.config.proposal_method == 'selective_search':
            return selective_search(x[0])
        elif self.config.proposal_method == 'edge_boxes':
            return edge_boxes(x[0])
        elif self.config.proposal_method == 'random_boxes':
            return random_boxes(self.config.n_random_proposals)
        else:
            raise ValueError("Invalid mode for proposal_method:",
                             self.config.proposal_method)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: Input batch of images (1 x C x H x W)

        :return cls_probs: Class probabilities (N x CL)
        """
        assert len(x) == 1, "Batch size must be 1"

        # Get box proposals
        box_proposals = self.get_box_proposals(x.cpu()).to(x.device)
        M = box_proposals.shape[0]

        # Get features from backbone
        features = self.backbone(x)[self.backbone_layer]  # (1, d, h, w)
        box_scale = features.shape[2]  # Boxes are [0, 1]

        # Map regions to features and get pooled features
        pooled_feats = roi_pool(features, [box_proposals], (1, 1),
                                spatial_scale=box_scale)  # (M, d, 1, 1)

        # Classification and detection
        # out = self.fc67(pooled_feats.view(M, -1))  # (M, hidden_dim)
        # classification_scores = F.softmax(self.fcc(out), dim=1)  # (M, CL)
        # detection_scores = F.softmax(self.fcd(out), dim=0)  # (M, CL)
        classification_scores = F.softmax(self.fcc(pooled_feats.view(M, -1)), dim=1)  # (M, CL)
        detection_scores = F.softmax(self.fcd(pooled_feats.view(M, -1)), dim=0)  # (M, CL)
        combined_scores = classification_scores * detection_scores  # (M, CL)
        cls_probs = combined_scores.sum(0, keepdim=True).clamp(0, 1)  # (1, CL)

        return (
            box_proposals,
            classification_scores,
            detection_scores,
            cls_probs
        )

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
        (box_proposals,
         classification_scores,
         detection_scores,
         cls_probs) = self(x)

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
            predictions = self.inference_from_scores(
                box_proposals,
                classification_scores,
                detection_scores,
                cls_probs,
            )
        else:
            predictions = None

        return loss, losses, predictions

    def inference_from_scores(
        self,
        box_proposals: Tensor,
        classification_scores: Tensor,
        detection_scores: Tensor,
        cls_probs: Tensor
    ) -> ObjectDetectorPrediction:
        """Get bounding box predictions for a batch of images
        :param box_proposals: Proposed bounding boxes (M, 4)
        :param classification_scores: Class score per box and class (M, CL)
        :param detection_scores: Objectness score per box and class (M, CL)
        :param cls_probs: Predicted class probabilites (1, CL)

        :return prediction: Predicted bounding boxes, probabilities, and classes
        """
        N, CL = cls_probs.shape

        # Get everything to CPU
        box_proposals = box_proposals.cpu()
        classification_scores = classification_scores.cpu()
        detection_scores = detection_scores.cpu()
        cls_probs = cls_probs.cpu()

        # Classification results
        cls_probs = cls_probs.float()
        global_prediction_probs = cls_probs[..., :-1]  # (1, CL - 1)
        classification_scores = classification_scores[:, :-1]  # (M, CL - 1)
        global_obj_probs = cls_probs[..., -1]  # (1)
        global_prediction_hard = (global_prediction_probs > 0.5).int()

        # Detection results
        obj_probs = detection_scores.sum(1)  # (M)
        box_keep_inds = nms(box_proposals, obj_probs, self.config.nms_threshold)  # (m)
        box_proposals = box_proposals[box_keep_inds]  # (m, 4)
        obj_probs = obj_probs[box_keep_inds, None]  # (m, 1)
        detection_scores = detection_scores[box_keep_inds]  # (m, CL)
        classification_scores = classification_scores[box_keep_inds]  # (m, CL)
        classification_preds = classification_scores.argmax(1, keepdim=True)  # (m, 1)

        return ObjectDetectorPrediction(
            global_prediction_hard=global_prediction_hard,
            global_prediction_probs=global_prediction_probs,
            global_obj_probs=global_obj_probs,
            box_prediction_hard=[torch.cat([
                box_proposals, classification_preds, obj_probs
            ], dim=-1)],
            box_prediction_probs=[torch.cat([
                box_proposals, obj_probs, classification_scores
            ], dim=-1)],
        )

    def inference(self, x: Tensor, **kwargs) -> ObjectDetectorPrediction:
        (box_proposals,
         classification_scores,
         detection_scores,
         cls_probs) = self(x)
        return self.inference_from_scores(
            box_proposals,
            classification_scores,
            detection_scores,
            cls_probs,
        )
