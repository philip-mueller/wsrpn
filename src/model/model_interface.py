import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Type, Tuple, Dict

import torch
from omegaconf import MISSING, OmegaConf
from torch import nn


log = logging.getLogger(__name__)


@dataclass
class ObjectDetectorPrediction:
    global_prediction_hard: torch.Tensor  # (N x C) multi-one-hot encoded
    global_prediction_probs: torch.Tensor  # (N x C) probs in [0, 1]
    global_obj_probs: torch.Tensor  # (N) obj_probs in [0, 1]

    # List of length N
    # Each element is of shape (M_i x 6)
    # (x, y, w, h, class_id, obj_score)
    # where
    #   x, y is the upper-left point of the box (in image pixel coordinates)
    #   class_id is an integer in [0, C)
    #   confidence is a float in [0, 1]
    box_prediction_hard: List[torch.Tensor]

    # List of length N
    # Each element is of shape (M_i x (5 + C))
    # (x, y, w, h, obj_score, class probs...)
    # where
    #   x, y is the upper-left point of the box (in image pixel coordinates)
    #   obj_score is prob in [0, 1]
    #   sum(class_probs...) = 1
    box_prediction_probs: List[torch.Tensor]

    aggregated_patch_prediction_hard: Optional[torch.Tensor] = None
    aggregated_patch_prediction_probs: Optional[torch.Tensor] = None
    aggregated_roi_prediction_hard: Optional[torch.Tensor] = None
    aggregated_roi_prediction_probs: Optional[torch.Tensor] = None

    seg_masks_from_rois: Optional[torch.Tensor] = None
    seg_masks_from_patches: Optional[torch.Tensor] = None
    seg_masks_from_superpixels: Optional[torch.Tensor] = None


@dataclass
class ModelConfig:
    model_type: str = MISSING


class ObjectDetectorModelInterface(ABC, nn.Module):
    CONFIG_CLS: Type = None

    def __init__(self, config):
        super(ObjectDetectorModelInterface, self).__init__()
        assert OmegaConf.get_type(config) is self.CONFIG_CLS
        self.config = config

    @abstractmethod
    def train_step(
        self,
        x: torch.Tensor,
        global_label: torch.Tensor,
        return_predictions=False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], ObjectDetectorPrediction]:
        """

        :param x: Images (N x H x W x 3)
        :param global_label: Image class labels, multi-one-hot encoded (N x C)
        :param return_predictions:
        :return: (loss, losses, predictions)
        - loss: total loss to opimizer
        - losses: dict of individual losses if the loss function consists of multiple losses
        - predictions
        """
        ...

    @abstractmethod
    def inference(self, x: torch.Tensor) -> ObjectDetectorPrediction:
        """

        :param x: Images (N x H x W x 3)
        :return: predictions (detached and on CPU)
        """
        ...

    def save_model(self, checkpoint_path: str, **kwargs):
        config_dict: dict = OmegaConf.to_container(self.config)
        state_dict = self.state_dict()
        ckpt_dict = {'config_dict': config_dict, 'state_dict': state_dict, **kwargs}
        torch.save(ckpt_dict, checkpoint_path)
        log.info(f'Saved checkpoint: {checkpoint_path}')

    @classmethod
    def register(cls):
        from src.model.model_loader import register_model_class
        register_model_class(cls)
