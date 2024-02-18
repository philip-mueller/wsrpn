import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional

import torch
from omegaconf import MISSING
from src.utils.utils import config_to_dict, prepare_config
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class BackboneConfig:
    backbone_architecture: str = MISSING
    backbone_pretrained: bool = MISSING
    pretraining_repo: str = MISSING
    backbone_checkpoint: Optional[str] = MISSING
    backbone_checkpoint_prefix: str = ''
    backbone_kwargs: Optional[Dict[str, Any]] = MISSING
    freeze_backbone: bool = MISSING
    ignore_last_stride: bool = MISSING


class Backbone(ABC, nn.Module):
    LAYERS = None
    DEFAULT_LAYER = None

    def __init__(self, d: Dict[str, int], downscale_factor: Dict[str, int], config: BackboneConfig):
        super(Backbone, self).__init__()
        self.d = d
        self.downscale_factor = downscale_factor
        self.config = config

    @abstractmethod
    def set_extracted_feature_layers(self, extracted_layers: Collection[str]):
        ...

    @classmethod
    @abstractmethod
    def register(cls):
        ...

    def save(self, checkpoint_path: str):
        saved_dict = {
            'config': config_to_dict(self.config),
            'state_dict': self.state_dict()
        }
        torch.save(saved_dict, checkpoint_path)


BACKBONE_LOADERS = dict()


def load_backbone(config: BackboneConfig) -> Backbone:
    config = prepare_config(config, BackboneConfig, log)

    # Check if backbone is registered
    assert config.backbone_architecture in BACKBONE_LOADERS, \
        f'{config.backbone_architecture} is an unknwon architecture. Options are: {set(BACKBONE_LOADERS.keys())}'

    # Load backbone
    config.backbone_kwargs = config.backbone_kwargs if config.backbone_kwargs is not None else {}
    backbone = BACKBONE_LOADERS[config.backbone_architecture](config)

    # Load pretrained weights
    if config.backbone_checkpoint is not None:
        log.info(f"Loading backbone weights from {config.backbone_checkpoint}")
        missing, unexpected = backbone.load_state_dict({config.backbone_checkpoint_prefix + key: value 
            for key, value in torch.load(config.backbone_checkpoint).items()}, strict=False)
        if len(missing) > 0:
            log.warn(f'Missing backbone weights: {missing}')
        if len(unexpected) > 0:
            log.warn(f'Unexpected backbone weights: {unexpected}')

    # Freeze backbone
    if config.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    return backbone
