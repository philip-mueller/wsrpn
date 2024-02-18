from collections import OrderedDict
from itertools import islice
from typing import List

import torch
from torch import nn
from torchvision.models import VGG

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig


class VGGBackbone(Backbone):
    """Layers grouped as in https://pytorch.org/assets/images/vgg.png"""
    LAYERS = {'input', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5'}
    DEFAULT_LAYER = 'layer5'

    def __init__(self, config: BackboneConfig, vgg: VGG):

        is_vgg16 = 'vgg16' in config.backbone_architecture.lower()
        is_vgg19 = 'vgg19' in config.backbone_architecture.lower()
        if not is_vgg16 and not is_vgg19:
            raise NotImplementedError("Currently only supports vgg16 and vgg19")

        d = {
            'input': 3,
            'layer1': 64,
            'layer2': 128,
            'layer3': 256,
            'layer4': 512,
            'layer5': 512,
        }
        downscale_factor = {
            'input': 1,
            'layer1': 1,
            'layer2': 2,
            'layer3': 4,
            'layer4': 8,
            'layer5': 16
        }
        super(VGGBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)

        self.layer_name_to_index = {
            'input': -1,
            'layer1': 3,
            'layer2': 8,
            'layer3': 15 if is_vgg16 else 17,
            'layer4': 22 if is_vgg16 else 26,
            'layer5': 29 if is_vgg16 else 35
        }
        self.backbone_layers = nn.ModuleDict(list(vgg.features.named_children()))
        self.extracted_layers_by_index = None
        self.num_encoder_layers = 0

    def set_extracted_feature_layers(self, extracted_layers: List[str]):
        self.extracted_layers_by_index = {
            self.layer_name_to_index[extracted_name]: extracted_name for extracted_name in extracted_layers
        }
        self.num_encoder_layers = max(self.layer_name_to_index[layer_name] for layer_name in extracted_layers) + 1
        assert len(self.extracted_layers_by_index) > 0

    def forward(self, x) -> OrderedDict[str, torch.Tensor]:
        assert self.extracted_layers_by_index is not None, \
            'set_extracted_feature_layers() has to be called before calling forward()'
        extracted_features = OrderedDict()

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # save input feature map if required (index = -1)
        extracted_name = self.extracted_layers_by_index.get(-1)
        if extracted_name is not None:
            extracted_features[extracted_name] = x

        for index, layer in enumerate(islice(self.backbone_layers.values(), self.num_encoder_layers)):
            # apply layer
            x = layer(x)

            # extract the output feature map of this layer if required
            extracted_name = self.extracted_layers_by_index.get(index)

            if extracted_name is not None:
                extracted_features[extracted_name] = x

        return extracted_features

    @staticmethod
    def load(config: BackboneConfig):
        resnet = torch.hub.load(config.pretraining_repo, config.backbone_architecture,
                                pretrained=config.backbone_pretrained)
        assert isinstance(resnet, VGG), type(resnet)
        return VGGBackbone(config, resnet)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['vgg16'] = VGGBackbone.load
        BACKBONE_LOADERS['vgg19'] = VGGBackbone.load
