from collections import OrderedDict
from itertools import islice
from typing import List

import torch
from torch import nn
from torchvision.models import ResNet

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig


class ResNetBackbone(Backbone):
    LAYERS = {'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}
    DEFAULT_LAYER = 'conv5'

    def __init__(self, config: BackboneConfig, resnet: ResNet):

        def get_d(layer: nn.Module):
            last_layer = list(layer[-1].children())[-2]
            if isinstance(last_layer, nn.Conv2d):
                return last_layer.out_channels
            elif isinstance(last_layer, nn.BatchNorm2d):
                return last_layer.num_features
            else:
                raise ValueError(f'Unknown layer type: {last_layer}')

        d = {
            'input': 3,
            'conv1': resnet.conv1.out_channels,  # 64,
            'conv2': get_d(resnet.layer1),
            'conv3': get_d(resnet.layer2),
            'conv4': get_d(resnet.layer3),
            'conv5': get_d(resnet.layer4),
        }
        downscale_factor = {
            'input': 1,
            'conv1': 2,
            'conv2': 4,
            'conv3': 8,
            'conv4': 16,
            'conv5': 32
        }
        super(ResNetBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)

        self.layer_name_to_index = {
            'input': -1,
            'conv1': 2,  # 0: conv1, 1: bn1, 2: relu
            'conv2': 4,  # 3: maxpool, 4: layer1
            'conv3': 5,  # layer2
            'conv4': 6,  # layer3
            'conv5': 7   # layer4
        }
        self.backbone_layers = nn.ModuleDict(list(resnet.named_children())[:-2])  # avg pool and final fc layers are never used
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
        assert isinstance(resnet, ResNet), type(resnet)
        return ResNetBackbone(config, resnet)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['resnet18'] = ResNetBackbone.load
        BACKBONE_LOADERS['resnet34'] = ResNetBackbone.load
        BACKBONE_LOADERS['resnet50'] = ResNetBackbone.load
        BACKBONE_LOADERS['resnet101'] = ResNetBackbone.load
        BACKBONE_LOADERS['resnet152'] = ResNetBackbone.load
        BACKBONE_LOADERS['dino_resnet50'] = ResNetBackbone.load
