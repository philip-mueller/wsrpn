
from collections import OrderedDict
from itertools import islice
from typing import List

import torch
from torch import nn
from torchvision.models import DenseNet

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig


class DenseNetBackbone(Backbone):
    LAYERS = {
        'input',
        'conv0',
        'denseblock1',
        'denseblock2',
        'denseblock3',
        'denseblock4',
        'norm5',
    }
    DEFAULT_LAYER = 'denseblock4'

    def __init__(self, config: BackboneConfig, densenet: DenseNet):

        def d_denseblock(block):
            layer = list(block.children())[-1]
            d_norm = layer.norm1.num_features
            d_conv = layer.conv2.out_channels
            return d_norm + d_conv

        d = {
            'input': 3,
            'conv0': densenet.features.conv0.out_channels,  # 64,
            'denseblock1': d_denseblock(densenet.features.denseblock1),  # 256
            'denseblock2': d_denseblock(densenet.features.denseblock2),  # 512
            'denseblock3': d_denseblock(densenet.features.denseblock3),  # 1024
            'denseblock4': d_denseblock(densenet.features.denseblock4),  # 1024
            'norm5': densenet.features.norm5.num_features,  # 1024
        }
        downscale_factor = {
            'input': 1,
            'conv0': 2,
            'denseblock1': 4,
            'denseblock2': 8,
            'denseblock3': 16,
            'denseblock4': 32,
            'norm5': 32,
        }
        super(DenseNetBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)

        self.layer_name_to_index = {
            'input': -1,
            'conv0': 0,
            'denseblock1': 4,
            'denseblock2': 6,
            'denseblock3': 8,
            'denseblock4': 10,
            'norm5': 11
        }
        self.backbone_layers = nn.ModuleDict(list(densenet.features.named_children()))
        self.extracted_layers_by_index = None
        self.num_encoder_layers = 0

        if config.ignore_last_stride:
            self.backbone_layers.transition3.pool.kernel_size = 1
            self.backbone_layers.transition3.pool.stride = 1

    def set_extracted_feature_layers(self, extracted_layers: List[str]):
        self.extracted_layers_by_index = {
            self.layer_name_to_index[name]: name for name in extracted_layers
        }
        self.num_encoder_layers = max(
            self.layer_name_to_index[name] for name in extracted_layers
        ) + 1
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

        for index, layer in enumerate(islice(self.backbone_layers.values(),
                                             self.num_encoder_layers)):
            # apply layer
            x = layer(x)

            # extract the output feature map of this layer if required
            extracted_name = self.extracted_layers_by_index.get(index)

            if extracted_name is not None:
                extracted_features[extracted_name] = x

        return extracted_features

    @staticmethod
    def load(config: BackboneConfig):
        densenet = torch.hub.load(
            config.pretraining_repo,
            config.backbone_architecture,
            pretrained=config.backbone_pretrained)
        assert isinstance(densenet, DenseNet), type(densenet)
        return DenseNetBackbone(config, densenet)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['densenet121'] = DenseNetBackbone.load
