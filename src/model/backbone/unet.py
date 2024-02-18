from collections import OrderedDict
from typing import List, Collection

import torch

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig


class UNetBackbone(Backbone):
    LAYERS = {'input', 'conv', 'down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4'}
    DEFAULT_LAYER = 'up4'

    def __init__(self, config: BackboneConfig):
        d = {
            'input': 3,
            'conv': 64,
            'down1': 128,
            'down2': 256,
            'down3': 512,
            'down4': 1024,
            'up1': 512,
            'up2': 256,
            'up3': 128,
            'up4': 64,
        }
        downscale_factor = {
            'input': 1,
            'conv': 1,
            'down1': 2,
            'down2': 4,
            'down3': 8,
            'down4': 16,
            'up1': 8,
            'up2': 4,
            'up3': 2,
            'up4': 1,
        }
        super(UNetBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)
        raise NotImplementedError  # TODO: add pl_bolts lib or replace UNet implementation
        from pl_bolts.models.vision import UNet
        unet = UNet(num_classes=1)  # classifier layer will not be used
        self.layers = unet.layers[:-1]  # remove classifier layer
        assert len(self.layers) == 9

        self.extracted_layers = None

    def set_extracted_feature_layers(self, extracted_layers: Collection[str]):
        self.extracted_layers = set(extracted_layers)
        assert len(self.extracted_layers) > 0
        assert all(layer in set(self.d.keys()) for layer in extracted_layers)

    def forward(self, x) -> OrderedDict[str, torch.Tensor]:
        assert self.extracted_layers is not None, \
            'set_extracted_feature_layers() has to be called before calling forward()'
        extracted_features = OrderedDict()
        if 'input' in self.extracted_layers:
            extracted_features['input'] = x

        xi = [self.layers[0](x)]
        if 'conv' in self.extracted_layers:
            extracted_features['conv'] = xi[0]

        # Down path
        for i, layer in enumerate(self.layers[1:self.num_layers]):
            xi.append(layer(xi[-1]))
            if f'down{i+1}' in self.extracted_layers:
                extracted_features[f'down{i+1}'] = xi[-1]

        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
            if f'up{i+1}' in self.extracted_layers:
                extracted_features[f'up{i+1}'] = xi[-1]

        return extracted_features

    @staticmethod
    def load(config: BackboneConfig):
        assert config.backbone_architecture == 'unet'
        assert not config.backbone_pretrained
        return UNetBackbone(config)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['unet'] = UNetBackbone.load
