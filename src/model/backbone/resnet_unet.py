import logging
from collections import OrderedDict
from functools import partial
from itertools import chain
from typing import List, Collection, Set, Tuple, Dict

import torch
from torch import nn
from torchvision.models import ResNet
from torch.nn import functional as F

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig
from src.model.backbone.resnet import ResNetBackbone

log = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if with_nonlinearity else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py

    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResUNetBackbone(Backbone):
    LAYERS = {'input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'bridge', 'up1', 'up2', 'up3', 'up4', 'up5'}
    DEFAULT_LAYER = 'up5'
    """
    Based on https://github.com/kevinlu1211/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
    """
    def __init__(self, config: BackboneConfig, resnet: ResNetBackbone, up_dims: Dict[str, int] = None):
        assert isinstance(resnet, ResNetBackbone)

        d = dict(resnet.d)  # init downsampling d from ResNet backbone
        default_up_dims = {
            'bridge': 2048,
            'up1': 1024,
            'up2': 512,
            'up3': 256,
            'up4': 128,
            'up5': 64,
        }
        up_dims = dict(default_up_dims, **up_dims) if up_dims is not None else default_up_dims
        d.update(up_dims)
        log.info(f'Using the following dims in ResUNet: {self.d}')
        downscale_factor = {
            'input': 1,
            'conv1': 2,
            'conv2': 4,
            'conv3': 8,
            'conv4': 16,
            'conv5': 32,
            'bridge': 32,
            'up1': 16,
            'up2': 8,
            'up3': 4,
            'up4': 2,
            'up5': 1,
        }

        super(ResUNetBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)

        resnet.set_extracted_feature_layers(['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'])
        self.resnet = resnet
        self.bridge = Bridge(self.d['conv5'], self.d['bridge'])
        self.up_blocks = nn.ModuleList([
            UpBlockForUNetWithResNet50(in_channels=self.d['up1'] + self.d['conv4'], out_channels=self.d['up1'],
                                       up_conv_in_channels=self.d['bridge'], up_conv_out_channels=self.d['up1']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up2'] + self.d['conv3'], out_channels=self.d['up2'],
                                       up_conv_in_channels=self.d['up1'], up_conv_out_channels=self.d['up2']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up3'] + self.d['conv2'], out_channels=self.d['up3'],
                                       up_conv_in_channels=self.d['up2'], up_conv_out_channels=self.d['up3']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up4'] + self.d['conv1'], out_channels=self.d['up4'],
                                       up_conv_in_channels=self.d['up3'], up_conv_out_channels=self.d['up4']),
            UpBlockForUNetWithResNet50(in_channels=self.d['up5'] + self.d['input'], out_channels=self.d['up5'],  # concatenated with input
                                       up_conv_in_channels=self.d['up4'], up_conv_out_channels=self.d['up5'])
        ])
        self.extracted_layers = None

    def set_extracted_feature_layers(self, extracted_layers: Collection[str]):
        self.extracted_layers = set(extracted_layers)
        assert len(self.extracted_layers) > 0
        assert all(layer in set(self.d.keys()) for layer in extracted_layers)

    def forward(self, x, frozen_backbone=False) -> OrderedDict[str, torch.Tensor]:
        assert self.extracted_layers is not None, \
            'set_extracted_feature_layers() has to be called before calling forward()'
        extracted_features = OrderedDict()
        if frozen_backbone:
            with torch.no_grad():
                x, down_outputs, extracted_features = self.downsample(x, extracted_features)
        else:
            x, down_outputs, extracted_features = self.downsample(x, extracted_features)

        x = self.bridge(x)
        if 'bridge' in self.extracted_layers:
            extracted_features['bridge'] = x

        extracted_features = self.upsample(x, down_outputs, extracted_features)
        return extracted_features

    def downsample(self, x, extracted_features):
        resnet_features = self.resnet(x)

        x = resnet_features['conv5']
        down_outputs = [resnet_features[feature] for feature in ('input', 'conv1', 'conv2', 'conv3', 'conv4')]
        for layer, value in resnet_features.items():
            if layer in self.extracted_layers:
                extracted_features[layer] = value
        return x, down_outputs, extracted_features

    def upsample(self, x, down_outputs, extracted_features):
        for i, block in enumerate(self.up_blocks):
            x = block(x, down_outputs.pop())

            if f'up{i + 1}' in self.extracted_layers:
                extracted_features[f'up{i + 1}'] = x
        return extracted_features

    def backbone_params(self):
        return self.resnet.parameters()

    def non_backbone_params(self):
        return chain(self.bridge.parameters(), self.up_blocks.parameters())

    @staticmethod
    def for_torchvision_resnet(resnet: ResNet, up_dims=None):
        resnet = ResNetBackbone(resnet=resnet)
        return ResUNetBackbone(resnet, up_dims=up_dims)

    @staticmethod
    def load(config: BackboneConfig):
        # load the corresponding resnet
        resnet_mappings = {
            'resnet18_unet': 'resnet18',
            'resnet34_unet': 'resnet34',
            'resnet50_unet': 'resnet50',
            'resnet101_unet': 'resnet101',
            'resnet152_unet': 'resnet152'
        }
        assert config.backbone_architecture in resnet_mappings.keys()
        resnet_architecture = resnet_mappings[config.backbone_architecture]
        resnet = torch.hub.load('pytorch/vision:v0.6.0', resnet_architecture,
                                pretrained=config.backbone_pretrained)
        assert isinstance(resnet, ResNet), type(resnet)
        resnet = ResNetBackbone(config, resnet)

        # build resunet from the given resnet
        return ResUNetBackbone(config, resnet, **config.backbone_kwargs)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['resnet18_unet'] = ResUNetBackbone.load
        BACKBONE_LOADERS['resnet34_unet'] = ResUNetBackbone.load
        BACKBONE_LOADERS['resnet50_unet'] = ResUNetBackbone.load
        BACKBONE_LOADERS['resnet101_unet'] = ResUNetBackbone.load
        BACKBONE_LOADERS['resnet152_unet'] = ResUNetBackbone.load
