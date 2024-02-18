
from collections import OrderedDict
from functools import partial
from typing import List

from torch import nn
from src.model.backbone.backbone_loader import BACKBONE_LOADERS, Backbone, BackboneConfig


import torchxrayvision as xrv
xrv.models.ResNet


model = xrv.models.DenseNet(weights="nih")


class CXRDenseNetBackbone(Backbone):
    LAYERS = ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']
    DEFAULT_LAYER = 'denseblock4'

    def __init__(self, config: BackboneConfig, dense_net: xrv.models.DenseNet):
        module_indices_by_name = {
            name: i for i, (name, module) in enumerate(dense_net.features.named_children())
            if name in CXRDenseNetBackbone.LAYERS
        }

        def get_d(name):
            next_module = dense_net.features[module_indices_by_name[name] + 1]
            if isinstance(next_module, xrv.models._Transition):
                return next_module.norm.num_features
            elif isinstance(next_module, nn.BatchNorm2d):
                return next_module.num_features

        d = {layer: get_d(layer) for layer in self.LAYERS}
        downscale_factor = {
            f'denseblock{layer_index+1}': 2 ** (layer_index + 1) for layer_index in range(4)
        }
        super(CXRDenseNetBackbone, self).__init__(
            d=d, downscale_factor=downscale_factor, config=config)

        self.dense_net = dense_net
        self.extracted_layers = None
        self.feat_out = {}

    def set_extracted_feature_layers(self, extracted_layers: List[str]):
        assert all(layer in CXRDenseNetBackbone.LAYERS for layer in extracted_layers), \
            f'Some extracted layers ({extracted_layers}) are not in {CXRDenseNetBackbone.LAYERS}'
        self.extracted_layers = extracted_layers
        assert len(self.extracted_layers) > 0

        def hook_fn_forward(module, input, output, layer_name):
            self.feat_out[layer_name] = output

        for layer_name in self.extracted_layers:
            hook_fn = partial(hook_fn_forward, layer_name=layer_name)
            getattr(self.dense_net.features, layer_name).register_forward_hook(hook_fn)

    def forward(self, x):
        x = xrv.models.fix_resolution(x, 224, self.dense_net)
        xrv.models.warn_normalization(x)

        assert self.extracted_layers is not None, \
            'set_extracted_feature_layers() has to be called before calling forward()'

        self.dense_net(x)
        extracted_features = OrderedDict()
        for name in self.extracted_layers:
            extracted_features[name] = self.feat_out[name]
        self.feat_out.clear()

        return extracted_features

    @staticmethod
    def load(config: BackboneConfig):
        assert config.pretraining_repo == 'xrv'
        if config.backbone_pretrained:
            weights = None
        else:
            assert config.backbone_architecture.startswith('densenet121')
            weights = config.backbone_architecture
        dense_net = xrv.models.DenseNet(weights=weights)
        return CXRDenseNetBackbone(config, dense_net)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['densenet121-res224-all'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-rsna'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-nih'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-pc'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-chex'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-mimic_nb'] = CXRDenseNetBackbone.load
        BACKBONE_LOADERS['densenet121-res224-mimic_ch'] = CXRDenseNetBackbone.load
