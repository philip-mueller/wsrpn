from collections import OrderedDict
from functools import partial
from typing import List

import torch
from torch import Tensor

from src.model.backbone.backbone_loader import Backbone, BACKBONE_LOADERS, BackboneConfig


class ViTBackbone(Backbone):
    LAYERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    DEFAULT_LAYER = '11'

    def __init__(self, config: BackboneConfig, vit: torch.nn.Module):
        patch_size = vit.patch_embed.patch_size
        patch_size = patch_size[0] if hasattr(patch_size, '__getitem__') else patch_size
        self.patch_size = patch_size
        self.num_heads = vit.blocks[0].attn.num_heads

        d = {layer: vit.embed_dim for layer in self.LAYERS}
        downscale_factor = {layer: patch_size for layer in self.LAYERS}
        super(ViTBackbone, self).__init__(d=d, downscale_factor=downscale_factor, config=config)

        self.vit = vit
        self.extracted_layers = None
        self.num_encoder_layers = 0
        self.feat_out = {}

    def set_extracted_feature_layers(self, extracted_layers: List[str]):
        self.extracted_layers = extracted_layers
        self.num_encoder_layers = max(int(layer_name) for layer_name in extracted_layers) + 1
        assert len(self.extracted_layers) > 0

        def hook_fn_forward_qkv(module, input, output, layer_name):
            # Use key values like in https://arxiv.org/pdf/2205.07839.pdf
            self.feat_out[layer_name] = output

        for layer_index in self.extracted_layers:
            hook_fn = partial(hook_fn_forward_qkv, layer_name=layer_index)
            self.vit.blocks._modules[str(layer_index)].attn.qkv.register_forward_hook(hook_fn)

    def forward(self, x: Tensor):
        assert self.extracted_layers is not None, \
            'set_extracted_feature_layers() has to be called before calling forward()'
        extracted_features = OrderedDict()

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b, _, h, w = x.shape
        h_, w_ = h // self.patch_size, w // self.patch_size
        t = h_ * w_ + 1  # +1 for [CLS] token

        self.vit(x)
        for i in self.extracted_layers:
            # Use key values like in https://arxiv.org/pdf/2205.07839.pdf
            f = self.feat_out[i]  # [b, t, 3 * d]
            f = f.reshape(b, t, 3, -1)  # [b, t, 3, d]
            f = f[:, 1:, 1, :]  # [b, t - 1, d] select only key and omit [CLS] token
            f = f.transpose(1, 2).unflatten(2, (h_, w_))  # [b, d, h_, w_]
            extracted_features[i] = f

        self.feat_out.clear()

        return extracted_features

    @staticmethod
    def load(config: BackboneConfig):
        vit = torch.hub.load(config.pretraining_repo, config.backbone_architecture,
                             pretrained=config.backbone_pretrained)
        return ViTBackbone(config, vit)

    @classmethod
    def register(cls):
        BACKBONE_LOADERS['dino_vits8'] = ViTBackbone.load
        BACKBONE_LOADERS['vit_base_patch8_224'] = ViTBackbone.load
