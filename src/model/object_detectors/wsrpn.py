from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import MISSING

from src.model.backbone.backbone_loader import BackboneConfig, load_backbone
from src.model.inference import apply_nms, assemble_box_predictions
from src.model.losses import (RoiPatchClassConsistencyLoss, SupConPerClassLoss,
                              weighted_binary_cross_entropy_wsrpn)
from src.model.model_components import (MLP,
                                        RoiTokenAttentionLayer)
from src.model.model_interface import (ModelConfig,
                                       ObjectDetectorModelInterface,
                                       ObjectDetectorPrediction)
from src.model.positional_embedding import positional_embedding_2d
from src.model.soft_roi_pool import BBoxMLP, SoftRoiPool, get_sample_grid
from src.utils.utils import to_device
from torch import Tensor, autocast


@dataclass
class WSRPNConfig(ModelConfig):
    model_type: str = MISSING
    num_classes: int = MISSING
    
    # Encoding
    backbone: BackboneConfig = MISSING
    backbone_layer: str = MISSING
    upsample_factor: int = MISSING
    pos_emb_type: str = MISSING
    d_emb: int = MISSING
    mlp_d_hidden: int = MISSING

    # Tokens and Token attention
    n_roi_tokens: int = MISSING
    stop_grad_to_features: bool = MISSING

    # soft ROI pooling
    # gpp is Gaussian Parameter Predictor
    n_gpp_layers: int = MISSING  # 0 means no MLP just a linear classifier (before sigmoid)
    gpp_d_hidden: int = MISSING
    gpp_use_offsets: bool = MISSING
    gpp_pos_emb: bool = MISSING
    gpp_use_ratios: bool = MISSING
    generalized_gaussian_beta: float = MISSING

    # Aggregation (per-class features from patches/ROIs)
    patch_aggregation: str = MISSING
    roi_aggregation: str = MISSING
    apply_obj_probs_to_classes: bool = MISSING
    prob_mode: str = MISSING
    use_cls_tokens: bool = MISSING
    lse_r: float = MISSING
    
    # Loss functions
    supcon_loss: Dict[str, Any] = MISSING
    patch_bce_loss: Dict[str, Any] = MISSING
    roi_bce_loss: Dict[str, Any] = MISSING
    roi_patch_cls_consistency_loss: Dict[str, Any] = MISSING
    use_patch_supcon: bool = MISSING
    use_patch_bce: bool = MISSING
    use_roi_supcon: bool = MISSING
    use_roi_bce: bool = MISSING
    use_roi_patch_cls_consistency: bool = MISSING

    # if None -> consider it if no finding is not the majority class
    obj_threshold: Optional[float] = None

class WSRPN(ObjectDetectorModelInterface):
    CONFIG_CLS = WSRPNConfig

    @dataclass
    class Features:
        features: Tensor
        cls_probs: Tensor
        aggregated_cls_features: Tensor
        aggregated_cls_probs: Tensor
        aggregated_or_probs: Tensor
        aggregated_and_probs: Tensor

    def __init__(self, config: WSRPNConfig):
        super(WSRPN, self).__init__(config)
        self.config = config

        # --- Encoding of patches ---
        # Init backbone
        self.backbone = load_backbone(config.backbone)
        self.backbone.set_extracted_feature_layers([config.backbone_layer])
        self.d_backbone = self.backbone.d[config.backbone_layer]
        self.d_emb = self.config.d_emb
        # Projection and upsampling
        if config.upsample_factor == 1:
            self.upsample_project = nn.Sequential(
                nn.BatchNorm2d(self.d_backbone),
                nn.Conv2d(self.d_backbone, self.d_emb, kernel_size=1)
            )
        else:
            self.upsample_project = nn.Sequential(
                nn.BatchNorm2d(self.d_backbone),
                nn.ConvTranspose2d(
                    self.d_backbone, self.d_emb,
                    kernel_size=config.upsample_factor,
                    stride=config.upsample_factor
                ),
            )
        # Position embeddings for patches
        self.pos_emb = positional_embedding_2d(config.pos_emb_type,
                                               channels=self.d_emb)

        # --- ROI tokens and ROI pooling ---
        self.predict_rois = config.use_roi_bce or config.use_roi_supcon or config.use_roi_patch_cls_consistency
        # Init bounding box (ROI) prediction tokens
        self.register_parameter(
            'roi_tokens',
            nn.Parameter(torch.randn(1, self.config.n_roi_tokens, self.d_emb))
        )
        self.roi_token_att = RoiTokenAttentionLayer(
            d_embedding=self.d_emb,
            d_hidden=config.mlp_d_hidden,
            num_cross_att_heads=8,
            use_cross_att=True,
            skip_class_emb=False,
            use_mlp_out=False,
            gumbel=False,
            hard=False,
            intra_temporal_attention=False,
            dropout=0.3
        )
        # Predictor for ROI parameters
        self.norm_roi_tokens = nn.LayerNorm(self.d_emb)
        self.gpp = BBoxMLP(
            num_hidden_layers=config.n_gpp_layers,
            d_in=self.d_emb,
            d_hidden=config.gpp_d_hidden,
            use_ratios=config.gpp_use_ratios
        )
        # ROI pooling
        self.norm_before_roi_pool = nn.LayerNorm(self.d_emb)
        self.roi_feature_projector = nn.Linear(self.d_emb, self.d_emb)
        self.roi_pool = SoftRoiPool(beta=config.generalized_gaussian_beta)

        # --- Classifier of patches/superpixels/ROIs ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_emb),
            MLP(
                num_hidden_layers=1,
                d_in=self.d_emb,
                d_hidden=config.mlp_d_hidden,
                d_out=config.num_classes,
                dropout=0.3, dropout_last_layer=True
            )
        )

        # --- Loss functions ---
        if config.use_patch_supcon or config.use_roi_supcon:
            self.out_projector = MLP(
                num_hidden_layers=1,
                d_in=self.d_emb,
                d_hidden=config.mlp_d_hidden,
                d_out=config.d_emb,
                dropout=0.3, dropout_last_layer=True
            )
            self.supcon_loss = SupConPerClassLoss(**config.supcon_loss)
        if config.use_roi_patch_cls_consistency:
            self.roi_patch_cls_consistency_loss = RoiPatchClassConsistencyLoss(
                **config.roi_patch_cls_consistency_loss)
        
    def encode_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encodes the input image using the backbone, projectes/upsamples the
        feature map and adds pos embeddings
        :param x: input image (N x 3 x H x W)
        :return: (feature_map, pos_emb)
          - feature_map: (N x H x W x d) Encoded, upsampled and projected
                         feature map with pos embeddings
          - pos_emb: (N x H x W x 2) Position embeddings of feature map
        """
        feature_map = self.backbone(x)[self.config.backbone_layer]  # (N x d_backbone x H' x W')
        feature_map = self.upsample_project(feature_map)  # (N x d x H x W)
        feature_map = rearrange(feature_map, 'n d h w -> n h w d')  # (N x H x W x d)
        pos_emb = self.pos_emb(feature_map)  # (N x H x W x 2)
        feature_map = feature_map + pos_emb  # (N x H x W x d)
        return feature_map, pos_emb

    def classify(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Classify patches, superpixels, or ROIs based on their features
        :param features: patch/superpixel/ROI features
                         (N x H x W x d) / (N x n_sp x d) / (N x n_roi_tokens x d)
        :return cls_probs: Class distribution, sum over classes <= 1 (not necessarily == 1)
                           (N x H x W x C) / (N x n_sp x C) / (N x n_roi_tokens x C)
        """
        N, *dims, d = features.shape
        features = features.view(N, -1, d)

        cls_probs = self.classifier(features)  # (N x n_features x C)
        if self.config.prob_mode == 'sigmoid':
            cls_probs = torch.sigmoid(cls_probs)  # (N x n_features x C)
        elif self.config.prob_mode == 'softmax':
            cls_probs = torch.softmax(cls_probs, dim=-1)  # (N x n_features x C)
        else:
            raise ValueError(self.config.prob_mode)

        if self.config.apply_obj_probs_to_classes:
            # multiply each cls probs with 1.- no-finding-probs
            obj_probs = 1. - cls_probs[..., -1]
            cls_probs = cls_probs[..., :-1] * obj_probs[..., None]
            cls_probs = torch.cat([cls_probs, 1. - obj_probs[..., None]], dim=-1)
        return cls_probs.view(N, *dims, -1)  # (N x n_features x C)

    def apply_roi_token_att(self, patch_features, patch_pos_emb) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applies ROI token attention to get token features from patch features
        :param patch_features: Encoded feature map (N x H x W x d)
        :param patch_pos_emb: Position embeddings of feature map (N x H x W x 2)
        :return (tokens, reference_positions, reference_pos_emb)
            - tokens: Token features (N x n_roi_tokens x d)
            - reference_positions: Position of patch with highest attention per
                                   ROI (N x n_roi_tokens x 2)
            - reference_pos_emb: Position embedding of patch with highest
                                 attention per ROI (N x n_roi_tokens x d)
        """
        if self.config.stop_grad_to_features:
            patch_features = patch_features.detach()
            patch_pos_emb = patch_pos_emb.detach()
        n, h, w, d = patch_features.shape
        roi_tokens = self.roi_tokens.repeat(n, 1, 1)  # (N x n_roi_tokens x d)

        flattened_features = patch_features.view(n, -1, d)  # (N x (H * W) x d)

        # (N x n_roi_tokens x d), (N x n_roi_tokens x (H * W))
        tokens, token_map = self.roi_token_att(
            roi_tokens, flattened_features, flattened_features)

        # compute reference positions
        if self.config.gpp_use_offsets:
            token_map_index = token_map.argmax(-1, keepdim=True)  # (N x n_roi_tokens x 1)
            token_map_hard = torch.zeros_like(
                token_map,
                memory_format=torch.legacy_contiguous_format
            ).scatter_(-1, token_map_index, 1.0)  # (N x n_roi_tokens x (H_backbone * W_backbone))
            # straight through trick to backprop gradients
            token_map_hard = token_map_hard - token_map.detach() + token_map  # (N x n_roi_tokens x (H_backbone * W_backbone))
            token_map = token_map_hard  # (N x n_roi_tokens x (H_backbone * W_backbone))

            # Get reference points by attending on positions
            patch_positions = get_sample_grid(h, w, device=token_map.device,
                                              dtype=token_map.dtype)  # (H x W x 2)
            patch_positions = repeat(
                patch_positions, 'h w d -> n (h w) d', n=n)  # (N x (H_backbone * W_backbone) x 2)
            reference_positions = torch.bmm(token_map, patch_positions)  # (N x n_roi_tokens x 2)
            if self.config.gpp_pos_emb:
                patch_pos_emb = rearrange(patch_pos_emb, 'n h w d -> n (h w) d')
                reference_pos_emb = torch.bmm(token_map, patch_pos_emb)  # (N x n_roi_tokens x d)
            else:
                reference_pos_emb = None
        else:
            reference_positions = None
            reference_pos_emb = None

        return tokens, reference_positions, reference_pos_emb

    def apply_roi_pool(self, tokens, features, cls_probs,
                       reference_pos, reference_pos_emb):
        """
        Applies soft ROI pooling based on token features over patches or superpixels
        :param tokens: Token features from ROI token attention (N x n_roi_tokens x d)
        :param features: Patch or superpixel features (N x H x W x d)/(N x n_sp x d)
        :param cls_probs: Class distribution of patches or superpixels (N x H x W x C)/(N x n_sp x C)
        :param reference_pos: Reference position for each ROI token (N x n_roi_tokens x 2)
        :param reference_pos_emb: Reference position embedding for each ROI token (N x n_roi_tokens x d)
        :return (roi_features,  roi_cls_probs, roi_bboxes, roi_patch_map)
            - roi_features: Pooled features (N x n_roi_tokens x d)
            - roi_cls_probs: Pooled class probabilities (N x n_roi_tokens x C)
            - roi_bboxes: Predicted bounding boxes per ROI (N x n_roi_tokens x 4)
                format: (x_center, y_center, w, h) with relative coordinates in [0, 1]
                Note: not differentiable if using superpixels
            - roi_patch_map: Assignment probs of patch to ROI (N x n_roi_tokens x H x W)
        """
        if self.config.stop_grad_to_features:
            features = features.detach()
            cls_probs = cls_probs.detach()

        # compute ROI params and apply Guassian attention (ROI pool)
        features = self.norm_before_roi_pool(features)
        features = self.roi_feature_projector(features)   # (N x H x W x d)

        tokens = self.norm_roi_tokens(tokens)
        with autocast(device_type='cuda', enabled=False):
            # (N x n_roi_tokens x 4)
            roi_params = self.gpp(
                tokens.float(),
                reference_positions=reference_pos,
                pos_emb=reference_pos_emb)

            # roi_features: (N x n_roi_tokens x d)
            # roi_bboxes: (N x n_roi_tokens x 4)
            # roi_patch_map: (N x n_roi_tokens x H x W)
            (roi_features,
                roi_bboxes,
                roi_patch_map) = self.roi_pool(features.float(), roi_params)

        # -> compute ROI obj and cls probs by classifying the roi_features
        # (N x n_roi_tokens x C), (N x n_roi_tokens)
        roi_cls_probs = self.classify(roi_features)

        return roi_features, roi_cls_probs, roi_bboxes, roi_patch_map

    def aggregate(self, features: Tensor, cls_probs: Tensor, prob_mode: str, use_cls_tokens: bool = False) -> Tuple[Tensor, Tensor]:
        """
        Aggregates the features and cls_probs of patches/superpixels/ROIs
        to get per-sample features (for supcon) and cls_probs (for multilabel BCE)
        :param features: patch/superpixel/ROI features
            (N x H x W x d) / (N x n_sp x d) / (N x n_roi_tokens x d)
        :param cls_probs: Class distribution, sum over classes <= 1 (not necessarily == 1)
                (N x H x W x C) / (N x n_sp x C) / (N x n_roi_tokens x C)
        :param feature_mode: Aggregation mode for class-features
            (obj_cls_weighted_sumnorm)
        :param prob_mode: Aggregation mode for class-probs
            (max, max_straight_through, obj_weighted_sumnorm, obj_weighted_nonorm)
        """
        N, *dims, d = features.shape
        N, *dims, C = cls_probs.shape
        n_features = prod(dims)
        features = features.view(N, n_features, d)  # (N x n_features x d)
        cls_probs = cls_probs.view(N, n_features, C)  # (N x n_features x C)

        if use_cls_tokens:
            prob_mode = 'cls_tokens'
            assert n_features == C - 1
            # all cls features ecxept no finding
            aggregated_no_find_features = features.new_zeros(N, 1, d)  # (N)
            # (N x C x d)
            aggregated_cls_features = torch.cat([features, aggregated_no_find_features], dim=1)
        else:
            # Class features as weighted average over patch/superpixel/ROI features
            # with weights based on class probs (normalized over all patches/superpixels/ROIs) and obj probs
            eps = 1.0  # TODO: 1.0 seems to work better than small eps, WHY??
            normalized_cls_probs = cls_probs / (cls_probs.sum(1, keepdim=True) + eps)  # (N x n_features x C)
            # Apply obj_probs to reweight the individual contributions of different boxes (obj_probs normalized to sum=1 per sample)
            normalized_cls_probs = normalized_cls_probs.transpose(1, 2)  # (N x C x n_features)
            # normalized_cls_probs = normalized_cls_probs * aggregated_cls_probs[:, :, None]
            aggregated_cls_features = torch.bmm(normalized_cls_probs, features)  # (N x C x d)

        if prob_mode is None:
            aggregated_cls_probs = None
            aggregated_or_probs = None
            aggregated_and_probs = None
        elif prob_mode == 'mean':
            # Use avg probabilities over all patches/superpixels/ROIs per class
            aggregated_cls_probs = cls_probs.mean(dim=1)  # (N x C)
            aggregated_or_probs = aggregated_cls_probs[:, -1]
            aggregated_and_probs = aggregated_cls_probs[:, -1]
            aggregated_cls_probs = aggregated_cls_probs[:, :-1]
        elif prob_mode == 'max':
            # Use max probabilities over all patches/superpixels/ROIs per class
            aggregated_cls_probs = cls_probs[..., :-1].amax(dim=1)  # (N x C)
            aggregated_and_probs = cls_probs[..., -1].amin(dim=1)
            aggregated_or_probs = cls_probs[..., -1].amax(dim=1)
        elif prob_mode == 'MIL_noisyOR':
            # Use probabilistic OR to aggregate over all patches/superpixels/ROIs per class
            # Note: we treat each class independently
            # we need to treat no-finding differently as it is logically inverted
            other_cls_probs, no_finding_probs = cls_probs[..., :-1], cls_probs[..., -1]
            # each class is true if there is any patch/sp/ROI with that class
            # -> OR over all patches/sps/ROIs -> OR = NOT (AND NOT) where AND is realized as product
            aggregated_cls_probs = 1. - (1. - other_cls_probs).prod(dim=1)
            # similar with OR-no-finding
            aggregated_or_probs = 1. - (1. - no_finding_probs).prod(dim=1)
            # AND-no-finding is true if all patches/sps/ROIs contain no finding
            # -> AND over all patches/sps/ROIs where AND is realized as product
            aggregated_and_probs = no_finding_probs.prod(dim=1)
        elif prob_mode == 'lse':
            lse_r = self.config.lse_r
            cls_probs_nofinding_inversed = -1. * cls_probs[..., -1]
            # (N x n_features x C+1)
            cls_probs_orand_nofinding = torch.cat([cls_probs, cls_probs_nofinding_inversed[..., None]], dim=-1)
            cls_probs_max = cls_probs_orand_nofinding.amax(dim=1, keepdim=True)  # (N, 1, C+1)
            aggregated_cls_probs = cls_probs_orand_nofinding - cls_probs_max  # (N x n_features x C+1)
            # (N x C)
            aggregated_cls_probs = torch.div((lse_r * aggregated_cls_probs).exp().mean(dim=1).log(), lse_r)
            aggregated_cls_probs = aggregated_cls_probs + cls_probs_max[:, 0, :]
            aggregated_and_probs = -1. * aggregated_cls_probs[..., -1]
            aggregated_or_probs = aggregated_cls_probs[..., -2]
            aggregated_cls_probs = aggregated_cls_probs[..., :-2]
        elif prob_mode == 'cls_tokens':
            assert n_features == C - 1
            # all cls probs ecxept no finding
            # (N x C-1)
            aggregated_cls_probs = cls_probs[:, :, :-1].diagonal(dim1=-2, dim2=-1)
            aggregated_and_probs = 1. - aggregated_cls_probs.amax(1)  # (N)
            aggregated_or_probs = 1. - aggregated_cls_probs.amin(1)  # (N)
        else:
            raise ValueError(prob_mode)

        return aggregated_cls_features, aggregated_cls_probs, aggregated_or_probs, aggregated_and_probs

    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        # --- Encode, classify, and aggregate patches ---
        # Encode the patches using the backbone + projection/upsampling
        # (N x H x W x d), (N x H x W x 2)
        patch_features, patch_pos_emb = self.encode_features(x)

        # Classify each patch + get obj probs
        # (N x H x W x C), (N x H x W)
        patch_cls_probs = self.classify(patch_features)

        # Compute per-sample cls_probs and per-sample/class features by aggregating over all patches
        # (N x C x d), (N x C)
        (patch_aggregated_cls_features,
         patch_aggregated_cls_probs,
         patch_aggregated_or_probs,
         patch_aggregated_and_probs) = self.aggregate(
            patch_features,
            patch_cls_probs,
            prob_mode=self.config.patch_aggregation
        )
        encoded_patches = WSRPN.Features(
            patch_features,
            patch_cls_probs,
            patch_aggregated_cls_features,
            patch_aggregated_cls_probs,
            aggregated_or_probs=patch_aggregated_or_probs,
            aggregated_and_probs=patch_aggregated_and_probs
        )

        # --- ROI prediction and pooling ---
        if not self.predict_rois:
            encoded_rois = None
            roi_bboxes = None
            roi_patch_map = None
        else:
            # Apply cross attention to get token features from patch features
            # Also get reference position and pos embedding based on the patch with
            # highest attention
            # (N x n_roi_tokens x d), (N x n_roi_tokens x 2), (N x n_roi_tokens x d)
            tokens, reference_pos, reference_pos_emb = self.apply_roi_token_att(
                patch_features, patch_pos_emb)

            # Compute ROI features and probs using soft ROI pooling on patches
            # roi_features: (N x n_roi_tokens x d)
            # roi_cls_probs: (N x n_roi_tokens x C)
            # roi_bboxes: (N x n_roi_tokens x 4)
            # roi_patch_map: (N x n_roi_tokens x H_s x W_s)
            (roi_features,
                roi_cls_probs,
                roi_bboxes,
                roi_patch_map) = self.apply_roi_pool(
                tokens,
                patch_features, patch_cls_probs,
                reference_pos=reference_pos,
                reference_pos_emb=reference_pos_emb
            )

            # Compute per-sample cls_probs and per-sample/class features by aggregating over all ROIs
            # (N x C x d), (N x C)
            (roi_aggregated_cls_features,
             roi_aggregated_cls_probs,
             roi_aggregated_or_probs,
             roi_aggregated_and_probs) = self.aggregate(
                roi_features,
                roi_cls_probs,
                prob_mode=self.config.roi_aggregation,
                use_cls_tokens=self.config.use_cls_tokens
            )
            encoded_rois = WSRPN.Features(
                roi_features,
                roi_cls_probs,
                roi_aggregated_cls_features,
                roi_aggregated_cls_probs,
                aggregated_or_probs=roi_aggregated_or_probs,
                aggregated_and_probs=roi_aggregated_and_probs
            )

        return encoded_patches, encoded_rois, roi_bboxes, roi_patch_map

    def train_step(self, x: Tensor, global_label: Tensor,
                   target_boxes: Optional[List[Tensor]] = None,
                   return_predictions=False, step=None, **kwargs):
        (encoded_patches,
         encoded_rois,
         roi_bboxes,
         roi_patch_map) = self(x)
        encoded_patches: WSRPN.Features
        encoded_rois: WSRPN.Features
        _, _, H, W = x.shape

        # ------------------------------ Losses ------------------------------
        losses = {}
        step_metrics = {}
        # ---------- Patch losses ----------
        if self.config.use_patch_supcon:
            # Supervised contrastive loss on class features aggregated from patches
            projected_cls_features = self.out_projector(encoded_patches.aggregated_cls_features)  # (N x C x d)
            losses['patch_supcon'] = self.supcon_loss(projected_cls_features, global_label)
        if self.config.use_patch_bce:
            # Multilabel BCE on class probabilities aggregated from patches (-> Multi-instance learning)
            losses['patch_bce'] = weighted_binary_cross_entropy_wsrpn(
                encoded_patches.aggregated_cls_probs,
                or_probs=encoded_patches.aggregated_or_probs,
                and_probs=encoded_patches.aggregated_and_probs,
                global_label=global_label,
                clamp_min=1e-7,
                **self.config.patch_bce_loss)
        
        # ---------- ROI losses ----------
        if self.config.use_roi_supcon:
            # Supervised contrastive loss on class features aggregated from ROIs
            assert encoded_rois.aggregated_cls_features is not None, \
                "roi_aggregation[0] (feature mode) can't be null when use_roi_supcon is True"
            projected_cls_features = self.out_projector(encoded_rois.aggregated_cls_features)  # (N x C x d)
            losses['roi_supcon'] = self.supcon_loss(projected_cls_features, global_label)
        if self.config.use_roi_bce:
            assert encoded_rois.aggregated_cls_probs is not None, \
                "roi_aggregation[1] (prob mode) can't be null when use_roi_bce is True"
            # Multilabel BCE on class probabilities aggregated from ROIs (-> Multi-instance learning)
            losses['roi_bce'] = weighted_binary_cross_entropy_wsrpn(
                encoded_rois.aggregated_cls_probs,
                or_probs=encoded_rois.aggregated_or_probs,
                and_probs=encoded_rois.aggregated_and_probs,
                global_label=global_label,
                clamp_min=1e-7,
                **self.config.roi_bce_loss)
        if self.config.use_roi_patch_cls_consistency:
            patch_cls_probs = encoded_patches.cls_probs
            roi_cls_probs = encoded_rois.cls_probs
            with autocast(device_type='cuda', enabled=False):
                losses['roi_patch_cls_consist'] = \
                    self.roi_patch_cls_consistency_loss(
                        patch_cls_probs.float(),
                        roi_cls_probs.float(),
                        roi_patch_map.float())
        
        # Aggregate losses
        assert len(losses) > 0
        loss = sum(losses.values()) / len(losses)
        step_metrics.update(losses)  # log the deatiled losses

        if return_predictions:
            predictions = self.inference_from_scores(
                roi_bboxes,
                encoded_rois,
                encoded_patches,
                roi_patch_map,
                img_size=(H, W)
            )
        else:
            predictions = None

        return loss, step_metrics, predictions

    @torch.no_grad()
    def inference_from_scores(
            self,
            roi_bboxes,
            encoded_rois: 'WSRPN.Features',
            encoded_patches: 'WSRPN.Features',
            roi_patch_map,
            img_size) -> ObjectDetectorPrediction:
        """
        :param roi_bboxes: (N x n_roi_tokens x 4)
        :param roi_cls_probs: (N x n_roi_tokens x C)
        :param aggregated_probs: (N x C)
        :param roi_patch_map: (N x n_roi_tokens x H x W)
        :param img_size: Tuple[h, w]
        """
        # --- Global predictions (just for debugging, not relevant for object detection) ---
        # Global prediction from aggregated patch probs (ignore no-finding)
        patch_aggregated_cls_probs = encoded_patches.aggregated_cls_probs  # (N x C - 1)
        patch_aggregated_cls_preds = patch_aggregated_cls_probs > 0.5  # (N x C - 1)

        # Global prediction from aggregated ROI probs (ignore no-finding)
        if encoded_rois.aggregated_cls_probs is not None:
            roi_aggregated_obj_probs = encoded_rois.aggregated_and_probs  # (N)
            roi_aggregated_cls_probs = encoded_rois.aggregated_cls_probs  # (N x C - 1)
            roi_aggregated_cls_preds = roi_aggregated_cls_probs > 0.5  # (N x C - 1)
        else:
            # use patch probs as default
            roi_aggregated_cls_probs = patch_aggregated_cls_probs
            roi_aggregated_cls_preds = patch_aggregated_cls_preds

        # --- Box predictions ---
        roi_cls_probs = encoded_rois.cls_probs
        if roi_cls_probs is not None:
            # Separate in actual class probs and obj_probs
            # (N x n_roi_tokens), (N x n_roi_tokens x C - 1)
            obj_probs, roi_cls_probs = split_cls_probs(roi_cls_probs)
            # Get hard predictions from soft probabilities
            roi_cls_preds = torch.argmax(roi_cls_probs, dim=2)  # (N x n_roi_tokens)
            # Filter out predictions where no finding prob is larger than any other class
            if self.config.obj_threshold is None:
                roi_cls_masks = (1. - obj_probs[:, :, None]) < roi_cls_probs  # (N x n_roi_tokens x C-1)
            else:
                roi_cls_masks = roi_cls_probs > 0.05  # self.config.obj_threshold
            obj_masks = roi_cls_masks.any(dim=2)  # (N x n_roi_tokens)

            # --- Only keep highest scoring box per class ---
            # (N x n_roi_tokens x C - 1)
            roi_cls_pred_mask = torch.zeros_like(roi_cls_probs).scatter_(dim=2, index=roi_cls_preds[:, :, None], value=1.0)
            roi_cls_probs_masked = roi_cls_probs * roi_cls_pred_mask  # (N x n_roi_tokens x C - 1)
            top_indices = torch.argmax(roi_cls_probs_masked, dim=1)  # (N x C-1)
            # this mask will have 1. at best box per class and 0 at others
            # (N x n_roi_tokens x C - 1)
            obj_masks_topboxes = torch.zeros_like(roi_cls_probs).scatter_(dim=1, index=top_indices[:, None, :], value=1.)
            # (N x n_roi_tokens)
            obj_masks_topboxes = obj_masks_topboxes.any(dim=-1)
            # (N x n_roi_tokens)
            ##
            obj_masks = obj_masks * obj_masks_topboxes

            # Normalize over class dimension
            roi_cls_probs = F.normalize(roi_cls_probs, p=1.0, dim=-1)  # (N x n_roi_tokens x C-1)

            # compute boxes
            H, W = img_size
            box_params = roi_bboxes.clone()
            box_params[:, :, 0:2] = box_params[:, :, 0:2] - (box_params[:, :, 2:4] / 2)  # convert center pos to upper-left
            box_params[:, :, 0] *= W
            box_params[:, :, 1] *= H
            box_params[:, :, 2] *= W
            box_params[:, :, 3] *= H
            box_probs_list, box_prediction_list = assemble_box_predictions(
                box_params,
                obj_masks,
                obj_probs,
                roi_cls_preds,
                roi_cls_probs
            )
            box_probs_list = to_device(box_probs_list, 'cpu')
            box_prediction_list = to_device(box_prediction_list, 'cpu')

            ## apply nms
            #box_probs_list, box_prediction_list = apply_nms(
            #    box_probs_list, box_prediction_list, iou_threshold=0.5
            #)

            # --- Global prediction from ROI predictions ---
            global_cls_preds = roi_cls_pred_mask.any(dim=1)  # (N x C-1)
        else:
            global_cls_preds = patch_aggregated_cls_preds
            box_probs_list = []
            box_prediction_list = []

        # --- Patch Mask predictions ---
        # Compute seg masks from patch probs
        patch_cls_probs = encoded_patches.cls_probs
        if patch_cls_probs is not None:
            if self.config.obj_threshold is None:
                # just take the highest scoring class
                seg_masks_from_patches = patch_cls_probs.argmax(dim=-1)  # (N x H x W)
            else:
                C = patch_cls_probs.shape[-1]
                # get class prediction
                seg_mask_probs, seg_masks_from_patches = patch_cls_probs[..., :-1].max(dim=-1)  # (N x H x W)
                obj_mask = seg_mask_probs > self.config.obj_threshold  # (N x H x W)
                seg_masks_from_patches[~obj_mask] = C - 1  # set no-finding
            seg_masks_from_patches = F.interpolate(
                seg_masks_from_patches.unsqueeze(1).to(torch.uint8),
                size=img_size,
                mode='nearest'
            ).squeeze(1)
        else:
            seg_masks_from_patches = None

        # Compute seg masks from rois and ROI-patch map
        if roi_cls_probs is not None:
            patch_cls_probs_from_rois = roi_cls_probs[:, :, :, None, None] * roi_patch_map[:, :, None, :, :]  # (N x n_roi_tokens x C-1 x H x W)
            patch_cls_probs_from_rois = patch_cls_probs_from_rois.mean(1)  # (N x C-1 x H x W)
            seg_masks_from_rois = patch_cls_probs_from_rois.argmax(1)  # (N x H x W)
            roi_patch_map_hard = roi_patch_map > 0.5  # (N x n_roi_tokens x H x W)
            patch_obj_masks_from_rois = roi_patch_map_hard * obj_masks[:, :, None, None]  # (N x n_roi_tokens x H x W)
            patch_obj_masks_from_rois = patch_obj_masks_from_rois.any(1)  # (N x H x W)
            # class index -1 -> no-finding
            C_without_nolabel = roi_cls_probs.shape[-1]
            seg_masks_from_rois[~patch_obj_masks_from_rois] = C_without_nolabel  # (N x H x W)
            seg_masks_from_rois = F.interpolate(
                seg_masks_from_rois.unsqueeze(1).to(torch.uint8),
                size=img_size,
                mode='nearest'
            ).squeeze(1)
        else:
            seg_masks_from_rois = None

        predictions = ObjectDetectorPrediction(
            global_prediction_hard=to_device(global_cls_preds, 'cpu'),
            global_prediction_probs=to_device(roi_aggregated_cls_probs, 'cpu'),
            global_obj_probs=to_device(roi_aggregated_obj_probs, 'cpu'),
            box_prediction_hard=box_prediction_list,
            box_prediction_probs=box_probs_list,
            aggregated_patch_prediction_hard=to_device(patch_aggregated_cls_preds, 'cpu'),
            aggregated_patch_prediction_probs=to_device(patch_aggregated_cls_probs, 'cpu'),
            aggregated_roi_prediction_hard=to_device(roi_aggregated_cls_preds, 'cpu'),
            aggregated_roi_prediction_probs=to_device(roi_aggregated_cls_probs, 'cpu'),
            seg_masks_from_rois=seg_masks_from_rois,
            seg_masks_from_patches=seg_masks_from_patches
        )
        #predictions = filter_top1_box_per_class(predictions)
        return predictions

    def inference(self, x, **kwargs) -> ObjectDetectorPrediction:
        _, _, H, W = x.shape
        (encoded_patches,
         encoded_rois,
         roi_bboxes,
         roi_patch_map) = self(x)
        return self.inference_from_scores(
            roi_bboxes,
            encoded_rois,
            encoded_patches,
            roi_patch_map,
            img_size=(H, W)
        )


def split_cls_probs(roi_cls_probs):
    obj_probs = 1. - roi_cls_probs[..., -1]  # (N x n_roi_tokens)
    roi_cls_probs = roi_cls_probs[..., :-1]  # (N x n_roi_tokens x C - 1)
    return obj_probs, roi_cls_probs


def merge_cls_probs(obj_probs, roi_cls_probs):
    nofind_probs = 1. - obj_probs
    return torch.cat([roi_cls_probs, nofind_probs.unsqueeze(-1)], dim=-1)
