"""Runtime patches for the DRIS + MS-VLAM experiment.

This module mirrors the experiment-specific vision tower and projector used by
``geochat.train.train_dris_msvlam_lora`` without enabling the training-only
FlashAttention monkey patch.  It is intended for inference and LoRA merging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import geochat.model.geochat_arch as geochat_arch
import geochat.model.multimodal_encoder.builder as vision_builder
import geochat.model.multimodal_projector.builder as projector_builder
from geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower


class DRISVisionTowerAdapter(nn.Module):
    """Token-level coarse-to-fine adapter around the existing CLIP tower."""

    def __init__(self, vision_tower_name, args, delay_load=False):
        super().__init__()
        self.base_tower = CLIPVisionTower(
            vision_tower_name,
            args=args,
            delay_load=delay_load,
        )
        self.low_res_size = getattr(args, "dris_low_res_size", 224)
        self.high_res_size = getattr(args, "dris_high_res_size", 504)
        self.top_k_rois = getattr(args, "dris_top_k_rois", 5)

    def load_model(self):
        if not self.base_tower.is_loaded:
            self.base_tower.load_model()

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            return [self.forward(image.unsqueeze(0)).squeeze(0) for image in images]

        images = images.to(device=self.device, dtype=self.dtype)
        high_features = self.base_tower(images)

        coarse_images = F.interpolate(
            images.float(),
            size=(self.low_res_size, self.low_res_size),
            mode="bilinear",
            align_corners=False,
        )
        coarse_images = F.interpolate(
            coarse_images,
            size=(self.high_res_size, self.high_res_size),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=images.dtype)
        low_features = self.base_tower(coarse_images)

        token_scores = high_features.norm(dim=-1)
        top_k = min(max(self.top_k_rois, 1), token_scores.shape[1])
        roi_indices = token_scores.topk(top_k, dim=1).indices
        roi_mask = torch.zeros_like(token_scores).scatter_(1, roi_indices, 1.0)
        roi_mask = roi_mask.unsqueeze(-1)

        return low_features * (1.0 - roi_mask) + high_features * roi_mask

    @property
    def image_processor(self):
        return self.base_tower.image_processor

    @property
    def is_loaded(self):
        return self.base_tower.is_loaded

    @property
    def dtype(self):
        return self.base_tower.dtype

    @property
    def device(self):
        return self.base_tower.device

    @property
    def config(self):
        return self.base_tower.config

    @property
    def hidden_size(self):
        return self.base_tower.hidden_size

    @property
    def num_patches(self):
        return self.base_tower.num_patches


class MSVLAMProjectorAdapter(nn.Module):
    """Project visual tokens to LLM space with local, region, and global context."""

    def __init__(
        self,
        vision_hidden_size,
        text_hidden_size,
        num_regions=16,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.local_proj = nn.Linear(vision_hidden_size, text_hidden_size)
        self.region_proj = nn.Linear(vision_hidden_size, text_hidden_size)
        self.global_proj = nn.Linear(vision_hidden_size, text_hidden_size)
        self.local_norm = nn.LayerNorm(text_hidden_size)
        self.region_norm = nn.LayerNorm(text_hidden_size)
        self.global_norm = nn.LayerNorm(text_hidden_size)

    def forward(self, image_features):
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)

        local_features = self.local_norm(self.local_proj(image_features))

        token_scores = image_features.norm(dim=-1)
        region_k = min(max(self.num_regions, 1), token_scores.shape[1])
        region_indices = token_scores.topk(region_k, dim=1).indices
        region_mask = torch.zeros_like(token_scores).scatter_(1, region_indices, 1.0)
        region_mask = region_mask / region_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        region_features = torch.bmm(region_mask.unsqueeze(1), image_features).squeeze(1)
        region_features = self.region_norm(self.region_proj(region_features)).unsqueeze(1)

        global_features = image_features.mean(dim=1)
        global_features = self.global_norm(self.global_proj(global_features)).unsqueeze(1)

        return (
            self.alpha * local_features
            + self.beta * region_features
            + self.gamma * global_features
        )


_ORIGINAL_BUILD_VISION_TOWER = vision_builder.build_vision_tower
_ORIGINAL_BUILD_PROJECTOR = projector_builder.build_vision_projector


def _build_vision_tower_with_dris(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    if getattr(vision_tower_cfg, "use_dris", False):
        return DRISVisionTowerAdapter(vision_tower, args=vision_tower_cfg, **kwargs)
    return _ORIGINAL_BUILD_VISION_TOWER(vision_tower_cfg, **kwargs)


def _build_projector_with_msvlam(config, delay_load=False, **kwargs):
    if getattr(config, "use_msvlam", False):
        return MSVLAMProjectorAdapter(
            vision_hidden_size=getattr(config, "mm_hidden_size", 1024),
            text_hidden_size=getattr(config, "hidden_size", 4096),
            num_regions=getattr(config, "msvlam_num_regions", 16),
            alpha=getattr(config, "msvlam_alpha", 1.0),
            beta=getattr(config, "msvlam_beta", 1.0),
            gamma=getattr(config, "msvlam_gamma", 1.0),
        )
    return _ORIGINAL_BUILD_PROJECTOR(config, delay_load=delay_load, **kwargs)


def apply_runtime_patches():
    vision_builder.build_vision_tower = _build_vision_tower_with_dris
    geochat_arch.build_vision_tower = _build_vision_tower_with_dris
    projector_builder.build_vision_projector = _build_projector_with_msvlam
    geochat_arch.build_vision_projector = _build_projector_with_msvlam
