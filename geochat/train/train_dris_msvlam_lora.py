"""
Training entrypoint for the DRIS + MS-VLAM LoRA experiment.

This file intentionally does not modify GeoChat's original train.py.  It adds
the experiment-specific arguments described in README_DRIS_MSVLAM.md, patches
the builders inside this process, and then reuses the original training loop.
"""

from dataclasses import dataclass, field
from typing import Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from geochat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from geochat.train import train as geochat_train  # noqa: E402
import geochat.model.geochat_arch as geochat_arch  # noqa: E402
import geochat.model.multimodal_encoder.builder as vision_builder  # noqa: E402
import geochat.model.multimodal_projector.builder as projector_builder  # noqa: E402
from geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower  # noqa: E402


@dataclass
class DRISMSVLAMModelArguments(geochat_train.ModelArguments):
    use_dris: bool = field(default=False)
    dris_low_res_size: int = field(default=224)
    dris_high_res_size: int = field(default=504)
    dris_roi_threshold: float = field(default=0.5)
    dris_top_k_rois: int = field(default=5)
    dris_saliency_kernel_size: int = field(default=5)
    use_msvlam: bool = field(default=False)
    msvlam_alignment_dim: int = field(default=768)
    msvlam_num_regions: int = field(default=16)
    msvlam_alpha: float = field(default=1.0)
    msvlam_beta: float = field(default=1.0)
    msvlam_gamma: float = field(default=1.0)
    msvlam_mu: float = field(default=0.5)
    msvlam_temperature: float = field(default=0.07)


@dataclass
class DRISMSVLAMTrainingArguments(geochat_train.TrainingArguments):
    delta: float = field(default=1.0)
    label_smoothing: float = field(default=0.0)


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

        # PyTorch 2.0 CUDA does not implement bilinear upsample for bf16.
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
_ORIGINAL_INITIALIZE_VISION_MODULES = geochat_arch.GeoChatMetaModel.initialize_vision_modules
_ORIGINAL_SAVE_CHECKPOINT = geochat_train.GeoChatTrainer._save_checkpoint


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


def _initialize_vision_modules_with_experiment_config(self, model_args, fsdp=None):
    for attr in (
        "use_dris",
        "dris_low_res_size",
        "dris_high_res_size",
        "dris_roi_threshold",
        "dris_top_k_rois",
        "dris_saliency_kernel_size",
        "use_msvlam",
        "msvlam_alignment_dim",
        "msvlam_num_regions",
        "msvlam_alpha",
        "msvlam_beta",
        "msvlam_gamma",
        "msvlam_mu",
        "msvlam_temperature",
    ):
        setattr(self.config, attr, getattr(model_args, attr))
    return _ORIGINAL_INITIALIZE_VISION_MODULES(self, model_args, fsdp=fsdp)


def _save_lora_only_checkpoint(self, model, trial, metrics=None):
    if not getattr(self.args, "lora_enable", False):
        return _ORIGINAL_SAVE_CHECKPOINT(self, model, trial, metrics=metrics)

    if self.args.local_rank not in (0, -1):
        return

    checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
    run_dir = self._get_output_dir(trial=trial)
    output_dir = os.path.join(run_dir, checkpoint_folder)
    os.makedirs(output_dir, exist_ok=True)

    state_dict = geochat_train.get_peft_state_maybe_zero_3(
        self.model.named_parameters(), self.args.lora_bias
    )
    non_lora_state_dict = geochat_train.get_peft_state_non_lora_maybe_zero_3(
        self.model.named_parameters()
    )

    self.model.config.save_pretrained(output_dir)
    self.model.save_pretrained(output_dir, state_dict=state_dict)
    torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))

    if self.tokenizer is not None:
        self.tokenizer.save_pretrained(output_dir)
    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


def apply_experiment_patches():
    geochat_train.ModelArguments = DRISMSVLAMModelArguments
    geochat_train.TrainingArguments = DRISMSVLAMTrainingArguments
    vision_builder.build_vision_tower = _build_vision_tower_with_dris
    geochat_arch.build_vision_tower = _build_vision_tower_with_dris
    projector_builder.build_vision_projector = _build_projector_with_msvlam
    geochat_arch.build_vision_projector = _build_projector_with_msvlam
    geochat_arch.GeoChatMetaModel.initialize_vision_modules = (
        _initialize_vision_modules_with_experiment_config
    )
    geochat_train.GeoChatTrainer._save_checkpoint = _save_lora_only_checkpoint


if __name__ == "__main__":
    apply_experiment_patches()
    geochat_train.train()
