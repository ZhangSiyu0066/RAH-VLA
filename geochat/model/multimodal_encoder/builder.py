import os
from .clip_encoder import CLIPVisionTower
from .dris_builder import build_dris_vision_tower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_dris = getattr(vision_tower_cfg, 'use_dris', False)
    
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        # Use DRIS if enabled, otherwise use standard CLIP
        if use_dris:
            return build_dris_vision_tower(vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
