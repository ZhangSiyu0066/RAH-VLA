"""
Builder for Dynamic Resolution Input Strategy (DRIS) Vision Tower
"""

import os
from .dris_encoder import DRISVisionTower
from .clip_encoder import CLIPVisionTower


def build_dris_vision_tower(vision_tower_cfg, **kwargs):
    """
    Build DRIS vision tower with base CLIP encoder
    
    Args:
        vision_tower_cfg: Configuration object with attributes:
            - mm_vision_tower: Base vision tower name/path
            - use_dris: Whether to use DRIS
            - dris_low_res_size: Low resolution size
            - dris_high_res_size: High resolution size
            - dris_roi_threshold: ROI selection threshold
            - dris_top_k_rois: Number of ROIs to select
            - dris_saliency_kernel_size: Gaussian kernel size for saliency
            - dris_fpn_channels: FPN channel dimensions
        **kwargs: Additional arguments
            
    Returns:
        DRISVisionTower or CLIPVisionTower instance
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', 
                          getattr(vision_tower_cfg, 'vision_tower', None))
    use_dris = getattr(vision_tower_cfg, 'use_dris', False)
    
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        # Build base CLIP vision tower
        base_vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
        # Wrap with DRIS if enabled
        if use_dris:
            low_res_size = getattr(vision_tower_cfg, 'dris_low_res_size', 224)
            high_res_size = getattr(vision_tower_cfg, 'dris_high_res_size', 504)
            roi_threshold = getattr(vision_tower_cfg, 'dris_roi_threshold', 0.5)
            top_k_rois = getattr(vision_tower_cfg, 'dris_top_k_rois', 5)
            saliency_kernel_size = getattr(vision_tower_cfg, 'dris_saliency_kernel_size', 5)
            fpn_channels = getattr(vision_tower_cfg, 'dris_fpn_channels', [256, 512, 1024, 2048])
            device = kwargs.get('device', 'cuda')
            
            dris_tower = DRISVisionTower(
                base_vision_tower=base_vision_tower,
                low_res_size=low_res_size,
                high_res_size=high_res_size,
                roi_threshold=roi_threshold,
                top_k_rois=top_k_rois,
                saliency_kernel_size=saliency_kernel_size,
                fpn_channels=fpn_channels,
                device=device
            )
            return dris_tower
        else:
            return base_vision_tower
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')