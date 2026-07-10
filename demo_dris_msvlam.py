"""
Demo script for DRIS and MS-VLAM features
Demonstrates how to use the new Dynamic Resolution Input Strategy
and Multi-scale Vision-language Alignment Mechanism
"""

import torch
import argparse
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geochat.model.multimodal_encoder.dris_encoder import DRISVisionTower
from geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from geochat.model.multimodal_projector.ms_vlam_projector import MSVLAMProjector
from geochat.model.alignment_loss import MultiScaleAlignmentLoss


def demo_dris(image_path: str):
    """
    Demonstrate DRIS (Dynamic Resolution Input Strategy)
    
    Args:
        image_path: Path to input image
    """
    print("=" * 80)
    print("DRIS (Dynamic Resolution Input Strategy) Demo")
    print("=" * 80)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"\nLoaded image: {image_path}")
    print(f"Image size: {image.size}")
    
    # Create base CLIP vision tower
    base_vision_tower = CLIPVisionTower(
        'openai/clip-vit-large-patch14',
        delay_load=False
    )
    
    # Wrap with DRIS
    dris_tower = DRISVisionTower(
        base_vision_tower=base_vision_tower,
        low_res_size=224,
        high_res_size=504,
        roi_threshold=0.5,
        top_k_rois=5,
        saliency_kernel_size=5,
        fpn_channels=[256, 512, 1024, 2048],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nDRIS Configuration:")
    print(f"  Low resolution: {dris_tower.low_res_size}x{dris_tower.low_res_size}")
    print(f"  High resolution: {dris_tower.high_res_size}x{dris_tower.high_res_size}")
    print(f"  ROI threshold: {dris_tower.roi_threshold}")
    print(f"  Top-K ROIs: {dris_tower.top_k_rois}")
    
    # Process image through DRIS
    print(f"\nProcessing image through DRIS...")
    with torch.no_grad():
        features = dris_tower(image)
    
    print(f"\nOutput features shape: {features.shape}")
    print(f"Feature dimension: {dris_tower.hidden_size}")
    
    print("\nDRIS processing completed successfully!")
    print("  - Low-resolution global analysis: ✓")
    print("  - Saliency-based ROI selection: ✓")
    print("  - High-resolution local refinement: ✓")
    print("  - FPN feature fusion: ✓")
    
    return features


def demo_msvlam(vision_features: torch.Tensor):
    """
    Demonstrate MS-VLAM (Multi-scale Vision-language Alignment Mechanism)
    
    Args:
        vision_features: Visual features from DRIS
    """
    print("\n" + "=" * 80)
    print("MS-VLAM (Multi-scale Vision-language Alignment Mechanism) Demo")
    print("=" * 80)
    
    # Create MS-VLAM projector
    msvlam = MSVLAMProjector(
        vision_hidden_size=vision_features.shape[-1],
        text_hidden_size=4096,
        alignment_dim=768,
        num_sam_regions=16,
        spp_pool_sizes=[1, 2, 4],
        alpha=1.0,  # Object-level weight
        beta=1.0,   # Region-level weight
        gamma=1.0,  # Global-level weight
        mu=0.5,      # Hard vs NCE weight
        temperature=0.07
    )
    
    print(f"\nMS-VLAM Configuration:")
    print(f"  Vision hidden size: {msvlam.vision_hidden_size}")
    print(f"  Text hidden size: {msvlam.text_hidden_size}")
    print(f"  Alignment dimension: {msvlam.alignment_dim}")
    print(f"  Number of regions: {msvlam.num_sam_regions}")
    print(f"  SPP pool sizes: {msvlam.spp_pool_sizes}")
    print(f"  Loss weights: α={msvlam.alpha}, β={msvlam.beta}, γ={msvlam.gamma}")
    
    # Create dummy text features
    batch_size = vision_features.shape[0]
    text_features = {
        'object': torch.randn(5, 4096),  # 5 objects
        'phrase': torch.randn(16, 4096),  # 16 phrases
        'global': torch.randn(batch_size, 4096)  # Global features
    }
    
    # Forward pass
    print(f"\nProcessing features through MS-VLAM...")
    outputs = msvlam(
        vision_features=vision_features.unsqueeze(1).unsqueeze(1),  # Add H, W dims
        text_features=text_features,
        return_losses=True
    )
    
    print(f"\nExtracted features:")
    print(f"  Object-level: {outputs['object_features'].shape}")
    print(f"  Region-level: {outputs['region_features'].shape}")
    print(f"  Global-level: {outputs['global_features'].shape}")
    
    print(f"\nAlignment losses:")
    print(f"  L_obj (object-level): {outputs['loss_obj']:.4f}")
    print(f"  L_reg (region-level): {outputs['loss_reg']:.4f}")
    print(f"  L_glob (global-level): {outputs['loss_glob']:.4f}")
    print(f"  L_align (total): {outputs['loss_align']:.4f}")
    
    print("\nMS-VLAM processing completed successfully!")
    print("  - Object-level alignment: ✓")
    print("  - Region-level alignment: ✓")
    print("  - Global-level alignment: ✓")
    
    return outputs


def demo_alignment_loss():
    """
    Demonstrate combined loss computation
    """
    print("\n" + "=" * 80)
    print("Multi-scale Alignment Loss Demo")
    print("=" * 80)
    
    # Create loss function
    loss_fn = MultiScaleAlignmentLoss(
        delta=1.0,  # Weight for alignment loss
        ignore_index=-100,
        label_smoothing=0.0
    )
    
    print(f"\nLoss Configuration:")
    print(f"  Delta (alignment weight): {loss_fn.delta}")
    print(f"  Ignore index: {loss_fn.ignore_index}")
    print(f"  Label smoothing: {loss_fn.label_smoothing}")
    
    # Create dummy data
    batch_size = 2
    seq_len = 32
    vocab_size = 32000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    alignment_losses = {
        'loss_align': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_reg': torch.tensor(0.4),
        'loss_glob': torch.tensor(0.2)
    }
    
    # Compute losses
    print(f"\nComputing losses...")
    outputs = loss_fn(
        logits=logits,
        labels=labels,
        alignment_losses=alignment_losses
    )
    
    print(f"\nLoss values:")
    print(f"  L_caption: {outputs['loss_caption']:.4f}")
    print(f"  L_align: {outputs['loss_align']:.4f}")
    print(f"  L (total): {outputs['loss']:.4f}")
    
    print(f"\nIndividual alignment losses:")
    print(f"  L_obj: {outputs['loss_obj']:.4f}")
    print(f"  L_reg: {outputs['loss_reg']:.4f}")
    print(f"  L_glob: {outputs['loss_glob']:.4f}")
    
    print("\nLoss computation completed!")
    print("  Eq. (14) - Caption loss: ✓")
    print("  Eq. (15) - Total loss: ✓")


def main():
    parser = argparse.ArgumentParser(description='Demo DRIS and MS-VLAM')
    parser.add_argument('--image', type=str, default='demo_images/04444.png',
                       help='Path to input image')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['dris', 'msvlam', 'loss', 'all'],
                       help='Which demo to run')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("DRIS + MS-VLAM Demonstration")
    print("=" * 80)
    print("\nThis demo showcases the new features added to GeoChat:")
    print("  1. Dynamic Resolution Input Strategy (DRIS)")
    print("  2. Multi-scale Vision-language Alignment Mechanism (MS-VLAM)")
    print("  3. Multi-scale Alignment Loss")
    print("\nThese implementations correspond to the mathematical formulations")
    print("presented in the research paper.")
    print("=" * 80)
    
    try:
        # Run selected demos
        if args.demo in ['dris', 'all']:
            if os.path.exists(args.image):
                features = demo_dris(args.image)
            else:
                print(f"\nWarning: Image not found at {args.image}")
                print("Using dummy features for remaining demos...")
                features = torch.randn(1, 1024)
        
        if args.demo in ['msvlam', 'all']:
            if 'features' in locals():
                demo_msvlam(features)
            else:
                # Create dummy features
                features = torch.randn(1, 1024)
                demo_msvlam(features)
        
        if args.demo in ['loss', 'all']:
            demo_alignment_loss()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        print("\nFor training with DRIS and MS-VLAM, see:")
        print("  - config/dris_msvlam_config.py")
        print("  - docs/DRIS_MSVLAM.md")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()