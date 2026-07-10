"""
Configuration for DRIS and MS-VLAM training
Example configuration demonstrating how to enable and configure the new features
"""

# DRIS (Dynamic Resolution Input Strategy) Configuration
DRIS_CONFIG = {
    'use_dris': True,                      # Enable DRIS
    'dris_low_res_size': 224,              # Low resolution for coarse processing
    'dris_high_res_size': 504,             # High resolution for ROI processing
    'dris_roi_threshold': 0.5,              # Saliency threshold for ROI selection
    'dris_top_k_rois': 5,                   # Number of ROIs to select
    'dris_saliency_kernel_size': 5,          # Gaussian kernel size for saliency smoothing
    'dris_fpn_channels': [256, 512, 1024, 2048],  # FPN channel dimensions
}

# MS-VLAM (Multi-scale Vision-language Alignment Mechanism) Configuration
MSVLAM_CONFIG = {
    'use_msvlam': True,                     # Enable MS-VLAM
    'msvlam_alignment_dim': 768,             # Alignment space dimension
    'msvlam_num_regions': 16,               # Number of regions for local alignment
    'msvlam_spp_sizes': [1, 2, 4],        # Spatial Pyramid Pooling sizes
    'msvlam_alpha': 1.0,                    # Weight for object-level loss (Eq. 13)
    'msvlam_beta': 1.0,                     # Weight for region-level loss (Eq. 13)
    'msvlam_gamma': 1.0,                    # Weight for global-level loss (Eq. 13)
    'msvlam_mu': 0.5,                       # Weight for hard vs NCE loss (Eq. 11)
    'msvlam_temperature': 0.07,              # Temperature for contrastive learning (Eq. 9)
}

# Loss Configuration (Eq. 15)
LOSS_CONFIG = {
    'delta': 1.0,                            # Weight for alignment loss (Eq. 15)
    'label_smoothing': 0.0,                  # Label smoothing for caption loss
    'ignore_index': -100,                       # Ignore index for loss computation
}

# Training Configuration
TRAINING_CONFIG = {
    # Model settings
    'model_name_or_path': 'lmsys/vicuna-7b-v1.5',
    'version': 'v1',
    'vision_tower': 'openai/clip-vit-large-patch14',
    'mm_projector_type': 'linear',             # 'linear' or 'mlp2x_gelu'
    
    # DRIS settings
    **DRIS_CONFIG,
    
    # MS-VLAM settings
    **MSVLAM_CONFIG,
    
    # Loss settings
    **LOSS_CONFIG,
    
    # Training parameters (from paper Section 3.4)
    'per_device_train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'learning_rate': 3e-4,
    'warmup_steps': 100,
    'max_steps': 1000,
    'weight_decay': 0.0,
    'lr_scheduler_type': 'linear',
    
    # Precision and memory
    'bf16': True,                             # Use bfloat16 mixed precision
    'fp16': False,
    'bits': 16,
    
    # LoRA settings (optional)
    'lora_enable': False,
    'lora_r': 64,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'lora_bias': 'none',
    
    # Data settings
    'data_path': 'data/your_data.json',
    'image_folder': 'data/images',
    'image_aspect_ratio': 'pad',
    
    # Output
    'output_dir': 'output/dris_msvlam_model',
    'logging_steps': 10,
    'save_steps': 500,
}

# Ablation Study Configurations
ABLATION_CONFIGS = {
    'baseline': {
        'use_dris': False,
        'use_msvlam': False,
    },
    'dris_only': {
        'use_dris': True,
        'use_msvlam': False,
    },
    'msvlam_only': {
        'use_dris': False,
        'use_msvlam': True,
    },
    'full_model': {
        'use_dris': True,
        'use_msvlam': True,
    },
    'dris_msvlam_no_obj': {
        'use_dris': True,
        'use_msvlam': True,
        'msvlam_alpha': 0.0,  # Disable object-level alignment
    },
    'dris_msvlam_no_region': {
        'use_dris': True,
        'use_msvlam': True,
        'msvlam_beta': 0.0,   # Disable region-level alignment
    },
    'dris_msvlam_no_global': {
        'use_dris': True,
        'use_msvlam': True,
        'msvlam_gamma': 0.0,  # Disable global-level alignment
    },
}

# Evaluation Metrics Configuration (from paper Section 4.3)
EVALUATION_CONFIG = {
    # Image captioning metrics
    'caption_metrics': [
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
        'METEOR', 'ROUGE-L', 'CIDEr', 'SPICE'
    ],
    
    # Cross-modal retrieval metrics
    'retrieval_metrics': [
        'R@1', 'R@5', 'R@10'
    ],
    
    # Visual grounding metrics
    'grounding_metrics': [
        'Accuracy@0.5'
    ],
}

# Dataset Configuration
DATASET_CONFIG = {
    'RS-GPT4V': {
        'name': 'RS-GPT4V',
        'tasks': [
            'image_captioning',
            'visual_grounding',
            'visual_qa',
            'detailed_description',
            'complex_reasoning'
        ],
        'train_images': 91937,
        'train_pairs': 991206,
        'test_images': 15999,
        'test_pairs': 258419,
    },
}


def get_config(config_name='full_model'):
    """
    Get configuration by name
    
    Args:
        config_name: Name of configuration ('full_model', 'baseline', etc.)
        
    Returns:
        Configuration dictionary
    """
    base_config = TRAINING_CONFIG.copy()
    
    if config_name in ABLATION_CONFIGS:
        ablation_config = ABLATION_CONFIGS[config_name]
        base_config.update(ablation_config)
    
    return base_config


def print_config(config):
    """Print configuration in a readable format"""
    print("=" * 80)
    print("DRIS + MS-VLAM Configuration")
    print("=" * 80)
    
    sections = [
        ('DRIS Settings', DRIS_CONFIG.keys()),
        ('MS-VLAM Settings', MSVLAM_CONFIG.keys()),
        ('Loss Settings', LOSS_CONFIG.keys()),
        ('Training Settings', ['model_name_or_path', 'vision_tower', 'learning_rate', 
                           'batch_size', 'warmup_steps', 'max_steps']),
    ]
    
    for section_name, keys in sections:
        print(f"\n{section_name}:")
        print("-" * 40)
        for key in keys:
            if key in config:
                print(f"  {key:30s}: {config[key]}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    # Example usage
    config = get_config('full_model')
    print_config(config)