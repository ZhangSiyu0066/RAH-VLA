# DRIS and MS-VLAM Implementation Guide

This document provides comprehensive documentation for the newly implemented features:
- **Dynamic Resolution Input Strategy (DRIS)**
- **Multi-scale Vision-language Alignment Mechanism (MS-VLAM)**

These implementations are based on the research paper's mathematical formulations and provide efficient and accurate multimodal understanding for remote sensing images.

## Table of Contents

1. [Overview](#overview)
2. [Dynamic Resolution Input Strategy (DRIS)](#dynamic-resolution-input-strategy-dris)
3. [Multi-scale Vision-language Alignment Mechanism (MS-VLAM)](#multi-scale-vision-language-alignment-mechanism-ms-vlam)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Ablation Studies](#ablation-studies)
10. [Mathematical Formulations](#mathematical-formulations)

---

## Overview

### Problem Statement

Existing remote sensing vision-language models face two major challenges:

1. **Resolution Efficiency Trade-off**: Fixed-resolution inputs cannot balance computational efficiency and fine-detail preservation
2. **Scale Inconsistency**: Single-scale alignment fails to capture multi-layered semantic structures

### Our Solution

1. **DRIS**: Coarse-to-fine dynamic resolution processing with saliency-based ROI selection
2. **MS-VLAM**: Three-tier alignment (object, local-region, global) for comprehensive semantic understanding

---

## Dynamic Resolution Input Strategy (DRIS)

### Concept

DRIS implements a two-stage coarse-to-fine approach:

1. **Stage 1 - Low-resolution global analysis**: Process entire image at low resolution to extract global context and identify regions of interest (ROIs)
2. **Stage 2 - High-resolution local refinement**: Apply high-resolution processing only to selected ROIs for fine-grained detail extraction

### Key Components

#### 1. Saliency Map Computation

```python
# Eq. (3) from paper
P_ROI = Topk(GaussianBlur(A_coarse), k)
```

- Computes attention heatmap from low-resolution features
- Applies Gaussian smoothing for noise reduction
- Selects top-K regions with highest saliency scores

#### 2. FPN Feature Fusion

```python
# Eq. (2) from paper
F_fused = Upsample(F_coarse) + Crop(F_fine, Ω)
```

- Feature Pyramid Network for multi-scale feature integration
- Lateral connections between different resolution levels
- Top-down pathway for context propagation

#### 3. Dynamic Resolution Allocation

```python
# Eq. (1) from paper
R(x,y) = {
    R_high if s(x,y) ≥ τ
    R_low otherwise
}
```

- Adaptive resolution based on saliency scores
- Threshold-based ROI selection
- Efficient computational resource allocation

### Implementation

**File**: `geochat/model/multimodal_encoder/dris_encoder.py`

**Key Classes**:
- `DRISVisionTower`: Main DRIS implementation
- `build_dris_vision_tower()`: Builder function for integration

**Usage Example**:

```python
from geochat.model.multimodal_encoder.dris_builder import build_dris_vision_tower

# Configure DRIS
config = type('Config', (), {
    'mm_vision_tower': 'openai/clip-vit-large-patch14',
    'use_dris': True,
    'dris_low_res_size': 224,
    'dris_high_res_size': 504,
    'dris_roi_threshold': 0.5,
    'dris_top_k_rois': 5,
    'dris_saliency_kernel_size': 5,
    'dris_fpn_channels': [256, 512, 1024, 2048]
})()

# Build DRIS tower
vision_tower = build_dris_vision_tower(config, device='cuda')

# Process image
features = vision_tower(image)
```

---

## Multi-scale Vision-language Alignment Mechanism (MS-VLAM)

### Concept

MS-VLAM implements three-level hierarchical alignment:

1. **Object-level**: Individual objects ↔ Entity descriptions
2. **Local-region-level**: Semantic regions ↔ Phrase descriptions
3. **Global-level**: Full image ↔ Complete text

### Key Components

#### 1. Object-level Alignment

```python
# Eq. (4) and (5)
v_obj^(p) = f_v(RoI(V, B_p))
o_obj^(p) = f_t(e_p)

# Eq. (7)
L_obj = 1 - (1/P) Σ w_p · cos(v_obj^(p), o_obj^(p))
```

- Uses object detector (Faster R-CNN, DETR) for bounding boxes
- RoI pooling for feature extraction
- IoU-based dynamic weighting for accurate localization

#### 2. Local-region-level Alignment

```python
# Eq. (8) - Hard matching
L_reg^hard = 1 - (1/K) Σ cos(v_k, p_π(k))

# Eq. (10) - Contrastive learning
L_reg^NCE = - (1/K) Σ log(exp(s_{k,y(k)}) / Σ_j exp(s_{kj}))

# Eq. (11) - Combined
L_reg = μ · L_reg^hard + (1-μ) · L_reg^NCE
```

- Uses SAM (Segment Anything Model) for region segmentation
- Hard matching for interpretable alignment
- InfoNCE loss for robust soft matching

#### 3. Global-level Alignment

```python
# Eq. (12)
L_glob = 1 - cos(SPP(V), t_[CLS])
```

- Spatial Pyramid Pooling (SPP) for multi-scale global features
- CLS token alignment with sentence-level representation
- Layer normalization for stability

#### 4. Total Alignment Loss

```python
# Eq. (13)
L_align = α · L_obj + β · L_reg + γ · L_glob
```

### Implementation

**File**: `geochat/model/multimodal_projector/ms_vlam_projector.py`

**Key Classes**:
- `MSVLAMProjector`: Main MS-VLAM implementation

**Usage Example**:

```python
from geochat.model.multimodal_projector.ms_vlam_projector import MSVLAMProjector

# Create MS-VLAM projector
msvlam = MSVLAMProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    alignment_dim=768,
    num_sam_regions=16,
    spp_pool_sizes=[1, 2, 4],
    alpha=1.0,  # Object-level weight
    beta=1.0,   # Region-level weight
    gamma=1.0,  # Global-level weight
    mu=0.5,     # Hard vs NCE weight
    temperature=0.07
)

# Forward pass
outputs = msvlam(
    vision_features=vision_feat,
    text_features={
        'object': object_text_feat,
        'phrase': phrase_text_feat,
        'global': global_text_feat
    },
    return_losses=True
)

# Get losses
L_obj = outputs['loss_obj']
L_reg = outputs['loss_reg']
L_glob = outputs['loss_glob']
L_align = outputs['loss_align']
```

---

## Installation and Setup

### Prerequisites

```bash
pip install torch torchvision
pip install pillow numpy
pip install transformers
```

### Repository Structure

```
GeoChat/
├── geochat/
│   ├── model/
│   │   ├── multimodal_encoder/
│   │   │   ├── dris_encoder.py          # DRIS implementation
│   │   │   ├── dris_builder.py         # DRIS builder
│   │   │   └── builder.py              # Updated to support DRIS
│   │   ├── multimodal_projector/
│   │   │   ├── ms_vlam_projector.py    # MS-VLAM implementation
│   │   │   └── builder.py              # Updated to support MS-VLAM
│   │   └── alignment_loss.py           # Loss functions
├── config/
│   └── dris_msvlam_config.py          # Configuration file
├── demo_dris_msvlam.py                # Demo script
└── docs/
    └── DRIS_MSVLAM.md                # This document
```

---

## Usage

### Quick Start

#### 1. Run Demo

```bash
# Run all demos
python demo_dris_msvlam.py

# Run specific demo
python demo_dris_msvlam.py --demo dris
python demo_dris_msvlam.py --demo msvlam
python demo_dris_msvlam.py --demo loss

# Use custom image
python demo_dris_msvlam.py --image path/to/image.png --demo all
```

#### 2. Load Configuration

```python
from config.dris_msvlam_config import get_config, print_config

# Get full model configuration
config = get_config('full_model')

# Print configuration
print_config(config)

# Get ablation configuration
ablation_config = get_config('dris_msvlam_no_obj')
```

#### 3. Training

```bash
# Using training script with DRIS and MS-VLAM
python geochat/train/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --vision_tower openai/clip-vit-large-patch14 \
    --use_dris True \
    --use_msvlam True \
    --data_path data/your_data.json \
    --output_dir output/dris_msvlam_model
```

---

## Configuration

### DRIS Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `use_dris` | `False` | Enable DRIS |
| `dris_low_res_size` | `224` | Low resolution for global analysis |
| `dris_high_res_size` | `504` | High resolution for ROI processing |
| `dris_roi_threshold` | `0.5` | Saliency threshold for ROI selection |
| `dris_top_k_rois` | `5` | Number of ROIs to select |
| `dris_saliency_kernel_size` | `5` | Gaussian kernel size |
| `dris_fpn_channels` | `[256, 512, 1024, 2048]` | FPN channel dimensions |

### MS-VLAM Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `use_msvlam` | `False` | Enable MS-VLAM |
| `msvlam_alignment_dim` | `768` | Alignment space dimension |
| `msvlam_num_regions` | `16` | Number of regions for local alignment |
| `msvlam_spp_sizes` | `[1, 2, 4]` | Spatial Pyramid Pooling sizes |
| `msvlam_alpha` | `1.0` | Object-level loss weight (α in Eq. 13) |
| `msvlam_beta` | `1.0` | Region-level loss weight (β in Eq. 13) |
| `msvlam_gamma` | `1.0` | Global-level loss weight (γ in Eq. 13) |
| `msvlam_mu` | `0.5` | Hard vs NCE loss weight (μ in Eq. 11) |
| `msvlam_temperature` | `0.07` | Temperature for contrastive learning (τ in Eq. 9) |

### Loss Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `delta` | `1.0` | Alignment loss weight (δ in Eq. 15) |
| `label_smoothing` | `0.0` | Label smoothing for caption loss |
| `ignore_index` | `-100` | Ignore index for loss computation |

---

## Training

### Training Configuration

```python
from config.dris_msvlam_config import TRAINING_CONFIG

# Full model configuration
config = TRAINING_CONFIG.copy()

# Key training parameters (from paper Section 3.4)
config.update({
    'per_device_train_batch_size': 8,
    'learning_rate': 3e-4,
    'warmup_steps': 100,
    'max_steps': 1000,
    'bf16': True,  # Use bfloat16 mixed precision
})
```

### Training Script

Modify existing training script to support new features:

```bash
python geochat/train/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type linear \
    --use_dris True \
    --use_msvlam True \
    --dris_low_res_size 224 \
    --dris_high_res_size 504 \
    --dris_roi_threshold 0.5 \
    --dris_top_k_rois 5 \
    --msvlam_alignment_dim 768 \
    --msvlam_alpha 1.0 \
    --msvlam_beta 1.0 \
    --msvlam_gamma 1.0 \
    --delta 1.0 \
    --data_path data/your_data.json \
    --image_folder data/images \
    --output_dir output/dris_msvlam_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --max_steps 1000 \
    --bf16 True
```

### Training Tips

1. **Memory Optimization**: Use gradient accumulation if batch size is too large
2. **Mixed Precision**: Enable `--bf16` for faster training and reduced memory
3. **Warmup**: Use 100-step warmup as specified in paper
4. **Learning Rate**: Start with 3e-4 and use linear decay

---

## Evaluation

### Supported Metrics

#### Image Captioning
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- SPICE

#### Cross-modal Retrieval
- R@1, R@5, R@10

#### Visual Grounding
- Accuracy@0.5

### Evaluation Script

```bash
# Evaluate image captioning
python geochat/eval/batch_geochat_scene.py \
    --model_path output/dris_msvlam_model \
    --data_path data/test.json

# Evaluate visual grounding
python geochat/eval/batch_geochat_grounding.py \
    --model_path output/dris_msvlam_model \
    --data_path data/grounding.json
```

---

## Ablation Studies

### Available Configurations

```python
from config.dris_msvlam_config import ABLATION_CONFIGS

# Baseline (no DRIS, no MS-VLAM)
config_baseline = get_config('baseline')

# DRIS only
config_dris_only = get_config('dris_only')

# MS-VLAM only
config_msvlam_only = get_config('msvlam_only')

# Full model (DRIS + MS-VLAM)
config_full = get_config('full_model')

# MS-VLAM ablations
config_no_obj = get_config('dris_msvlam_no_obj')
config_no_region = get_config('dris_msvlam_no_region')
config_no_global = get_config('dris_msvlam_no_global')
```

### Running Ablation Experiments

```bash
# Baseline
python train.py --config baseline --output_dir output/baseline

# DRIS only
python train.py --config dris_only --output_dir output/dris_only

# MS-VLAM only
python train.py --config msvlam_only --output_dir output/msvlam_only

# Full model
python train.py --config full_model --output_dir output/full_model

# MS-VLAM ablations
python train.py --config dris_msvlam_no_obj --output_dir output/no_obj
python train.py --config dris_msvlam_no_region --output_dir output/no_region
python train.py --config dris_msvlam_no_global --output_dir output/no_global
```

### Expected Results (from paper)

| Configuration | Accuracy@0.5 (DIOR-RSVG) |
|--------------|----------------------------|
| Baseline (LoRA) | 33.15 |
| DRIS only | 35.62 (+2.47) |
| MS-VLAM only | 37.04 (+3.89) |
| Full model | **40.27** (+7.12) |
| w/o object-level | 38.91 (-1.36) |
| w/o region-level | 39.10 (-1.17) |
| w/o global-level | 38.76 (-1.51) |

---

## Mathematical Formulations

### Dynamic Resolution Input Strategy

#### Eq. (1) - Dynamic Resolution Allocation
```
R(x,y) = {
    R_high if s(x,y) ≥ τ
    R_low otherwise
}
```

#### Eq. (2) - FPN Feature Fusion
```
F_fused = Upsample(F_coarse) + Crop(F_fine, Ω)
```

#### Eq. (3) - ROI Selection
```
P_ROI = Topk(GaussianBlur(A_coarse), k)
```

### Multi-scale Vision-language Alignment

#### Eq. (4)-(5) - Object-level Features
```
v_obj^(p) = f_v(RoI(V, B_p))
o_obj^(p) = f_t(e_p)
```

#### Eq. (6) - Cosine Similarity
```
cos(u, v) = (u^T v) / (||u|| ||v||)
```

#### Eq. (7) - Object-level Loss
```
L_obj = 1 - (1/P) Σ w_p · cos(v_obj^(p), o_obj^(p))
```

#### Eq. (8)-(11) - Region-level Loss
```
L_reg^hard = 1 - (1/K) Σ cos(v_k, p_π(k))
s_{kj} = v_k^T p_j / τ
L_reg^NCE = -(1/K) Σ log(exp(s_{k,y(k)}) / Σ_j exp(s_{kj}))
L_reg = μ · L_reg^hard + (1-μ) · L_reg^NCE
```

#### Eq. (12) - Global-level Loss
```
L_glob = 1 - cos(SPP(V), t_[CLS])
```

#### Eq. (13) - Total Alignment Loss
```
L_align = α · L_obj + β · L_reg + γ · L_glob
```

#### Eq. (14) - Caption Loss
```
L_caption = - Σ log P_L(x_k | V, x_<t)
```

#### Eq. (15) - Total Loss
```
L = L_caption + δ · L_align
```

---

## FAQ

### Q: Can I use DRIS and MS-VLAM independently?

A: Yes! Both features can be enabled/disabled independently:
- Enable only DRIS: `use_dris=True, use_msvlam=False`
- Enable only MS-VLAM: `use_dris=False, use_msvlam=True`

### Q: What is the computational overhead of DRIS?

A: DRIS reduces overall computation by:
- Processing full image at low resolution (224×224)
- Applying high resolution only to selected ROIs (5 regions at 504×504)
- Expected speedup: 2-3x compared to full 504×504 processing

### Q: How do I adjust the number of alignment levels?

A: Use loss weights (α, β, γ) to control contribution:
- Disable object-level: `msvlam_alpha=0.0`
- Disable region-level: `msvlam_beta=0.0`
- Disable global-level: `msvlam_gamma=0.0`

### Q: Can I use custom object detectors with MS-VLAM?

A: Yes! The MS-VLAM object-level alignment accepts pre-computed bbox features. Simply pass them via the `bbox_features` parameter.

### Q: What datasets are supported?

A: The implementation is designed for RS-GPT4V but can work with any remote sensing vision-language dataset with image-caption pairs.

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**:
1. Reduce batch size
2. Use gradient accumulation
3. Reduce `dris_top_k_rois`
4. Reduce `dris_high_res_size`

### Issue: Slow training

**Solution**:
1. Enable mixed precision (`--bf16 True`)
2. Reduce `dris_high_res_size` from 504 to 336
3. Reduce `msvlam_num_regions` from 16 to 8

### Issue: Poor convergence

**Solution**:
1. Increase warmup steps
2. Adjust learning rate
3. Tune loss weights (α, β, γ, δ)
4. Ensure data quality

---

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Multimodal Interpretation of Remote Sensing Images: Dynamic Resolution Input Strategy and Multi-scale Vision–Language Alignment Mechanism},
  author={Zhang, Siyu and Shan, Lianlei and Qiu, Runhe},
  journal={ICLR 2025},
  year={2025}
}
```

---

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

## License

This implementation follows the same license as the original GeoChat project.