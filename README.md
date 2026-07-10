# RAH-VLA: Resolution-Adaptive Hierarchical Vision-Language Alignment for Multimodal Remote Sensing Understanding

<p align="center">
  <a href="https://github.com/ZhangSiyu0066/RAH-VLA">
    <img src="https://img.shields.io/badge/GitHub-RAH--VLA-blue" alt="GitHub">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  </a>
</p>

RAH-VLA is a resolution-adaptive hierarchical vision-language alignment framework for multimodal remote sensing understanding. It is built on the GeoChat codebase and introduces two core components:

- **DRIS**: Dynamic Resolution Input Strategy for coarse-to-fine, token-level visual encoding.
- **MS-VLAM**: Multi-scale Vision-Language Alignment Mechanism for object-, region-, and global-level semantic alignment.

The framework is evaluated on remote sensing captioning, visual grounding, GPT-4V-based multimodal evaluation, hyperspectral captioning, ablation studies, and efficiency analysis.

**Paper**: [RAH-VLA PDF](RAH_VLA__Resolution_Adaptive_Hierarchical_Vision_Language_Alignment_for_Multimodal_Remote_Sensing_Understanding.pdf)

## Overview

<p align="center">
  <img src="images/Fig.1.png" width="95%" alt="Motivation and overview of RAH-VLA">
</p>

RAH-VLA targets two common limitations in remote sensing vision-language models: fixed-resolution visual processing and single-level image-text alignment. DRIS allocates more visual computation to informative regions while preserving global context, and MS-VLAM aligns visual and textual representations at multiple semantic levels.

<p align="center">
  <img src="images/Fig.2.png" width="95%" alt="RAH-VLA framework">
</p>

## Main Results

### Image Captioning

RAH-VLA achieves strong captioning performance on four benchmark remote sensing datasets. The table below reports the RAH-VLA rows from the paper.

| Dataset | Visual Encoder | Text Decoder | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NWPU-Captions | CLIP ViT-L/14 | Vicuna-7B-v1.5 | **0.889** | **0.812** | **0.748** | **0.689** | **0.458** | **0.791** | **2.061** | **0.334** |
| RSICD | CLIP ViT-L/14 | Vicuna-7B-v1.5 | 0.772 | 0.662 | 0.578 | 0.505 | **0.392** | **0.672** | **2.781** | **0.501** |
| UCM | CLIP ViT-L/14 | Vicuna-7B-v1.5 | **0.918** | **0.868** | **0.829** | **0.798** | **0.489** | **0.832** | **3.701** | **0.547** |
| Sydney | CLIP ViT-L/14 | Vicuna-7B-v1.5 | **0.927** | **0.872** | **0.826** | **0.793** | **0.482** | **0.789** | **2.864** | **0.458** |

### Visual Grounding on DIOR-RSVG

| Method | Accuracy@0.5 |
|---|---:|
| Qwen-VL-Chat | 25.05 |
| LLaVA-1.5 | 9.52 |
| Full-FT | 36.31 |
| LoRA | 33.15 |
| MoE-LoRA | 37.86 |
| **RAH-VLA (Ours)** | **40.27** |

### GPT-4V-Based Evaluation

| Method | Complex Reasoning & Conversation | Detailed Description | Overall Score |
|---|---:|---:|---:|
| LLaVA-1.5 | 5.210 | 5.088 | 5.194 |
| Qwen-VL-Chat | 2.648 | 2.282 | 2.599 |
| InternLM-XC2 | 5.312 | 4.392 | 5.189 |
| Full-FT | 6.270 | 6.530 | 6.304 |
| LoRA | 6.061 | 6.374 | 6.103 |
| MoE-LoRA | 6.108 | 6.468 | 6.156 |
| **RAH-VLA (Ours)** | **6.512** | **6.781** | **6.574** |

### Generalization on HyperCAP

| Dataset | Method | BLEU-1 | METEOR | ROUGE-L | CIDEr |
|---|---|---:|---:|---:|---:|
| Botswana | GeoChat-7B | 0.0245 | 0.0563 | 0.0332 | 0.0128 |
| Botswana | **RAH-VLA (Ours)** | **0.0304** | **0.0815** | **0.0433** | **0.0370** |
| Houston13 | GeoChat-7B | 0.0239 | 0.0572 | 0.0324 | 0.0146 |
| Houston13 | **RAH-VLA (Ours)** | **0.0339** | **0.1028** | **0.0515** | **0.0383** |
| Indian Pines | GeoChat-7B | 0.0263 | 0.0581 | 0.0341 | 0.0155 |
| Indian Pines | **RAH-VLA (Ours)** | **0.0319** | **0.0758** | **0.0433** | **0.0329** |
| KSC | GeoChat-7B | 0.0087 | 0.0206 | 0.0123 | 0.0082 |
| KSC | **RAH-VLA (Ours)** | **0.0109** | **0.0349** | **0.0169** | **0.0239** |
| Overall | GeoChat-7B | 0.0214 | 0.0485 | 0.0284 | 0.0124 |
| Overall | **RAH-VLA (Ours)** | **0.0267** | **0.0698** | **0.0378** | **0.0310** |

Overall relative improvements over GeoChat-7B are **+24.9% BLEU-1**, **+43.9% METEOR**, **+33.2% ROUGE-L**, and **+149.8% CIDEr**.

### Ablation on DIOR-RSVG

| ID | DRIS | MS-VLAM | MS-VLAM Sub-modules | Accuracy@0.5 |
|---|---|---|---|---:|
| A | No | No | - | 33.15 |
| B | Yes | No | - | 35.62 |
| C | No | Yes | Full | 37.04 |
| D | Yes | Yes | Full | **40.27** |
| E | Yes | Yes | w/o Object-level | 38.91 |
| F | Yes | Yes | w/o Local-region-level | 39.10 |
| G | Yes | Yes | w/o Global-level | 38.76 |

### Efficiency

| Method | FLOPs (G) | Time (ms) | Memory (GB) |
|---|---:|---:|---:|
| Baseline | 286.4 | 164.2 | 21.7 |
| Token Pruning | 238.7 | 139.5 | 19.4 |
| **RAH-VLA (Ours)** | **176.3** | **105.8** | **16.2** |

Compared with the baseline, RAH-VLA reduces FLOPs by **38.4%**, latency by **35.6%**, and memory consumption by **25.3%**.

## Visualization

<p align="center">
  <img src="images/visualization1.png" width="95%" alt="RAH-VLA visualization on optical remote sensing images">
</p>

<p align="center">
  <img src="images/visualization2.png" width="95%" alt="RAH-VLA visualization on hyperspectral remote sensing images">
</p>

## 📁 项目结构

```
RAH-VLA/
├── geochat/
│   ├── model/
│   │   ├── multimodal_encoder/
│   │   │   ├── dris_encoder.py          # DRIS 编码器
│   │   │   ├── dris_builder.py         # DRIS 构建器
│   │   │   ├── clip_encoder.py         # CLIP 编码器
│   │   │   └── builder.py              # 编码器构建器
│   │   ├── multimodal_projector/
│   │   │   ├── ms_vlam_projector.py    # MS-VLAM 投影器
│   │   │   └── builder.py              # 投影器构建器
│   │   ├── alignment_loss.py           # 多尺度对齐损失
│   │   └── geochat_arch.py             # 模型架构
│   ├── eval/                            # 评估脚本
│   ├── train/                           # 训练脚本
│   └── serve/                           # 推理服务
├── config/
│   └── dris_msvlam_config.py          # DRIS-MSVLAM 配置
├── demo_dris_msvlam.py                # 演示脚本
├── README_DRIS_MSVLAM.md              # DRIS-MSVLAM 详细文档
├── docs/
│   └── DRIS_MSVLAM.md                 # 详细技术文档
├── scripts/                           # 训练脚本
└── images/                            # 图像资源
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone git@github.com:ZhangSiyu0066/RAH-VLA.git
cd RAH-VLA

# 创建 conda 环境
conda create -n rsvlm python=3.10 -y
conda activate rsvlm

# 安装包
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# 安装训练所需包
pip install ninja
pip install flash-attn --no-build-isolation
```

### 2. 运行 DRIS-MSVLAM 演示

```bash
# 运行所有演示
python demo_dris_msvlam.py

# 运行特定演示
python demo_dris_msvlam.py --demo dris    # 仅 DRIS
python demo_dris_msvlam.py --demo msvlam  # 仅 MS-VLAM
python demo_dris_msvlam.py --demo loss    # 仅损失函数

# 使用自定义图像
python demo_dris_msvlam.py --image path/to/image.png --demo all
```

### 3. 配置模型

```python
from config.dris_msvlam_config import get_config

# 获取完整模型配置
config = get_config('full_model')

# 获取消融实验配置
config = get_config('dris_only')      # 仅 DRIS
config = get_config('msvlam_only')    # 仅 MS-VLAM
config = get_config('dris_msvlam_no_obj')  # MS-VLAM 无对象级
```

---

## 🔧 训练模型

### 使用 DRIS-MSVLAM 训练

```bash
python geochat/train/train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --vision_tower openai/clip-vit-large-patch14 \
    --use_dris True \
    --use_msvlam True \
    --dris_low_res_size 224 \
    --dris_high_res_size 504 \
    --dris_top_k_rois 5 \
    --msvlam_alpha 1.0 \
    --msvlam_beta 1.0 \
    --msvlam_gamma 1.0 \
    --delta 1.0 \
    --data_path data/your_data.json \
    --output_dir output/dris_msvlam_model \
    --bf16 True
```

### 训练参数说明

#### DRIS 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_dris` | `False` | 启用 DRIS |
| `dris_low_res_size` | `224` | 低分辨率大小 |
| `dris_high_res_size` | `504` | 高分辨率大小 |
| `dris_roi_threshold` | `0.5` | ROI 选择阈值 |
| `dris_top_k_rois` | `5` | 选择 ROI 数量 |
| `dris_saliency_kernel_size` | `5` | 高斯核大小 |

#### MS-VLAM 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_msvlam` | `False` | 启用 MS-VLAM |
| `msvlam_alignment_dim` | `768` | 对齐空间维度 |
| `msvlam_num_regions` | `16` | 区域数量 |
| `msvlam_alpha` | `1.0` | 对象级损失权重 (α) |
| `msvlam_beta` | `1.0` | 区域级损失权重 (β) |
| `msvlam_gamma` | `1.0` | 全局级损失权重 (γ) |
| `msvlam_mu` | `0.5` | 硬匹配 vs 对比学习权重 (μ) |
| `msvlam_temperature` | `0.07` | 温度参数 (τ) |

#### 损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `delta` | `1.0` | 对齐损失权重 (δ) |
| `label_smoothing` | `0.0` | 标签平滑 |
| `ignore_index` | `-100` | 忽略索引 |

---

## 📚 详细文档

### DRIS-MSVLAM 详细文档

完整的技术文档和实现细节请参考：
- **[README_DRIS_MSVLAM.md](README_DRIS_MSVLAM.md)** - DRIS-MSVLAM 实现总结
- **[docs/DRIS_MSVLAM.md](docs/DRIS_MSVLAM.md)** - 详细技术文档

包含内容：
- 详细算法描述
- 数学公式和方程
- 使用示例和代码
- 配置说明
- 训练指南
- 评估方法
- 消融实验
- 故障排除
- FAQ

### 其他文档

- **[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)** - 模型权重和下载
- **[docs/Evaluation.md](docs/Evaluation.md)** - 评估指南
- **[docs/LoRA.md](docs/LoRA.md)** - LoRA 训练说明
- **[docs/Data.md](docs/Data.md)** - 数据集说明
- **[docs/Customize_Component.md](docs/Customize_Component.md)** - 自定义组件指南

---

## 🧪 消融实验

### 可用配置

```python
from config.dris_msvlam_config import ABLATION_CONFIGS

# 基线 (无 DRIS, 无 MS-VLAM)
config_baseline = get_config('baseline')

# 仅 DRIS
config_dris_only = get_config('dris_only')

# 仅 MS-VLAM
config_msvlam_only = get_config('msvlam_only')

# 完整模型 (DRIS + MS-VLAM)
config_full = get_config('full_model')

# MS-VLAM 消融
config_no_obj = get_config('dris_msvlam_no_obj')
config_no_region = get_config('dris_msvlam_no_region')
config_no_global = get_config('dris_msvlam_no_global')
```

---

## 💡 使用建议

### 内存优化
- 使用梯度累积增加有效批次大小
- 减少批次大小
- 降低 `dris_high_res_size`
- 减少 ROI 数量

### 训练加速
- 启用混合精度 (`--bf16 True`)
- 使用低分辨率作为起点
- 减少区域数量

### 性能调优
- 调整损失权重 (α, β, γ, δ)
- 增加 warmup 步数
- 使用学习率调度
- 确保数据质量

---

## 🔬 核心算法

### DRIS (动态分辨率输入)

**原理**: 实现粗到细的两阶段处理策略

```python
# Eq. (1) - 动态分辨率分配
R(x,y) = {
    R_high if s(x,y) ≥ τ
    R_low otherwise
}

# Eq. (2) - FPN 特征融合
F_fused = Upsample(F_coarse) + Crop(F_fine, Ω)

# Eq. (3) - ROI 选择
P_ROI = Topk(GaussianBlur(A_coarse), k)
```

**优势**:
- 减少计算开销 2-3 倍
- 保留关键细节信息
- 自适应资源分配

### MS-VLAM (多尺度视觉-语言对齐)

**原理**: 实现三层级对齐机制

```python
# Eq. (7) - 对象级损失
L_obj = 1 - (1/P) Σ w_p · cos(v_obj^(p), o_obj^(p))

# Eq. (11) - 区域级损失
L_reg = μ · L_reg^hard + (1-μ) · L_reg^NCE

# Eq. (12) - 全局级损失
L_glob = 1 - cos(SPP(V), t_[CLS])

# Eq. (13) - 总对齐损失
L_align = α · L_obj + β · L_reg + γ · L_glob
```

**优势**:
- 捕获多粒度语义
- 增强跨模态一致性
- 提升细粒度理解能力

---

## 🏗️ 架构说明

RAH-VLA 基于 GeoChat (CVPR 2024) 架构，增强了以下能力：

### 原始 GeoChat 能力
- 图像和区域描述
- 视觉问答 (VQA)
- 场景分类
- 视觉定位对话
- 指代对象检测

### 增强 (DRIS-MSVLAM)
- 动态分辨率处理，提升效率
- 多尺度对齐机制，提升性能
- 更好的细粒度理解
- 改进的跨模态一致性

---

## 📜 引用

如果您在研究中使用了 RAH-VLA 或 DRIS-MSVLAM，请引用以下论文：

### GeoChat (基线)
```bibtex
@article{kuckreja2023geochat,
        title={GeoChat: Grounded Large Vision-Language Model for Remote Sensing},
        author={Kuckreja, Kartik and Danish, Muhammad S. and Naseer, Muzammal and Das, Abhijit and Khan, Salman and Khan, Fahad S.},
        journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
}
```

### DRIS-MSVLAM (本实现)
```bibtex
@inproceedings{zhang2025multimodal,
  title={Multimodal Interpretation of Remote Sensing Images: Dynamic Resolution Input Strategy and Multi-scale Vision–Language Alignment Mechanism},
  author={Zhang, Siyu and Shan, Lianlei and Qiu, Runhe},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

## 🙏 致谢

本项目基于以下开源项目：
- **[GeoChat](https://github.com/mbzuai-oryx/GeoChat)** - CVPR 2024
- **[LLaVA](https://github.com/haotian-liu/LLaVA)** - Large Language-and-Vision Assistant
- **[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)** - Large Language Model

感谢这些项目的作者发布的开源代码和模型。

---

## 📄 许可证

本项目遵循 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📧 联系方式

如有问题或建议，请在 GitHub 上创建 issue。

---

## 🔗 相关链接

- [GeoChat 原始仓库](https://github.com/mbzuai-oryx/GeoChat)
- [GeoChat 项目页面](https://mbzuai-oryx.github.io/GeoChat)
- [GeoChat 论文 (arXiv)](https://arxiv.org/abs/2311.15826)

---

**注意**: 本实现用于研究和教育目的。如需用于商业用途，请联系原作者获取许可。
