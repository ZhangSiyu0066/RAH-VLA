# DRIS + MS-VLAM Implementation Summary

本文档总结了基于研究论文实现的两个核心功能模块。

## 📋 实现内容

### 1. 动态分辨率输入策略 (DRIS)
- **文件位置**: `geochat/model/multimodal_encoder/dris_encoder.py`
- **构建器**: `geochat/model/multimodal_encoder/dris_builder.py`
- **核心特性**:
  - 粗到细的两阶段处理
  - 基于显著性的ROI选择
  - FPN特征融合
  - 注意力增强的ROI筛选

### 2. 多尺度视觉-语言对齐机制 (MS-VLAM)
- **文件位置**: `geochat/model/multimodal_projector/ms_vlam_projector.py`
- **核心特性**:
  - 对象级对齐 (Object-level)
  - 局部区域级对齐 (Local-region-level)
  - 全局级对齐 (Global-level)
  - 灵活的损失函数组合

### 3. 多尺度对齐损失函数
- **文件位置**: `geochat/model/alignment_loss.py`
- **包含内容**:
  - `MultiScaleAlignmentLoss`: 组合损失 (Eq. 15)
  - `CaptionLoss`: 标题生成损失 (Eq. 14)
  - `ContrastiveAlignmentLoss`: 对比学习损失
  - `AlignmentLossWrapper`: 灵活的损失组合器

## 📁 文件结构

```
GeoChat/
├── geochat/
│   ├── model/
│   │   ├── multimodal_encoder/
│   │   │   ├── dris_encoder.py          # DRIS实现
│   │   │   ├── dris_builder.py         # DRIS构建器
│   │   │   └── builder.py              # 更新以支持DRIS
│   │   ├── multimodal_projector/
│   │   │   ├── ms_vlam_projector.py    # MS-VLAM实现
│   │   │   └── builder.py              # 更新以支持MS-VLAM
│   │   └── alignment_loss.py           # 损失函数
├── config/
│   └── dris_msvlam_config.py          # 配置文件
├── demo_dris_msvlam.py                # 演示脚本
└── docs/
    └── DRIS_MSVLAM.md                # 详细文档
```

## 🚀 快速开始

### 1. 运行演示

```bash
# 运行所有演示
python demo_dris_msvlam.py

# 运行特定演示
python demo_dris_msvlam.py --demo dris    # 仅DRIS
python demo_dris_msvlam.py --demo msvlam  # 仅MS-VLAM
python demo_dris_msvlam.py --demo loss    # 仅损失函数

# 使用自定义图像
python demo_dris_msvlam.py --image path/to/image.png --demo all
```

### 2. 配置模型

```python
from config.dris_msvlam_config import get_config

# 获取完整模型配置
config = get_config('full_model')

# 获取消融实验配置
config = get_config('dris_only')      # 仅DRIS
config = get_config('msvlam_only')    # 仅MS-VLAM
config = get_config('dris_msvlam_no_obj')  # MS-VLAM无对象级
```

### 3. 训练模型

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

## 🔧 核心算法

### DRIS (动态分辨率输入)

**原理**: 实现粗到细的两阶段处理策略

```python
# Eq. (1) - 动态分辨率分配
R(x,y) = {
    R_high if s(x,y) ≥ τ
    R_low otherwise
}

# Eq. (2) - FPN特征融合
F_fused = Upsample(F_coarse) + Crop(F_fine, Ω)

# Eq. (3) - ROI选择
P_ROI = Topk(GaussianBlur(A_coarse), k)
```

**优势**:
- 减少计算开销2-3倍
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

### 损失函数

```python
# Eq. (14) - 标题生成损失
L_caption = - Σ log P_L(x_k | V, x_<t)

# Eq. (15) - 总损失
L = L_caption + δ · L_align
```

## 📊 预期性能提升

根据论文实验结果：

| 配置 | Accuracy@0.5 (DIOR-RSVG) | 提升 |
|------|---------------------------|------|
| 基线 (LoRA) | 33.15 | - |
| 仅DRIS | 35.62 | +2.47 |
| 仅MS-VLAM | 37.04 | +3.89 |
| 完整模型 | **40.27** | **+7.12** |
| 无对象级对齐 | 38.91 | -1.36 |
| 无区域级对齐 | 39.10 | -1.17 |
| 无全局级对齐 | 38.76 | -1.51 |

## ⚙️ 配置参数

### DRIS参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_dris` | `False` | 启用DRIS |
| `dris_low_res_size` | `224` | 低分辨率大小 |
| `dris_high_res_size` | `504` | 高分辨率大小 |
| `dris_roi_threshold` | `0.5` | ROI选择阈值 |
| `dris_top_k_rois` | `5` | 选择ROI数量 |
| `dris_saliency_kernel_size` | `5` | 高斯核大小 |

### MS-VLAM参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_msvlam` | `False` | 启用MS-VLAM |
| `msvlam_alignment_dim` | `768` | 对齐空间维度 |
| `msvlam_num_regions` | `16` | 区域数量 |
| `msvlam_alpha` | `1.0` | 对象级损失权重 (α) |
| `msvlam_beta` | `1.0` | 区域级损失权重 (β) |
| `msvlam_gamma` | `1.0` | 全局级损失权重 (γ) |
| `msvlam_mu` | `0.5` | 硬匹配vs对比学习权重 (μ) |
| `msvlam_temperature` | `0.07` | 温度参数 (τ) |

### 损失参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `delta` | `1.0` | 对齐损失权重 (δ) |
| `label_smoothing` | `0.0` | 标签平滑 |
| `ignore_index` | `-100` | 忽略索引 |

## 🧪 消融实验

### 可用配置

```python
from config.dris_msvlam_config import ABLATION_CONFIGS

# 基线 (无DRIS, 无MS-VLAM)
config_baseline = get_config('baseline')

# 仅DRIS
config_dris_only = get_config('dris_only')

# 仅MS-VLAM
config_msvlam_only = get_config('msvlam_only')

# 完整模型 (DRIS + MS-VLAM)
config_full = get_config('full_model')

# MS-VLAM消融
config_no_obj = get_config('dris_msvlam_no_obj')
config_no_region = get_config('dris_msvlam_no_region')
config_no_global = get_config('dris_msvlam_no_global')
```

## 📚 详细文档

完整的实现和使用文档请参考:
- **[docs/DRIS_MSVLAM.md](docs/DRIS_MSVLAM.md)**

包含内容:
- 详细算法描述
- 数学公式
- 使用示例
- 配置说明
- 训练指南
- 评估方法
- 故障排除
- FAQ

## 🔗 相关论文

本研究基于以下论文的数学公式:

**Multimodal Interpretation of Remote Sensing Images: Dynamic Resolution Input Strategy and Multi-scale Vision–Language Alignment Mechanism**

作者: Siyu Zhang, Lianlei Shan, Runhe Qiu  
会议: ICLR 2025

## 💡 使用建议

### 内存优化
- 使用梯度累积增加有效批次大小
- 减少批次大小
- 降低 `dris_high_res_size`
- 减少ROI数量

### 训练加速
- 启用混合精度 (`--bf16 True`)
- 使用低分辨率作为起点
- 减少区域数量

### 性能调优
- 调整损失权重 (α, β, γ, δ)
- 增加warmup步数
- 使用学习率调度
- 确保数据质量

## 📝 代码示例

### 使用DRIS编码器

```python
from geochat.model.multimodal_encoder.dris_encoder import DRISVisionTower
from geochat.model.multimodal_encoder.clip_encoder import CLIPVisionTower

# 创建基础CLIP编码器
base_tower = CLIPVisionTower('openai/clip-vit-large-patch14')

# 包装为DRIS
dris_tower = DRISVisionTower(
    base_vision_tower=base_tower,
    low_res_size=224,
    high_res_size=504,
    top_k_rois=5
)

# 处理图像
features = dris_tower(image)
```

### 使用MS-VLAM投影器

```python
from geochat.model.multimodal_projector.ms_vlam_projector import MSVLAMProjector

# 创建MS-VLAM投影器
msvlam = MSVLAMProjector(
    vision_hidden_size=1024,
    text_hidden_size=4096,
    alignment_dim=768,
    alpha=1.0,
    beta=1.0,
    gamma=1.0
)

# 前向传播
outputs = msvlam(
    vision_features=vision_feat,
    text_features={
        'object': obj_feat,
        'phrase': phrase_feat,
        'global': global_feat
    },
    return_losses=True
)

# 获取损失
L_align = outputs['loss_align']
```

### 使用损失函数

```python
from geochat.model.alignment_loss import MultiScaleAlignmentLoss

# 创建损失函数
loss_fn = MultiScaleAlignmentLoss(delta=1.0)

# 计算损失
outputs = loss_fn(
    logits=logits,
    labels=labels,
    alignment_losses={'loss_align': align_loss}
)

total_loss = outputs['loss']
```

## 🤝 贡献

欢迎贡献! 请遵循以下步骤:

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本实现遵循原始GeoChat项目的相同许可证。

## 📧 联系方式

如有问题或建议，请在GitHub上创建issue。

---

**注意**: 本实现仅用于研究和教育目的。如需用于商业用途，请联系原作者获取许可。