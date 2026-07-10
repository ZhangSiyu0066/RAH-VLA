#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/RAH-VLA

export HF_HOME=/root/autodl-fs/huggingface
export TRANSFORMERS_CACHE=/root/autodl-fs/huggingface
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=RAH-VLA-DRIS-MSVLAM
export WANDB_NAME=dris_msvlam_lora_vicuna7b_clip14
export WANDB_DIR=/root/autodl-fs/geochat/wandb
mkdir -p "${WANDB_DIR}"

MODEL_PATH=/root/autodl-fs/geochat/models/vicuna-7b-v1.5
VISION_TOWER=/root/autodl-fs/geochat/models/clip-vit-large-patch14
DATA_PATH=/root/autodl-fs/geochat/dataset/GeoChat_Instruct.json
IMAGE_FOLDER=/root/autodl-fs/geochat/images/final_images_llava
OUTPUT_DIR=/root/autodl-fs/geochat/checkpoints/dris_msvlam_lora

deepspeed --master_port "${MASTER_PORT:-29600}" geochat/train/train_dris_msvlam_lora.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path "${MODEL_PATH}" \
    --version v1 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --use_dris True \
    --dris_low_res_size 224 \
    --dris_high_res_size 504 \
    --dris_roi_threshold 0.5 \
    --dris_top_k_rois 5 \
    --dris_saliency_kernel_size 5 \
    --use_msvlam True \
    --msvlam_alignment_dim 768 \
    --msvlam_num_regions 16 \
    --msvlam_alpha 1.0 \
    --msvlam_beta 1.0 \
    --msvlam_gamma 1.0 \
    --msvlam_mu 0.5 \
    --msvlam_temperature 0.07 \
    --delta 1.0 \
    --bf16 True \
    --tf32 True \
    --bits 16 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --warmup_steps 100 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --save_steps 1000 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
