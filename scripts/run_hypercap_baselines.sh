#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/autodl-tmp/RAH-VLA}
CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
ENV_NAME=${ENV_NAME:-rsvlm}

QUESTION_FILE=${QUESTION_FILE:-$REPO_ROOT/hypercap_eval/questions/all.jsonl}
IMAGE_FOLDER=${IMAGE_FOLDER:-$REPO_ROOT/hypercap_eval/images}
ANSWER_DIR=${ANSWER_DIR:-$REPO_ROOT/hypercap_eval/answers}
LOG_DIR=${LOG_DIR:-$REPO_ROOT/hypercap_eval/logs}

OURS_MODEL=${OURS_MODEL:-/root/autodl-fs/geochat/models/dris_msvlam_lora_merged}
GEOCHAT_MODEL=${GEOCHAT_MODEL:-/root/autodl-fs/geochat/models/geochat-7b}
LLAVA_MODEL=${LLAVA_MODEL:-$REPO_ROOT/models/llava-v1.5-7b}
QWEN_MODEL=${QWEN_MODEL:-$REPO_ROOT/models/Qwen-VL-Chat}

mkdir -p "$ANSWER_DIR" "$LOG_DIR"
source "$CONDA_SH"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

wait_for_no_process() {
  local pattern=$1
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "Waiting for running job matching: $pattern"
    sleep 60
  done
}

wait_for_download() {
  local pattern=$1
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    echo "Waiting for download matching: $pattern"
    sleep 60
  done
}

run_geochat_like() {
  local name=$1
  local model_path=$2
  local batch_size=$3
  local model_name=$4
  local answers=$ANSWER_DIR/${name}_predictions.jsonl
  local log=$LOG_DIR/${name}_predictions.log
  echo "Running $name -> $answers"
  python geochat/eval/batch_geochat_hypercap_caption.py \
    --model-path "$model_path" \
    --model-name "$model_name" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$answers" \
    --max-new-tokens 64 \
    --batch-size "$batch_size" \
    --resume \
    > "$log" 2>&1
  python scripts/evaluate_hypercap_caption.py \
    --predictions "$answers" \
    --output-csv "$ANSWER_DIR/${name}_metrics.csv"
}

run_qwen() {
  local name=qwen_vl_chat
  local answers=$ANSWER_DIR/${name}_predictions.jsonl
  local log=$LOG_DIR/${name}_predictions.log
  echo "Running $name -> $answers"
  python geochat/eval/batch_qwen_vl_hypercap_caption.py \
    --model-path "$QWEN_MODEL" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$answers" \
    --resume \
    > "$log" 2>&1
  python scripts/evaluate_hypercap_caption.py \
    --predictions "$answers" \
    --output-csv "$ANSWER_DIR/${name}_metrics.csv"
}

echo "Waiting for current Ours generation to finish."
wait_for_no_process "batch_geochat_hypercap_caption.py --model-path $OURS_MODEL"

if [[ -s "$ANSWER_DIR/all_predictions.jsonl" ]]; then
  python scripts/evaluate_hypercap_caption.py \
    --predictions "$ANSWER_DIR/all_predictions.jsonl" \
    --output-csv "$ANSWER_DIR/ours_metrics.csv"
fi

echo "Running GeoChat baseline."
run_geochat_like geochat_7b "$GEOCHAT_MODEL" 8 geochat_7b

echo "Waiting for LLaVA download to finish."
wait_for_download "huggingface-cli download liuhaotian/llava-v1.5-7b"
echo "Running LLaVA-1.5 baseline."
run_geochat_like llava_v1_5_7b "$LLAVA_MODEL" 8 geochat_llava_v1_5_7b

echo "Waiting for Qwen-VL-Chat download to finish."
wait_for_download "huggingface-cli download Qwen/Qwen-VL-Chat"
echo "Running Qwen-VL-Chat baseline."
run_qwen

echo "All HyperCap baseline jobs finished."
