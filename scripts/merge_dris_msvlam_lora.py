import argparse
import os

import torch

from geochat.mm_utils import get_model_name_from_path
from geochat.model.builder import load_pretrained_model
from geochat.model.dris_msvlam_runtime import apply_runtime_patches


def merge_lora(args):
    apply_runtime_patches()

    model_name = args.model_name or get_model_name_from_path(args.model_path)
    if "geochat" not in model_name.lower() or "lora" not in model_name.lower():
        model_name = "geochat_dris_msvlam_lora"

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map=args.device_map,
        device=args.device,
    )

    os.makedirs(args.save_model_path, exist_ok=True)
    model.config.save_pretrained(args.save_model_path)
    model.save_pretrained(
        args.save_model_path,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.save_model_path)

    if image_processor is not None and args.save_image_processor:
        image_processor.save_pretrained(args.save_model_path)

    info_path = os.path.join(args.save_model_path, "MERGE_INFO.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Merged DRIS-MS-VLAM LoRA model.\n")
        f.write(f"LoRA path: {args.model_path}\n")
        f.write(f"Base model: {args.model_base}\n")
        f.write(f"Model name used for loading: {model_name}\n")
        f.write("LoRA weights were merged with merge_and_unload().\n")
        f.write("non_lora_trainables.bin was loaded before merging.\n")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="geochat_dris_msvlam_lora")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-shard-size", type=str, default="2GB")
    parser.add_argument("--safe-serialization", action="store_true")
    parser.add_argument("--save-image-processor", action="store_true")

    merge_lora(parser.parse_args())
