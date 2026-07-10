import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from geochat.model.dris_msvlam_runtime import apply_runtime_patches

apply_runtime_patches()

from geochat.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from geochat.conversation import SeparatorStyle, conv_templates  # noqa: E402
from geochat.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token  # noqa: E402
from geochat.model.builder import load_pretrained_model  # noqa: E402
from geochat.utils import disable_torch_init  # noqa: E402


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids(path):
    done = set()
    if not path or not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question_id" in row:
                done.add(row["question_id"])
    return done


def build_prompt(question, model, conv_mode):
    qs = question
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def eval_model(args):
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        args.model_name,
        device=args.device,
    )
    model.eval()

    questions = load_questions(args.question_file)
    if args.resume:
        done_ids = load_done_ids(args.answers_file)
        questions = [q for q in questions if q.get("question_id") not in done_ids]
        print(f"Resume enabled: skipped {len(done_ids)} completed samples.")
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    mode = "a" if args.resume else "w"
    with open(args.answers_file, mode, encoding="utf-8") as out:
        for start in tqdm(range(0, len(questions), args.batch_size), desc="Generating captions"):
            batch = questions[start : start + args.batch_size]
            prompts = []
            stop_strs = []
            for sample in batch:
                conv, prompt = build_prompt(sample["text"], model, args.conv_mode)
                prompts.append(prompt)
                stop_strs.append(conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2)

            input_rows = [
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                for prompt in prompts
            ]
            input_lens = [row.shape[0] for row in input_rows]
            if len(set(input_lens)) != 1:
                raise ValueError("Batch prompts must have the same token length.")
            input_ids = torch.stack(input_rows, dim=0).cuda()

            images = [
                Image.open(os.path.join(args.image_folder, sample["image"])).convert("RGB")
                for sample in batch
            ]
            image_tensor = image_processor.preprocess(
                images,
                crop_size={"height": 504, "width": 504},
                size={"shortest_edge": 504},
                return_tensors="pt",
            )["pixel_values"].half().cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            input_len = input_lens[0]
            answers = tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
            for sample, answer, stop_str in zip(batch, answers, stop_strs):
                answer = answer.strip()
                if stop_str in answer:
                    answer = answer.split(stop_str, 1)[0].strip()
                result = dict(sample)
                result["answer"] = answer
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--model-name", default="geochat_dris_msvlam_merged")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    eval_model(parser.parse_args())
