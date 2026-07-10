import argparse
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def eval_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        trust_remote_code=True,
        fp16=True,
    ).eval()

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
        for sample in tqdm(questions, desc="Generating captions"):
            image_path = os.path.join(args.image_folder, sample["image"])
            # Validate image early so corrupt samples are easier to locate.
            Image.open(image_path).convert("RGB").close()
            query = tokenizer.from_list_format(
                [
                    {"image": image_path},
                    {"text": sample["text"]},
                ]
            )
            with torch.inference_mode():
                answer, _ = model.chat(tokenizer, query=query, history=None)
            result = dict(sample)
            result["answer"] = answer.strip()
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    eval_model(parser.parse_args())
