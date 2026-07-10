import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image


DATASET_CSV = {
    "Botswana": "Bostwana.csv",
    "Houston13": "houston13.csv",
    "Indian Pines": "Indian_pines.csv",
    "KSC": "KSC.csv",
}


def load_mat_array(path, key):
    mat = sio.loadmat(path)
    if key in mat:
        return mat[key]
    keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(keys) == 1:
        return mat[keys[0]]
    raise KeyError(f"{path} does not contain key {key}; found {keys}")


def find_file(folder, suffix):
    matches = [p for p in Path(folder).iterdir() if p.name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No *{suffix} file found in {folder}")
    return matches[0]


def fit_pca_rgb(cube, seed=42, max_pixels=200000):
    # cube: (bands, height, width)
    bands, height, width = cube.shape
    x = cube.reshape(bands, -1).T.astype(np.float32)
    rng = np.random.default_rng(seed)
    if x.shape[0] > max_pixels:
        idx = rng.choice(x.shape[0], size=max_pixels, replace=False)
        sample = x[idx]
    else:
        sample = x

    band_min = np.percentile(sample, 1, axis=0)
    band_max = np.percentile(sample, 99, axis=0)
    denom = np.maximum(band_max - band_min, 1e-6)
    sample_norm = np.clip((sample - band_min) / denom, 0.0, 1.0)
    mean = sample_norm.mean(axis=0, keepdims=True)
    centered = sample_norm - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T

    x_norm = np.clip((x - band_min) / denom, 0.0, 1.0)
    rgb = (x_norm - mean) @ components
    rgb = rgb.reshape(height, width, 3)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [1, 99])
        rgb[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def build_caption_queues(csv_path):
    df = pd.read_csv(csv_path)
    df["Class name"] = df["Class name"].str.strip().str.lower()
    queues = defaultdict(list)
    for _, row in df.iterrows():
        queues[row["Class name"]].append(str(row["Description"]))
    return queues


def save_patch(rgb_image, row, col, patch_size, image_size, output_path, image_format):
    pad = patch_size // 2
    padded = np.pad(rgb_image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    patch = padded[row : row + patch_size, col : col + patch_size]
    image = Image.fromarray(patch, mode="RGB").resize((image_size, image_size), Image.BICUBIC)
    if image_format.lower() in {"jpg", "jpeg"}:
        image.save(output_path, quality=95, subsampling=0)
    else:
        image.save(output_path)


def prepare_dataset(dataset_dir, output_root, args):
    dataset_name = Path(dataset_dir).name
    print(f"Preparing {dataset_name}...")
    data_path = find_file(dataset_dir, "_data.mat")
    gt_path = find_file(dataset_dir, "_gt.mat")
    csv_path = Path(dataset_dir) / DATASET_CSV.get(dataset_name, "")
    if not csv_path.exists():
        csv_path = next(Path(dataset_dir).glob("*.csv"))
    class_map_path = Path(dataset_dir) / "class_map.json"

    cube = load_mat_array(data_path, "data")
    gt = load_mat_array(gt_path, "gt")
    if cube.shape[0] != gt.shape[0] and cube.shape[0] != gt.shape[1]:
        pass
    if cube.shape[0] == gt.shape[0] and cube.shape[1] == gt.shape[1]:
        cube = np.moveaxis(cube, -1, 0)
    gt = gt.astype(np.int64)

    with open(class_map_path, "r", encoding="utf-8") as f:
        class_map = json.load(f)
    caption_queues = build_caption_queues(csv_path)
    caption_offsets = defaultdict(int)

    rgb = fit_pca_rgb(cube, seed=args.seed, max_pixels=args.pca_max_pixels)
    labeled = [(int(r), int(c), int(gt[r, c])) for r, c in zip(*np.nonzero(gt))]

    if args.max_per_dataset and len(labeled) > args.max_per_dataset:
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.choice(len(labeled), size=args.max_per_dataset, replace=False).tolist())
        labeled = [labeled[i] for i in keep]

    image_dir = Path(output_root) / "images" / dataset_name.replace(" ", "_")
    image_dir.mkdir(parents=True, exist_ok=True)
    question_dir = Path(output_root) / "questions"
    question_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = question_dir / f"{dataset_name.replace(' ', '_')}.jsonl"

    records = []
    ext = "jpg" if args.image_format.lower() in {"jpg", "jpeg"} else "png"
    for idx, (row, col, raw_label) in enumerate(labeled):
        zero_label = raw_label - 1
        class_name = class_map.get(str(zero_label), class_map.get(str(raw_label), "Unknown"))
        class_key = class_name.strip().lower()
        captions = caption_queues.get(class_key, ["No description available."])
        offset = caption_offsets[class_key]
        caption = captions[offset % len(captions)]
        caption_offsets[class_key] += 1

        image_rel = f"{dataset_name.replace(' ', '_')}/{idx:06d}.{ext}"
        image_path = Path(output_root) / "images" / image_rel
        save_patch(rgb, row, col, args.patch_size, args.image_size, image_path, args.image_format)
        records.append(
            {
                "question_id": f"{dataset_name.replace(' ', '_')}_{idx:06d}",
                "dataset": dataset_name,
                "image": image_rel,
                "text": args.prompt,
                "ground_truth": caption,
                "class_name": class_name,
                "label": int(zero_label),
                "position": [row, col],
            }
        )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{dataset_name}: wrote {len(records)} samples to {out_jsonl}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypercap-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="*", default=["Botswana", "Houston13", "Indian Pines", "KSC"])
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--max-per-dataset", type=int, default=0)
    parser.add_argument("--pca-max-pixels", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Describe this remote sensing image in detail.")
    args = parser.parse_args()

    root = Path(args.hypercap_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    for name in args.datasets:
        all_records.extend(prepare_dataset(root / "Datasets" / name, output_root, args))

    all_path = output_root / "questions" / "all.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Total: wrote {len(all_records)} samples to {all_path}")


if __name__ == "__main__":
    main()
