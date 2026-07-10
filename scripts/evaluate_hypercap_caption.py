import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clipped_ngram_precision(ref_tokens, hyp_tokens, n):
    if len(hyp_tokens) < n:
        return 0.0
    ref_counts = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    hyp_counts = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
    clipped = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    total = sum(hyp_counts.values())
    return clipped / total if total else 0.0


def bleu_n(ref_tokens, hyp_tokens, n):
    precisions = [clipped_ngram_precision(ref_tokens, hyp_tokens, i) for i in range(1, n + 1)]
    smoothed = [max(p, 1e-9) for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in smoothed) / n)
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    return bp * geo_mean


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = lcs_len(ref_tokens, hyp_tokens)
    r = lcs / len(ref_tokens)
    p = lcs / len(hyp_tokens)
    return (2 * p * r / (p + r)) if p + r else 0.0


def meteor_lite(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return (10 * precision * recall) / (recall + 9 * precision) if recall + 9 * precision else 0.0


def cider_lite(groups):
    refs = []
    hyps = []
    for row in groups:
        refs.append(tokenize(row["ground_truth"]))
        hyps.append(tokenize(row["answer"]))

    doc_freq = defaultdict(int)
    for ref in refs:
        grams = set()
        for n in range(1, 5):
            grams.update(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        for gram in grams:
            doc_freq[gram] += 1

    total_docs = max(len(refs), 1)

    def vector(tokens, n):
        counts = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        vec = {}
        for gram, count in counts.items():
            tf = count / max(sum(counts.values()), 1)
            idf = math.log((total_docs + 1) / (doc_freq.get(gram, 0) + 1))
            vec[gram] = tf * idf
        return vec

    scores = []
    for ref, hyp in zip(refs, hyps):
        per_n = []
        for n in range(1, 5):
            rv = vector(ref, n)
            hv = vector(hyp, n)
            dot = sum(hv[g] * rv.get(g, 0.0) for g in hv)
            hn = math.sqrt(sum(v * v for v in hv.values()))
            rn = math.sqrt(sum(v * v for v in rv.values()))
            per_n.append(dot / (hn * rn) if hn and rn else 0.0)
        scores.append(10.0 * sum(per_n) / 4)
    return sum(scores) / len(scores) if scores else 0.0


def compute_lite(rows):
    metrics = defaultdict(list)
    by_dataset = defaultdict(list)
    for row in rows:
        ref = tokenize(row["ground_truth"])
        hyp = tokenize(row["answer"])
        values = {
            "BLEU-1": bleu_n(ref, hyp, 1),
            "BLEU-2": bleu_n(ref, hyp, 2),
            "BLEU-3": bleu_n(ref, hyp, 3),
            "BLEU-4": bleu_n(ref, hyp, 4),
            "METEOR": meteor_lite(ref, hyp),
            "ROUGE-L": rouge_l(ref, hyp),
        }
        for name, value in values.items():
            metrics[name].append(value)
        by_dataset[row.get("dataset", "all")].append(row)

    summary = {
        name: sum(values) / len(values) if values else 0.0
        for name, values in metrics.items()
    }
    summary["CIDEr"] = cider_lite(rows)
    summary["SPICE"] = None
    dataset_summary = {}
    for dataset, dataset_rows in by_dataset.items():
        dataset_summary[dataset] = compute_lite_no_group(dataset_rows)
    return summary, dataset_summary


def compute_lite_no_group(rows):
    metrics = defaultdict(list)
    for row in rows:
        ref = tokenize(row["ground_truth"])
        hyp = tokenize(row["answer"])
        for n in range(1, 5):
            metrics[f"BLEU-{n}"].append(bleu_n(ref, hyp, n))
        metrics["METEOR"].append(meteor_lite(ref, hyp))
        metrics["ROUGE-L"].append(rouge_l(ref, hyp))
    summary = {name: sum(values) / len(values) if values else 0.0 for name, values in metrics.items()}
    summary["CIDEr"] = cider_lite(rows)
    summary["SPICE"] = None
    return summary


def try_coco_eval(rows):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.spice.spice import Spice

    refs = {i: [row["ground_truth"]] for i, row in enumerate(rows)}
    hyps = {i: [row["answer"]] for i, row in enumerate(rows)}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]
    out = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(refs, hyps)
        if isinstance(names, list):
            for name, value in zip(names, score):
                out[name] = value
        else:
            out[names] = score
    return out


def write_csv(path, rows):
    fieldnames = ["Dataset", "Samples", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--standard-coco", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    if not rows:
        raise ValueError(f"No predictions found in {args.predictions}")

    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "all")].append(row)

    table = []
    if args.standard_coco:
        try:
            overall = try_coco_eval(rows)
            dataset_scores = {name: try_coco_eval(ds_rows) for name, ds_rows in by_dataset.items()}
            metric_note = "standard pycocoevalcap"
        except Exception as exc:
            print(f"Standard COCO evaluation failed: {exc}")
            overall, dataset_scores = compute_lite(rows)
            metric_note = "lite fallback"
    else:
        overall, dataset_scores = compute_lite(rows)
        metric_note = "lite fallback"

    for dataset, score in sorted(dataset_scores.items()):
        row = {"Dataset": dataset, "Samples": len(by_dataset[dataset])}
        row.update(score)
        table.append(row)
    row = {"Dataset": "ALL", "Samples": len(rows)}
    row.update(overall)
    table.append(row)
    write_csv(args.output_csv, table)
    print(f"Wrote {args.output_csv} using {metric_note} metrics.")


if __name__ == "__main__":
    main()
