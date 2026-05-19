# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-15
# Prompt : Write a script to analyze the prompt selection folder, comparing
#          mIoU of each prompt result file against SA-1B ground truth masks.

"""
Usage
-----
    python -m scripts.analyze_prompt_selection

Output
------
    experiments/propmt-selection/metrics.json
    experiments/propmt-selection/metrics_per_image.json
"""

import json
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
from tqdm import tqdm

EXP_DIR  = Path("experiments/propmt-selection")
GT_DIR   = Path("data/SA-1B-Part-000999")
RESULT_FILES = [
    "prompt_visual_results.jsonl",
    "prompt_things_results.jsonl",
    "prompt_objects_results.jsonl",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enc(rle: dict) -> dict:
    return {
        "size":   rle["size"],
        "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"],
    }


def mean_miou_rles(pred_rles: list, gt_rles: list) -> float:
    """Best-match mIoU: for each prediction find max-IoU GT mask, then average."""
    if not pred_rles or not gt_rles:
        return 0.0
    pred_enc = [_enc(r) for r in pred_rles]
    gt_enc   = [_enc(r) for r in gt_rles]
    iou_mat  = np.array(mask_utils.iou(pred_enc, gt_enc, [0] * len(gt_enc)))
    if iou_mat.ndim == 1:
        iou_mat = iou_mat.reshape(len(pred_enc), -1)
    return float(iou_mat.max(axis=1).mean())


def load_gt(image_name: str) -> list[dict]:
    """Return list of COCO RLE dicts from the SA-1B ground-truth JSON."""
    gt_path = GT_DIR / f"{image_name}.json"
    if not gt_path.exists():
        return []
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)
    return [ann["segmentation"] for ann in data.get("annotations", [])]


def load_jsonl(path: Path) -> dict[str, dict]:
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            result[entry["image_name"]] = entry
    return result


def prompt_label(path: Path) -> str:
    """Extract a short label from the filename, e.g. 'visual', 'things', 'objects'."""
    stem = path.stem  # e.g. prompt_visual_results
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else stem


# ── Analysis ──────────────────────────────────────────────────────────────────

def run() -> None:
    summary      = []
    per_image    = {}

    for fname in RESULT_FILES:
        fpath = EXP_DIR / fname
        if not fpath.exists():
            print(f"[SKIP] {fname} not found")
            continue

        label = prompt_label(fpath)
        print(f"\n── Prompt: {label}  ({fname}) ──")
        data  = load_jsonl(fpath)

        miou_sum, count, skipped_no_pred, skipped_no_gt = 0.0, 0, 0, 0
        image_miou: dict[str, float] = {}

        for image_name, entry in tqdm(data.items(), desc=label):
            pred_rles = [p["segmentation"] for p in entry.get("predictions", [])]
            if not pred_rles:
                skipped_no_pred += 1
                image_miou[image_name] = None
                continue

            gt_rles = load_gt(image_name)
            if not gt_rles:
                skipped_no_gt += 1
                image_miou[image_name] = None
                continue

            miou = mean_miou_rles(pred_rles, gt_rles)
            image_miou[image_name] = round(miou, 6)
            miou_sum += miou
            count    += 1

        mean_miou = miou_sum / count if count else 0.0
        print(f"  mean mIoU = {mean_miou:.4f}  |  n={count}  "
              f"skipped_no_pred={skipped_no_pred}  skipped_no_gt={skipped_no_gt}")

        summary.append({
            "prompt":           label,
            "file":             fname,
            "n_evaluated":      count,
            "skipped_no_pred":  skipped_no_pred,
            "skipped_no_gt":    skipped_no_gt,
            "mean_miou":        round(mean_miou, 6),
        })
        per_image[label] = image_miou

    # ── Save outputs ──────────────────────────────────────────────────────────
    metrics_path = EXP_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, indent=2)
    print(f"\nSaved → {metrics_path}")

    per_image_path = EXP_DIR / "metrics_per_image.json"
    with open(per_image_path, "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)
    print(f"Saved → {per_image_path}")

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Prompt':<12}  {'mIoU':>8}  {'n':>6}  {'no_pred':>8}  {'no_gt':>6}")
    print("-" * 52)
    for row in summary:
        print(f"{row['prompt']:<12}  {row['mean_miou']:>8.4f}  "
              f"{row['n_evaluated']:>6}  {row['skipped_no_pred']:>8}  "
              f"{row['skipped_no_gt']:>6}")


if __name__ == "__main__":
    run()
