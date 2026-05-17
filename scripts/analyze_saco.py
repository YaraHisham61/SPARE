# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-17
# Prompt : Generic mIoU analysis script for SACo-format JSONL experiments.
#          Entry format: {image_id, image_name, original_prompt, noise_level, predictions}.
#          Takes any experiment directory + baseline file; computes mIoU for every
#          other JSONL vs baseline. Works for visual noise, flip, and text degradation.
#          Supports --invert-flip to flip masks back before IoU (spatial equivariance metric).

"""
Usage
-----
    python -m scripts.analyze_saco <experiment_dir> <baseline_jsonl> [--invert-flip]

Examples
--------
    # SACo flip — output stability (direct comparison, no spatial alignment)
    python -m scripts.analyze_saco \\
        experiments/2026-5-16_visual-degradation-flip-task-SACo \\
        visual_degradation_flip_none.jsonl

    # SACo flip — spatial equivariance (flip masks back before IoU)
    python -m scripts.analyze_saco \\
        experiments/2026-5-16_visual-degradation-flip-task-SACo \\
        visual_degradation_flip_none.jsonl --invert-flip

    # SACo visual noise (once L0 file exists)
    python -m scripts.analyze_saco \\
        experiments/2026-05-16_visual-degradation-task-SACo \\
        visual_degradation_L0.jsonl

    # Text degradation
    python -m scripts.analyze_saco \\
        experiments/2026-05-9_text-degradation-task \\
        text_degradation_L0.jsonl

Output
------
    <experiment_dir>/metrics.json          (direct comparison)
    <experiment_dir>/metrics_flipped.json  (with --invert-flip)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from pycocotools import mask as mask_utils
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enc(rle: dict) -> dict:
    return {
        "size":   rle["size"],
        "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"],
    }


def mean_miou_rles(pred_rles: list, gt_rles: list) -> float:
    """Best-match mIoU using pycocotools C backend — fast path for non-flip comparisons."""
    if not pred_rles or not gt_rles:
        return 0.0
    pred_enc = [_enc(r) for r in pred_rles]
    gt_enc   = [_enc(r) for r in gt_rles]
    iou_mat  = np.array(mask_utils.iou(pred_enc, gt_enc, [0] * len(gt_enc)))
    if iou_mat.ndim == 1:
        iou_mat = iou_mat.reshape(len(pred_enc), -1)
    return float(iou_mat.max(axis=1).mean())


# flip_type → inverse transform applied to (H, W) mask arrays
# Integer keys match the mode used in the data generation script (0=h, 1=v, 2=both)
_FLIP_FN = {
    0:           lambda m: m[:, ::-1],
    1:           lambda m: m[::-1, :],
    2:           lambda m: m[::-1, ::-1],
    "h":         lambda m: m[:, ::-1],
    "v":         lambda m: m[::-1, :],
    "both":      lambda m: m[::-1, ::-1],
    "flip_h":    lambda m: m[:, ::-1],
    "flip_v":    lambda m: m[::-1, :],
    "flip_both": lambda m: m[::-1, ::-1],
}


def _detect_flip(level) -> int | str | None:
    """Return the _FLIP_FN key for this level, or None if it is not a flip condition."""
    if level in _FLIP_FN:
        return level
    if isinstance(level, str):
        key = level.lower().replace(" ", "_")
        return key if key in _FLIP_FN else None
    return None


def mean_miou_rles_flipped(pred_rles: list, gt_rles: list, flip_key: str) -> float:
    """Decode pred masks, invert the flip, then compute vectorised best-match mIoU."""
    if not pred_rles or not gt_rles:
        return 0.0

    flip_fn = _FLIP_FN[flip_key]

    # Decode and flip pred masks back to original orientation
    pred_masks = np.stack([flip_fn(mask_utils.decode(_enc(r))) for r in pred_rles])  # (N, H, W)
    gt_masks   = np.stack([mask_utils.decode(_enc(r))           for r in gt_rles  ])  # (M, H, W)

    N = pred_masks.shape[0]
    M = gt_masks.shape[0]

    pred_f = pred_masks.reshape(N, -1).astype(np.float32)  # (N, H*W)
    gt_f   = gt_masks.reshape(M, -1).astype(np.float32)    # (M, H*W)

    inter  = pred_f @ gt_f.T                                          # (N, M)
    union  = pred_f.sum(1, keepdims=True) + gt_f.sum(1) - inter      # (N, M)
    iou_mat = inter / np.maximum(union, 1e-10)
    return float(iou_mat.max(axis=1).mean())


def entry_key(entry: dict) -> tuple:
    """Unique key per (image, prompt) pair in SACo-format files."""
    return (entry["image_id"], entry["original_prompt"])


def load_jsonl(path: Path) -> dict:
    """Load SACo JSONL into {(image_id, original_prompt): entry}."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data[entry_key(entry)] = entry
    return data


def get_noise_level(entry: dict):
    """Extract noise_level (may be int or str)."""
    return entry.get("noise_level") or entry.get("noisy_prompt")


# ── Analysis ──────────────────────────────────────────────────────────────────

def run(exp_dir: Path, baseline_path: Path, invert_flip: bool = False) -> None:
    mode = "spatial equivariance (flip-back)" if invert_flip else "output stability (direct)"
    print(f"Experiment : {exp_dir}")
    print(f"Baseline   : {baseline_path.name}")
    print(f"Mode       : {mode}\n")

    print("Loading baseline...")
    baseline = load_jsonl(baseline_path)
    print(f"  {len(baseline)} (image, prompt) pairs loaded.")

    other_files = sorted(
        p for p in exp_dir.glob("*.jsonl")
        if p.resolve() != baseline_path.resolve() and "temp" not in p.stem
    )
    if not other_files:
        print("No comparison files found. Exiting.")
        return

    baseline_level = get_noise_level(next(iter(baseline.values())))
    levels = [{
        "file":              baseline_path.name,
        "noise_level":       baseline_level,
        "n_entries":         len(baseline),
        "mean_miou":         1.0,
        "sensitivity_index": 0.0,
    }]

    for fpath in other_files:
        print(f"\nLoading {fpath.name}...")
        data    = load_jsonl(fpath)
        common  = sorted(set(baseline.keys()) & set(data.keys()))
        missing = len(baseline) - len(common)
        if missing:
            print(f"  ⚠  {missing} (image, prompt) pairs missing — skipped.")

        miou_sum, count, skipped = 0.0, 0, 0
        level     = None
        flip_key  = None

        for key in tqdm(common, desc=fpath.stem):
            base_rles = [p["segmentation"] for p in baseline[key]["predictions"]]
            if not base_rles:
                skipped += 1
                continue
            pred_rles = [p["segmentation"] for p in data[key]["predictions"]]

            if level is None:
                level    = get_noise_level(data[key])
                flip_key = _detect_flip(level) if invert_flip else None

            if flip_key is not None:
                miou_sum += mean_miou_rles_flipped(pred_rles, base_rles, flip_key)
            else:
                miou_sum += mean_miou_rles(pred_rles, base_rles)
            count += 1

        miou = miou_sum / (count or 1)
        si   = round((miou - 1.0) / level, 6) if isinstance(level, (int, float)) and level != 0 else round(miou - 1.0, 6)

        levels.append({
            "file":              fpath.name,
            "noise_level":       level,
            "n_entries":         count,
            "skipped_no_base":   skipped,
            "mean_miou":         round(miou, 6),
            "sensitivity_index": si,
        })
        print(f"  level={level}  mIoU={miou:.4f}  S={si}  n={count}  skipped={skipped}")

    suffix   = "_flipped" if invert_flip else ""
    out_path = exp_dir / f"metrics{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":  str(exp_dir),
            "baseline":    baseline_path.name,
            "mode":        mode,
            "levels":      levels,
        }, f, indent=2)
    print(f"\nSaved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mIoU analysis for SACo-format experiments.")
    parser.add_argument("experiment_dir", type=Path, help="Path to experiment directory")
    parser.add_argument("baseline_jsonl", type=str,  help="Baseline JSONL filename (inside experiment_dir)")
    parser.add_argument(
        "--invert-flip", action="store_true",
        help="Flip predicted masks back to original orientation before IoU (spatial equivariance metric). "
             "Auto-detects flip type from noise_level field. Saves to metrics_flipped.json.",
    )
    args = parser.parse_args()

    exp_dir       = args.experiment_dir
    baseline_path = exp_dir / args.baseline_jsonl

    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    run(exp_dir, baseline_path, invert_flip=args.invert_flip)
