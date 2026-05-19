# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-19
# Prompt : Per-concept fragility analysis for text degradation: partition SACo results by
#          prompt word-count (1-word, 2-word, 3+-word) and compute mIoU + SI per bucket
#          per noise level. Tests hypothesis that shorter prompts are more fragile to
#          character-level substitutions (L1 on a 4-char word = 25% corruption vs ~6%
#          on a 15-char phrase). Outputs metrics JSON + line plot using C2A_PALETTE.

"""
Usage
-----
    python -m scripts.analyze_text_prompt_length <experiment_dir> <baseline_jsonl>

Example
-------
    python -m scripts.analyze_text_prompt_length \\
        experiments/2026-05-9_text-degradation-task \\
        text_degradation_L0.jsonl

Output
------
    <experiment_dir>/metrics_by_prompt_length.json
    <experiment_dir>/prompt_length_fragility.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycocotools import mask as mask_utils
from tqdm import tqdm


# Click2Act canonical palette
C2A_PALETTE = {
    "primary"   : "#2E86AB",
    "secondary" : "#A23B72",
    "tertiary"  : "#F18F01",
    "neutral"   : "#6C757D",
    "success"   : "#3BB273",
    "warning"   : "#E84855",
    "bg"        : "#F8F9FA",
    "text"      : "#212529",
}
C2A_ORDER = [
    C2A_PALETTE["primary"],
    C2A_PALETTE["secondary"],
    C2A_PALETTE["tertiary"],
    C2A_PALETTE["success"],
    C2A_PALETTE["warning"],
    C2A_PALETTE["neutral"],
]

BUCKETS = ["1-word", "2-word", "3+-word"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bucket(prompt: str) -> str:
    n = len(prompt.strip().split())
    if n == 1:
        return "1-word"
    if n == 2:
        return "2-word"
    return "3+-word"


def _enc(rle: dict) -> dict:
    return {
        "size":   rle["size"],
        "counts": rle["counts"].encode("utf-8") if isinstance(rle["counts"], str) else rle["counts"],
    }


def mean_miou_rles(pred_rles: list, gt_rles: list) -> float:
    """Best-match mIoU (Eq. 1–2): for each pred mask find max-IoU baseline mask, average."""
    if not pred_rles or not gt_rles:
        return 0.0
    pred_enc = [_enc(r) for r in pred_rles]
    gt_enc   = [_enc(r) for r in gt_rles]
    iou_mat  = np.array(mask_utils.iou(pred_enc, gt_enc, [0] * len(gt_enc)))
    if iou_mat.ndim == 1:
        iou_mat = iou_mat.reshape(len(pred_enc), -1)
    return float(iou_mat.max(axis=1).mean())


def entry_key(entry: dict) -> tuple:
    return (entry["image_id"], entry["original_prompt"])


def load_jsonl(path: Path) -> dict:
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data[entry_key(entry)] = entry
    return data


def get_noise_level(entry: dict):
    if entry.get("noise_level") is not None:
        return entry["noise_level"]
    return entry.get("noisy_prompt")


def _bucket_char_stats(baseline: dict) -> dict[str, dict]:
    """Compute mean/min/max non-space char length per bucket across the baseline."""
    acc = {b: [] for b in BUCKETS}
    for (_, prompt) in baseline:
        acc[_bucket(prompt)].append(len(prompt.replace(" ", "")))
    return {
        b: {
            "n":          len(acc[b]),
            "mean_chars": round(float(np.mean(acc[b])), 1) if acc[b] else 0.0,
            "min_chars":  int(min(acc[b])) if acc[b] else 0,
            "max_chars":  int(max(acc[b])) if acc[b] else 0,
        }
        for b in BUCKETS
    }


# ── Analysis ──────────────────────────────────────────────────────────────────

def run(exp_dir: Path, baseline_path: Path) -> None:
    print(f"Experiment : {exp_dir}")
    print(f"Baseline   : {baseline_path.name}\n")

    print("Loading baseline...")
    baseline = load_jsonl(baseline_path)
    print(f"  {len(baseline)} (image, prompt) pairs loaded.")

    char_stats = _bucket_char_stats(baseline)
    print("\n  Baseline distribution (non-space chars):")
    print(f"  {'Bucket':<12}  {'n':>6}  {'mean_chars':>10}  {'min':>4}  {'max':>4}")
    for b in BUCKETS:
        s = char_stats[b]
        print(f"  {b:<12}  {s['n']:>6}  {s['mean_chars']:>10.1f}  {s['min_chars']:>4}  {s['max_chars']:>4}")

    other_files = sorted(
        p for p in exp_dir.glob("*.jsonl")
        if p.resolve() != baseline_path.resolve() and "temp" not in p.stem
    )
    if not other_files:
        print("No comparison files found. Exiting.")
        return

    # results[level][bucket] = {n, skipped, mean_miou, sensitivity_index, eff_corruption_rate}
    results: dict = {}

    for fpath in other_files:
        print(f"\nLoading {fpath.name}...")
        data    = load_jsonl(fpath)
        common  = sorted(set(baseline.keys()) & set(data.keys()))
        missing = len(baseline) - len(common)
        if missing:
            print(f"  ⚠  {missing} pairs not in baseline — skipped.")

        level     = None
        bucket_acc = {b: {"sum": 0.0, "count": 0, "skipped": 0} for b in BUCKETS}

        for key in tqdm(common, desc=fpath.stem):
            base_rles = [p["segmentation"] for p in baseline[key]["predictions"]]
            if not base_rles:
                bucket_acc[_bucket(key[1])]["skipped"] += 1
                continue

            pred_rles = [p["segmentation"] for p in data[key]["predictions"]]
            if level is None:
                level = get_noise_level(data[key])

            b = _bucket(key[1])
            bucket_acc[b]["sum"]   += mean_miou_rles(pred_rles, base_rles)
            bucket_acc[b]["count"] += 1

        if level is None:
            print("  Could not determine noise level — skipping file.")
            continue

        results[level] = {}
        print(f"  Level {level}  (ℓ = {level} substitution{'s' if level != 1 else ''}):")
        print(f"  {'Bucket':<12}  {'n':>6}  {'mIoU':>8}  {'SI':>10}  {'eff_corr%':>10}")
        for b in BUCKETS:
            acc  = bucket_acc[b]
            n    = acc["count"]
            miou = acc["sum"] / n if n else 0.0
            si   = round((miou - 1.0) / level, 6) if isinstance(level, (int, float)) and level != 0 else 0.0
            # effective corruption rate = substitutions / mean non-space chars in bucket
            mean_chars = char_stats[b]["mean_chars"]
            eff_corr   = round(level / mean_chars, 4) if mean_chars > 0 else None
            results[level][b] = {
                "n":                    n,
                "skipped":              acc["skipped"],
                "mean_miou":            round(miou, 6),
                "sensitivity_index":    si,
                "eff_corruption_rate":  eff_corr,
            }
            print(f"  {b:<12}  {n:>6}  {miou:>8.4f}  {si:>10.6f}  {eff_corr*100:>9.1f}%")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = exp_dir / "metrics_by_prompt_length.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "experiment":   str(exp_dir),
            "baseline":     baseline_path.name,
            "char_stats":   char_stats,
            "levels":       {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"\nSaved → {out_json}")

    # ── SI summary table (paper-ready) ────────────────────────────────────────
    numeric_levels = sorted(l for l in results if isinstance(l, (int, float)))
    if numeric_levels:
        print("\n── Sensitivity Index by bucket ─────────────────────────────────────")
        header = f"{'Bucket':<12}" + "".join(f"  SI(L={l})" for l in numeric_levels)
        print(header)
        print("-" * len(header))
        for b in BUCKETS:
            row = f"{b:<12}" + "".join(
                f"  {results[l][b]['sensitivity_index']:>8.4f}" for l in numeric_levels
            )
            print(row)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not numeric_levels:
        print("No numeric levels found — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=C2A_PALETTE["bg"])
    ax.set_facecolor(C2A_PALETTE["bg"])

    for idx, b in enumerate(BUCKETS):
        miou_vals = [results[l][b]["mean_miou"] for l in numeric_levels]
        # Prepend the L=0 anchor (mIoU = 1.0 by construction)
        xs = [0] + numeric_levels
        ys = [1.0] + miou_vals
        ax.plot(xs, ys, marker="o", color=C2A_ORDER[idx], linewidth=2.0, label=b)

    # Annotate effective corruption rate for L=1 (most illustrative)
    if 1 in results:
        for idx, b in enumerate(BUCKETS):
            eff = results[1][b]["eff_corruption_rate"]
            miou_at_1 = results[1][b]["mean_miou"]
            ax.annotate(
                f"{eff*100:.0f}% chars\ncorrupted",
                xy=(1, miou_at_1),
                xytext=(1.15, miou_at_1 + 0.02 * (1 - idx)),
                fontsize=7,
                color=C2A_ORDER[idx],
            )

    ax.set_xlabel("Noise level ℓ  (character substitutions)", color=C2A_PALETTE["text"], fontsize=11)
    ax.set_ylabel("Mean IoU  (output stability)", color=C2A_PALETTE["text"], fontsize=11)
    ax.set_title(
        "Per-prompt-length fragility under text corruption\n(SACo-Gold pipeline)",
        color=C2A_PALETTE["text"], fontsize=12,
    )
    ax.set_xticks([0] + numeric_levels)
    ax.set_xlim(-0.1, max(numeric_levels) + 0.7)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(1.0, color=C2A_PALETTE["neutral"], linewidth=0.8, linestyle="--", label="clean baseline")
    ax.legend(title="Prompt length", framealpha=0.9, fontsize=9)
    ax.tick_params(colors=C2A_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C2A_PALETTE["neutral"])

    plt.tight_layout()
    out_fig = exp_dir / "prompt_length_fragility.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {out_fig}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-prompt-length mIoU fragility under text degradation (SACo pipeline)."
    )
    parser.add_argument("experiment_dir", type=Path, help="Path to text-degradation experiment directory")
    parser.add_argument("baseline_jsonl", type=str,  help="Baseline JSONL filename (inside experiment_dir)")
    args = parser.parse_args()

    exp_dir       = args.experiment_dir
    baseline_path = exp_dir / args.baseline_jsonl

    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    run(exp_dir, baseline_path)
