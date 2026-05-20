# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create systematic evaluation pipeline for SAM3 robustness testing

import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
import cv2

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_helper import load_model, run_sam3
from scripts.evaluation_metrics import compute_miou
from scripts.degradations import apply_combined_degradation
from scripts.data_visualization import load_sample
from constants import DATA_DIR

def load_sa1b_sample(image_name):
    """
    Load a sample from SA-1B dataset.
    Assumes format: {image_name}.jpg and {image_name}.json
    """
    try:
        img, meta = load_sample(image_name)

        # Extract ground truth information
        gt_masks = []
        gt_boxes = []
        gt_texts = []

        for ann in meta.get('annotations', []):
            # For synthetic data, create simple rectangular masks
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Create binary mask for the bounding box
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
                gt_masks.append(mask)
                gt_boxes.append(ann['bbox'])  # [x, y, w, h]

            # Extract text labels
            if 'category' in ann:
                gt_texts.append(ann['category'])

        # Use first text as prompt, or default
        text_prompt = gt_texts[0] if gt_texts else 'object'

        return img, gt_masks, text_prompt, gt_boxes

    except Exception as e:
        print(f"Error loading sample {image_name}: {e}")
        return None, None, None, None

def get_dataset_samples(data_dir, max_samples=100):
    """
    Get list of available samples from data directory.
    """
    samples = []
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, data_dir)
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.jpg'):
                sample_name = file.replace('.jpg', '')
                samples.append(sample_name)

    # If no data, use placeholder samples for testing
    if not samples:
        samples = ['sa_11177068', 'sa_11177073']  # From existing notebooks

    return samples[:max_samples]

def run_robustness_evaluation(output_dir='experiments/robustness_eval_2026-05-07',
                             max_samples=50):
    """
    Run systematic robustness evaluation across all degradation types.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define degradation parameter combinations
    degradation_configs = []

    # Geometric degradation levels
    geo_configs = [
        (0, 0),    # No degradation
        (10, 5),   # 10% expansion, 5% shift
        (20, 10),  # 20% expansion, 10% shift
        (30, 15),  # 30% expansion, 15% shift
    ]

    # Linguistic degradation levels
    ling_configs = [0.0, 0.1, 0.2, 0.3]  # Error rates

    # Visual degradation levels
    vis_configs = [0, 10, 25, 50]  # Sigma values

    # Generate all combinations
    for geo_exp, geo_shift in geo_configs:
        for ling_err in ling_configs:
            for vis_sigma in vis_configs:
                degradation_configs.append({
                    'geo_exp': geo_exp,
                    'geo_shift': geo_shift,
                    'ling_err': ling_err,
                    'vis_sigma': vis_sigma,
                    'config_name': f"geo_{geo_exp}_{geo_shift}_ling_{int(ling_err*100)}_vis_{vis_sigma}"
                })

    print(f"Total degradation configurations: {len(degradation_configs)}")

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        model, processor = load_model(device)
    except RuntimeError as exc:
        print(exc)
        print("Evaluation aborted because the SAM3 model could not be loaded.")
        return None

    # Get dataset samples
    samples = get_dataset_samples(DATA_DIR, max_samples)
    print(f"Evaluating on {len(samples)} samples")

    # Results storage
    results = []

    # Run evaluation
    for sample_name in tqdm(samples, desc="Processing samples"):
        img, gt_masks, gt_text, gt_boxes = load_sa1b_sample(sample_name)

        if img is None:
            continue

        # Baseline evaluation (no degradation)
        baseline_results = run_sam3(model, processor, img, device, gt_text, gt_boxes)
        baseline_miou = compute_miou(
            [mask.cpu().numpy() for mask in baseline_results['masks']],
            gt_masks
        )

        # Apply each degradation configuration
        for config in degradation_configs:
            # Apply combined degradation
            deg_img, deg_text, deg_boxes = apply_combined_degradation(
                img, gt_text, gt_boxes,
                config['geo_exp'], config['geo_shift'], config['ling_err'], config['vis_sigma']
            )

            # Run inference on degraded inputs
            pred_results = run_sam3(model, processor, deg_img, device, deg_text, deg_boxes)
            pred_masks = [mask.cpu().numpy() for mask in pred_results['masks']]

            # Compute metrics
            miou = compute_miou(pred_masks, gt_masks)
            sensitivity = (baseline_miou - miou) / max(
                config['geo_exp'] + config['geo_shift'] + config['ling_err']*100 + config['vis_sigma'], 1
            )

            # Store results
            result_entry = {
                'sample': sample_name,
                'config': config['config_name'],
                'baseline_miou': baseline_miou,
                'degraded_miou': miou,
                'sensitivity': sensitivity,
                'geo_exp': config['geo_exp'],
                'geo_shift': config['geo_shift'],
                'ling_err': config['ling_err'],
                'vis_sigma': config['vis_sigma'],
                'n_pred_masks': len(pred_masks),
                'n_gt_masks': len(gt_masks)
            }
            results.append(result_entry)

    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, 'robustness_results.csv')
    results_df.to_csv(results_file, index=False)

    # Compute summary statistics
    summary = results_df.groupby('config').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'sensitivity': ['mean', 'std']
    }).round(4)

    summary_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(summary_file)

    # Generate analysis report
    generate_analysis_report(results_df, output_dir)

    print(f"Evaluation complete. Results saved to {output_dir}")
    return results_df

def generate_analysis_report(results_df, output_dir):
    """
    Generate a comprehensive analysis report.
    """
    report = []

    report.append("# SAM3 Robustness Evaluation Report")
    report.append(f"Generated: 2026-05-07")
    report.append(f"Samples evaluated: {len(results_df['sample'].unique())}")
    report.append("")

    # Overall statistics
    baseline_mean = results_df['baseline_miou'].mean()
    report.append("## Overall Performance")
    report.append(f"Baseline mIoU: {baseline_mean:.4f}")
    report.append("")

    # Sensitivity by degradation type
    report.append("## Sensitivity Analysis")
    report.append("")

    # Geometric sensitivity
    geo_sens = results_df.groupby('geo_exp')['sensitivity'].mean()
    report.append("### Geometric Degradation Sensitivity")
    for exp, sens in geo_sens.items():
        if exp > 0:
            report.append(f"Expansion {exp}%: sensitivity = {sens:.4f}")
    report.append("")

    # Linguistic sensitivity
    ling_sens = results_df.groupby('ling_err')['sensitivity'].mean()
    report.append("### Linguistic Degradation Sensitivity")
    for err, sens in ling_sens.items():
        if err > 0:
            report.append(f"Error rate {err:.1f}: sensitivity = {sens:.4f}")
    report.append("")

    # Visual sensitivity
    vis_sens = results_df.groupby('vis_sigma')['sensitivity'].mean()
    report.append("### Visual Degradation Sensitivity")
    for sigma, sens in vis_sens.items():
        if sigma > 0:
            report.append(f"Sigma {sigma}: sensitivity = {sens:.4f}")
    report.append("")

    # Save report
    report_file = os.path.join(output_dir, 'analysis_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

if __name__ == "__main__":
    run_robustness_evaluation(max_samples=1)  # Just 1 sample for speed