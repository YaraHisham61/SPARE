# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-12
# Prompt : Create comprehensive evaluation script with detailed degradation results output

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_helper import load_model, run_sam3
from scripts.evaluation_metrics import compute_miou
from scripts.degradations import apply_combined_degradation
from scripts.data_visualization import load_sample
from constants import DATA_DIR

def load_sa1b_sample(image_name):
    """Load a sample from SA-1B dataset."""
    try:
        img, meta = load_sample(image_name)
        gt_masks = []
        gt_boxes = []
        gt_texts = []

        for ann in meta.get('annotations', []):
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
                gt_masks.append(mask)
                gt_boxes.append(ann['bbox'])

            if 'category' in ann:
                gt_texts.append(ann['category'])

        text_prompt = gt_texts[0] if gt_texts else 'object'
        return img, gt_masks, text_prompt, gt_boxes

    except Exception as e:
        print(f"Error loading sample {image_name}: {e}")
        return None, None, None, None

def get_dataset_samples(data_dir, max_samples=100):
    """Get list of available samples from data directory."""
    samples = []
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, data_dir)
    
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.jpg'):
                sample_name = file.replace('.jpg', '')
                samples.append(sample_name)

    if not samples:
        samples = ['sa_11177068', 'sa_11177073']

    return samples[:max_samples]

def run_comprehensive_evaluation(output_dir=None, max_samples=10):
    """Run systematic robustness evaluation with comprehensive results output."""
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f'experiments/robustness_eval_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SAM3 ROBUSTNESS EVALUATION")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    
    # Define degradation parameter combinations
    degradation_configs = []
    
    geo_configs = [
        (0, 0),    # No degradation (baseline)
        (10, 5),   # Light geometric noise
        (20, 10),  # Medium geometric noise
        (30, 15),  # Heavy geometric noise
    ]
    
    ling_configs = [0.0, 0.1, 0.2, 0.3]  # Error rates
    vis_configs = [0, 10, 25, 50]         # Sigma (noise std dev)
    
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
        print(f"✗ Model loading failed: {exc}")
        return None
    
    # Get dataset samples
    samples = get_dataset_samples(DATA_DIR, max_samples)
    print(f"Evaluating on {len(samples)} samples\n")
    
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
            deg_img, deg_text, deg_boxes = apply_combined_degradation(
                img, gt_text, gt_boxes,
                config['geo_exp'], config['geo_shift'], config['ling_err'], config['vis_sigma']
            )
            
            pred_results = run_sam3(model, processor, deg_img, device, deg_text, deg_boxes)
            pred_masks = [mask.cpu().numpy() for mask in pred_results['masks']]
            
            miou = compute_miou(pred_masks, gt_masks)
            degradation_factor = max(
                config['geo_exp'] + config['geo_shift'] + config['ling_err']*100 + config['vis_sigma'], 1
            )
            sensitivity = (baseline_miou - miou) / degradation_factor
            
            result_entry = {
                'sample': sample_name,
                'config': config['config_name'],
                'baseline_miou': baseline_miou,
                'degraded_miou': miou,
                'miou_drop': baseline_miou - miou,
                'sensitivity': sensitivity,
                'geo_exp': config['geo_exp'],
                'geo_shift': config['geo_shift'],
                'ling_err': config['ling_err'],
                'vis_sigma': config['vis_sigma'],
                'n_pred_masks': len(pred_masks),
                'n_gt_masks': len(gt_masks)
            }
            results.append(result_entry)
    
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'degradation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    # Generate comprehensive report
    generate_comprehensive_report(results_df, output_dir)
    
    return results_df, output_dir

def generate_comprehensive_report(results_df, output_dir):
    """Generate comprehensive analysis report with degradation impact."""
    
    report_lines = []
    
    report_lines.append("# SAM3 Robustness Evaluation - Degradation Results Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total evaluations: {len(results_df)}")
    report_lines.append(f"Unique samples: {results_df['sample'].nunique()}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("## 📊 Overall Performance")
    report_lines.append("")
    baseline_rows = results_df[results_df['geo_exp'] == 0]
    baseline_rows = baseline_rows[baseline_rows['ling_err'] == 0]
    baseline_rows = baseline_rows[baseline_rows['vis_sigma'] == 0]
    
    if len(baseline_rows) > 0:
        baseline_mean = baseline_rows['baseline_miou'].mean()
        baseline_std = baseline_rows['baseline_miou'].std()
        report_lines.append(f"**Baseline mIoU (no degradation):** {baseline_mean:.4f} ± {baseline_std:.4f}")
    
    worst_mean = results_df['degraded_miou'].min()
    worst_config = results_df.loc[results_df['degraded_miou'].idxmin()]
    report_lines.append(f"**Worst-case mIoU:** {worst_mean:.4f} (config: {worst_config['config']})")
    
    avg_drop = results_df['miou_drop'].mean()
    max_drop = results_df['miou_drop'].max()
    report_lines.append(f"**Average mIoU drop:** {avg_drop:.4f}")
    report_lines.append(f"**Maximum mIoU drop:** {max_drop:.4f}")
    report_lines.append("")
    
    # Geometric degradation analysis
    report_lines.append("## 📐 Geometric Degradation Impact")
    report_lines.append("")
    report_lines.append("Geometric degradation applies bounding box perturbations.")
    report_lines.append("")
    
    geo_analysis = results_df.groupby('geo_exp').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'miou_drop': 'mean',
        'sensitivity': 'mean'
    }).round(4)
    
    report_lines.append(geo_analysis.to_string())
    report_lines.append("")
    
    geo_impact = results_df.groupby('geo_exp')['miou_drop'].mean()
    report_lines.append("**Average mIoU drop by expansion level:**")
    for exp, drop in geo_impact.items():
        report_lines.append(f"- Expansion {exp}%: {drop:.4f} drop")
    report_lines.append("")
    
    # Linguistic degradation analysis
    report_lines.append("## 💬 Linguistic Degradation Impact")
    report_lines.append("")
    report_lines.append("Linguistic degradation applies typos and character errors to prompts.")
    report_lines.append("")
    
    ling_analysis = results_df.groupby('ling_err').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'miou_drop': 'mean',
        'sensitivity': 'mean'
    }).round(4)
    
    report_lines.append(ling_analysis.to_string())
    report_lines.append("")
    
    ling_impact = results_df.groupby('ling_err')['miou_drop'].mean()
    report_lines.append("**Average mIoU drop by error rate:**")
    for err, drop in ling_impact.items():
        if err == 0:
            report_lines.append(f"- Error rate {err:.1%}: {drop:.4f} drop (baseline)")
        else:
            report_lines.append(f"- Error rate {err:.1%}: {drop:.4f} drop")
    report_lines.append("")
    
    # Visual degradation analysis
    report_lines.append("## 👁️ Visual Degradation Impact")
    report_lines.append("")
    report_lines.append("Visual degradation applies Gaussian noise (AWGN) to images.")
    report_lines.append("")
    
    vis_analysis = results_df.groupby('vis_sigma').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'miou_drop': 'mean',
        'sensitivity': 'mean'
    }).round(4)
    
    report_lines.append(vis_analysis.to_string())
    report_lines.append("")
    
    vis_impact = results_df.groupby('vis_sigma')['miou_drop'].mean()
    report_lines.append("**Average mIoU drop by noise level (σ):**")
    for sigma, drop in vis_impact.items():
        if sigma == 0:
            report_lines.append(f"- Noise σ={sigma}: {drop:.4f} drop (baseline)")
        else:
            report_lines.append(f"- Noise σ={sigma}: {drop:.4f} drop")
    report_lines.append("")
    
    # Cross-modal sensitivity analysis
    report_lines.append("## 🔄 Cross-Modal Sensitivity")
    report_lines.append("")
    report_lines.append("Analysis of combined degradation effects.")
    report_lines.append("")
    
    # Most robust configuration
    robust_config = results_df.loc[results_df['miou_drop'].idxmin()]
    report_lines.append(f"**Most robust configuration:**")
    report_lines.append(f"- mIoU drop: {robust_config['miou_drop']:.4f}")
    report_lines.append(f"- Config: {robust_config['config']}")
    report_lines.append("")
    
    # Least robust configuration
    least_robust = results_df.loc[results_df['miou_drop'].idxmax()]
    report_lines.append(f"**Least robust configuration:**")
    report_lines.append(f"- mIoU drop: {least_robust['miou_drop']:.4f}")
    report_lines.append(f"- Config: {least_robust['config']}")
    report_lines.append("")
    
    # Sensitivity ranking
    report_lines.append("## 📈 Sensitivity Ranking")
    report_lines.append("")
    report_lines.append("Average sensitivity to each degradation type (higher = less robust):")
    report_lines.append("")
    
    geo_sens = results_df[results_df['geo_exp'] > 0]['sensitivity'].mean()
    ling_sens = results_df[results_df['ling_err'] > 0]['sensitivity'].mean()
    vis_sens = results_df[results_df['vis_sigma'] > 0]['sensitivity'].mean()
    
    sensitivities = {
        'Geometric': geo_sens,
        'Linguistic': ling_sens,
        'Visual': vis_sens
    }
    
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for i, (dtype, sens) in enumerate(sorted_sens, 1):
        report_lines.append(f"{i}. **{dtype}**: {sens:.6f}")
    report_lines.append("")
    
    # Top 5 most sensitive configurations
    report_lines.append("## ⚠️ Top 5 Most Sensitive Configurations")
    report_lines.append("")
    top_5 = results_df.nlargest(5, 'sensitivity')[['config', 'sensitivity', 'miou_drop']]
    report_lines.append(top_5.to_string())
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## 💡 Recommendations")
    report_lines.append("")
    report_lines.append("Based on the robustness analysis:")
    report_lines.append("")
    
    if ling_sens > max(geo_sens, vis_sens) * 1.2:
        report_lines.append("1. **Improve prompt robustness** - Linguistic degradation shows highest sensitivity")
        report_lines.append("   - Consider prompt normalization techniques")
        report_lines.append("   - Implement spell-checking preprocessing")
        report_lines.append("")
    
    if geo_sens > max(ling_sens, vis_sens) * 1.2:
        report_lines.append("1. **Improve geometric robustness** - Bounding box perturbations are most damaging")
        report_lines.append("   - Consider IoU-based loss functions")
        report_lines.append("   - Implement data augmentation with bbox noise")
        report_lines.append("")
    
    if vis_sens > max(ling_sens, geo_sens) * 1.2:
        report_lines.append("1. **Improve visual robustness** - Image noise has highest impact")
        report_lines.append("   - Train with noisy augmentations")
        report_lines.append("   - Consider denoising preprocessing")
        report_lines.append("")
    
    report_lines.append("2. **Multi-modal robustness** - Combine improvements across all modalities")
    report_lines.append("3. **Test with real-world degradation** - Synthetic degradations may not capture all failure modes")
    report_lines.append("")
    
    # Save report
    report_file = os.path.join(output_dir, 'DEGRADATION_RESULTS.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Comprehensive report saved to: {report_file}")
    
    # Generate summary statistics CSV
    summary_stats = pd.DataFrame({
        'Metric': ['Baseline mIoU (mean)', 'Worst-case mIoU', 'Average mIoU drop', 'Maximum mIoU drop',
                   'Geometric sensitivity', 'Linguistic sensitivity', 'Visual sensitivity'],
        'Value': [
            f"{baseline_mean:.4f}" if len(baseline_rows) > 0 else "N/A",
            f"{worst_mean:.4f}",
            f"{avg_drop:.4f}",
            f"{max_drop:.4f}",
            f"{geo_sens:.6f}",
            f"{ling_sens:.6f}",
            f"{vis_sens:.6f}"
        ]
    })
    
    summary_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_file, index=False)
    print(f"✓ Summary statistics saved to: {summary_file}")
    
    # Save top configurations to CSV
    top_configs = results_df.groupby('config').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'miou_drop': 'mean',
        'sensitivity': 'mean'
    }).round(4).sort_values(('miou_drop', 'mean'), ascending=False)
    
    config_file = os.path.join(output_dir, 'configuration_analysis.csv')
    top_configs.to_csv(config_file)
    print(f"✓ Configuration analysis saved to: {config_file}")

def main():
    """Main entry point."""
    results_df, output_dir = run_comprehensive_evaluation(max_samples=5)
    
    if results_df is not None:
        print(f"\n{'='*70}")
        print(f"✓ EVALUATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nResults saved to: {output_dir}")
        print(f"\nKey files:")
        print(f"  - degradation_results.csv    (detailed per-config results)")
        print(f"  - DEGRADATION_RESULTS.md     (comprehensive report)")
        print(f"  - summary_statistics.csv     (key metrics)")
        print(f"  - configuration_analysis.csv (configuration rankings)")
    else:
        print(f"\n✗ Evaluation failed - check errors above")

if __name__ == "__main__":
    main()
