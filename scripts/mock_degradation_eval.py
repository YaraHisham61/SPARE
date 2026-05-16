# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-12
# Prompt : Create mock evaluation generator with realistic degradation results

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

def generate_mock_results(output_dir=None, num_samples=5, seed=42):
    """Generate realistic mock degradation results for demonstration."""
    
    np.random.seed(seed)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f'experiments/degradation_results_{timestamp}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING MOCK DEGRADATION RESULTS")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}\n")
    
    # Sample names
    samples = [f'synthetic_{i:03d}' for i in range(num_samples)]
    
    # Degradation configurations
    degradation_configs = []
    
    geo_configs = [(0, 0), (10, 5), (20, 10), (30, 15)]
    ling_configs = [0.0, 0.1, 0.2, 0.3]
    vis_configs = [0, 10, 25, 50]
    
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
    
    # Generate results
    results = []
    
    for sample_name in samples:
        # Baseline mIoU: randomly between 0.75 and 0.95
        baseline_miou = np.random.uniform(0.75, 0.95)
        
        for config in degradation_configs:
            # Degradation severity
            deg_severity = (config['geo_exp'] + config['geo_shift']) / 30 + \
                          config['ling_err'] + \
                          config['vis_sigma'] / 50
            
            # mIoU drop increases with degradation severity
            # Add some noise for realism
            base_drop = deg_severity * np.random.uniform(0.15, 0.35)
            noise = np.random.normal(0, 0.02)
            miou_drop = max(0, base_drop + noise)
            
            degraded_miou = baseline_miou - miou_drop
            degraded_miou = max(0, min(1, degraded_miou))  # Clamp [0, 1]
            
            # Sensitivity
            degradation_factor = max(deg_severity * 100, 1)
            sensitivity = miou_drop / degradation_factor
            
            result_entry = {
                'sample': sample_name,
                'config': config['config_name'],
                'baseline_miou': round(baseline_miou, 4),
                'degraded_miou': round(degraded_miou, 4),
                'miou_drop': round(miou_drop, 4),
                'sensitivity': round(sensitivity, 6),
                'geo_exp': config['geo_exp'],
                'geo_shift': config['geo_shift'],
                'ling_err': config['ling_err'],
                'vis_sigma': config['vis_sigma'],
                'n_pred_masks': np.random.randint(2, 8),
                'n_gt_masks': 1
            }
            results.append(result_entry)
    
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'degradation_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"✓ Detailed results saved to: degradation_results.csv")
    print(f"  ({len(results_df)} total configurations)")
    
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
    report_lines.append("## OVERALL PERFORMANCE")
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
    report_lines.append(f"**Performance retention:** {(1 - avg_drop / baseline_mean) * 100:.2f}% on average")
    report_lines.append("")
    
    # Geometric degradation analysis
    report_lines.append("## GEOMETRIC DEGRADATION IMPACT")
    report_lines.append("")
    report_lines.append("Geometric degradation applies bounding box perturbations (expansion & center shift).")
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
        if exp == 0:
            report_lines.append(f"- Expansion {exp}% (baseline): {drop:.4f} drop")
        else:
            report_lines.append(f"- Expansion {exp}%: {drop:.4f} drop")
    report_lines.append("")
    
    # Linguistic degradation analysis
    report_lines.append("## LINGUISTIC DEGRADATION IMPACT")
    report_lines.append("")
    report_lines.append("Linguistic degradation applies keyboard-adjacent typos to text prompts.")
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
            report_lines.append(f"- Error rate {err:.1%} (baseline): {drop:.4f} drop")
        else:
            report_lines.append(f"- Error rate {err:.1%}: {drop:.4f} drop")
    report_lines.append("")
    
    # Visual degradation analysis
    report_lines.append("## VISUAL DEGRADATION IMPACT")
    report_lines.append("")
    report_lines.append("Visual degradation applies Gaussian noise (AWGN) to images at various σ levels.")
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
            report_lines.append(f"- Noise σ={sigma} (baseline): {drop:.4f} drop")
        else:
            report_lines.append(f"- Noise σ={sigma}: {drop:.4f} drop")
    report_lines.append("")
    
    # Cross-modal sensitivity analysis
    report_lines.append("## CROSS-MODAL SENSITIVITY ANALYSIS")
    report_lines.append("")
    report_lines.append("Analysis of combined degradation effects across modalities.")
    report_lines.append("")
    
    # Most robust configuration
    robust_config = results_df.loc[results_df['miou_drop'].idxmin()]
    report_lines.append(f"**Most robust configuration:**")
    report_lines.append(f"- mIoU drop: {robust_config['miou_drop']:.4f}")
    report_lines.append(f"- Config: {robust_config['config']}")
    report_lines.append(f"- Degradation: geo_exp={robust_config['geo_exp']}, ling_err={robust_config['ling_err']:.1f}, vis_sigma={robust_config['vis_sigma']}")
    report_lines.append("")
    
    # Least robust configuration
    least_robust = results_df.loc[results_df['miou_drop'].idxmax()]
    report_lines.append(f"**Least robust configuration:**")
    report_lines.append(f"- mIoU drop: {least_robust['miou_drop']:.4f}")
    report_lines.append(f"- Config: {least_robust['config']}")
    report_lines.append(f"- Degradation: geo_exp={least_robust['geo_exp']}, ling_err={least_robust['ling_err']:.1f}, vis_sigma={least_robust['vis_sigma']}")
    report_lines.append("")
    
    # Sensitivity ranking
    report_lines.append("## SENSITIVITY RANKING")
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
        print(f"{i}. **{dtype}**: {sens:.6f}")
        report_lines.append(f"{i}. **{dtype}**: {sens:.6f}")
    report_lines.append("")
    
    # Top 5 most sensitive configurations
    report_lines.append("## TOP 5 MOST SENSITIVE CONFIGURATIONS")
    report_lines.append("")
    top_5 = results_df.nlargest(5, 'sensitivity')[['config', 'sensitivity', 'miou_drop']]
    report_lines.append(top_5.to_string())
    report_lines.append("")
    
    # Recommendations
    report_lines.append("## RECOMMENDATIONS FOR IMPROVING ROBUSTNESS")
    report_lines.append("")
    report_lines.append("Based on the sensitivity analysis:")
    report_lines.append("")
    
    most_sensitive = sorted_sens[0][0]
    
    if most_sensitive == 'Linguistic':
        report_lines.append("### Priority 1: Linguistic Robustness")
        report_lines.append("- **Issue:** Model is highly sensitive to prompt perturbations")
        report_lines.append("- **Solutions:**")
        report_lines.append("  - Implement prompt normalization (lowercase, remove extra spaces)")
        report_lines.append("  - Add spell-checking preprocessing")
        report_lines.append("  - Train on augmented prompts with character-level noise")
        report_lines.append("  - Consider prompt embeddings that are robust to typos")
        report_lines.append("")
    
    if most_sensitive == 'Geometric':
        report_lines.append("### Priority 1: Geometric Robustness")
        report_lines.append("- **Issue:** Model is highly sensitive to bounding box perturbations")
        report_lines.append("- **Solutions:**")
        report_lines.append("  - Use IoU-based loss functions during training")
        report_lines.append("  - Implement data augmentation with bbox noise")
        report_lines.append("  - Train with adversarial box perturbations")
        report_lines.append("  - Consider soft coordinate regression for bbox offsets")
        report_lines.append("")
    
    if most_sensitive == 'Visual':
        report_lines.append("### Priority 1: Visual Robustness")
        report_lines.append("- **Issue:** Model is highly sensitive to image noise")
        report_lines.append("- **Solutions:**")
        report_lines.append("  - Train with heavy image augmentations (AWGN, blur, compression)")
        report_lines.append("  - Implement denoising preprocessing (e.g., bilateral filter)")
        report_lines.append("  - Use domain randomization for visual degradation")
        report_lines.append("  - Consider noise-adaptive normalization layers")
        report_lines.append("")
    
    report_lines.append("### Cross-Modal Strategy")
    report_lines.append("- Combine robustness improvements across all three modalities")
    report_lines.append("- Validate on combinations of degradations, not just individual types")
    report_lines.append("- Consider ensemble methods that vote across modalities")
    report_lines.append("")
    
    report_lines.append("### Testing and Validation")
    report_lines.append("- Test with real-world degradation patterns (not just synthetic)")
    report_lines.append("- Evaluate on in-the-wild data with natural variations")
    report_lines.append("- Monitor degradation performance in production deployments")
    report_lines.append("")
    
    # Save report
    report_file = os.path.join(output_dir, 'DEGRADATION_RESULTS.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Comprehensive report saved to: DEGRADATION_RESULTS.md")
    
    # Generate summary statistics CSV
    summary_stats = pd.DataFrame({
        'Metric': ['Baseline mIoU (mean)', 'Worst-case mIoU', 'Average mIoU drop', 'Maximum mIoU drop',
                   'Geometric sensitivity', 'Linguistic sensitivity', 'Visual sensitivity',
                   'Most robust config', 'Least robust config'],
        'Value': [
            f"{baseline_mean:.4f}" if len(baseline_rows) > 0 else "N/A",
            f"{worst_mean:.4f}",
            f"{avg_drop:.4f}",
            f"{max_drop:.4f}",
            f"{geo_sens:.6f}",
            f"{ling_sens:.6f}",
            f"{vis_sens:.6f}",
            robust_config['config'],
            least_robust['config']
        ]
    })
    
    summary_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_file, index=False)
    print(f"✓ Summary statistics saved to: summary_statistics.csv")
    
    # Save top configurations to CSV
    top_configs = results_df.groupby('config').agg({
        'degraded_miou': ['mean', 'std', 'min', 'max'],
        'miou_drop': 'mean',
        'sensitivity': 'mean'
    }).round(4).sort_values(('miou_drop', 'mean'), ascending=False)
    
    config_file = os.path.join(output_dir, 'configuration_analysis.csv')
    top_configs.to_csv(config_file)
    print(f"✓ Configuration analysis saved to: configuration_analysis.csv")

def main():
    """Main entry point."""
    results_df, output_dir = generate_mock_results(num_samples=5)
    
    print(f"\n{'='*70}")
    print(f"✓ MOCK EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey files created:")
    print(f"  [CSV] degradation_results.csv    - detailed per-config results")
    print(f"  [MD]  DEGRADATION_RESULTS.md     - comprehensive analysis report")
    print(f"  [CSV] summary_statistics.csv     - key metrics summary")
    print(f"  [CSV] configuration_analysis.csv - configuration rankings")
    print(f"\n💡 To use real SAM3 evaluation, set HF_TOKEN and run:")
    print(f"   uv run python scripts/comprehensive_eval.py")

if __name__ == "__main__":
    main()
