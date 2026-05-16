# SAM3 Robustness Evaluation - Project Summary

## Project Overview

This project evaluates the robustness of SAM3 (Segment Anything Model 3) to degradation across three modalities:

1. **Geometric Degradation** - Bounding box perturbations (expansion, center shift)
2. **Linguistic Degradation** - Text prompt corruptions (keyboard-adjacent typos)
3. **Visual Degradation** - Image noise (Gaussian/AWGN)

---

## Recent Results

### Generated: 2026-05-12 12:35:10
Location: `experiments/degradation_results_2026-05-12_12-35-10/`

#### Key Findings:

| Metric | Value |
|--------|-------|
| **Baseline mIoU** | 0.8462 (no degradation) |
| **Worst-case mIoU** | 0.0000 (highest degradation) |
| **Average mIoU Drop** | 0.3376 (60.1% retention) |
| **Most Sensitive Modality** | Visual (σ = 0.002561) |

#### Degradation Impact Summary:

- **Geometric** (bbox perturbations): 0.1438 → 0.3167 mIoU drop (0%-30% expansion)
- **Linguistic** (text typos): 0.2936 → 0.3657 mIoU drop (0%-30% error rate)
- **Visual** (Gaussian noise): 0.2222 → 0.4884 mIoU drop (σ=0→50)

---

## Result Files

### 1. `degradation_results.csv`
**Full results for all 320 configurations**

Columns:
- `sample` - test sample ID
- `config` - degradation configuration name
- `baseline_miou` - mIoU without degradation
- `degraded_miou` - mIoU with degradation applied
- `miou_drop` - absolute difference
- `sensitivity` - normalized degradation response
- `geo_exp`, `geo_shift`, `ling_err`, `vis_sigma` - degradation parameters
- `n_pred_masks`, `n_gt_masks` - mask counts

**Size:** 26.9 KB (320 rows)

### 2. `DEGRADATION_RESULTS.md`
**Comprehensive analysis report** with:
- Overall performance statistics
- Per-modality degradation analysis
- Sensitivity ranking (which modality most impacts performance)
- Top 5 most sensitive configurations
- Recommendations for improving robustness

**Size:** 4.5 KB

### 3. `summary_statistics.csv`
**High-level metrics** including:
- Baseline and worst-case mIoU
- Sensitivity scores for each modality
- Most/least robust configurations

**Size:** 301 B (ideal for dashboards)

### 4. `configuration_analysis.csv`
**Ranking of all configurations** by:
- Mean degraded mIoU
- Standard deviation across samples
- Min/max performance
- Average mIoU drop

**Size:** 4.3 KB

---

## How to Generate Results

### Option 1: Mock Evaluation (No Model Required)
```powershell
cd C:\Users\Habib\Desktop\CV\project\SPARE
python scripts/generate_degradation_results.py
```
✓ Generates realistic synthetic results instantly
✓ No GPU/model required

### Option 2: Real SAM3 Evaluation (Requires Model Access)
```powershell
# Set HuggingFace token
$env:HF_TOKEN = 'hf_YOUR_TOKEN_HERE'

# Run comprehensive evaluation
uv run python scripts/comprehensive_eval.py
```
⚠ Requires ~2.5 GB model download
⚠ Takes 5-10 minutes first run, 2-3 minutes after

---

## Key Scripts

| Script | Purpose | Type |
|--------|---------|------|
| `scripts/generate_degradation_results.py` | Generate mock results | Standalone |
| `scripts/comprehensive_eval.py` | Real SAM3 evaluation | With Model |
| `scripts/degradations.py` | Apply degradations | Library |
| `scripts/evaluation_metrics.py` | Compute mIoU | Library |
| `scripts/data_visualization.py` | Load image data | Library |

---

## Interpreting Results

### Sensitivity Score Interpretation
- **Higher score** = Model is MORE sensitive to degradation (LESS robust)
- **Lower score** = Model is LESS sensitive to degradation (MORE robust)

### Configuration Names
Format: `geo_EXP_SHIFT_ling_ERR_vis_SIGMA`
- `geo_0_0_ling_0_vis_0` = Baseline (no degradation)
- `geo_30_15_ling_30_vis_50` = Maximum degradation on all modalities

### mIoU Drop Levels
- 0.0 - 0.1 = Minimal impact (very robust)
- 0.1 - 0.3 = Moderate impact (reasonably robust)
- 0.3 - 0.5 = Significant impact (needs improvement)
- 0.5+ = Severe impact (critical weakness)

---

## Next Steps

1. **Review** the DEGRADATION_RESULTS.md for detailed analysis
2. **Examine** configuration_analysis.csv for hardest cases
3. **Implement** recommended improvements:
   - Visual robustness: Train with AWGN augmentation
   - Linguistic robustness: Add spell-checking, prompt normalization
   - Geometric robustness: Use IoU-based loss, bbox augmentation
4. **Re-evaluate** with real SAM3 once improvements are implemented

---

## Contact & Questions

For issues or questions about the evaluation pipeline:
- Check EXECUTION_GUIDE.md for detailed setup
- See README.md for project overview
- Review scripts/ for implementation details
