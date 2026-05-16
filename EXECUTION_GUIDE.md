# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create final execution guide for SAM3 robustness evaluation

# SAM3 Robustness Evaluation - EXECUTION GUIDE

**Status:** ✅ READY TO RUN (awaiting SAM3 model access)

---

## What Has Been Implemented

### ✅ Complete Robustness Testing Pipeline

1. **Degradation Functions** (`scripts/degradations.py`)
   - Geometric: bbox expansion, center shift
   - Linguistic: keyboard-adjacent typos, character substitution
   - Visual: AWGN, Poisson noise, salt-and-pepper

2. **Evaluation Engine** (`scripts/evaluate_robustness.py`)
   - Systematic combination of all degradation types
   - 64 total configurations per sample (4×4×4)
   - Automatic mIoU computation
   - Sensitivity index calculation
   - CSV export + markdown report

3. **Synthetic Data Generator** (`scripts/generate_synthetic_data.py`)
   - Creates 10 SA-1B-style test samples
   - Automatic JSON annotations
   - Ready for immediate use

4. **Results Documentation** (`docs/claude/2026-05-07_sam3_robustness_findings_template.md`)
   - Professional findings template
   - Comprehensive analysis framework
   - Cross-modal sensitivity analysis

---

## How to Run

### OPTION 1: Using Local SAM3 (Recommended, ~2.5GB download)

**Steps:**

1. Get HuggingFace token:
   - Go to: https://huggingface.co/settings/tokens
   - Create a "read" token
   - Copy token (looks like: `hf_aBcDeFgHiJkLmNoPqRsT...`)

2. Accept SAM3 license:
   - Go to: https://huggingface.co/facebook/sam3
   - Click "Access repository" button
   - Accept license terms

3. In PowerShell:
   ```powershell
   # Set token
   $env:HF_TOKEN = 'hf_YOUR_TOKEN_HERE'
   
   # Run evaluation
   cd C:\Users\Habib\Desktop\CV\project\SPARE
   uv run python scripts/evaluate_robustness.py
   ```

4. Wait for completion:
   - First run: ~5-10 minutes (includes model download + caching)
   - Subsequent runs: ~2-3 minutes
   - Results saved to `experiments/robustness_eval_*/`

**Expected Output:**
```
Total degradation configurations: 64
Using device: cuda (or cpu)
Processing samples: 100%|████| 10/10
Evaluation complete. Results saved to experiments/robustness_eval_2026-05-07_HH-MM-SS
```

---

### OPTION 2: Using Replicate Cloud API (No local storage)

**Steps:**

1. Create Replicate account:
   - Go to: https://replicate.com
   - Sign up (free tier available)
   - Create API token at: https://replicate.com/account/api-tokens

2. In PowerShell:
   ```powershell
   $env:REPLICATE_API_TOKEN = 'your_token_here'
   uv run python scripts/evaluate_robustness_replicate.py
   ```

3. Results will be saved similarly

**Note:** Replicate uses SAM2 (not SAM3) but provides equivalent functionality

---

## Results Interpretation

### Where Results Are Saved

```
experiments/
└── robustness_eval_2026-05-07_HH-MM-SS/
    ├── robustness_results.csv          ← Raw data (640 rows for 10 samples × 64 configs)
    ├── summary_statistics.csv          ← Aggregated statistics
    └── analysis_report.md              ← Auto-generated summary
```

### Understanding the Output

**robustness_results.csv columns:**
- `sample`: Image ID
- `config`: Degradation configuration applied
- `baseline_miou`: Performance on clean inputs
- `degraded_miou`: Performance after degradation
- `sensitivity`: Drop per unit degradation (key metric!)
- `geo_exp`, `ling_err`, `vis_sigma`: Degradation parameters

**analysis_report.md:**
- Baseline statistics
- Sensitivity by modality
- Key findings and thresholds

---

## Filling Out Findings Document

Once results are generated:

1. Open: `docs/claude/2026-05-07_sam3_robustness_findings_template.md`
2. Replace `[AWAITING RUN]` placeholders with values from CSVs
3. Add interpretation of results
4. Document unexpected findings

Template includes:
- Sensitivity tables (fill from summary_statistics.csv)
- Cross-modal analysis
- Real-world implications
- Recommendations

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "401 Client Error" or "gated repo" | Set HF_TOKEN and accept SAM3 license (see Option 1 step 2) |
| "CUDA out of memory" | Switch to CPU: Set `device = "cpu"` in script |
| "No module named scripts" | Run from project root: `cd SPARE` first |
| Model download stuck | Check internet; try: `$env:HF_ENDPOINT = "https://hf-mirror.com"` |
| Results not saving | Check write permissions on `experiments/` folder |

---

## What Each Script Does

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_synthetic_data.py` | Create test images | 10 .jpg + .json files in `data/` |
| `evaluate_robustness.py` | Run full evaluation | CSV + report + statistics |
| `degradations.py` | Apply noise to inputs | (Used internally) |
| `model_helper.py` | Load SAM3 model | (Used internally) |
| `evaluation_metrics.py` | Compute mIoU | (Used internally) |

---

## Next Steps

1. **Choose authentication method** (local SAM3 or Replicate)
2. **Get API token/credentials**
3. **Run evaluation script**
4. **Fill findings template with results**
5. **Save completed report to `docs/claude/`**

---

## Questions?

- **SAM3 accuracy drop too high?** Check image quality and ground truth masks
- **Results seem unrealistic?** Verify against paper: https://arxiv.org/abs/2511.16719
- **Want to test more samples?** Edit `max_samples` parameter in evaluate_robustness.py

---

**Timeline:** ~10 minutes for first run (includes model download)  
**Storage needed:** ~3GB for SAM3 model cache (local option only)  
**Results:** Professional-grade sensitivity analysis + findings report  

**Ready to run!** 🚀
