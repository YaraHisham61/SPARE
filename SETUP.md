# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create README for SAM3 authentication and robustness evaluation

# SAM3 Robustness Evaluation Setup

## Quick Start

### Option 1: Using Local SAM3 (Recommended)

1. **Get HuggingFace Token**
   - Go to: https://huggingface.co/settings/tokens
   - Create a "read" token
   - Copy the token (starts with `hf_`)

2. **Set Token in PowerShell**
   ```powershell
   $env:HF_TOKEN = 'hf_your_token_here'
   ```

3. **Run Evaluation**
   ```powershell
   uv run python scripts/evaluate_robustness.py
   ```
   
   This will:
   - Download SAM3 model (happens automatically with token)
   - Cache it locally in `models/` (~2.5GB)
   - Run robustness tests on 10 synthetic samples
   - Save results to `experiments/robustness_eval_*/`

### Option 2: Using Replicate API (Cloud)

If you don't want to download the model locally, use Replicate:

1. **Create Replicate Account**
   - Go to: https://replicate.com
   - Sign up and create an API token

2. **Set Token**
   ```powershell
   $env:REPLICATE_API_TOKEN = 'your_replicate_token'
   ```

3. **Run**
   ```powershell
   uv run python scripts/evaluate_robustness_replicate.py
   ```

## Troubleshooting

**Error: "401 Client Error" or "gated repo"**
- You need a HuggingFace token
- Token must have been created AFTER you accepted the SAM3 license
- Try creating a new token

**Error: "CUDA out of memory"**
- Switch to CPU: Set `device = "cpu"` in the script
- Or use smaller batch sizes

**Model download stuck**
- Check internet connection
- Try: `$env:HF_ENDPOINT = "https://hf-mirror.com"`

## Results

After evaluation completes, find results in:
- `experiments/robustness_eval_YYYY-MM-DD_HH-MM-SS/robustness_results.csv`
- `experiments/robustness_eval_YYYY-MM-DD_HH-MM-SS/summary_statistics.csv`
- `experiments/robustness_eval_YYYY-MM-DD_HH-MM-SS/analysis_report.md`

Results include sensitivity analysis for:
- Geometric degradation (bbox expansion/shift)
- Linguistic degradation (text corruption)
- Visual degradation (image noise)
