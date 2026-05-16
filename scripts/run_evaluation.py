# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create master script to setup, download SAM3, and run robustness evaluation

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"→ {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0

def main():
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)
    
    print("\n" + "="*70)
    print("SAM3 ROBUSTNESS EVALUATION - SETUP & EXECUTION")
    print("="*70)
    
    # Step 1: Check for HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_cache = Path.home() / ".cache" / "huggingface" / "token"
        if hf_cache.exists():
            with open(hf_cache) as f:
                hf_token = f.read().strip()
    
    if hf_token:
        print(f"\n✓ HuggingFace token configured")
    else:
        print(f"\n⚠ No HuggingFace token found. You need to:")
        print(f"  1. Visit: https://huggingface.co/facebook/sam3")
        print(f"  2. Accept the license")
        print(f"  3. Create a token: https://huggingface.co/settings/tokens")
        print(f"  4. Set it: $env:HF_TOKEN = 'hf_...'")
        print(f"\nAttempting to run anyway (may fail if model requires auth)...")
    
    # Step 2: Verify dependencies
    print(f"\n{'='*70}")
    print(f"→ Verifying dependencies")
    print(f"{'='*70}")
    try:
        import torch
        import transformers
        import pandas
        import tqdm
        print(f"✓ All dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print(f"Run: uv sync")
        return 1
    
    # Step 3: Ensure synthetic data exists
    data_dir = root_dir / "data" / "SA-1B-Part-000999"
    if not (data_dir / "synthetic_000.jpg").exists():
        print(f"\n{'='*70}")
        print(f"→ Generating synthetic test data")
        print(f"{'='*70}")
        if not run_command(f"uv run python scripts/generate_synthetic_data.py", 
                          "Generate synthetic data"):
            print(f"✗ Failed to generate synthetic data")
            return 1
    else:
        print(f"\n✓ Synthetic data already exists ({len(list(data_dir.glob('*.jpg')))} samples)")
    
    # Step 4: Run evaluation
    print(f"\n{'='*70}")
    print(f"→ Running robustness evaluation")
    print(f"{'='*70}")
    if not run_command(f"uv run python scripts/evaluate_robustness.py", 
                      "Execute robustness evaluation"):
        print(f"\n✗ Evaluation failed")
        print(f"\nTroubleshooting:")
        print(f"  - Check HuggingFace authentication (set HF_TOKEN)")
        print(f"  - Verify you have access to facebook/sam3")
        print(f"  - Check internet connection for model download")
        return 1
    
    print(f"\n{'='*70}")
    print(f"✓ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to:")
    print(f"  - experiments/robustness_eval_*/robustness_results.csv")
    print(f"  - experiments/robustness_eval_*/summary_statistics.csv")
    print(f"  - experiments/robustness_eval_*/analysis_report.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
