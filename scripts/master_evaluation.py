# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create interactive master script for complete SAM3 robustness evaluation workflow

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def check_hf_auth():
    """Check if HF authentication is available."""
    # Check environment
    if os.getenv("HF_TOKEN"):
        return True, "environment"
    
    # Check cache
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        return True, "cache"
    
    return False, None

def run_evaluation_local():
    """Run evaluation with local SAM3."""
    print_header("Running Local SAM3 Evaluation")
    
    has_auth, source = check_hf_auth()
    if not has_auth:
        print("❌ No HuggingFace authentication found")
        print("\nTo authenticate:")
        print("1. Go to: https://huggingface.co/facebook/sam3")
        print("2. Accept the license")
        print("3. Create token: https://huggingface.co/settings/tokens")
        print("4. In PowerShell set: $env:HF_TOKEN = 'hf_...'")
        print("5. Re-run this script")
        return False
    
    print(f"✓ HuggingFace auth found ({source})")
    print("Starting evaluation (this will download ~2.5GB model on first run)...\n")
    
    root_dir = Path(__file__).parent.parent
    result = subprocess.run(
        "uv run python scripts/evaluate_robustness.py",
        shell=True,
        cwd=root_dir
    )
    
    return result.returncode == 0

def run_evaluation_replicate():
    """Run evaluation with Replicate cloud API."""
    print_header("Running Replicate Cloud Evaluation (SAM2)")
    
    api_token = os.getenv("REPLICATE_API_TOKEN")
    if not api_token:
        print("❌ No Replicate API token found")
        print("\nTo setup Replicate:")
        print("1. Go to: https://replicate.com/account/api-tokens")
        print("2. Create API token")
        print("3. In PowerShell set: $env:REPLICATE_API_TOKEN = 'r8_...'")
        print("4. Re-run this script")
        return False
    
    print(f"✓ Replicate API token found")
    print("Starting cloud evaluation...\n")
    
    root_dir = Path(__file__).parent.parent
    result = subprocess.run(
        "uv run python scripts/evaluate_robustness_replicate.py",
        shell=True,
        cwd=root_dir
    )
    
    return result.returncode == 0

def generate_findings_template():
    """Copy findings template to results."""
    print_header("Generating Findings Document")
    
    template_path = Path(__file__).parent.parent / "docs" / "claude" / "2026-05-07_sam3_robustness_findings_template.md"
    
    if template_path.exists():
        # Find latest experiment dir
        exp_dirs = list((Path(__file__).parent.parent / "experiments").glob("robustness_eval_*"))
        if exp_dirs:
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            findings_path = latest_exp / "FINDINGS.md"
            
            with open(template_path, encoding='utf-8') as f:
                template = f.read()
            
            with open(findings_path, 'w', encoding='utf-8') as f:
                f.write(template)
            
            print(f"✓ Template generated at: {findings_path}")
            print("\nNext steps:")
            print("1. Open the findings document")
            print("2. Replace [AWAITING RUN] with values from robustness_results.csv")
            print("3. Add your interpretation")
            return True
    
    return False

def main():
    print_header("SAM3 ROBUSTNESS EVALUATION - MASTER SCRIPT")
    
    print("Choose evaluation method:\n")
    print("1. Local SAM3 (recommended, requires HF token)")
    print("   - High quality (actual SAM3 model)")
    print("   - Slower (model download + local GPU)")
    print("   - More accurate results")
    print()
    print("2. Replicate Cloud (SAM2, no local storage)")
    print("   - Medium quality (SAM2 instead of SAM3)")
    print("   - Faster (API-based)")
    print("   - Requires Replicate token")
    print()
    print("3. Show execution guide (manual setup)")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    root_dir = Path(__file__).parent.parent
    os.chdir(root_dir)
    
    # Ensure synthetic data exists
    data_dir = root_dir / "data" / "SA-1B-Part-000999"
    if not (data_dir / "synthetic_000.jpg").exists():
        print("\nGenerating synthetic data...")
        subprocess.run("uv run python scripts/generate_synthetic_data.py", shell=True, cwd=root_dir)
    
    # Run chosen option
    if choice == "1":
        success = run_evaluation_local()
    elif choice == "2":
        success = run_evaluation_replicate()
    elif choice == "3":
        with open(root_dir / "EXECUTION_GUIDE.md") as f:
            print(f.read())
        return 0
    else:
        print("Invalid choice")
        return 1
    
    if success:
        generate_findings_template()
        
        print_header("✓ EVALUATION COMPLETE")
        print("Check experiments/ folder for results:")
        print("  - robustness_results.csv")
        print("  - summary_statistics.csv")
        print("  - FINDINGS.md (template)")
        return 0
    else:
        print_header("❌ EVALUATION FAILED")
        print("Check error messages above for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
