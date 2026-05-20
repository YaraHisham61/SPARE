# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create authentication and model setup script for SAM3

import os
import sys
from pathlib import Path

def setup_hf_token():
    """
    Set up HuggingFace token from environment or cache.
    """
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN found in environment")
        return hf_token
    
    # Check if already cached
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    if cache_path.exists():
        with open(cache_path) as f:
            token = f.read().strip()
            if token:
                print(f"✓ HF token found in cache")
                return token
    
    return None

def prompt_for_token():
    """
    Prompt user for HF token.
    """
    print("\n" + "="*70)
    print("SAM3 MODEL AUTHENTICATION REQUIRED")
    print("="*70)
    print("\nThe SAM3 model is gated on Hugging Face. You need to:")
    print("\n1. Go to: https://huggingface.co/facebook/sam3")
    print("2. Accept the license agreement")
    print("3. Create an access token at: https://huggingface.co/settings/tokens")
    print("4. Paste the token below (it starts with 'hf_')")
    print("\nAlternatively, set HF_TOKEN environment variable")
    print("="*70 + "\n")
    
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    
    if token:
        # Save to cache for future use
        cache_dir = Path.home() / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_dir / "token", "w") as f:
            f.write(token)
        print("\n✓ Token saved to ~/.cache/huggingface/token")
        return token
    
    return None

def main():
    token = setup_hf_token()
    
    if not token:
        token = prompt_for_token()
    
    if token:
        # Set environment variable for transformers
        os.environ["HF_TOKEN"] = token
        
        # Also login with huggingface_hub if available
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("✓ Authenticated with Hugging Face")
            return 0
        except ImportError:
            print("✓ HF_TOKEN set (huggingface_hub not available for login)")
            return 0
    else:
        print("\n✗ No token provided. Cannot access SAM3 model.")
        print("Run this script again to authenticate.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
