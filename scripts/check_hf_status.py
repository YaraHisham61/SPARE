# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Check HuggingFace authentication status and attempt model download

import os
from pathlib import Path

# Check environment
hf_token = os.getenv("HF_TOKEN")
print(f"HF_TOKEN set: {bool(hf_token)}")

# Check cache
cache_path = Path.home() / ".cache" / "huggingface" / "token"
print(f"HF cache token exists: {cache_path.exists()}")

# Try to import and check authentication
try:
    from huggingface_hub import get_huggingface_hub_cache
    cache_dir = get_huggingface_hub_cache()
    print(f"HF cache directory: {cache_dir}")
except Exception as e:
    print(f"Error checking HF cache: {e}")

# List cached models
cache_models = Path.home() / ".cache" / "huggingface" / "hub"
if cache_models.exists():
    models = list(cache_models.glob("models--facebook--sam3*"))
    print(f"Cached SAM3 models: {len(models)}")
    for m in models:
        print(f"  - {m.name}")
else:
    print("No HF models cached yet")
