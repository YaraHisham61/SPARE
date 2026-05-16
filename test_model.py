#!/usr/bin/env python
# AI-GENERATED - Test SAM3 model loading

from scripts.model_helper import load_model

print("Testing SAM3 model loading...")
try:
    model, processor = load_model("cpu", use_cache=True)
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Processor type: {type(processor)}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
