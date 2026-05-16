# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create a simple test script to verify SAM3 robustness components

import os
import sys
import numpy as np

# Ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.degradations import apply_linguistic_noise, apply_geometric_noise, apply_visual_noise
from scripts.model_helper import load_model, run_sam3
from scripts.data_visualization import load_sample
from scripts.evaluation_metrics import compute_miou

def test_components():
    """Test individual components of the robustness evaluation."""

    print("Testing linguistic degradation...")
    text = "cat"
    degraded = apply_linguistic_noise(text, 0.5)
    print(f"Original: {text} -> Degraded: {degraded}")

    print("\nTesting geometric degradation...")
    bbox = [100, 100, 50, 50]
    degraded_bbox = apply_geometric_noise(bbox, 20, 10)
    print(f"Original bbox: {bbox} -> Degraded: {degraded_bbox}")

    print("\nTesting visual degradation...")
    # Create a simple test image
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    degraded_img = apply_visual_noise(test_img, 25)
    print(f"Image shape: {test_img.shape} -> Degraded shape: {degraded_img.shape}")

    print("\nTesting data loading...")
    try:
        img, meta = load_sample('synthetic_000')
        print(f"Loaded image shape: {img.shape}")
        print(f"Metadata keys: {list(meta.keys())}")
    except Exception as e:
        print(f"Error loading data: {e}")

    print("\nTesting model loading...")
    try:
        device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
        model, processor = load_model(device)
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")

    print("\nAll component tests completed!")

if __name__ == "__main__":
    test_components()