import json
import numpy as np
from pathlib import Path
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

from scripts.text_degradation_helper import inject_typos
from scripts.model_helper import load_model, run_sam3,process_results
from constants import *

print("📦 Loading SACo-Gold dataset...")
ds = load_dataset('facebook/SACo-Gold', 'SA-1B')
ds = ds['test']

OUTPUT_DIR = Path("experiments/2026-05-9_text-degradation-task")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SA1B_DIR_PATH = Path(SA1B_DIR)

print("🤖 Loading SAM3 Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model(device)

noise_levels = [0, 1, 2, 3]
file_paths = {level: OUTPUT_DIR / f"text_degradation_L{level}.jsonl" for level in noise_levels}

image_list = ds[0].get('images', [])

for img_info in tqdm(image_list, desc="Images"):
    original_prompt = img_info.get('text_input', '')
    img_path = SA1B_DIR_PATH / img_info['file_name']
    
    if not img_path.exists():
        continue
        
    try:
            img_arr = np.array(Image.open(img_path).convert('RGB'))
    except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    # Process each noise level for this image
    for level in noise_levels:
        noisy_prompt = inject_typos(original_prompt, level=level)
        results = run_sam3(model, processor, img_arr, device, noisy_prompt)
        
        entry = {
            "image_id": img_info['id'],
            "image_name" : img_info['file_name'],
            "original_prompt": original_prompt,
            "noisy_prompt": noisy_prompt,
            "noise_level": level,
            "predictions": process_results(results)
        }

        with open(file_paths[level], "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    torch.cuda.empty_cache()
    