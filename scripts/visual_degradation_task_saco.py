import json
import numpy as np
from pathlib import Path
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from scripts.visual_degradation_helper import add_awgn
from scripts.model_helper import load_model, run_sam3,process_results
from constants import *

print("📦 Loading SACo-Gold dataset...")
ds = load_dataset('facebook/SACo-Gold', 'SA-1B')
ds = ds['test']

OUTPUT_DIR = Path("experiments/2026-05-16_visual-degradation-task-SACo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SA1B_DIR_PATH = Path(SA1B_DIR)

print("🤖 Loading SAM3 Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = load_model(device)

sigma_levels = [0, 10,25,50]
file_paths = {level: OUTPUT_DIR / f"visual_degradation_L{level}.jsonl" for level in sigma_levels}

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

    for level in sigma_levels:
        img_arr = add_awgn(img_arr,level)
        results = run_sam3(model, processor, img_arr, device, original_prompt)
        
        entry = {
            "image_id": img_info['id'],
            "image_name" : img_info['file_name'],
            "original_prompt": original_prompt,
            "noise_level": level,
            "predictions": process_results(results)
        }

        with open(file_paths[level], "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    torch.cuda.empty_cache()
    