import torch
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils

from transformers import Sam3Model, Sam3Processor
from constants import *

def load_model(device: str):
    model = Sam3Model.from_pretrained(MODEL_DIR).to(device, dtype=torch.float16)
    processor = Sam3Processor.from_pretrained(MODEL_DIR)
    model.eval()
    return model, processor


def run_sam3(model,processor,img,device,text_prompt ='visual', threshold=0.5, mask_threshold=0.5):
    inputs = processor(images=Image.fromarray(img), text=text_prompt, return_tensors="pt").to(device, dtype=torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    return results

def process_results(results):
    """Converts numpy/torch results into JSON-serializable formats with RLE masks."""
    masks = results['masks'].cpu().numpy()
    scores = results['scores'].cpu().numpy()
    boxes = results['boxes'].cpu().numpy()
    
    instance_preds = []
    for i in range(len(masks)):
        mask_binary = (masks[i] > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mask_binary))
        rle['counts'] = rle['counts'].decode('utf-8')
        
        instance_preds.append({
            "segmentation": rle,
            "score": float(scores[i]),      
            "bbox": boxes[i].tolist()       
        })
    return instance_preds