import os
import torch
from pathlib import Path
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

def run_sam3(model, processor, img, device, text_prompt='object', bboxes=None,
             threshold=0.1, mask_threshold=0.5):
    """
    Run SAM3 inference with optional text and box prompts.

    Args:
        model: SAM3 model
        processor: SAM3 processor
        img: Input image array
        device: Device to run on
        text_prompt: Text prompt for segmentation
        bboxes: List of bounding boxes [x, y, w, h] format
        threshold: Instance segmentation threshold
        mask_threshold: Mask binarization threshold

    Returns:
        Dictionary with 'masks', 'boxes', 'scores'
    """
    # Convert bboxes to xyxy format if provided
    input_boxes = None
    if bboxes is not None and len(bboxes) > 0:
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        input_boxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            input_boxes.append([x, y, x + w, y + h])
        input_boxes = torch.tensor(input_boxes, dtype=torch.float32).unsqueeze(0).to(device)

    inputs = processor(
        images=Image.fromarray(img),
        text=text_prompt if text_prompt else None,
        boxes=input_boxes,
        return_tensors="pt"
    ).to(device)

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
