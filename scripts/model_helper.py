import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from constants import *

def load_model(device: str):
    model = Sam3Model.from_pretrained(MODEL_DIR).to(device)
    processor = Sam3Processor.from_pretrained(MODEL_DIR)
    model.eval()
    return model, processor


def run_sam3(model,processor,img,device,text_prompt = 'object',threshold=0.1,mask_threshold=0.5):
    inputs = processor(images=Image.fromarray(img), text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    return results