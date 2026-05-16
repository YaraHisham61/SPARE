import os
import torch
from pathlib import Path
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from constants import *

def load_model(device: str, use_cache=True):
    """
    Load SAM3 model from local cache or Hugging Face.
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
        use_cache: Whether to use cached model
    
    Returns:
        Tuple of (model, processor)
    
    Raises:
        RuntimeError: If model cannot be loaded
    """
    # Try to authenticate if token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print(f"✓ Authenticated with HuggingFace using token")
        except Exception as e:
            print(f"⚠ Could not authenticate: {e}")
    
    root_dir = Path(__file__).parent.parent
    local_model_dir = root_dir / MODEL_DIR
    
    # Try local model first if it exists
    if local_model_dir.exists():
        print(f"Loading model from local cache: {local_model_dir}")
        try:
            model = Sam3Model.from_pretrained(str(local_model_dir)).to(device)
            processor = Sam3Processor.from_pretrained(str(local_model_dir))
            model.eval()
            return model, processor
        except Exception as e:
            print(f"⚠ Failed to load local model: {e}")
    
    # Try HuggingFace model
    print(f"Loading model from Hugging Face: {MODEL_ID}")
    try:
        # First attempt: might have cached or token available
        model = Sam3Model.from_pretrained(
            MODEL_ID,
            cache_dir=str(root_dir / "models"),
            trust_remote_code=True
        ).to(device)
        processor = Sam3Processor.from_pretrained(
            MODEL_ID,
            cache_dir=str(root_dir / "models"),
            trust_remote_code=True
        )
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
            raise RuntimeError(
                f"\n{'='*70}\n"
                f"SAM3 MODEL ACCESS REQUIRED\n"
                f"{'='*70}\n"
                f"The SAM3 model is gated on Hugging Face.\n\n"
                f"Steps to authenticate:\n"
                f"  1. Go to: https://huggingface.co/facebook/sam3\n"
                f"  2. Accept the license agreement\n"
                f"  3. Create a token: https://huggingface.co/settings/tokens (select 'repo')\n"
                f"  4. Set token in PowerShell:\n"
                f"     [System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_your_token_here', 'User')\n"
                f"  5. Restart PowerShell\n"
                f"  6. Run this script again\n"
                f"\nAlternatively, cache locally:\n"
                f"  python scripts/download_model.py\n"
                f"{'='*70}\n"
            ) from e
        else:
            raise RuntimeError(
                f"Failed to load SAM3 model: {e}\n"
                f"Make sure you have internet and valid HF credentials."
            ) from e
    
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

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    return results