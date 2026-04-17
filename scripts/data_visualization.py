import json
import matplotlib
import numpy as np
import cv2

def load_sample(image_name):
    image_path = f"../data/SA-1B-Part-000999/{image_name}.jpg"
    json_path = f"../data/SA-1B-Part-000999/{image_name}.json"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found or could not be read: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with open(json_path) as f:
        meta = json.load(f)
    
    return image, meta

def overlay_boxes(image, boxes, color=(0, 255, 0), label="", max_boxes=30):
    """
    image: np.array (H, W, 3)
    boxes: list of [x, y, w, h]
    returns: np.array with boxes drawn
    """
    result = image.copy()
    
    for x, y, w, h in boxes[:max_boxes]:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness=2)
    
    return result

def overlay_masks(image, masks):
    """
    image: np.array (H, W, 3)
    masks: torch tensor (N, H, W)
    returns: np.array with masks overlaid
    """
    result = image.copy()
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        colored_mask = np.zeros_like(result)
        colored_mask[mask > 0] = color
        result = cv2.addWeighted(result, 1.0, colored_mask, 0.5, 0)

    return result