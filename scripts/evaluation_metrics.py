import numpy as np

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union else 0.0


def compute_miou(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    """Mean of best-matching IoU: for each prediction find the closest GT mask."""
    if not pred_masks or not gt_masks:
        return 0.0
    return float(np.mean([max(iou(p, g) for g in gt_masks) for p in pred_masks]))