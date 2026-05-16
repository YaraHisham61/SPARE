import numpy as np


def _clip_bbox(x1: float, y1: float, x2: float, y2: float,
               img_w: int, img_h: int) -> list[float]:
    """Clamp box coordinates to image bounds and ensure x2>x1, y2>y1."""
    x1 = max(0.0, min(x1, img_w - 1))
    y1 = max(0.0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return [x1, y1, x2 - x1, y2 - y1]


def expand_bbox(
    bbox: list[float],
    factor: float,
    img_w: int,
    img_h: int,
) -> list[float]:
    """Expand a COCO-format bbox [x, y, w, h] symmetrically around its center.

    Args:
        bbox:   [x, y, w, h] with (x, y) as the top-left corner.
        factor: Fractional expansion, e.g. 0.10 for 10 %, 0.20 for 20 %.
        img_w:  Image width  — used to clamp the result.
        img_h:  Image height — used to clamp the result.

    Returns:
        Expanded bbox as [x, y, w, h], clamped to image bounds.
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    new_w = w * (1 + factor)
    new_h = h * (1 + factor)

    x1 = cx - new_w / 2
    y1 = cy - new_h / 2
    x2 = cx + new_w / 2
    y2 = cy + new_h / 2

    return _clip_bbox(x1, y1, x2, y2, img_w, img_h)


def shift_bbox(
    bbox: list[float],
    factor: float,
    img_w: int,
    img_h: int,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Shift the center of a COCO-format bbox [x, y, w, h] by ±factor of its size.

    The shift magnitude along each axis is sampled uniformly from
    [-factor * dim, +factor * dim] where dim is the box width or height.

    Args:
        bbox:   [x, y, w, h] with (x, y) as the top-left corner.
        factor: Fractional shift, e.g. 0.10 for 10 %, 0.20 for 20 %.
        img_w:  Image width  — used to clamp the result.
        img_h:  Image height — used to clamp the result.
        rng:    Optional NumPy Generator for reproducibility.

    Returns:
        Shifted bbox as [x, y, w, h], clamped to image bounds.
    """
    if rng is None:
        rng = np.random.default_rng()

    x, y, w, h = bbox

    dx = rng.uniform(-factor * w, factor * w)
    dy = rng.uniform(-factor * h, factor * h)

    x1 = x + dx
    y1 = y + dy
    x2 = x1 + w
    y2 = y1 + h

    return _clip_bbox(x1, y1, x2, y2, img_w, img_h)


GEOM_LEVELS: dict[str, float] = {
    "L10": 0.10,
    "L20": 0.20,
}
