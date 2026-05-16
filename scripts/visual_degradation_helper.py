import numpy as np


def add_awgn(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def random_flip(image: np.ndarray, mode: str = "random") -> np.ndarray:
    """
    Flip an image horizontally, vertically, or both.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        mode: "horizontal", "vertical", "both", or "random"
    
    Returns:
        Flipped image as numpy array
    """
    if mode == "none":
        return image.copy()
    
    if mode == "random":
        mode = np.random.choice(["h", "v", "both"])
    
    if mode == "h":
        return image[:, ::-1].copy()
    elif mode == "v":
        return image[::-1, :].copy()
    elif mode == "both":
        return image[::-1, ::-1].copy()
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose: horizontal, vertical, both, random")