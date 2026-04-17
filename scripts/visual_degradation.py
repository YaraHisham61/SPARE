import numpy as np


def add_awgn(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
