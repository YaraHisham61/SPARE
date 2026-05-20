# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Implement linguistic and geometric degradation functions for SAM3 robustness evaluation

import numpy as np
import random
import string

def apply_linguistic_noise(text_prompt: str, error_rate: float, error_type: str = 'substitute') -> str:
    """
    Apply linguistic degradation to text prompts.

    Args:
        text_prompt: Original text prompt
        error_rate: Probability of corrupting each word (0.0 to 1.0)
        error_type: Type of corruption ('substitute', 'random', 'delete')

    Returns:
        Corrupted text prompt
    """
    if not text_prompt or error_rate <= 0:
        return text_prompt

    words = text_prompt.split()
    corrupted_words = []

    # Keyboard adjacency mapping for realistic typos
    keyboard_adj = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'swerfvcx',
        'e': 'wsdfr', 'f': 'dertgbvc', 'g': 'ftyhbv', 'h': 'gyujnb',
        'i': 'ujklo', 'j': 'huikm', 'k': 'jiolm', 'l': 'kop',
        'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
        'q': 'wa', 'r': 'edfgt', 's': 'awedxz', 't': 'rfghy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
        'y': 'tghu', 'z': 'asx'
    }

    for word in words:
        if np.random.random() < error_rate and len(word) > 1:
            word_chars = list(word.lower())
            n_errors = min(3, max(1, len(word) // 3))  # 1-3 errors based on word length

            error_positions = np.random.choice(len(word_chars), n_errors, replace=False)

            for pos in error_positions:
                if error_type == 'substitute':
                    # Keyboard-adjacent substitution
                    if word_chars[pos] in keyboard_adj:
                        word_chars[pos] = np.random.choice(list(keyboard_adj[word_chars[pos]]))
                elif error_type == 'random':
                    # Random letter substitution
                    word_chars[pos] = np.random.choice(list(string.ascii_lowercase))
                elif error_type == 'delete':
                    # Character deletion
                    word_chars[pos] = ''

            corrupted_word = ''.join(word_chars)
            # Preserve original capitalization pattern
            if word.istitle():
                corrupted_word = corrupted_word.capitalize()
            elif word.isupper():
                corrupted_word = corrupted_word.upper()

            corrupted_words.append(corrupted_word)
        else:
            corrupted_words.append(word)

    return ' '.join(corrupted_words)


def apply_geometric_noise(gt_bbox: list, expansion_pct: float, shift_pct: float) -> list:
    """
    Apply geometric degradation to bounding boxes.

    Args:
        gt_bbox: Ground truth bbox [x, y, w, h]
        expansion_pct: Percentage to expand bbox area
        shift_pct: Percentage to shift bbox center

    Returns:
        Degraded bbox [x, y, w, h]
    """
    x, y, w, h = gt_bbox

    # Expand dimensions
    w_exp = w * (1 + expansion_pct / 100)
    h_exp = h * (1 + expansion_pct / 100)

    # Shift center
    shift_x = w * (shift_pct / 100) * np.random.choice([-1, 1])
    shift_y = h * (shift_pct / 100) * np.random.choice([-1, 1])

    new_center_x = x + w/2 + shift_x
    new_center_y = y + h/2 + shift_y

    return [new_center_x - w_exp/2, new_center_y - h_exp/2, w_exp, h_exp]


def apply_visual_noise(image: np.ndarray, sigma: float, noise_type: str = 'awgn') -> np.ndarray:
    """
    Apply visual degradation to images.

    Args:
        image: Input image array
        sigma: Noise intensity parameter
        noise_type: Type of noise ('awgn', 'poisson', 'salt_pepper')

    Returns:
        Noisy image array
    """
    if sigma <= 0:
        return image

    noisy = image.astype(np.float32)

    if noise_type == 'awgn':
        # Additive White Gaussian Noise
        noise = np.random.normal(0, sigma, image.shape)
        noisy += noise

    elif noise_type == 'poisson':
        # Poisson noise (for low-light simulation)
        noisy = np.random.poisson(noisy * sigma) / sigma

    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        prob = sigma / 100  # sigma interpreted as percentage
        salt_pepper = np.random.random(image.shape[:2])
        noisy[salt_pepper < prob/2] = 0  # pepper
        noisy[salt_pepper > 1 - prob/2] = 255  # salt

    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_combined_degradation(image: np.ndarray, text_prompt: str, bboxes: list,
                              geo_exp: float, geo_shift: float, ling_err: float,
                              vis_sigma: float) -> tuple:
    """
    Apply all types of degradation simultaneously.

    Returns:
        Tuple of (degraded_image, degraded_text, degraded_bboxes)
    """
    degraded_image = apply_visual_noise(image, vis_sigma)
    degraded_text = apply_linguistic_noise(text_prompt, ling_err)
    degraded_bboxes = [apply_geometric_noise(bbox, geo_exp, geo_shift) for bbox in bboxes]

    return degraded_image, degraded_text, degraded_bboxes