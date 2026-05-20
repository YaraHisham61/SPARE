# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-07
# Prompt : Create synthetic data generator for testing SAM3 robustness evaluation

import os
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_sample(sample_id, output_dir):
    """
    Create a synthetic image with objects for testing.
    """
    # Create blank image
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White background

    # Define some objects
    objects = [
        {
            'name': 'cat',
            'bbox': [100, 100, 150, 120],
            'color': (255, 100, 100)
        },
        {
            'name': 'dog',
            'bbox': [300, 200, 120, 100],
            'color': (100, 255, 100)
        },
        {
            'name': 'bird',
            'bbox': [200, 350, 80, 60],
            'color': (100, 100, 255)
        }
    ]

    # Draw objects as colored rectangles
    for obj in objects:
        x, y, w, h = obj['bbox']
        cv2.rectangle(img, (x, y), (x+w, y+h), obj['color'], -1)  # Filled rectangle

        # Add text label
        cv2.putText(img, obj['name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Convert to RGB for saving
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save image
    image_path = os.path.join(output_dir, f'{sample_id}.jpg')
    Image.fromarray(img_rgb).save(image_path)

    # Create metadata JSON
    metadata = {
        'image_id': sample_id,
        'image_size': [512, 512],
        'annotations': []
    }

    for i, obj in enumerate(objects):
        metadata['annotations'].append({
            'id': i,
            'category': obj['name'],
            'bbox': obj['bbox'],  # [x, y, w, h]
            'area': obj['bbox'][2] * obj['bbox'][3],
            'segmentation': []  # Placeholder - would need actual mask data
        })

    # Save metadata
    json_path = os.path.join(output_dir, f'{sample_id}.json')
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return img_rgb, metadata

def generate_synthetic_dataset(n_samples=10, output_dir='data/SA-1B-Part-000999'):
    """
    Generate a small synthetic dataset for testing.
    """
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {os.path.abspath(output_dir)}")

    samples = []

    for i in range(n_samples):
        sample_id = f'synthetic_{i:03d}'
        print(f"Generating sample {sample_id}")
        img, meta = create_synthetic_sample(sample_id, output_dir)
        samples.append(sample_id)

    # Create sample list
    sample_list_path = os.path.join(output_dir, 'samples.txt')
    with open(sample_list_path, 'w') as f:
        f.write('\n'.join(samples))
    print(f"Sample list saved to: {sample_list_path}")

    print(f"Generated {n_samples} synthetic samples in {output_dir}")
    return samples

if __name__ == "__main__":
    generate_synthetic_dataset()