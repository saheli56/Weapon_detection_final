"""
Generate a small synthetic dataset for weapon detection fine-tuning.

This script creates 20 images with synthetic 'weapons' (drawn rectangles)
and corresponding YOLO-format labels. Saves to ml/dataset/.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_dataset(output_dir: Path, num_images: int = 20, img_size: tuple = (640, 640)):
    """Generate synthetic images with weapons and YOLO labels."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create blank image
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255  # White background

        # Draw 1-3 random 'weapons' (rectangles)
        num_weapons = random.randint(1, 3)
        labels = []
        for _ in range(num_weapons):
            # Random bbox (normalized)
            x_center = random.uniform(0.1, 0.9)
            y_center = random.uniform(0.1, 0.9)
            width = random.uniform(0.1, 0.3)
            height = random.uniform(0.1, 0.3)

            # Denormalize for drawing
            x1 = int((x_center - width / 2) * img_size[0])
            y1 = int((y_center - height / 2) * img_size[1])
            x2 = int((x_center + width / 2) * img_size[0])
            y2 = int((y_center + height / 2) * img_size[1])

            # Draw rectangle (weapon)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle

            # YOLO label: class x_center y_center width height
            labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save image
        img_path = images_dir / f"weapon_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        # Save label
        label_path = labels_dir / f"weapon_{i:03d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

    print(f"Generated {num_images} synthetic images and labels in {output_dir}")


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "dataset"
    generate_synthetic_dataset(output_dir, num_images=20)