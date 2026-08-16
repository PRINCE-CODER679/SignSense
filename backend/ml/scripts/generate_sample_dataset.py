"""
SignSense AI - Synthetic Sample Dataset Generator (Phase 3)

Generates test hand sample images in backend/ml/data/raw/A-Z
so the landmark feature extraction pipeline can be verified deterministically.
"""
import sys
from pathlib import Path

# Add backend and repository root directories to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
repo_dir = backend_dir.parent
for p in [str(backend_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import math
try:
    from ml.config import RAW_DATA_DIR, CLASSES
except ImportError:
    from backend.ml.config import RAW_DATA_DIR, CLASSES


def render_synthetic_hand_image(letter: str, sample_idx: int) -> np.ndarray:
    """
    Renders an image containing a hand shape with distinct finger positions 
    tailored to ASL fingerspelling letter shapes.
    """
    # 400x400 RGB canvas with light background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 230

    # Base palm center and wrist
    seed = (ord(letter) * 13 + sample_idx * 19) % 100
    wrist_x = 200 + (seed % 6 - 3)
    wrist_y = 330 + (seed % 6 - 3)

    palm_center_x = 200 + (seed % 6 - 3)
    palm_center_y = 230 + (seed % 6 - 3)

    # Vary finger extension angles according to ASL class characteristics
    angle_offset = (ord(letter) - ord('A')) * (2.0 * math.pi / 26.0)

    # 5 finger tip angles relative to palm center
    finger_angles = [
        -2.4 + math.sin(angle_offset) * 0.2,          # Thumb
        -1.75 + math.cos(angle_offset * 0.5) * 0.3,   # Index
        -1.45 + math.sin(angle_offset * 0.7) * 0.3,   # Middle
        -1.15 + math.cos(angle_offset * 0.9) * 0.3,   # Ring
        -0.80 + math.sin(angle_offset * 1.1) * 0.3,   # Pinky
    ]

    finger_lengths = [
        65 + (seed % 5),   # Thumb
        95 + (seed % 7),   # Index
        105 + (seed % 6),  # Middle
        90 + (seed % 8),   # Ring
        75 + (seed % 4)    # Pinky
    ]

    # Draw palm base
    cv2.circle(img, (int(palm_center_x), int(palm_center_y)), 55, (175, 190, 225), -1)
    cv2.line(img, (int(wrist_x), int(wrist_y)), (int(palm_center_x), int(palm_center_y)), (160, 175, 210), 38)

    # Draw fingers (MCP -> PIP -> DIP -> TIP joints)
    for angle, length in zip(finger_angles, finger_lengths):
        seg_len = length / 3.0
        curr_x, curr_y = palm_center_x, palm_center_y
        
        for segment in range(3):
            next_x = curr_x + seg_len * math.cos(angle)
            next_y = curr_y + seg_len * math.sin(angle)
            thickness = int(18 - segment * 3)
            cv2.line(img, (int(curr_x), int(curr_y)), (int(next_x), int(next_y)), (150, 165, 205), thickness)
            cv2.circle(img, (int(next_x), int(next_y)), thickness // 2 + 1, (130, 145, 185), -1)
            curr_x, curr_y = next_x, next_y

    return img


def generate_sample_dataset(samples_per_class: int = 5):
    """
    Generates synthetic sample images for each class A-Z in raw data directory.
    """
    print(f"Generating synthetic sample dataset ({samples_per_class} samples per class across 26 classes)...")
    total_generated = 0

    for letter in CLASSES:
        class_dir = RAW_DATA_DIR / letter
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, samples_per_class + 1):
            img = render_synthetic_hand_image(letter, i)
            img_path = class_dir / f"sample_{i:04d}.png"
            cv2.imwrite(str(img_path), img)
            total_generated += 1

    print(f"Successfully generated {total_generated} sample images in '{RAW_DATA_DIR}'.")


if __name__ == "__main__":
    generate_sample_dataset(samples_per_class=5)
