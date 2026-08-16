"""
SignSense AI - High-Quality Realistic ASL Landmark Dataset Generator (Phase 3 Fix)

Generates 100 realistic 3D hand landmark samples per class for all 26 letters (A-Z)
(2,600 samples total) using precise 21-joint anatomical ASL fingerspelling geometry.
Applies random 3D rotations, distance scaling, wrist offsets, and anatomical joint jitter
to build a rich, balanced feature dataset.
"""
import sys
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
from ml.config import (
    PROCESSED_CSV_PATH,
    CLASSES,
    FEATURE_COLUMNS,
    LABEL_COLUMN
)
from ml.preprocessing.normalizer import normalize_landmarks


def rotate_3d(points: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Applies 3D rotation matrix (yaw, pitch, roll in radians) to Nx3 points."""
    # Rotation around X (pitch)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    # Rotation around Y (yaw)
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    # Rotation around Z (roll)
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return (R @ points.T).T


def build_base_hand_skeleton(letter: str, variant: int = 0) -> np.ndarray:
    """
    Constructs base 21 3D MediaPipe hand landmarks (wrist=0, thumb=1..4, index=5..8,
    middle=9..12, ring=13..16, pinky=17..20) corresponding to ASL letter shape.
    Coordinates are in normalized units relative to palm center.
    """
    wrist = np.array([0.5, 0.7, 0.0])
    
    # Palm MCP (Knuckle) anchors
    mcp_thumb  = np.array([0.42, 0.62, -0.02])
    mcp_index  = np.array([0.46, 0.50, -0.01])
    mcp_middle = np.array([0.50, 0.49,  0.00])
    mcp_ring   = np.array([0.54, 0.51,  0.01])
    mcp_pinky  = np.array([0.58, 0.54,  0.02])

    # Helper vectors for finger extension states
    # Upwards (extended), Curled down into palm, Forward, Outward
    def extend_finger(mcp, angle_offset_x=0.0, length_scale=1.0):
        # 3 joint segments (PIP, DIP, TIP)
        seg = 0.030 * length_scale
        dx = angle_offset_x
        return [
            mcp + np.array([dx, -seg * 1, -0.005]),
            mcp + np.array([dx*1.5, -seg * 2, -0.010]),
            mcp + np.array([dx*2.0, -seg * 3, -0.015])
        ]

    def curl_finger(mcp, curl_x=0.0, curl_z=0.025):
        # Curled into palm
        return [
            mcp + np.array([curl_x, 0.015, curl_z * 0.5]),
            mcp + np.array([curl_x*0.8, 0.030, curl_z * 0.8]),
            mcp + np.array([curl_x*0.5, 0.020, curl_z * 0.4])
        ]

    def extend_horizontal(mcp, dx=0.03):
        return [
            mcp + np.array([dx * 1, -0.005, -0.005]),
            mcp + np.array([dx * 2, -0.010, -0.010]),
            mcp + np.array([dx * 3, -0.015, -0.015])
        ]

    def extend_downward(mcp, dx=0.0):
        return [
            mcp + np.array([dx, 0.025, 0.01]),
            mcp + np.array([dx, 0.050, 0.02]),
            mcp + np.array([dx, 0.075, 0.03])
        ]

    # Configure finger positions according to ASL shape
    if letter == 'A':
        thumb_pts = [mcp_thumb + np.array([0.02, -0.02, -0.01]), mcp_thumb + np.array([0.03, -0.04, -0.015]), mcp_thumb + np.array([0.03, -0.06, -0.02])]
        idx_pts = curl_finger(mcp_index)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'B':
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.035])]
        idx_pts = extend_finger(mcp_index, angle_offset_x=-0.005, length_scale=1.0)
        mid_pts = extend_finger(mcp_middle, angle_offset_x=0.0, length_scale=1.05)
        rng_pts = extend_finger(mcp_ring, angle_offset_x=0.005, length_scale=0.95)
        pnk_pts = extend_finger(mcp_pinky, angle_offset_x=0.010, length_scale=0.85)

    elif letter == 'C':
        thumb_pts = [mcp_thumb + np.array([-0.03, -0.03, -0.03]), mcp_thumb + np.array([-0.05, -0.05, -0.05]), mcp_thumb + np.array([-0.06, -0.06, -0.06])]
        c_curve = lambda mcp: [mcp + np.array([0.01, -0.02, -0.03]), mcp + np.array([0.02, -0.04, -0.05]), mcp + np.array([0.01, -0.05, -0.07])]
        idx_pts = c_curve(mcp_index)
        mid_pts = c_curve(mcp_middle)
        rng_pts = c_curve(mcp_ring)
        pnk_pts = c_curve(mcp_pinky)

    elif letter == 'D':
        # Variant 0: Straight index, Variant 1 & 2: Natural curved/arched index forming D loop
        thumb_pts = [mcp_thumb + np.array([0.03, -0.02, 0.01]), mcp_thumb + np.array([0.05, -0.03, 0.02]), mcp_thumb + np.array([0.06, -0.04, 0.02])]
        if variant == 1:
            idx_pts = [mcp_index + np.array([0.005, -0.025, -0.015]), mcp_index + np.array([0.010, -0.045, 0.005]), mcp_index + np.array([0.015, -0.060, 0.020])]
        elif variant == 2:
            idx_pts = [mcp_index + np.array([0.010, -0.030, -0.010]), mcp_index + np.array([0.015, -0.050, 0.010]), mcp_index + np.array([0.020, -0.065, 0.025])]
        else:
            idx_pts = extend_finger(mcp_index)
        mid_pts = [mcp_middle + np.array([0.0, -0.02, 0.02]), mcp_middle + np.array([0.0, -0.03, 0.03]), mcp_middle + np.array([0.0, -0.035, 0.025])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'E':
        # All fingers tightly curled, thumb tucked across bottom of fingers
        thumb_pts = [mcp_thumb + np.array([0.03, 0.02, 0.02]), mcp_thumb + np.array([0.05, 0.025, 0.025]), mcp_thumb + np.array([0.06, 0.025, 0.025])]
        idx_pts = curl_finger(mcp_index, curl_z=0.035)
        mid_pts = curl_finger(mcp_middle, curl_z=0.035)
        rng_pts = curl_finger(mcp_ring, curl_z=0.035)
        pnk_pts = curl_finger(mcp_pinky, curl_z=0.035)

    elif letter == 'F':
        # Index tip touching thumb tip, others extended up
        idx_pts = [mcp_index + np.array([-0.02, 0.01, 0.02]), mcp_index + np.array([-0.03, 0.02, 0.03]), mcp_index + np.array([-0.04, 0.03, 0.035])]
        thumb_pts = [mcp_thumb + np.array([0.01, -0.01, 0.015]), mcp_thumb + np.array([0.02, -0.02, 0.025]), mcp_thumb + np.array([0.025, -0.025, 0.035])]
        mid_pts = extend_finger(mcp_middle, angle_offset_x=-0.005)
        rng_pts = extend_finger(mcp_ring, angle_offset_x=0.005)
        pnk_pts = extend_finger(mcp_pinky, angle_offset_x=0.015)

    elif letter == 'G':
        # Index & thumb horizontal
        thumb_pts = extend_horizontal(mcp_thumb, dx=-0.025)
        idx_pts = extend_horizontal(mcp_index, dx=-0.030)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'H':
        # Index & middle horizontal together
        thumb_pts = [mcp_thumb + np.array([0.02, 0.01, 0.02]), mcp_thumb + np.array([0.03, 0.02, 0.02]), mcp_thumb + np.array([0.04, 0.02, 0.02])]
        idx_pts = extend_horizontal(mcp_index, dx=-0.028)
        mid_pts = extend_horizontal(mcp_middle, dx=-0.028)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'I':
        # Pinky extended up, others curled
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        idx_pts = curl_finger(mcp_index)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = extend_finger(mcp_pinky)

    elif letter == 'J':
        # Pinky extended up with slight curve/angle
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        idx_pts = curl_finger(mcp_index)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = [mcp_pinky + np.array([0.01, -0.02, -0.01]), mcp_pinky + np.array([0.025, -0.04, -0.015]), mcp_pinky + np.array([0.04, -0.05, -0.02])]

    elif letter == 'K':
        # Index up, middle forward, thumb resting on middle knuckle
        idx_pts = extend_finger(mcp_index)
        mid_pts = [mcp_middle + np.array([0.0, -0.02, -0.02]), mcp_middle + np.array([0.0, -0.04, -0.04]), mcp_middle + np.array([0.0, -0.05, -0.05])]
        thumb_pts = [mcp_thumb + np.array([0.02, -0.02, -0.01]), mcp_thumb + np.array([0.04, -0.03, -0.02]), mcp_thumb + np.array([0.05, -0.04, -0.02])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'L':
        # L shape: Thumb horizontal, Index vertical
        thumb_pts = [mcp_thumb + np.array([-0.03, -0.01, 0.0]), mcp_thumb + np.array([-0.05, -0.01, 0.0]), mcp_thumb + np.array([-0.07, -0.01, 0.0])]
        idx_pts = extend_finger(mcp_index)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'M':
        # Thumb tucked under index, middle, ring
        thumb_pts = [mcp_thumb + np.array([0.04, 0.02, 0.01]), mcp_thumb + np.array([0.07, 0.02, 0.01]), mcp_thumb + np.array([0.09, 0.02, 0.01])]
        idx_pts = curl_finger(mcp_index, curl_z=0.03)
        mid_pts = curl_finger(mcp_middle, curl_z=0.03)
        rng_pts = curl_finger(mcp_ring, curl_z=0.03)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'N':
        # Thumb tucked under index and middle
        thumb_pts = [mcp_thumb + np.array([0.04, 0.02, 0.01]), mcp_thumb + np.array([0.06, 0.02, 0.01]), mcp_thumb + np.array([0.075, 0.02, 0.01])]
        idx_pts = curl_finger(mcp_index, curl_z=0.03)
        mid_pts = curl_finger(mcp_middle, curl_z=0.03)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'O':
        # O shape: All finger tips touching thumb tip
        o_curve = lambda mcp: [mcp + np.array([-0.01, -0.02, -0.02]), mcp + np.array([-0.02, -0.04, -0.03]), mcp + np.array([-0.03, -0.05, -0.02])]
        idx_pts = o_curve(mcp_index)
        mid_pts = o_curve(mcp_middle)
        rng_pts = o_curve(mcp_ring)
        pnk_pts = o_curve(mcp_pinky)
        thumb_pts = [mcp_thumb + np.array([0.02, -0.02, -0.01]), mcp_thumb + np.array([0.03, -0.035, -0.015]), mcp_thumb + np.array([0.04, -0.045, -0.02])]

    elif letter == 'P':
        # Pointing down K
        idx_pts = extend_horizontal(mcp_index, dx=-0.025)
        mid_pts = extend_downward(mcp_middle)
        thumb_pts = [mcp_thumb + np.array([0.02, 0.01, 0.01]), mcp_thumb + np.array([0.03, 0.02, 0.02]), mcp_thumb + np.array([0.04, 0.03, 0.02])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'Q':
        # Pointing down G
        thumb_pts = extend_downward(mcp_thumb, dx=-0.01)
        idx_pts = extend_downward(mcp_index, dx=0.01)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'R':
        # Crossed index and middle
        idx_pts = extend_finger(mcp_index, angle_offset_x=0.015)
        mid_pts = extend_finger(mcp_middle, angle_offset_x=-0.015)
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'S':
        # Tight fist with thumb over front
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, -0.02]), mcp_thumb + np.array([0.06, 0.01, -0.03]), mcp_thumb + np.array([0.08, 0.01, -0.035])]
        idx_pts = curl_finger(mcp_index, curl_z=0.025)
        mid_pts = curl_finger(mcp_middle, curl_z=0.025)
        rng_pts = curl_finger(mcp_ring, curl_z=0.025)
        pnk_pts = curl_finger(mcp_pinky, curl_z=0.025)

    elif letter == 'T':
        # Thumb tucked under index finger
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.01]), mcp_thumb + np.array([0.05, 0.01, 0.01]), mcp_thumb + np.array([0.06, 0.01, 0.01])]
        idx_pts = curl_finger(mcp_index, curl_z=0.03)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'U':
        # Index & middle straight up together (closed V)
        idx_pts = extend_finger(mcp_index, angle_offset_x=0.005)
        mid_pts = extend_finger(mcp_middle, angle_offset_x=-0.005)
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'V':
        # Index & middle straight up spread apart (open V)
        idx_pts = extend_finger(mcp_index, angle_offset_x=-0.02)
        mid_pts = extend_finger(mcp_middle, angle_offset_x=0.02)
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'W':
        # Index, middle, ring straight up spread apart
        idx_pts = extend_finger(mcp_index, angle_offset_x=-0.025)
        mid_pts = extend_finger(mcp_middle, angle_offset_x=0.0)
        rng_pts = extend_finger(mcp_ring, angle_offset_x=0.025)
        thumb_pts = [mcp_thumb + np.array([0.03, 0.02, 0.02]), mcp_thumb + np.array([0.05, 0.02, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'X':
        # Index bent hook shape
        idx_pts = [mcp_index + np.array([0.0, -0.03, -0.01]), mcp_index + np.array([0.0, -0.04, 0.01]), mcp_index + np.array([0.0, -0.03, 0.02])]
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    elif letter == 'Y':
        # Thumb & pinky extended, middle 3 curled
        thumb_pts = [mcp_thumb + np.array([-0.03, -0.02, 0.0]), mcp_thumb + np.array([-0.05, -0.03, 0.0]), mcp_thumb + np.array([-0.07, -0.04, 0.0])]
        pnk_pts = [mcp_pinky + np.array([0.03, -0.02, 0.0]), mcp_pinky + np.array([0.05, -0.03, 0.0]), mcp_pinky + np.array([0.07, -0.04, 0.0])]
        idx_pts = curl_finger(mcp_index)
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)

    elif letter == 'Z':
        # Index extended up/forward for Z tracing
        idx_pts = [mcp_index + np.array([0.01, -0.04, -0.02]), mcp_index + np.array([0.02, -0.07, -0.03]), mcp_index + np.array([0.03, -0.09, -0.03])]
        thumb_pts = [mcp_thumb + np.array([0.03, 0.01, 0.02]), mcp_thumb + np.array([0.05, 0.015, 0.03]), mcp_thumb + np.array([0.06, 0.02, 0.03])]
        mid_pts = curl_finger(mcp_middle)
        rng_pts = curl_finger(mcp_ring)
        pnk_pts = curl_finger(mcp_pinky)

    else:
        idx_pts = extend_finger(mcp_index)
        mid_pts = extend_finger(mcp_middle)
        rng_pts = extend_finger(mcp_ring)
        pnk_pts = extend_finger(mcp_pinky)
        thumb_pts = [mcp_thumb + np.array([0.02, -0.02, 0.0]), mcp_thumb + np.array([0.04, -0.04, 0.0]), mcp_thumb + np.array([0.06, -0.06, 0.0])]

    landmarks = [wrist, mcp_thumb] + thumb_pts + [mcp_index] + idx_pts + [mcp_middle] + mid_pts + [mcp_ring] + rng_pts + [mcp_pinky] + pnk_pts
    return np.array(landmarks, dtype=np.float64)


def generate_samples_for_class(letter: str, n_samples: int = 100) -> List[List[float]]:
    """
    Generates n_samples normalized 63D feature vectors for a given letter class,
    applying realistic 3D spatial augmentations (rotations, distance scaling, joint noise).
    """
    samples_63d: List[List[float]] = []
    half_samples = n_samples // 2

    for i in range(n_samples):
        base_coords = build_base_hand_skeleton(letter, variant=i % 3)
        coords = base_coords.copy()
        
        # 1. Random 3D Rotation (yaw ±15 deg, pitch ±15 deg, roll ±15 deg)
        yaw = math.radians(random.uniform(-15.0, 15.0))
        pitch = math.radians(random.uniform(-15.0, 15.0))
        roll = math.radians(random.uniform(-15.0, 15.0))
        
        # Center at wrist before rotation
        wrist = coords[0].copy()
        coords_centered = coords - wrist
        coords_rotated = rotate_3d(coords_centered, yaw, pitch, roll)
        
        # 2. Camera distance scaling (hand size variation: 0.85x to 1.15x)
        scale = random.uniform(0.85, 1.15)
        coords_scaled = coords_rotated * scale
        
        # 3. Add random camera translation offset
        trans_x = random.uniform(-0.10, 0.10)
        trans_y = random.uniform(-0.10, 0.10)
        trans_z = random.uniform(-0.05, 0.05)
        coords_world = coords_scaled + wrist + np.array([trans_x, trans_y, trans_z])
        
        # 4. Add subtle joint anatomical noise (Gaussian jitter σ = 0.002)
        noise = np.random.normal(0.0, 0.002, coords_world.shape)
        # Keep wrist noise smaller
        noise[0] *= 0.2
        coords_final = coords_world + noise
        
        # 5. Apply exact project normalization pipeline -> 63D vector (canonical Right hand)
        feat_63 = normalize_landmarks(coords_final.tolist(), handedness="Right")
        samples_63d.append(feat_63)

    return samples_63d


def generate_full_dataset(samples_per_class: int = 200):
    """
    Builds and exports a clean canonical ASL landmark dataset (26 classes x 200 samples = 5,200 samples)
    to PROCESSED_CSV_PATH ('backend/ml/data/processed/asl_features.csv').
    """
    print(f"\nBuilding realistic canonical ASL landmark dataset ({samples_per_class} samples/class x 26 classes = {samples_per_class * 26} total samples)...")
    
    rows = []
    for letter in CLASSES:
        feats_list = generate_samples_for_class(letter, n_samples=samples_per_class)
        for vec in feats_list:
            row_dict = {LABEL_COLUMN: letter}
            for col, val in zip(FEATURE_COLUMNS, vec):
                row_dict[col] = float(val)
            rows.append(row_dict)
            
    df = pd.DataFrame(rows)
    # Reorder columns: label, x0, y0, z0, ..., x20, y20, z20
    df = df[[LABEL_COLUMN] + FEATURE_COLUMNS]
    
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    
    print(f"Successfully generated clean bilateral dataset at '{PROCESSED_CSV_PATH}'.")
    print(f"Total samples: {len(df):,} | Classes: {len(CLASSES)} | Features per sample: {len(FEATURE_COLUMNS)}")
    print(f"Class distribution:\n{df[LABEL_COLUMN].value_counts().to_dict()}\n")


if __name__ == "__main__":
    generate_full_dataset(samples_per_class=200)

