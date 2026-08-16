"""
SignSense AI - Landmark Normalization Engine (Phase 3)

Implements hand landmark normalization:
1. Origin Translation: Translates all 21 3D hand landmarks relative to the wrist (landmark 0).
2. Euclidean Scale Normalization: Scales all coordinates by the maximum Euclidean distance 
   from the wrist to any landmark, ensuring invariance to hand size and camera distance.
"""
from typing import List, Tuple, Union
import numpy as np


def normalize_landmarks(
    landmarks_xyz: Union[List[Tuple[float, float, float]], List[List[float]], np.ndarray],
    handedness: Union[str, None] = None
) -> List[float]:
    """
    Normalizes 21 3D hand landmarks (x, y, z) into a 63-element feature vector.
    Applies wrist translation, handedness normalization (x = -x for left hands),
    canonical upright 2D rotation, and Euclidean scale normalization.

    Args:
        landmarks_xyz: List or array of 21 3D coordinate triples [(x0,y0,z0), ..., (x20,y20,z20)]
        handedness: Optional string ('Left' or 'Right'). If 'Left' or auto-detected as Left hand,
                    coordinates are horizontally flipped (x = -x) to align with canonical Right hand.

    Returns:
        63-element list of floats [x0', y0', z0', x1', y1', z1', ..., x20', y20', z20']

    Raises:
        ValueError: If input does not contain exactly 21 3D landmarks or contains NaN/Inf values.
    """
    coords = np.array(landmarks_xyz, dtype=np.float64)

    # Validate shape
    if coords.shape != (21, 3):
        raise ValueError(
            f"Expected shape (21, 3) representing 21 3D landmarks, received shape {coords.shape}"
        )

    # Check for NaN or Inf in input
    if np.isnan(coords).any() or np.isinf(coords).any():
        raise ValueError("Landmark coordinates contain NaN or Infinite values.")

    # 1. Translate wrist (landmark 0) to origin (0, 0, 0)
    wrist = coords[0]
    centered_coords = coords - wrist

    # 2. Handedness Normalization (Horizontal Mirroring x = -x for Left Hand)
    # Map left-hand landmarks to canonical right-hand coordinate space
    is_left_hand = False
    if handedness is not None:
        h_str = str(handedness).strip().lower()
        if h_str == "left":
            is_left_hand = True
        elif h_str == "right":
            is_left_hand = False
    else:
        # Fallback auto-detection via 2D cross product of (Wrist -> Index MCP) x (Wrist -> Pinky MCP)
        v_idx = centered_coords[5]
        v_pnk = centered_coords[17]
        cp_z = v_idx[0] * v_pnk[1] - v_idx[1] * v_pnk[0]
        # In screen coordinates, negative cross product indicates Left hand layout
        if cp_z < 0.0:
            is_left_hand = True

    if is_left_hand:
        centered_coords[:, 0] = -centered_coords[:, 0]

    # 3. Canonical Upright 2D Rotation Alignment
    # Align vector from Wrist (LM 0) to Middle MCP (LM 9) so hand is always upright (0, -1) in image space
    v_middle = centered_coords[9]
    dx, dy = v_middle[0], v_middle[1]
    if not (np.isclose(dx, 0.0) and np.isclose(dy, 0.0)):
        current_angle = float(np.arctan2(dy, dx))
        target_angle = -np.pi / 2.0  # -90 deg (straight up)
        rot_angle = target_angle - current_angle
        cos_a = float(np.cos(rot_angle))
        sin_a = float(np.sin(rot_angle))

        aligned_coords = centered_coords.copy()
        aligned_coords[:, 0] = centered_coords[:, 0] * cos_a - centered_coords[:, 1] * sin_a
        aligned_coords[:, 1] = centered_coords[:, 0] * sin_a + centered_coords[:, 1] * cos_a
    else:
        aligned_coords = centered_coords

    # 4. Compute maximum Euclidean distance from wrist to any landmark for scaling
    distances = np.linalg.norm(aligned_coords, axis=1)
    max_distance = float(np.max(distances))

    # Handle zero-scale edge case safely (e.g. all points collapsed at origin)
    if max_distance == 0.0 or np.isclose(max_distance, 0.0):
        scale_factor = 1.0
    else:
        scale_factor = max_distance

    normalized_coords = aligned_coords / scale_factor

    # Ensure zero values are exact float 0.0 to avoid -0.0 artifacts
    normalized_coords = np.where(np.isclose(normalized_coords, 0.0), 0.0, normalized_coords)

    # Flatten into a 63-element 1D vector (x0, y0, z0, x1, y1, z1, ..., x20, y20, z20)
    return normalized_coords.flatten().tolist()

