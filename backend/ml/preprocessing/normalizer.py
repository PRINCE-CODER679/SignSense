"""
SignSense AI - Landmark Normalization Engine (Phase 3)

Implements hand landmark normalization:
1. Origin Translation: Translates all 21 3D hand landmarks relative to the wrist (landmark 0).
2. Euclidean Scale Normalization: Scales all coordinates by the maximum Euclidean distance 
   from the wrist to any landmark, ensuring invariance to hand size and camera distance.
"""
from typing import List, Tuple, Union
import numpy as np


def compute_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Computes angle in radians at p2 formed by vectors (p1-p2) and (p3-p2)."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def compute_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes angle in radians between two 3D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def normalize_landmarks(
    landmarks_xyz: Union[List[Tuple[float, float, float]], List[List[float]], np.ndarray],
    handedness: Union[str, None] = None
) -> List[float]:
    """
    Normalizes 21 3D hand landmarks (x, y, z) into a 115-element feature vector.
    Applies wrist translation, handedness normalization (x = -x for left hands),
    canonical upright 2D rotation, Euclidean scale normalization, and extracts
    52 rotation- and scale-invariant geometric features (distances, 3D joint angles, & depth).

    Args:
        landmarks_xyz: List or array of 21 3D coordinate triples [(x0,y0,z0), ..., (x20,y20,z20)]
        handedness: Optional string ('Left' or 'Right'). If 'Left' or auto-detected as Left hand,
                    coordinates are horizontally flipped (x = -x) to align with canonical Right hand.

    Returns:
        115-element list of floats [x0', y0', z0', ..., x20', y20', z20', dist_t_i, ..., thumb_index_pip_cross]

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

    # 5. Extract 40 Rotation & Scale Invariant Geometric Features
    pts = normalized_coords
    geo_features: List[float] = []

    def d(i, j):
        return float(np.linalg.norm(pts[i] - pts[j]))

    # Fingertip to thumb tip distances (4)
    geo_features.extend([d(4, 8), d(4, 12), d(4, 16), d(4, 20)])

    # Adjacent fingertip distances (3)
    geo_features.extend([d(8, 12), d(12, 16), d(16, 20)])

    # Fingertip to wrist distances (5)
    geo_features.extend([d(0, 4), d(0, 8), d(0, 12), d(0, 16), d(0, 20)])

    # Thumb tip to MCP knuckle distances (4)
    geo_features.extend([d(4, 5), d(4, 9), d(4, 13), d(4, 17)])

    # Fingertip to own MCP knuckle distances (5)
    geo_features.extend([d(2, 4), d(5, 8), d(9, 12), d(13, 16), d(17, 20)])

    # Joint flex angles (15)
    geo_features.extend([
        compute_joint_angle(pts[0], pts[1], pts[2]),
        compute_joint_angle(pts[1], pts[2], pts[3]),
        compute_joint_angle(pts[2], pts[3], pts[4]),
        compute_joint_angle(pts[0], pts[5], pts[6]),
        compute_joint_angle(pts[5], pts[6], pts[7]),
        compute_joint_angle(pts[6], pts[7], pts[8]),
        compute_joint_angle(pts[0], pts[9], pts[10]),
        compute_joint_angle(pts[9], pts[10], pts[11]),
        compute_joint_angle(pts[10], pts[11], pts[12]),
        compute_joint_angle(pts[0], pts[13], pts[14]),
        compute_joint_angle(pts[13], pts[14], pts[15]),
        compute_joint_angle(pts[14], pts[15], pts[16]),
        compute_joint_angle(pts[0], pts[17], pts[18]),
        compute_joint_angle(pts[17], pts[18], pts[19]),
        compute_joint_angle(pts[18], pts[19], pts[20]),
    ])

    # Inter-finger spread angles (4)
    v_t = pts[4] - pts[0]
    v_i = pts[8] - pts[0]
    v_m = pts[12] - pts[0]
    v_r = pts[16] - pts[0]
    v_p = pts[20] - pts[0]

    geo_features.extend([
        compute_vector_angle(v_t, v_i),
        compute_vector_angle(v_i, v_m),
        compute_vector_angle(v_m, v_r),
        compute_vector_angle(v_r, v_p),
    ])

    # Thumb tip to PIP joint distances for fist letter disambiguation (M, N, T, E, S, A) (4)
    geo_features.extend([d(4, 6), d(4, 10), d(4, 14), d(4, 18)])

    # Thumb tip to DIP joint distances (4)
    geo_features.extend([d(4, 7), d(4, 11), d(4, 15), d(4, 19)])

    # Thumb depth relative to palm plane formed by Wrist (0), Index MCP (5), Pinky MCP (17) (1)
    v_palm1 = pts[5] - pts[0]
    v_palm2 = pts[17] - pts[0]
    palm_normal = np.cross(v_palm1, v_palm2)
    norm_p = np.linalg.norm(palm_normal)
    if norm_p > 1e-7:
        palm_normal = palm_normal / norm_p
        thumb_depth = float(np.dot(pts[4] - pts[0], palm_normal))
    else:
        thumb_depth = 0.0
    geo_features.append(thumb_depth)

    # Targeted O vs C gap ratio: thumb tip to index tip distance d(4,8) / index finger length d(5,8) (1)
    len_index_finger = d(5, 8)
    gap_o_c = float(d(4, 8) / (len_index_finger if len_index_finger > 1e-6 else 1.0))
    geo_features.append(gap_o_c)

    # Targeted M vs N PIP differential distance: d(4, 14) - d(4, 10) (1)
    m_n_pip_diff = float(d(4, 14) - d(4, 10))
    geo_features.append(m_n_pip_diff)

    # Thumb tip to Index PIP cross projection (1)
    v_thumb_tip = pts[4] - pts[0]
    v_index_pip = pts[6] - pts[0]
    cross_ti = np.cross(v_thumb_tip, v_index_pip)
    thumb_index_pip_cross = float(np.linalg.norm(cross_ti))
    geo_features.append(thumb_index_pip_cross)

    # Combine 63 raw coordinates + 52 geometric features = 115 features
    full_vector = normalized_coords.flatten().tolist() + geo_features
    return full_vector

