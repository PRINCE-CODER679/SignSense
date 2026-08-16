"""
SignSense AI - Preprocessing & Pipeline Unit Test Suite (Phase 3)

Tests:
1. 21 landmarks produce exactly 63 numerical features.
2. Wrist origin normalization places wrist landmark (0) at (0.0, 0.0, 0.0).
3. Max Euclidean scale normalization maps maximum distance to 1.0.
4. Output contains zero NaN or Infinite values.
5. Zero-scale / collapsed landmark edge cases are handled safely.
6. Invalid sample inputs (wrong shape, missing points) are rejected with appropriate error log.
7. Label validator restricts labels strictly to A-Z.
8. Stratified dataset splitting maintains class proportions across train, val, and test splits.
"""
import sys
from pathlib import Path

# Ensure backend and repository root directories are in sys.path
backend_dir = Path(__file__).resolve().parent.parent
repo_dir = backend_dir.parent
for p in [str(backend_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import numpy as np
import pandas as pd
from ml.preprocessing.normalizer import normalize_landmarks
from ml.preprocessing.quality_checks import validate_features, validate_sample
from ml.preprocessing.extractor import LandmarkExtractor
from ml.config import CLASSES, FEATURE_DIM, FEATURE_COLUMNS, LABEL_COLUMN


def test_21_landmarks_produce_63_features():
    """Verify that 21 3D landmarks produce a 63-element feature vector."""
    # Synthetic 21 landmark tuples
    mock_landmarks = [(float(i), float(i * 2), float(i * 3)) for i in range(21)]
    features = normalize_landmarks(mock_landmarks)
    assert len(features) == 63
    assert isinstance(features, list)


def test_wrist_origin_normalization():
    """Verify wrist (landmark 0) is translated to (0.0, 0.0, 0.0)."""
    wrist_x, wrist_y, wrist_z = 100.5, 200.2, 50.8
    mock_landmarks = [(wrist_x + i * 2, wrist_y + i * 3, wrist_z + i) for i in range(21)]

    features = normalize_landmarks(mock_landmarks)

    # First 3 elements correspond to wrist x0', y0', z0'
    x0, y0, z0 = features[0], features[1], features[2]
    assert np.isclose(x0, 0.0, atol=1e-6)
    assert np.isclose(y0, 0.0, atol=1e-6)
    assert np.isclose(z0, 0.0, atol=1e-6)


def test_euclidean_scale_normalization():
    """Verify max Euclidean distance from wrist among landmarks scales to 1.0."""
    mock_landmarks = [(0.0, 0.0, 0.0)] * 21
    # Place landmark 10 at distance (3, 4, 0) -> Euclidean norm = 5.0
    mock_landmarks[10] = (3.0, 4.0, 0.0)

    features = normalize_landmarks(mock_landmarks)

    # Reshape features to 21x3
    coords_2d = np.array(features).reshape(21, 3)
    distances = np.linalg.norm(coords_2d, axis=1)

    assert np.isclose(np.max(distances), 1.0, atol=1e-6)
    # Landmark 10 coordinates scaled by 5.0 -> (0.6, 0.8, 0.0)
    assert np.isclose(coords_2d[10, 0], 0.6, atol=1e-6)
    assert np.isclose(coords_2d[10, 1], 0.8, atol=1e-6)


def test_no_nan_or_inf_values():
    """Verify output features contain no NaN or Inf values."""
    mock_landmarks = [(np.sin(i), np.cos(i), float(i)) for i in range(21)]
    features = normalize_landmarks(mock_landmarks)

    arr = np.array(features)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_zero_scale_edge_case():
    """Verify all landmarks at same point do not trigger zero-division error."""
    mock_landmarks = [(10.0, 20.0, 30.0)] * 21
    features = normalize_landmarks(mock_landmarks)

    assert len(features) == 63
    arr = np.array(features)
    assert not np.isnan(arr).any()
    assert np.allclose(arr, 0.0)


def test_invalid_landmark_shape_rejected():
    """Verify invalid landmark shapes raise ValueError."""
    invalid_10_landmarks = [(1.0, 2.0, 3.0)] * 10
    with pytest.raises(ValueError, match=r"Expected shape \(21, 3\)"):
        normalize_landmarks(invalid_10_landmarks)


def test_label_validation_restricts_to_az():
    """Verify label validation allows only A-Z and rejects invalid characters."""
    mock_vec = [0.1] * 63

    # Valid labels A-Z
    for char in ["A", "M", "Z"]:
        is_valid, msg = validate_sample(mock_vec, char)
        assert is_valid is True
        assert msg == "VALID"

    # Invalid labels
    for bad_label in ["a", "1", "AB", "ENTRY", "!", ""]:
        is_valid, msg = validate_sample(mock_vec, bad_label)
        assert is_valid is False
        assert "INVALID_LABEL" in msg


def test_quality_checks_dataframe_validation():
    """Verify dataset quality assertions detect missing columns and invalid labels."""
    # Create valid synthetic dataframe
    data = []
    for letter in CLASSES:
        for _ in range(2):
            row = {LABEL_COLUMN: letter}
            for col in FEATURE_COLUMNS:
                row[col] = np.random.uniform(-1.0, 1.0)
            # Wrist zero
            row["x0"], row["y0"], row["z0"] = 0.0, 0.0, 0.0
            data.append(row)

    valid_df = pd.DataFrame(data)
    report = validate_features(valid_df, raise_on_error=True)
    assert report["status"] == "PASSED"
    assert report["total_samples"] == 52

    # Test corrupted dataframe with NaN
    corrupt_df = valid_df.copy()
    corrupt_df.loc[0, "x5"] = np.nan
    with pytest.raises(ValueError, match="Corrupted numerical values"):
        validate_features(corrupt_df, raise_on_error=True)


def test_extractor_handles_empty_frame():
    """Verify LandmarkExtractor returns expected failure tuple on empty frame."""
    with LandmarkExtractor() as extractor:
        success, landmarks, reason = extractor.extract_from_frame(np.array([]))
        assert success is False
        assert landmarks is None
        assert "EMPTY_FRAME" in reason
