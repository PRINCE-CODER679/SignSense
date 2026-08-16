"""
SignSense AI - Data Quality Checks & Assertions Engine (Phase 3)

Fails loudly when corrupted data, NaN/Inf values, invalid feature dimensions,
or invalid class labels are detected.
"""
import logging
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import pandas as pd
try:
    from ml.config import CLASSES, FEATURE_DIM, FEATURE_COLUMNS, LABEL_COLUMN
except ImportError:
    from backend.ml.config import CLASSES, FEATURE_DIM, FEATURE_COLUMNS, LABEL_COLUMN


logger = logging.getLogger(__name__)


def validate_sample(
    feature_vector: Union[List[float], np.ndarray], label: str
) -> Tuple[bool, str]:
    """
    Validates a single processed landmark sample.

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if label not in CLASSES:
        return False, f"INVALID_LABEL: '{label}' is not in allowed A-Z classes."

    arr = np.array(feature_vector, dtype=np.float64)
    if arr.shape != (FEATURE_DIM,):
        return False, f"INVALID_DIMENSION: Expected {FEATURE_DIM} features, got {arr.shape}."

    if np.isnan(arr).any():
        return False, "NAN_DETECTED: Feature vector contains NaN values."

    if np.isinf(arr).any():
        return False, "INF_DETECTED: Feature vector contains Infinite values."

    return True, "VALID"


def validate_features(
    df: pd.DataFrame, raise_on_error: bool = True
) -> Dict[str, Any]:
    """
    Performs comprehensive dataset-level quality assertions on a pandas DataFrame.

    Checks:
    - Expected label and 63 feature columns exist
    - All feature columns contain numeric non-NaN, non-Inf values
    - All labels are valid A-Z characters
    - Duplicate sample detection
    - Severe class imbalance checking

    Args:
        df: DataFrame containing 'label' and 'x0'..'z20' feature columns.
        raise_on_error: If True, raises ValueError immediately on quality check failures.

    Returns:
        Dict containing quality metric statistics.
    """
    # 1. Check required columns
    missing_cols = [col for col in [LABEL_COLUMN] + FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        msg = f"Data quality failure: Missing required columns: {missing_cols}"
        if raise_on_error:
            raise ValueError(msg)
        return {"status": "FAILED", "error": msg}

    # 2. Check total sample count
    if len(df) == 0:
        msg = "Data quality failure: Dataset is empty (0 samples)."
        if raise_on_error:
            raise ValueError(msg)
        return {"status": "FAILED", "error": msg}

    # 3. Check for invalid labels
    invalid_labels = df[~df[LABEL_COLUMN].isin(CLASSES)][LABEL_COLUMN].unique().tolist()
    if invalid_labels:
        msg = f"Data quality failure: Found invalid labels in dataset: {invalid_labels}"
        if raise_on_error:
            raise ValueError(msg)
        return {"status": "FAILED", "error": msg}

    # 4. Check feature numerical matrix shape
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if features.shape[1] != FEATURE_DIM:
        msg = f"Data quality failure: Expected feature dimension {FEATURE_DIM}, got {features.shape[1]}"
        if raise_on_error:
            raise ValueError(msg)
        return {"status": "FAILED", "error": msg}

    # 5. Check NaN and Inf values
    nan_count = int(np.isnan(features).sum())
    inf_count = int(np.isinf(features).sum())
    if nan_count > 0 or inf_count > 0:
        msg = f"Data quality failure: Corrupted numerical values detected (NaN: {nan_count}, Inf: {inf_count})."
        if raise_on_error:
            raise ValueError(msg)
        return {"status": "FAILED", "error": msg}

    # 6. Check duplicate rows
    duplicate_count = int(df.duplicated(subset=FEATURE_COLUMNS).sum())

    # 7. Check class distribution & severe imbalance
    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    present_classes = set(class_counts.keys())
    missing_classes = [c for c in CLASSES if c not in present_classes]

    if missing_classes:
        logger.warning(f"Data quality warning: Missing samples for classes: {missing_classes}")

    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    imbalance_ratio = (max_count / min_count) if min_count > 0 else float("inf")

    if imbalance_ratio > 5.0:
        logger.warning(f"Data quality warning: Severe class imbalance detected (Ratio: {imbalance_ratio:.2f}).")

    report = {
        "status": "PASSED",
        "total_samples": len(df),
        "num_features": FEATURE_DIM,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_samples": duplicate_count,
        "unique_classes": len(present_classes),
        "missing_classes": missing_classes,
        "min_class_count": min_count,
        "max_class_count": max_count,
        "imbalance_ratio": round(imbalance_ratio, 2) if min_count > 0 else None,
        "class_distribution": class_counts
    }

    return report
