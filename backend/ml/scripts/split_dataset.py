"""
SignSense AI - Stratified Dataset Split Engine (Phase 3)

Splits processed ASL landmark feature CSV into:
- 70% Training set (train_features.csv)
- 15% Validation set (val_features.csv)
- 15% Testing set (test_features.csv)

Ensures exact class distribution preservation across splits using stratified sampling with fixed seed (42).
Prevents data leakage across sets.
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure backend and repository root directories are in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
repo_dir = backend_dir.parent
for p in [str(backend_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from sklearn.model_selection import train_test_split
try:
    from ml.config import (
        PROCESSED_CSV_PATH,
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        SPLIT_METADATA_PATH,
        LABEL_COLUMN,
        RANDOM_STATE,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        CLASSES,
    )
    from ml.preprocessing.quality_checks import validate_features
except ImportError:
    from backend.ml.config import (
        PROCESSED_CSV_PATH,
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        SPLIT_METADATA_PATH,
        LABEL_COLUMN,
        RANDOM_STATE,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        CLASSES,
    )
    from backend.ml.preprocessing.quality_checks import validate_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DatasetSplit")


def split_dataset() -> Dict[str, Any]:
    """
    Executes 70/15/15 stratified train/val/test dataset split.

    Returns:
        Dict containing dataset split metrics and output file locations.
    """
    logger.info(f"Loading processed feature dataset from '{PROCESSED_CSV_PATH}'...")
    if not PROCESSED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset file not found at '{PROCESSED_CSV_PATH}'. "
            "Please run 'python backend/ml/scripts/extract_landmarks.py' first."
        )

    df = pd.read_csv(PROCESSED_CSV_PATH)
    logger.info(f"Loaded dataset containing {len(df):,} samples.")

    # 1. Run quality check assertions before splitting
    validate_features(df, raise_on_error=True)

    # 2. Check if all present classes have at least 2 samples for stratified splitting
    class_counts = df[LABEL_COLUMN].value_counts()
    min_samples = class_counts.min() if len(class_counts) > 0 else 0
    use_stratify = df[LABEL_COLUMN] if min_samples >= 2 else None

    if use_stratify is None:
        logger.warning(
            f"Some classes have fewer than 2 samples (min count: {min_samples}). "
            "Splitting without strict stratification for low-sample classes."
        )

    temp_ratio = VAL_RATIO + TEST_RATIO
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=use_stratify,
        random_state=RANDOM_STATE,
    )

    temp_stratify = temp_df[LABEL_COLUMN] if use_stratify is not None and temp_df[LABEL_COLUMN].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_stratify,
        random_state=RANDOM_STATE,
    )

    # 4. Save split CSVs
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    val_df.to_csv(VAL_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    logger.info(f"Saved Train Set ({len(train_df):,} samples, {len(train_df)/len(df):.1%}) -> '{TRAIN_CSV_PATH}'")
    logger.info(f"Saved Val Set   ({len(val_df):,} samples, {len(val_df)/len(df):.1%}) -> '{VAL_CSV_PATH}'")
    logger.info(f"Saved Test Set  ({len(test_df):,} samples, {len(test_df)/len(df):.1%}) -> '{TEST_CSV_PATH}'")

    # 5. Calculate class distributions per set
    train_counts = train_df[LABEL_COLUMN].value_counts().to_dict()
    val_counts = val_df[LABEL_COLUMN].value_counts().to_dict()
    test_counts = test_df[LABEL_COLUMN].value_counts().to_dict()

    split_report = {
        "random_state": RANDOM_STATE,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "sample_counts": {
            "total": len(df),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "class_distributions": {
            "train": train_counts,
            "validation": val_counts,
            "test": test_counts,
        },
        "data_leakage_prevention": (
            "Samples were randomly shuffled using fixed seed 42 with class stratification. "
            "If video frame sequences are used in future raw datasets, samples from the same "
            "recording session or participant ID must be grouped at the sequence level prior to splitting."
        ),
    }

    SPLIT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_METADATA_PATH, "w") as f:
        json.dump(split_report, f, indent=2)

    # Print Summary Report
    print("\n" + "=" * 50)
    print("STRATIFIED DATASET SPLIT REPORT")
    print("=" * 50)
    print(f"Total Samples: {len(df):,}")
    print(f"Train (70%):   {len(train_df):,} samples")
    print(f"Val   (15%):   {len(val_df):,} samples")
    print(f"Test  (15%):   {len(test_df):,} samples")
    print("=" * 50 + "\n")

    return split_report


if __name__ == "__main__":
    split_dataset()
