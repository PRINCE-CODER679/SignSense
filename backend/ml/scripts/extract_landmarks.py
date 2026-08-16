"""
SignSense AI - Landmark Extraction & Dataset Processing Script (Phase 3)

Traverses A-Z raw image directories, extracts 21 MediaPipe hand landmarks,
applies wrist origin translation and Euclidean scale normalization, validates features,
exports processed dataset to CSV, and generates an empirical execution summary report.
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Ensure backend and repository root directories are in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
repo_dir = backend_dir.parent
for p in [str(backend_dir), str(repo_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
try:
    from ml.config import (
        RAW_DATA_DIR,
        PROCESSED_CSV_PATH,
        QUALITY_REPORT_PATH,
        CLASSES,
        FEATURE_COLUMNS,
        LABEL_COLUMN,
        SUPPORTED_IMAGE_EXTENSIONS,
    )
    from ml.preprocessing.extractor import LandmarkExtractor
    from ml.preprocessing.normalizer import normalize_landmarks
    from ml.preprocessing.quality_checks import validate_features, validate_sample
except ImportError:
    from backend.ml.config import (
        RAW_DATA_DIR,
        PROCESSED_CSV_PATH,
        QUALITY_REPORT_PATH,
        CLASSES,
        FEATURE_COLUMNS,
        LABEL_COLUMN,
        SUPPORTED_IMAGE_EXTENSIONS,
    )
    from backend.ml.preprocessing.extractor import LandmarkExtractor
    from backend.ml.preprocessing.normalizer import normalize_landmarks
    from backend.ml.preprocessing.quality_checks import validate_features, validate_sample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("LandmarkExtraction")


def process_dataset() -> Dict[str, Any]:
    """
    Main dataset processing routine.

    Returns:
        Dict containing extraction statistics and status summary.
    """
    logger.info("Starting ASL Landmark Feature Extraction Pipeline...")

    total_samples = 0
    valid_samples = 0
    skipped_samples = 0

    class_counts: Dict[str, int] = {c: 0 for c in CLASSES}
    skipped_details: List[Dict[str, str]] = []
    processed_rows: List[Dict[str, Any]] = []

    with LandmarkExtractor(static_image_mode=True, max_num_hands=1) as extractor:
        for letter in CLASSES:
            class_dir = RAW_DATA_DIR / letter
            if not class_dir.exists():
                logger.warning(f"Directory missing for class '{letter}': {class_dir}")
                continue

            # Search supported image files in class subfolder
            image_paths = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]

            logger.info(f"Processing Class '{letter}': {len(image_paths)} images found.")

            for img_path in image_paths:
                total_samples += 1

                # 1. Extract 21 3D landmarks
                success, landmarks_xyz, reason = extractor.extract_from_image_path(img_path)

                if not success or landmarks_xyz is None:
                    skipped_samples += 1
                    skipped_details.append({
                        "file": str(img_path.relative_to(RAW_DATA_DIR)),
                        "class": letter,
                        "reason": reason
                    })
                    continue

                # 2. Normalize 21 landmarks into 63 features
                try:
                    features_63 = normalize_landmarks(landmarks_xyz)
                except Exception as exc:
                    skipped_samples += 1
                    skipped_details.append({
                        "file": str(img_path.relative_to(RAW_DATA_DIR)),
                        "class": letter,
                        "reason": f"NORMALIZATION_ERROR: {str(exc)}"
                    })
                    continue

                # 3. Validate single sample
                is_valid, validation_msg = validate_sample(features_63, letter)
                if not is_valid:
                    skipped_samples += 1
                    skipped_details.append({
                        "file": str(img_path.relative_to(RAW_DATA_DIR)),
                        "class": letter,
                        "reason": validation_msg
                    })
                    continue

                # 4. Construct dataset row
                row_dict = {LABEL_COLUMN: letter}
                for col_name, val in zip(FEATURE_COLUMNS, features_63):
                    row_dict[col_name] = float(val)

                processed_rows.append(row_dict)
                valid_samples += 1
                class_counts[letter] += 1

    logger.info(f"Extraction completed. Valid: {valid_samples}, Skipped: {skipped_samples}")

    if not processed_rows:
        logger.error("No valid samples were extracted. Check raw dataset directory contents.")
        df = pd.DataFrame(columns=[LABEL_COLUMN] + FEATURE_COLUMNS)
    else:
        df = pd.DataFrame(processed_rows)

    # Reorder columns: label, x0, y0, z0, ..., x20, y20, z20
    cols = [LABEL_COLUMN] + FEATURE_COLUMNS
    df = df[cols]

    # Save processed features CSV
    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    logger.info(f"Saved processed feature dataset to '{PROCESSED_CSV_PATH}'")

    # Run dataset level quality assertions if samples exist
    if len(df) > 0:
        quality_summary = validate_features(df, raise_on_error=False)
    else:
        quality_summary = {"status": "EMPTY", "error": "No valid samples found."}

    # Prepare metadata JSON report
    report = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "skipped_samples": skipped_samples,
        "feature_dim": 63,
        "class_counts": class_counts,
        "quality_summary": quality_summary,
        "skipped_details": skipped_details[:50]  # Cap detailed skip log to 50 items
    }

    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # Print Final Execution Summary Report
    print("\n" + "=" * 50)
    print("LANDMARK FEATURE EXTRACTION REPORT")
    print("=" * 50)
    print(f"Total samples:   {total_samples:,}")
    print(f"Valid samples:   {valid_samples:,}")
    print(f"Skipped samples: {skipped_samples:,}")
    print("-" * 50)

    for letter in CLASSES:
        count = class_counts.get(letter, 0)
        print(f"{letter}: {count}")

    print("=" * 50 + "\n")

    return report


if __name__ == "__main__":
    process_dataset()
