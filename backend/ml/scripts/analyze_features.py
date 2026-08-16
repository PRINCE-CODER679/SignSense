"""
SignSense AI - Feature Analysis & Exploratory Data Inspection (Phase 3)

Analyzes processed landmark feature vectors:
- Feature dimensions & feature coordinate stats (min, max, mean, std)
- Class balance metrics & class counts
- Performs 2D PCA dimensionality reduction for visual inspection & inspection plot saving

Note: PCA is for visual analysis ONLY and is NOT part of the production inference pipeline.
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

import numpy as np
import pandas as pd
try:
    from ml.config import (
        PROCESSED_CSV_PATH,
        QUALITY_REPORT_PATH,
        EDA_REPORT_PATH,
        PCA_PLOT_PATH,
        LABEL_COLUMN,
        FEATURE_COLUMNS,
        FEATURE_DIM,
        CLASSES,
    )
except ImportError:
    from backend.ml.config import (
        PROCESSED_CSV_PATH,
        QUALITY_REPORT_PATH,
        EDA_REPORT_PATH,
        PCA_PLOT_PATH,
        LABEL_COLUMN,
        FEATURE_COLUMNS,
        FEATURE_DIM,
        CLASSES,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("FeatureAnalysis")


def analyze_features() -> Dict[str, Any]:
    """
    Runs exploratory feature inspection and exports statistical EDA report.

    Returns:
        Dict containing statistical metrics and EDA report.
    """
    logger.info(f"Loading processed feature dataset from '{PROCESSED_CSV_PATH}'...")
    if not PROCESSED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset file not found at '{PROCESSED_CSV_PATH}'. "
            "Please run 'python backend/ml/scripts/extract_landmarks.py' first."
        )

    df = pd.read_csv(PROCESSED_CSV_PATH)
    total_samples = len(df)
    logger.info(f"Loaded dataset containing {total_samples:,} samples.")

    # 1. Feature dimensions & column validation
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    # 2. Coordinate range statistics across normalized 63 features
    feature_min = float(np.min(features))
    feature_max = float(np.max(features))
    feature_mean = float(np.mean(features))
    feature_std = float(np.std(features))

    # Wrist origin assertion check: x0, y0, z0 should be identically 0.0
    wrist_coords = features[:, 0:3]
    wrist_max_err = float(np.max(np.abs(wrist_coords)))

    # 3. Class distribution & balance analysis
    class_counts = df[LABEL_COLUMN].value_counts().to_dict()
    min_class_count = min(class_counts.values()) if class_counts else 0
    max_class_count = max(class_counts.values()) if class_counts else 0
    balance_ratio = (min_class_count / max_class_count) if max_class_count > 0 else 0.0

    # 4. Load quality report if available for valid/invalid stats
    quality_data = {}
    if QUALITY_REPORT_PATH.exists():
        with open(QUALITY_REPORT_PATH, "r") as f:
            quality_data = json.load(f)

    # 5. Perform 2D PCA for visual inspection (if matplotlib/sklearn available)
    pca_stats = {}
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(features)
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        pca_stats = {
            "explained_variance_ratio": explained_variance_ratio,
            "total_explained_variance": float(sum(explained_variance_ratio))
        }

        # Save plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            classes_present = df[LABEL_COLUMN].unique()
            for label in sorted(classes_present):
                mask = (df[LABEL_COLUMN] == label)
                plt.scatter(
                    pca_features[mask, 0],
                    pca_features[mask, 1],
                    label=label,
                    alpha=0.6,
                    s=15
                )

            plt.title("SignSense AI - ASL Landmark Feature 2D PCA Projection (Inspection Only)")
            plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
            plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            PCA_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(PCA_PLOT_PATH, dpi=150)
            plt.close()
            logger.info(f"Saved PCA inspection plot to '{PCA_PLOT_PATH}'")
        except Exception as plot_err:
            logger.warning(f"Could not save PCA plot image: {plot_err}")

    except Exception as pca_err:
        logger.warning(f"PCA computation skipped: {pca_err}")

    # Compile EDA report
    eda_report = {
        "dataset_path": str(PROCESSED_CSV_PATH),
        "total_samples": total_samples,
        "valid_samples": quality_data.get("valid_samples", total_samples),
        "skipped_samples": quality_data.get("skipped_samples", 0),
        "feature_dimensions": FEATURE_DIM,
        "wrist_origin_error_max": wrist_max_err,
        "coordinate_stats": {
            "min": feature_min,
            "max": feature_max,
            "mean": feature_mean,
            "std": feature_std,
        },
        "dataset_balance": {
            "unique_classes": len(class_counts),
            "min_class_count": min_class_count,
            "max_class_count": max_class_count,
            "balance_ratio": round(balance_ratio, 4),
        },
        "pca_analysis": pca_stats,
        "class_distribution": class_counts
    }

    EDA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_REPORT_PATH, "w") as f:
        json.dump(eda_report, f, indent=2)

    # Print Summary Console Report
    print("\n" + "=" * 50)
    print("ASL FEATURE ANALYSIS & INSPECTION REPORT")
    print("=" * 50)
    print(f"Total Samples:            {total_samples:,}")
    print(f"Feature Dimensions:       {FEATURE_DIM}")
    print(f"Wrist Origin Error (Max): {wrist_max_err:.6e}")
    print(f"Normalized Coordinate Range: [{feature_min:.4f}, {feature_max:.4f}]")
    print(f"Mean ± Std:               {feature_mean:.4f} ± {feature_std:.4f}")
    print(f"Class Balance Ratio:      {balance_ratio:.2%}")
    if pca_stats:
        print(f"PCA Total Variance (2D):  {pca_stats['total_explained_variance']:.1%}")
    print("=" * 50 + "\n")

    return eda_report


if __name__ == "__main__":
    analyze_features()
