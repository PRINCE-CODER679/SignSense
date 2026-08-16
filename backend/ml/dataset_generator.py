"""
SignSense AI - Landmark Dataset Generator & Preprocessing Pipeline Entrypoint (Phase 3)

Runs sample dataset rendering, MediaPipe landmark extraction, normalization,
stratified train/val/test dataset splitting, and exploratory data analysis.
"""
import sys
import logging
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from ml.scripts.generate_sample_dataset import generate_sample_dataset
    from ml.scripts.extract_landmarks import process_dataset
    from ml.scripts.split_dataset import split_dataset
    from ml.scripts.analyze_features import analyze_features
except ImportError:
    from backend.ml.scripts.generate_sample_dataset import generate_sample_dataset
    from backend.ml.scripts.extract_landmarks import process_dataset
    from backend.ml.scripts.split_dataset import split_dataset
    from backend.ml.scripts.analyze_features import analyze_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DatasetGenerator")


def run_full_pipeline(samples_per_class: int = 15):
    """
    Executes Phase 3 end-to-end dataset generation and preprocessing:
    1. Generates raw synthetic sample hand images for classes A-Z
    2. Extracts 21 MediaPipe hand landmarks and normalizes them into 63 features
    3. Splits dataset stratify-wise (70/15/15) into train/val/test CSVs
    4. Runs feature analysis and exports quality/EDA metadata reports
    """
    print("\n" + "=" * 65)
    print(" SIGNSENSE AI - PHASE 3 DATASET & PREPROCESSING PIPELINE ")
    print("=" * 65 + "\n")

    # Step 1: Generate sample dataset
    logger.info("Step 1: Generating sample raw dataset...")
    generate_sample_dataset(samples_per_class=samples_per_class)

    # Step 2: Extract & normalize MediaPipe landmarks
    logger.info("Step 2: Extracting MediaPipe landmarks and normalizing features...")
    extract_report = process_dataset()

    # Step 3: Stratified dataset split
    logger.info("Step 3: Performing stratified Train / Val / Test split...")
    split_report = split_dataset()

    # Step 4: Feature analysis
    logger.info("Step 4: Running feature statistical analysis...")
    eda_report = analyze_features()

    print("\n" + "=" * 65)
    print(" PHASE 3 PIPELINE EXECUTION COMPLETE ")
    print(f" Valid Samples Processed: {extract_report.get('valid_samples', 0)}")
    print(f" Train / Val / Test Split: {split_report.get('train_samples', 0)} / {split_report.get('val_samples', 0)} / {split_report.get('test_samples', 0)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignSense AI Dataset Generator Pipeline")
    parser.add_argument("--samples", type=int, default=15, help="Number of synthetic samples per class")
    args = parser.parse_args()

    run_full_pipeline(samples_per_class=args.samples)

