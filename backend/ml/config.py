"""
SignSense AI - Machine Learning Configuration & Constants (Phase 3)
"""
import os
from pathlib import Path

# Base Directory Paths
ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
PREPROCESSING_DIR = ML_DIR / "preprocessing"
SCRIPTS_DIR = ML_DIR / "scripts"
MODELS_DIR = ML_DIR / "models"

# Ensure all required directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR, PREPROCESSING_DIR, SCRIPTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset & Class Constants
CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', ..., 'Z']
NUM_CLASSES = len(CLASSES)  # 26
LABEL_COLUMN = "label"

# MediaPipe Hand Landmark Feature Constants
NUM_LANDMARKS = 21
COORDINATES_PER_LANDMARK = 3  # (x, y, z)
FEATURE_DIM = NUM_LANDMARKS * COORDINATES_PER_LANDMARK  # 63

# Feature Column Names: ['x0', 'y0', 'z0', 'x1', 'y1', 'z1', ..., 'x20', 'y20', 'z20']
FEATURE_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ('x', 'y', 'z')]

# Supported Image Formats
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Train / Validation / Test Split Parameters
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Exported Dataset & Metadata File Paths
PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "asl_features.csv"
TRAIN_CSV_PATH = PROCESSED_DATA_DIR / "train_features.csv"
VAL_CSV_PATH = PROCESSED_DATA_DIR / "val_features.csv"
TEST_CSV_PATH = PROCESSED_DATA_DIR / "test_features.csv"

QUALITY_REPORT_PATH = METADATA_DIR / "quality_report.json"
SPLIT_METADATA_PATH = METADATA_DIR / "split_metadata.json"
EDA_REPORT_PATH = METADATA_DIR / "eda_report.json"
PCA_PLOT_PATH = METADATA_DIR / "pca_inspection.png"
