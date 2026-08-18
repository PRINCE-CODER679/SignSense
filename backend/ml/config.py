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
RAW_COORDINATE_DIM = NUM_LANDMARKS * COORDINATES_PER_LANDMARK  # 63

# Geometric Invariant Feature Names (49 features)
GEOMETRIC_FEATURE_COLUMNS = [
    # Fingertip to thumb tip distances (4)
    "dist_t_i", "dist_t_m", "dist_t_r", "dist_t_p",
    # Adjacent fingertip distances (3)
    "dist_i_m", "dist_m_r", "dist_r_p",
    # Fingertip to wrist distances (5)
    "dist_w_t", "dist_w_i", "dist_w_m", "dist_w_r", "dist_w_p",
    # Thumb tip to MCP knuckle distances (4)
    "dist_t_mcp_i", "dist_t_mcp_m", "dist_t_mcp_r", "dist_t_mcp_p",
    # Fingertip to own MCP knuckle distances (5)
    "dist_mcp_t", "dist_mcp_i", "dist_mcp_m", "dist_mcp_r", "dist_mcp_p",
    # Joint flex angles (15)
    "angle_thumb_1", "angle_thumb_2", "angle_thumb_mcp",
    "angle_index_1", "angle_index_2", "angle_index_3",
    "angle_middle_1", "angle_middle_2", "angle_middle_3",
    "angle_ring_1", "angle_ring_2", "angle_ring_3",
    "angle_pinky_1", "angle_pinky_2", "angle_pinky_3",
    # Inter-finger spread angles (4)
    "spread_t_i", "spread_i_m", "spread_m_r", "spread_r_p",
    # Thumb tip to PIP joint distances for fist letter disambiguation (M, N, T, E, S, A) (4)
    "dist_t_pip_i", "dist_t_pip_m", "dist_t_pip_r", "dist_t_pip_p",
    # Thumb tip to DIP joint distances (4)
    "dist_t_dip_i", "dist_t_dip_m", "dist_t_dip_r", "dist_t_dip_p",
    # Thumb palm depth relative to palm plane (1)
    "thumb_palm_depth",
    # Targeted O vs C gap ratio (1)
    "gap_o_c",
    # Targeted M vs N PIP differential distance (1)
    "m_n_pip_diff",
    # Thumb to Index PIP cross projection (1)
    "thumb_index_pip_cross"
]

NUM_GEOMETRIC_FEATURES = len(GEOMETRIC_FEATURE_COLUMNS)  # 52
FEATURE_DIM = RAW_COORDINATE_DIM + NUM_GEOMETRIC_FEATURES  # 115

# Raw coordinate feature column names ['x0', 'y0', 'z0', ..., 'x20', 'y20', 'z20']
RAW_FEATURE_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ('x', 'y', 'z')]

# Full Feature Column Names (115 features)
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + GEOMETRIC_FEATURE_COLUMNS

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
