"""
Configuration settings for the tree species classification pipeline.

Simplified with fixed hyperparameters (no grid search / Optuna).
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "dataset_base.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

# Data settings
TARGET_COLUMN = "species"
ID_COLUMN = "tree_id"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Class mapping (for encoding)
CLASS_NAMES = ["birch", "cedar", "fir", "larch", "pine", "spruce"]
N_CLASSES = len(CLASS_NAMES)

# Preprocessing settings
CORRELATION_THRESHOLD = 0.90  # Remove features with correlation > this
OUTLIER_IQR_MULTIPLIER = 1.5  # For outlier detection (not removal for tree-based)

# Feature selection settings
MIN_VARIANCE_THRESHOLD = 0.01  # Drop near-constant features
USE_FEATURE_SELECTION = True  # Enable/disable automatic feature selection

# Features to drop based on domain knowledge and correlation analysis
FEATURES_TO_DROP = [
    # Highly correlated height percentiles (keep p99, p50, p25)
    'height_p95', 'height_p90', 'height_p75',
    # Redundant intensity percentiles (keep p50, p75, p95)
    'intensity_p25',
    # Redundant range features (x_range and y_range are captured in crown_area_2d)
    'x_range', 'y_range',
    # Intensity range is redundant with max-min
    'intensity_range',
]

# Cross-validation settings
N_SPLITS = 3  # Reduced for speed
SCORING_METRIC = "f1_macro"

# ============================================================================
# FIXED HYPERPARAMETERS - FAST TRAINING
# ============================================================================

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.15,
}

CATBOOST_PARAMS = {
    'iterations': 100,
    'depth': 5,
    'learning_rate': 0.15,
}

BALANCED_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 12,
}

LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.15,
    'num_leaves': 31,
}

