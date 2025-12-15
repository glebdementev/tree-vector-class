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
VAL_SIZE = 0.2  # Held-out validation split (used for model selection / early stopping)

# Class mapping (for encoding)
CLASS_NAMES = ["birch", "cedar", "fir", "larch", "pine", "spruce"]
N_CLASSES = len(CLASS_NAMES)

# Preprocessing settings
CORRELATION_THRESHOLD = 0.90  # Remove features with correlation > this
OUTLIER_IQR_MULTIPLIER = 1.5  # For outlier detection (not removal for tree-based)
REMOVE_CORRELATED_FEATURES = True  # Keep the feature space smaller/less redundant for stability

# Feature selection settings
MIN_VARIANCE_THRESHOLD = 0.01  # Drop near-constant features
USE_FEATURE_SELECTION = True  # Enable/disable automatic feature selection
USE_PERMUTATION_FEATURE_SELECTION = True
PERM_IMPORTANCE_THRESHOLD = 0.0  # conservative: keep features with positive permutation importance

# Imbalance handling
USE_ADASYN = True

# Optional training-time tuning (kept off by default to avoid overfitting the split)
TUNE_XGBOOST = False
TUNE_XGBOOST_N_ITER = 12

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
N_SPLITS = 5  # Increased for better generalization estimates
SCORING_METRIC = "f1_macro"

# Early stopping settings
EARLY_STOPPING_ROUNDS = 50

# ============================================================================
# OPTIMIZED HYPERPARAMETERS - WITH REGULARIZATION
# ============================================================================

XGBOOST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'border_count': 128,
}

BALANCED_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}

LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

