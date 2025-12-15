#!/usr/bin/env python3
"""
Tree species classification pipeline - single "best model" path.

Usage:
    python main.py                    # Train, select best, refit, save one artifact
    python main.py --no-plots         # Skip plots
    python main.py --data-path PATH   # Custom dataset path
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MODELS_DIR, PLOTS_DIR, RANDOM_STATE
from config import USE_PERMUTATION_FEATURE_SELECTION, PERM_IMPORTANCE_THRESHOLD
from config import USE_ADASYN, TUNE_XGBOOST, TUNE_XGBOOST_N_ITER
from data_loader import load_data, print_data_report
from preprocessing import (
    DataPreprocessor, create_train_test_split, 
    apply_adasyn, select_features_permutation, create_train_val_test_split
)
from models import ModelTrainer, LIGHTGBM_AVAILABLE
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Tree Species Classification')
    parser.add_argument('--data-path', '-d', type=str, default=None,
                       help='Path to dataset CSV')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n🌲 TREE SPECIES CLASSIFICATION")
    print("="*50)
    print("   ✓ Single-path training: ADASYN → feature selection → model selection → refit → save best_model.joblib")
    
    # Load data
    data_path = args.data_path or DATA_PATH
    df = load_data(data_path)
    print_data_report(df)
    
    # ---------------------------------------------------------------------
    # Proper split BEFORE fitting preprocessing to avoid leakage
    # ---------------------------------------------------------------------
    df_train, df_val, df_test = create_train_val_test_split(df)

    # Fit preprocessing ONLY on train, then transform val/test
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(df_train)
    X_val, y_val = preprocessor.transform(df_val)
    X_test, y_test = preprocessor.transform(df_test)
    label_mapping = preprocessor.get_label_mapping()

    # ---------------------------------------------------------------------
    # Feature selection WITHOUT leaking into model-selection validation
    # We carve out a feature-selection split from training only.
    # ---------------------------------------------------------------------
    selected_features = list(X_train.columns)
    if USE_PERMUTATION_FEATURE_SELECTION:
        X_fs_fit, X_fs_val, y_fs_fit, y_fs_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
        print(f"\n📊 Feature-selection split (train only):")
        print(f"   FS-Fit: {len(y_fs_fit)}, FS-Val: {len(y_fs_val)}")

        fs_class_weights = preprocessor.get_class_weights(y_fs_fit)
        fs_trainer = ModelTrainer(class_weights=fs_class_weights)
        fs_trainer.train_catboost(X_fs_fit, y_fs_fit, eval_set=(X_fs_val, y_fs_val))
        selected_features = select_features_permutation(
            fs_trainer.models['catboost'], X_fs_val, y_fs_val, threshold=PERM_IMPORTANCE_THRESHOLD
        )

        if len(selected_features) < X_train.shape[1]:
            print(f"\n✅ Using {len(selected_features)} selected features (from {X_train.shape[1]})")
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]
        else:
            selected_features = list(X_train.columns)
            print(f"\n✅ Feature selection kept all {len(selected_features)} features")
    else:
        print("\nℹ️  Skipping permutation feature selection")

    # ---------------------------------------------------------------------
    # Imbalance handling: optionally apply ADASYN ONLY on training (never on val/test)
    # ---------------------------------------------------------------------
    if USE_ADASYN:
        X_train, y_train = apply_adasyn(X_train, y_train, target_classes=[1, 2])
    else:
        print("\nℹ️  Skipping ADASYN (using class weights instead)")

    # Compute class weights on (possibly) resampled training labels
    class_weights = preprocessor.get_class_weights(y_train)

    # Train candidate models on X_train, select best on X_val.
    include_lgb = LIGHTGBM_AVAILABLE
    trainer = ModelTrainer(class_weights=class_weights)

    # Optional: tune XGBoost using CV on TRAIN ONLY (keeps held-out val clean)
    xgb_params_override = None
    if TUNE_XGBOOST:
        xgb_params_override = trainer.tune_xgboost_cv(
            X_train, y_train, n_iter=TUNE_XGBOOST_N_ITER, cv_splits=3
        )
    trainer.train_all_models(
        X_train, y_train,
        include_lightgbm=include_lgb,
        include_stacking=True,
        eval_set=(X_val, y_val),
        xgb_params_override=xgb_params_override,
    )

    # Choose best by validation F1 (on a clean, untouched validation set)
    val_scores = {}
    for name, model in trainer.models.items():
        y_val_pred = model.predict(X_val)
        val_scores[name] = float(
            ModelEvaluator(label_mapping=label_mapping).evaluate_model(
                name, y_val, y_val_pred, None
            )['f1_macro']
        )
    best_model_name = max(val_scores.keys(), key=lambda k: val_scores[k])
    print(f"\n🏆 Selected best model (by val F1): {best_model_name} (F1={val_scores[best_model_name]:.4f})")

    # Keep the selected model trained on TRAIN ONLY (val is a true holdout for evaluation/tuning).
    final_model = trainer.models[best_model_name]

    # Cleanup + save ONLY the best model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "best_model.joblib"

    # Remove old model artifacts to keep the folder clean
    for p in MODELS_DIR.glob("*_model.joblib"):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        (MODELS_DIR / "best_model.joblib").unlink()
    except Exception:
        pass

    import joblib
    bundle = {
        "model_name": best_model_name,
        "model": final_model,
        "preprocessor": preprocessor,
        "selected_features": selected_features,
        "label_mapping": label_mapping,
        "model_params": {
            "xgboost_params_override": xgb_params_override if best_model_name == "xgboost" else None,
        },
        "postprocess": {
            "thresholds": None,
            "use_thresholds": False,
        },
    }
    joblib.dump(bundle, best_path)
    print(f"\n✅ Saved best model bundle to {best_path}")

    # Evaluate on validation + test
    print("\n📊 Evaluating best model on validation + test set...")
    evaluator = ModelEvaluator(label_mapping=label_mapping)

    y_val_pred = final_model.predict(X_val)
    y_val_proba = final_model.predict_proba(X_val)
    evaluator.evaluate_model(best_model_name + "_val", y_val, y_val_pred, y_val_proba)

    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)
    evaluator.evaluate_model(best_model_name, y_test, y_test_pred, y_test_proba)
    evaluator.print_classification_report(best_model_name)
    evaluator.print_comparison_summary()

    # Threshold optimization: tune on val, evaluate on test (can improve macro F1 for rare classes)
    try:
        thresh_result = evaluator.evaluate_with_threshold_optimization(
            best_model_name,
            y_val=y_val,
            y_val_proba=y_val_proba,
            y_test=y_test,
            y_test_proba=y_test_proba,
        )
        bundle["postprocess"]["thresholds"] = thresh_result["thresholds"]
        bundle["postprocess"]["use_thresholds"] = True
        joblib.dump(bundle, best_path)  # update saved bundle with thresholds
    except Exception as e:
        print(f"\n   ⚠️  Threshold optimization skipped: {e}")
    
    # Plots
    if not args.no_plots:
        print("\n📈 Generating plots...")
        evaluator.plot_confusion_matrix(best_model_name, normalize=True)
        evaluator.plot_precision_recall_curves()
        evaluator.plot_roc_curves()

        # Feature importance only if supported
        try:
            tmp_trainer = ModelTrainer()
            tmp_trainer.models[best_model_name] = final_model
            importance_df = tmp_trainer.get_feature_importance(best_model_name, selected_features)
            evaluator.plot_feature_importance(importance_df, best_model_name)
        except Exception as e:
            print(f"   ⚠️  Skipping feature importance plot: {e}")
    
    # Summary
    best_metrics = evaluator.results[best_model_name]['metrics']
    
    print(f"\n🏆 Best: {best_model_name.upper()} (F1={best_metrics['f1_macro']:.3f})")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"📁 Plots: {PLOTS_DIR}")
    print("\n✅ Done!")
    
    return best_model_name, best_metrics


if __name__ == "__main__":
    best_name, best_metrics = main()
