#!/usr/bin/env python3
"""
Tree species classification pipeline - simplified and fast.

Usage:
    python main.py                    # Full pipeline
    python main.py --skip-training    # Load saved models
    python main.py --no-plots         # Skip plots
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from config import DATA_PATH, MODELS_DIR, PLOTS_DIR, RANDOM_STATE, SCORING_METRIC
from data_loader import load_data, print_data_report
from preprocessing import DataPreprocessor, create_train_test_split
from models import ModelTrainer, LIGHTGBM_AVAILABLE
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Tree Species Classification')
    parser.add_argument('--skip-training', '-s', action='store_true')
    parser.add_argument('--data-path', '-d', type=str, default=None)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-lightgbm', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n🌲 TREE SPECIES CLASSIFICATION")
    print("="*50)
    
    # Load data
    data_path = args.data_path or DATA_PATH
    df = load_data(data_path)
    print_data_report(df)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    class_weights = preprocessor.get_class_weights(y)
    label_mapping = preprocessor.get_label_mapping()
    
    # Split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    feature_names = list(X_train.columns)
    
    # Train
    trainer = ModelTrainer(class_weights=class_weights)
    
    if args.skip_training:
        print("\n⏭️  Loading saved models...")
        trainer.load_models()
    else:
        include_lgb = not args.no_lightgbm and LIGHTGBM_AVAILABLE
        trainer.train_all_models(X_train, y_train, include_lightgbm=include_lgb)
        trainer.save_models()
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    evaluator = ModelEvaluator(label_mapping=label_mapping)
    
    for model_name in trainer.models.keys():
        y_pred = trainer.predict(model_name, X_test)
        y_proba = trainer.predict_proba(model_name, X_test)
        evaluator.evaluate_model(model_name, y_test, y_pred, y_proba)
    
    # Results
    evaluator.print_comparison_summary()
    
    # Plots
    if not args.no_plots:
        print("\n📈 Generating plots...")
        evaluator.plot_all_confusion_matrices(normalize=True)
        evaluator.plot_metrics_comparison()
        evaluator.plot_roc_curves()
        evaluator.plot_precision_recall_curves()
        
        best_model = evaluator.get_best_model()
        importance_df = trainer.get_feature_importance(best_model, feature_names)
        evaluator.plot_feature_importance(importance_df, best_model)
    
    # Summary
    best_model = evaluator.get_best_model()
    best_metrics = evaluator.results[best_model]['metrics']
    
    print(f"\n🏆 Best: {best_model.upper()} (F1={best_metrics['f1_macro']:.3f})")
    print(f"📁 Models: {MODELS_DIR}")
    print(f"📁 Plots: {PLOTS_DIR}")
    print("\n✅ Done!")
    
    return evaluator.get_comparison_table(), best_model


if __name__ == "__main__":
    results, best = main()
