"""
Model definitions and training module for tree species classification.

Simplified approach with fixed hyperparameters - no grid search or Optuna.
Fast and cheap to train.

Models:
1. XGBoost - Excellent performance with scale_pos_weight for imbalance
2. CatBoost - Often achieves highest balanced accuracy
3. BalancedRandomForest - Specifically designed for imbalanced data
4. LightGBM - Fast and efficient gradient boosting
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import joblib

from config import (
    RANDOM_STATE, N_SPLITS, SCORING_METRIC, MODELS_DIR,
    XGBOOST_PARAMS, CATBOOST_PARAMS, BALANCED_RF_PARAMS, LIGHTGBM_PARAMS,
    EARLY_STOPPING_ROUNDS
)

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")


class ModelTrainer:
    """
    Handles model training with fixed hyperparameters.
    Simple and fast - no hyperparameter search.
    """
    
    def __init__(self, class_weights: Dict[int, float] = None):
        """
        Initialize the model trainer.
        
        Args:
            class_weights: Dictionary mapping class indices to weights.
        """
        self.class_weights = class_weights
        self.models = {}
        self.cv_scores = {}
        
        # Set up cross-validation strategy
        self.cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    def _get_xgboost_model(self):
        """Create XGBoost classifier with fixed params."""
        from xgboost import XGBClassifier
        
        return XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **XGBOOST_PARAMS
        )
    
    def _get_catboost_model(self):
        """Create CatBoost classifier with fixed params."""
        from catboost import CatBoostClassifier
        
        return CatBoostClassifier(
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=-1,
            **CATBOOST_PARAMS
        )
    
    def _get_balanced_rf_model(self):
        """Create BalancedRandomForest classifier with fixed params."""
        from imblearn.ensemble import BalancedRandomForestClassifier
        
        return BalancedRandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            sampling_strategy='auto',
            replacement=True,
            **BALANCED_RF_PARAMS
        )
    
    def _get_lightgbm_model(self):
        """Create LightGBM classifier with fixed params."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")
        
        return lgb.LGBMClassifier(
            objective='multiclass',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
            class_weight='balanced',
            **LIGHTGBM_PARAMS
        )
    
    def _quick_cv_score(self, model, X_train, y_train, sample_weights=None) -> float:
        """Quick single-fold validation for speed."""
        from sklearn.model_selection import train_test_split
        
        # Single 80/20 split instead of full CV
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
        
        if sample_weights is not None:
            # Get corresponding weights
            train_indices = X_tr.index if hasattr(X_tr, 'index') else range(len(X_tr))
            sw_tr = np.array([sample_weights[i] for i in range(len(y_tr))])
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average='macro')
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """Train XGBoost with early stopping."""
        from sklearn.model_selection import train_test_split
        print("🚀 Training XGBoost...", end=" ", flush=True)
        
        sample_weights = None
        if self.class_weights:
            sample_weights = np.array([self.class_weights[y] for y in y_train])
        
        # Split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
        )
        
        model = self._get_xgboost_model()
        
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'verbose': False,
        }
        
        if sample_weights is not None:
            sw_tr = np.array([self.class_weights[y] for y in y_tr])
            fit_params['sample_weight'] = sw_tr
        
        model.fit(X_tr, y_tr, **fit_params)
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['xgboost'] = model
        self.cv_scores['xgboost'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """Train CatBoost with early stopping."""
        from sklearn.model_selection import train_test_split
        from catboost import Pool
        print("🚀 Training CatBoost...", end=" ", flush=True)
        
        # Split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
        )
        
        model = self._get_catboost_model()
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['catboost'] = model
        self.cv_scores['catboost'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_balanced_rf(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """Train BalancedRandomForest (no early stopping for RF)."""
        from sklearn.model_selection import train_test_split
        print("🚀 Training BalancedRF...", end=" ", flush=True)
        
        # Split for validation score
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
        )
        
        model = self._get_balanced_rf_model()
        model.fit(X_tr, y_tr)
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['balanced_rf'] = model
        self.cv_scores['balanced_rf'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """Train LightGBM with early stopping."""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available, skipping...")
            return None
        
        from sklearn.model_selection import train_test_split
        print("🚀 Training LightGBM...", end=" ", flush=True)
        
        # Split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
        )
        
        model = self._get_lightgbm_model()
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
        )
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['lightgbm'] = model
        self.cv_scores['lightgbm'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        include_lightgbm: bool = True,
        include_stacking: bool = True
    ) -> Dict[str, Any]:
        """Train all models with early stopping."""
        print("\n🎯 Training models...")
        
        self.train_xgboost(X_train, y_train)
        self.train_catboost(X_train, y_train)
        self.train_balanced_rf(X_train, y_train)
        
        if include_lightgbm and LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train)
        
        if include_stacking:
            self.train_stacking_ensemble(X_train, y_train, include_lightgbm)
        
        return self.models
    
    def train_stacking_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        include_lightgbm: bool = True
    ) -> Any:
        """Train stacking ensemble with base models."""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        
        print("🚀 Training Stacking Ensemble...", end=" ", flush=True)
        
        # Build estimators list
        estimators = [
            ('xgboost', self._get_xgboost_model()),
            ('catboost', self._get_catboost_model()),
            ('balanced_rf', self._get_balanced_rf_model()),
        ]
        
        if include_lightgbm and LIGHTGBM_AVAILABLE:
            estimators.append(('lightgbm', self._get_lightgbm_model()))
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=RANDOM_STATE
            ),
            cv=5,
            stack_method='predict_proba',
            passthrough=False,
            n_jobs=-1,
        )
        
        # Split for validation score
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
        )
        
        stacking.fit(X_tr, y_tr)
        
        # Validation score
        y_pred = stacking.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['stacking'] = stacking
        self.cv_scores['stacking'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return stacking
    
    def get_cv_scores_summary(self) -> pd.DataFrame:
        """Get summary of cross-validation scores for all models."""
        summary_data = []
        
        for model_name, score in self.cv_scores.items():
            summary_data.append({
                'Model': model_name,
                f'CV {SCORING_METRIC}': score,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(f'CV {SCORING_METRIC}', ascending=False)
        
        return summary_df
    
    def save_models(self, prefix: str = "") -> None:
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            filename = MODELS_DIR / f"{prefix}{model_name}_model.joblib"
            joblib.dump(model, filename)
            print(f"✅ Saved {model_name} to {filename}")
    
    def load_models(self, prefix: str = "") -> Dict[str, Any]:
        """Load all models from disk."""
        model_names = ['xgboost', 'catboost', 'balanced_rf', 'lightgbm', 'stacking']
        
        for model_name in model_names:
            filename = MODELS_DIR / f"{prefix}{model_name}_model.joblib"
            if filename.exists():
                self.models[model_name] = joblib.load(filename)
                print(f"✅ Loaded {model_name} from {filename}")
        
        return self.models
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        return self.models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name: str, feature_names: list) -> pd.DataFrame:
        """Get feature importance from a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            raise ValueError(f"Model '{model_name}' doesn't support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    # Test the module
    from data_loader import load_data
    from preprocessing import DataPreprocessor, create_train_test_split
    
    # Load and preprocess
    df = load_data()
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    class_weights = preprocessor.get_class_weights(y)
    
    # Split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Train models
    trainer = ModelTrainer(class_weights=class_weights)
    trainer.train_all_models(X_train, y_train)
    
    # Show CV scores
    print("\n" + "="*60)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*60)
    print(trainer.get_cv_scores_summary().to_string(index=False))
