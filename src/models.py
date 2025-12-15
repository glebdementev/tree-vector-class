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
from sklearn.model_selection import ParameterSampler

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
    
    def _get_xgboost_model(self, params_override: Optional[Dict[str, Any]] = None):
        """Create XGBoost classifier with fixed params (optionally overridden)."""
        from xgboost import XGBClassifier

        params = dict(XGBOOST_PARAMS)
        if params_override:
            params.update(params_override)

        return XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params
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
    
    def train_xgboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
        params_override: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Train XGBoost with early stopping (optionally using a provided eval_set)."""
        from sklearn.model_selection import train_test_split
        print("🚀 Training XGBoost...", end=" ", flush=True)
        
        sample_weights = None
        if self.class_weights:
            sample_weights = np.array([self.class_weights[y] for y in y_train])
        
        if eval_set is None:
            # Split for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = eval_set
        
        model = self._get_xgboost_model(params_override=params_override)
        
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'verbose': False,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
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

    def tune_xgboost_cv(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_iter: int = 15,
        cv_splits: int = 3,
    ) -> Dict[str, Any]:
        """
        Lightweight random search for XGBoost params using CV on TRAIN ONLY.

        This avoids tuning directly on the held-out validation set, reducing the
        risk of an overly-optimistic val score vs test.
        """
        print(f"\n🧪 Tuning XGBoost (random search, {cv_splits}-fold CV, n_iter={n_iter})...")

        # Small, safe search space for generalization
        param_dist = {
            "max_depth": [3, 4, 5, 6, 7],
            "min_child_weight": [1, 2, 3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.07, 0.1],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.05, 0.1, 0.3],
            "reg_lambda": [0.8, 1.0, 2.0, 5.0],
            "gamma": [0.0, 0.25, 0.5, 1.0],
            # Let early stopping decide effective number of trees; keep ceiling high enough
            "n_estimators": [800, 1200, 1600],
        }

        sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

        best_score = -1.0
        best_params: Dict[str, Any] = {}

        for i, params in enumerate(sampler, start=1):
            fold_scores = []
            for tr_idx, va_idx in cv.split(X_train, y_train):
                X_tr = X_train.iloc[tr_idx] if hasattr(X_train, "iloc") else X_train[tr_idx]
                y_tr = y_train[tr_idx]
                X_va = X_train.iloc[va_idx] if hasattr(X_train, "iloc") else X_train[va_idx]
                y_va = y_train[va_idx]

                model = self._get_xgboost_model(params_override=params)

                fit_params = {
                    "eval_set": [(X_va, y_va)],
                    "verbose": False,
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                }

                if self.class_weights:
                    sw_tr = np.array([self.class_weights[y] for y in y_tr])
                    fit_params["sample_weight"] = sw_tr

                model.fit(X_tr, y_tr, **fit_params)
                y_pred = model.predict(X_va)
                fold_scores.append(f1_score(y_va, y_pred, average="macro", zero_division=0))

            mean_score = float(np.mean(fold_scores)) if fold_scores else -1.0
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

            print(f"   {i:02d}/{len(sampler)} mean_f1_macro={mean_score:.4f} best={best_score:.4f}", end="\r")

        print(" " * 60, end="\r")
        print(f"✅ Best XGBoost CV macro F1: {best_score:.4f}")
        print(f"   Best params: {best_params}")
        return best_params
    
    def train_catboost(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    ) -> Any:
        """Train CatBoost with early stopping (optionally using a provided eval_set)."""
        from sklearn.model_selection import train_test_split
        print("🚀 Training CatBoost...", end=" ", flush=True)
        
        if eval_set is None:
            # Split for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = eval_set
        
        model = self._get_catboost_model()
        
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            verbose=False
        )
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['catboost'] = model
        self.cv_scores['catboost'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_balanced_rf(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    ) -> Any:
        """Train BalancedRandomForest (no early stopping)."""
        from sklearn.model_selection import train_test_split
        print("🚀 Training BalancedRF...", end=" ", flush=True)
        
        if eval_set is None:
            # Split for validation score
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = eval_set
        
        model = self._get_balanced_rf_model()
        model.fit(X_tr, y_tr)
        
        # Validation score
        y_pred = model.predict(X_val)
        val_score = f1_score(y_val, y_pred, average='macro')
        
        self.models['balanced_rf'] = model
        self.cv_scores['balanced_rf'] = val_score
        
        print(f"Done! Val Score: {val_score:.4f}")
        return model
    
    def train_lightgbm(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    ) -> Any:
        """Train LightGBM with early stopping (optionally using a provided eval_set)."""
        if not LIGHTGBM_AVAILABLE:
            print("⚠️  LightGBM not available, skipping...")
            return None
        
        from sklearn.model_selection import train_test_split
        print("🚀 Training LightGBM...", end=" ", flush=True)
        
        if eval_set is None:
            # Split for early stopping
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
            )
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = eval_set
        
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
        include_stacking: bool = True,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
        xgb_params_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train all models with early stopping."""
        print("\n🎯 Training models...")
        
        self.train_xgboost(X_train, y_train, eval_set=eval_set, params_override=xgb_params_override)
        self.train_catboost(X_train, y_train, eval_set=eval_set)
        self.train_balanced_rf(X_train, y_train, eval_set=eval_set)
        
        if include_lightgbm and LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train, eval_set=eval_set)
        
        if include_stacking:
            self.train_stacking_ensemble(X_train, y_train, include_lightgbm, eval_set=eval_set)
        
        return self.models
    
    def train_stacking_ensemble(
        self, 
        X_train: pd.DataFrame, 
        y_train: np.ndarray,
        include_lightgbm: bool = True,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
    ) -> Any:
        """Train stacking ensemble with base models."""
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
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
        
        # Fit on provided training set; score on provided eval_set if present.
        stacking.fit(X_train, y_train)
        if eval_set is not None:
            X_val, y_val = eval_set
            y_pred = stacking.predict(X_val)
            val_score = f1_score(y_val, y_pred, average='macro')
        else:
            # Fallback: compute a quick internal score (avoid reporting train score)
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
            )
            stacking.fit(X_tr, y_tr)
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
