# Model Improvement Plan

**Current:** F1=0.511, Balanced Accuracy=0.518  
**Target:** F1=0.55-0.60

---

## Phase 1: Hyperparameter Optimization with Early Stopping

### 1.1 Update `config.py` hyperparameters

```python
CATBOOST_PARAMS = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'border_count': 128,
}

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

N_SPLITS = 5  # Increase from 3
```

### 1.2 Add early stopping to `models.py`

- CatBoost: `early_stopping_rounds=50` in fit with eval_set
- XGBoost: `early_stopping_rounds=50` in fit with eval_set
- LightGBM: `callbacks=[lgb.early_stopping(50)]` in fit

Split training data 80/20 for train/eval inside each training method.

---

## Phase 2: Feature Engineering

### 2.1 Add to `preprocessing.py` - new method `_create_derived_features()`

Create these features BEFORE correlation removal:

```python
# Crown shape
df['crown_compactness'] = df['crown_volume_3d'] / (df['crown_area_2d'] * df['height_range'] + 1e-6)
df['crown_elongation'] = df['height_range'] / (df['crown_diameter'] + 1e-6)

# Vertical distribution
df['upper_to_total_density'] = df['density_upper'] / (df['density_lower'] + df['density_middle'] + df['density_upper'] + 1e-6)
df['canopy_shape_index'] = df['vertical_centroid'] * df['height_width_ratio']

# Color-intensity
df['green_intensity_ratio'] = df['green_mean'] / (df['intensity_mean'] + 1e-6)
df['rgb_balance'] = (df['red_mean'] - df['blue_mean']) / (df['green_mean'] + 1e-6)

# Height distribution
df['height_quartile_ratio'] = (df['height_p99'] - df['height_p50']) / (df['height_p50'] - df['height_p25'] + 1e-6)

# Penetration
df['penetration_density_ratio'] = df['laser_penetration_proxy'] / (df['point_density'] + 1e-6)
```

### 2.2 Call in `fit_transform()` after step 4 (split X/y), before step 6

---

## Phase 3: Stacking Ensemble

### 3.1 Add to `models.py` - new method `train_stacking_ensemble()`

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def train_stacking_ensemble(self, X_train, y_train):
    estimators = [
        ('xgboost', self._get_xgboost_model()),
        ('catboost', self._get_catboost_model()),
        ('lightgbm', self._get_lightgbm_model()),
        ('balanced_rf', self._get_balanced_rf_model()),
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    self.models['stacking'] = stacking
```

### 3.2 Update `train_all_models()` to call stacking after individual models

### 3.3 Update `main.py` to include stacking in evaluation

---

## Phase 4: ADASYN for Minority Classes

### 4.1 Add to `preprocessing.py` - new function `apply_adasyn()`

```python
from imblearn.over_sampling import ADASYN

def apply_adasyn(X_train, y_train, target_classes=[1, 2]):
    """Oversample cedar (1) and fir (2) only."""
    class_counts = pd.Series(y_train).value_counts()
    target_count = int(class_counts.median())
    
    sampling_strategy = {
        cls: target_count for cls in target_classes 
        if class_counts[cls] < target_count
    }
    
    if not sampling_strategy:
        return X_train, y_train
    
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_jobs=-1)
    return adasyn.fit_resample(X_train, y_train)
```

### 4.2 Call in `main.py` after train/test split, before training

Add flag `--use-adasyn` to enable/disable.

---

## Phase 5: Threshold Optimization

### 5.1 Add to `evaluation.py` - new method `optimize_thresholds()`

```python
def optimize_thresholds(self, y_true, y_proba, n_classes=6):
    """Find optimal threshold per class on validation set."""
    thresholds = {}
    for cls in range(n_classes):
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.2, 0.8, 0.05):
            y_pred_cls = (y_proba[:, cls] >= thresh).astype(int)
            y_true_cls = (y_true == cls).astype(int)
            f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        thresholds[cls] = best_thresh
    return thresholds
```

### 5.2 Add `predict_with_thresholds()` method

```python
def predict_with_thresholds(self, y_proba, thresholds):
    """Apply per-class thresholds to probabilities."""
    adjusted = y_proba.copy()
    for cls, thresh in thresholds.items():
        adjusted[:, cls] = adjusted[:, cls] / thresh
    return adjusted.argmax(axis=1)
```

### 5.3 Use validation set (split from train) to optimize, apply to test

---

## Phase 6: Permutation Feature Selection

### 6.1 Add to `preprocessing.py` - new function `select_features_permutation()`

```python
from sklearn.inspection import permutation_importance

def select_features_permutation(model, X_val, y_val, threshold=0.0):
    """Remove features with negative permutation importance."""
    perm = permutation_importance(model, X_val, y_val, n_repeats=10, 
                                   random_state=42, scoring='f1_macro', n_jobs=-1)
    keep_mask = perm.importances_mean > threshold
    return X_val.columns[keep_mask].tolist()
```

### 6.2 Run after initial model training, retrain with selected features

Optional step - add flag `--feature-selection` to enable.

---

## Implementation Order

1. **Phase 1** - Hyperparameters + early stopping (low effort, high impact)
2. **Phase 2** - Feature engineering (medium effort, high impact)
3. **Phase 3** - Stacking ensemble (medium effort, high impact)
4. **Phase 4** - ADASYN (low effort, medium impact)
5. **Phase 5** - Threshold optimization (low effort, medium impact)
6. **Phase 6** - Feature selection (optional, run if overfitting suspected)

---

## Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Update hyperparameters, N_SPLITS=5 |
| `preprocessing.py` | Add `_create_derived_features()`, `apply_adasyn()`, `select_features_permutation()` |
| `models.py` | Add early stopping, `train_stacking_ensemble()` |
| `evaluation.py` | Add `optimize_thresholds()`, `predict_with_thresholds()` |
| `main.py` | Add flags, integrate new methods |

---

## Expected Results

| Phase | Cumulative F1 |
|-------|---------------|
| Baseline | 0.511 |
| +Phase 1 | 0.53-0.54 |
| +Phase 2 | 0.55-0.56 |
| +Phase 3 | 0.57-0.59 |
| +Phase 4,5 | 0.58-0.60 |

