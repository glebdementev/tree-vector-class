# Tree Species Classification - Improvement Plan

## Current Status
- **Best Model**: CatBoost (F1=0.511, Balanced Acc=0.518)
- **Dataset**: 1,950 samples, 35 features (after preprocessing), 6 classes
- **Imbalance**: 2.89:1 (cedar 8.1%, fir 10.1% are minorities)
- **Main Confusion**: Conifers confused with each other (fir↔spruce, larch↔pine)

---

## Improvement Plan

### Step 1: Feature Engineering (preprocessing.py)
Add derived features in `DataPreprocessor.fit_transform()` before correlation removal:
- `crown_compactness` = crown_volume_3d / (crown_area_2d * height_range)
- `crown_elongation` = height_range / crown_diameter
- `upper_density_ratio` = density_upper / (density_lower + density_middle + density_upper)
- `canopy_shape_index` = vertical_centroid * height_width_ratio
- `green_intensity_ratio` = green_mean / intensity_mean
- `rgb_balance` = (red_mean - blue_mean) / green_mean
- `height_quartile_ratio` = (height_p99 - height_p50) / (height_p50 - height_p25)
- `penetration_density_ratio` = laser_penetration_proxy / point_density

### Step 2: Update Hyperparameters (config.py)
```python
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

CATBOOST_PARAMS = {
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3,
    'border_count': 128,
}

LIGHTGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

BALANCED_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}

N_SPLITS = 5  # Was 3
```

### Step 3: Add Early Stopping (models.py)
Modify training methods to use early stopping with validation set:
- XGBoost: `early_stopping_rounds=50`, pass `eval_set`
- CatBoost: `early_stopping_rounds=50`, pass `eval_set`
- LightGBM: `callbacks=[lgb.early_stopping(50)]`, pass `eval_set`

### Step 4: Add Stacking Ensemble (models.py)
Create new `StackingEnsemble` class:
- Base estimators: XGBoost, CatBoost, LightGBM (not BalancedRF - too slow)
- Meta-learner: LogisticRegression(class_weight='balanced', max_iter=1000)
- Use `cv=5` in StackingClassifier to prevent overfitting
- Use `stack_method='predict_proba'`

### Step 5: ADASYN for Minority Classes (preprocessing.py)
Add optional ADASYN resampling for cedar (class 1) and fir (class 2):
- Only apply to training data
- Target: bring cedar/fir to ~250 samples each
- Use `SMOTEENN` (SMOTE + ENN cleaning) as alternative

### Step 6: Threshold Optimization (evaluation.py)
Add per-class threshold tuning after training:
- Get probability predictions on validation set
- For each class, find threshold that maximizes F1
- Store optimal thresholds, use in final prediction

### Step 7: Permutation Feature Selection (preprocessing.py)
After initial training, compute permutation importance:
- Remove features with negative importance (hurt performance)
- Keep minimum 25 features
- Retrain with reduced feature set

### Step 8: Update Evaluation (main.py)
- Add stacking model to comparison
- Report per-class metrics (especially cedar, fir recall)
- Save best model with threshold config

---

## Implementation Order
1. **config.py**: Update hyperparameters + N_SPLITS
2. **preprocessing.py**: Add feature engineering function
3. **models.py**: Add early stopping to all models
4. **models.py**: Add StackingEnsemble class
5. **preprocessing.py**: Add optional ADASYN
6. **evaluation.py**: Add threshold optimization
7. **main.py**: Integrate all changes
8. **preprocessing.py**: Add permutation feature selection (optional, run after baseline)

---

## Expected Results
| Metric | Current | Target |
|--------|---------|--------|
| F1 Macro | 0.511 | 0.55-0.60 |
| Balanced Accuracy | 0.518 | 0.55-0.62 |
| Cedar Recall | 0.41 | 0.50+ |
| Fir Recall | 0.51 | 0.55+ |

---

## Files to Modify
- `src/config.py` - hyperparameters
- `src/preprocessing.py` - feature engineering, ADASYN
- `src/models.py` - early stopping, stacking
- `src/evaluation.py` - threshold optimization
- `src/main.py` - pipeline integration
