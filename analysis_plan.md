# Classification Model Analysis Plan

## Project Overview
- **Dataset**: `train_data (2025).csv`
- **Task**: Binary Classification (Target: -1 or 1)
- **Samples**: 4,041 rows
- **Features**: 83 numeric features (V1-V83) + ID column
- **Goal**: Build and compare 3 classification models

---

# Phase 1-3 Results Summary (COMPLETED ✅)

## Key Findings from EDA

### 📊 Dataset Overview
- **Total samples**: 4,041
- **Total features**: 83 (all numeric, all int64)
- **Target classes**: -1 and 1
- **Memory usage**: ~2.6 MB

### 📋 Data Quality - EXCELLENT
- ✅ **Missing values**: 0 (none!)
- ✅ **Duplicate rows**: 0 (none!)
- ✅ **All features are numeric**: Yes
- ✅ **Unique IDs**: 4,041 (all unique)
- ✅ **No imputation needed**

### ⚠️ CLASS IMBALANCE - CRITICAL FINDING
| Class | Count | Percentage |
|-------|-------|------------|
| -1 | 3,773 | 93.4% |
| 1 | 268 | 6.6% |

**Imbalance Ratio**: 14.08:1 (HIGHLY IMBALANCED!)

This is the most important finding and will significantly affect our modeling approach.

### 📈 Distribution Insights
- **Highly skewed features (|skewness| > 2)**: 5 features
- **Features with outliers**: 72 out of 83 features
- **Top features by outlier percentage**:
  - V19: 16.95% outliers
  - V38: 12.40% outliers
  - V11: 10.37% outliers
  - V35: 10.05% outliers
  - V43: 10.00% outliers

### 🔗 Correlation Analysis
- **Highly correlated pairs (|r| > 0.9)**: 18 pairs
- **Features significantly related to target (p < 0.05)**: 60 out of 83

### 🎯 Top 10 Most Important Features (Random Forest)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | V60 | 0.1202 |
| 2 | V63 | 0.0509 |
| 3 | V4 | 0.0507 |
| 4 | V19 | 0.0404 |
| 5 | V64 | 0.0249 |
| 6 | V61 | 0.0247 |
| 7 | V62 | 0.0228 |
| 8 | V3 | 0.0216 |
| 9 | V22 | 0.0214 |
| 10 | V21 | 0.0192 |

---

# Updated Plan Based on Findings

## Phase 4: Data Preprocessing (ADJUSTED)

### 4.1 Handle Class Imbalance ⭐ CRITICAL
Since we have a 14:1 imbalance ratio, we MUST address this. Options:
1. **Class Weights** ✅ (PREFERRED) - Use `class_weight='balanced'` or `scale_pos_weight` in models
2. **ADASYN** - Better than SMOTE, focuses on harder-to-learn minority examples
3. **SMOTE** - ⚠️ Caution: Research shows it can produce poorly calibrated probabilities
4. **Specialized Ensembles** - BalancedRandomForest, EasyEnsemble (from `imbalanced-learn`)

**Recommended approach (based on 2024-2025 research)**:
- **Primary**: Use class weights (cleaner, no synthetic data issues)
- **Secondary**: Try ADASYN if class weights alone aren't sufficient
- **Avoid**: Heavy reliance on SMOTE (calibration issues noted in literature)

### 4.2 Feature Scaling
- **No scaling needed** for tree-based models (XGBoost, CatBoost, BalancedRF)

### 4.3 Handle Highly Correlated Features
- Remove one feature from each pair with |r| > 0.95
- Reduces noise and speeds up training
- Expected to remove ~5-10 features

### 4.4 Handle Outliers
- **Keep outliers** - all our models are tree-based and robust to them

### 4.5 Train-Test Split
- **Stratified split** to maintain class proportions
- 80% train, 20% test
- Use `stratify=y` parameter

---

## Phase 5: Model Selection (RESEARCH-BASED UPDATE)

### 📚 Research Findings (Nature 2025, ML Best Practices)
- **Boosting algorithms** (XGBoost, CatBoost, LightGBM) achieve **balanced accuracy > 0.75** on imbalanced data
- **CatBoost** often achieves the **highest balanced accuracy** in comparative studies
- **BalancedRandomForest** outperforms standard RandomForest for imbalanced classification
- **Tree-based models** are inherently more robust to class imbalance than linear models

### Selected Models (3 TOP PERFORMERS):

| Model | Why This Model? | Handling Imbalance | Research Support |
|-------|----------------|-------------------|------------------|
| **1. XGBoost** | Top performer for imbalanced data, built-in handling | `scale_pos_weight=14` | ⭐⭐⭐ Excellent |
| **2. CatBoost** | Often outperforms XGBoost, handles categorical features | `auto_class_weights='Balanced'` | ⭐⭐⭐ Excellent (Nature 2025) |
| **3. BalancedRandomForest** | Designed specifically for imbalanced data, undersamples majority in each bootstrap | Built-in balancing | ⭐⭐⭐ Specialized |

### Why These Models?
1. **XGBoost** - Proven excellent performance on imbalanced datasets
2. **CatBoost** - Studies show it often achieves highest balanced accuracy
3. **BalancedRandomForest** (from `imbalanced-learn`) - specifically designed for imbalanced data

### Alternative Models to Consider:
- **LightGBM** - Faster than XGBoost, `is_unbalance=True` parameter
- **EasyEnsembleClassifier** - Trains multiple AdaBoost on balanced subsets
- **RUSBoostClassifier** - Random undersampling + boosting

---

## Phase 6: Hyperparameter Tuning (RESEARCH-BASED UPDATE)

### 6.1 XGBoost (Primary Model)
```python
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [14],  # ratio of negative to positive
    'min_child_weight': [1, 3, 5]
}
```

### 6.2 CatBoost (NEW - Research-backed)
```python
param_grid_catboost = {
    'iterations': [100, 200, 300],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'l2_leaf_reg': [1, 3, 5],
    'auto_class_weights': ['Balanced'],  # handles imbalance automatically
    'random_seed': [42]
}
```

### 6.3 BalancedRandomForest (from imbalanced-learn)
```python
from imblearn.ensemble import BalancedRandomForestClassifier

param_grid_brf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'sampling_strategy': ['auto'],  # automatically balances
    'replacement': [True, False]
}
```

### Tuning Strategy
- **StratifiedKFold** cross-validation (5-fold) to maintain class balance
- Use **RandomizedSearchCV** for efficiency (especially boosting models)
- Primary metric: **F1-Score** (most appropriate for imbalanced data)
- Secondary metrics: ROC-AUC, Precision, Recall, **Balanced Accuracy**
- **Threshold Tuning**: After training, optimize decision threshold for best F1

---

## Phase 7: Model Evaluation (ADJUSTED for Imbalanced Data)

### 7.1 Primary Metrics (for imbalanced classification):

| Metric | Why Important |
|--------|---------------|
| **F1-Score** | Harmonic mean of precision and recall - best for imbalanced data |
| **ROC-AUC** | Measures discrimination ability across all thresholds |
| **Precision** | Important if false positives are costly |
| **Recall** | Important if false negatives are costly |
| **Balanced Accuracy** | Average of recall for each class |

### 7.2 DO NOT rely heavily on:
- **Accuracy** - Can be misleading (e.g., 93.4% accuracy by predicting all -1)

### 7.3 Visualization
- Confusion matrices (with normalized values)
- ROC curves for all models
- Precision-Recall curves (more informative for imbalanced data)
- Feature importance comparison

---

## Phase 8: Final Model Selection

### Selection Criteria (prioritized):
1. **F1-Score on minority class (1)** - Most important
2. **ROC-AUC** - Overall discrimination ability
3. **Balanced Accuracy** - Fair assessment across classes
4. **Model interpretability** - For business understanding
5. **Training/prediction speed** - For practical deployment

---

## Implementation Considerations

### For BalancedRandomForest (RECOMMENDED over standard RF):
```python
from imblearn.ensemble import BalancedRandomForestClassifier

brf = BalancedRandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
brf.fit(X_train, y_train)
```

### For CatBoost (NEW):
```python
from catboost import CatBoostClassifier

catboost = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=False
)
catboost.fit(X_train, y_train)
```

### For ADASYN (if resampling needed - better than SMOTE):
```python
from imblearn.over_sampling import ADASYN

# Apply ADASYN only to training data, not test data
adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
```

### For handling highly correlated features:
```python
# Remove features with correlation > 0.95
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop
```

### For XGBoost scale_pos_weight:
```python
# Calculate automatically
scale_pos_weight = sum(y_train == -1) / sum(y_train == 1)
```

### For Threshold Tuning (POST-TRAINING - IMPORTANT):
```python
from sklearn.metrics import precision_recall_curve

# Get predicted probabilities
y_proba = model.predict_proba(X_val)[:, 1]

# Find optimal threshold for F1
precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold:.3f}")
```

---

## Summary of Key Adjustments (RESEARCH-BASED)

| Original Plan | Adjusted Plan (Based on Research) |
|---------------|-----------------------------------|
| Standard train-test split | Stratified split to maintain class balance |
| Accuracy as primary metric | F1-Score as primary metric |
| Random Forest | **BalancedRandomForest** (specialized for imbalanced data) |
| SMOTE as primary resampling | **Class weights preferred** (SMOTE can cause calibration issues) |
| No CatBoost | **Added CatBoost** (highest balanced accuracy in studies) |
| GridSearchCV | RandomizedSearchCV with stratified CV |
| Fixed threshold (0.5) | **Threshold tuning** post-training for optimal F1 |
| Standard confusion matrix | Normalized confusion matrix + PR curves |

### Key Research Insights Applied:
1. **Boosting > Bagging** for imbalanced data (Nature 2025)
2. **Class weights > SMOTE** for probability calibration
3. **CatBoost** often achieves highest balanced accuracy
4. **BalancedRandomForest** specifically designed for this problem
5. **Threshold tuning** is critical for imbalanced classification

---

## Expected Challenges

1. **Class imbalance** may cause models to favor majority class
2. **Large number of features (83)** may benefit from feature selection
3. **Highly correlated features** - addressed by removing pairs with |r| > 0.95

---

## Next Steps

1. ✅ Phases 1-3: Data Loading, Quality Check, EDA - COMPLETED
2. ⏳ Phase 4: Preprocessing (scaling, feature selection, class weight setup)
3. ⏳ Phase 5: Model Training (XGBoost, CatBoost, BalancedRandomForest)
4. ⏳ Phase 6: Hyperparameter Tuning with stratified CV
5. ⏳ Phase 7: Model Comparison (F1-Score + Balanced Accuracy focus)
6. ⏳ Phase 8: Threshold Tuning + Final Selection and Conclusion

**Ready to proceed with Phases 4-8?**

---

## Required Libraries

```python
# Core ML
pip install scikit-learn xgboost catboost lightgbm

# Imbalanced learning (for BalancedRandomForest, ADASYN)
pip install imbalanced-learn
```
