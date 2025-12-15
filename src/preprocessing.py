"""
Data preprocessing module for cleaning and feature engineering.

Enhanced with advanced feature selection and variance filtering.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from scipy import stats

from config import (
    RANDOM_STATE, TEST_SIZE, VAL_SIZE, TARGET_COLUMN, ID_COLUMN,
    CORRELATION_THRESHOLD, OUTLIER_IQR_MULTIPLIER,
    FEATURES_TO_DROP, MIN_VARIANCE_THRESHOLD, USE_FEATURE_SELECTION,
    REMOVE_CORRELATED_FEATURES
)


class DataPreprocessor:
    """
    Handles all preprocessing steps for the tree species classification dataset.
    
    Enhanced with:
    - Domain-knowledge based feature dropping
    - Low-variance feature removal
    - Mutual information based feature selection
    - Advanced correlation handling
    """
    
    def __init__(self, use_feature_selection: bool = USE_FEATURE_SELECTION):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.removed_correlated_features = []
        self.removed_low_variance_features = []
        self.manually_dropped_features = []
        self.feature_stats = {}
        self.mutual_info_scores = {}
        self.is_fitted = False
        self.use_feature_selection = use_feature_selection
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame with features and target.
            
        Returns:
            Tuple of (X_processed, y_encoded).
        """
        print("\n🔧 Preprocessing data...")
        
        # Step 1: Remove ID column
        df_clean = self._remove_id_column(df)
        
        # Step 2: Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Step 3: Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Step 4: Split features and target
        X = df_clean.drop(columns=[TARGET_COLUMN])
        y = df_clean[TARGET_COLUMN]
        
        # Step 5: Encode target labels
        y_encoded = self._encode_target(y)
        
        # Step 5.5: Create derived features (NEW)
        X = self._create_derived_features(X)
        
        if self.use_feature_selection:
            # Step 6: Drop known redundant features (domain knowledge)
            X = self._drop_manual_features(X)
            
            # Step 7: Remove low-variance features
            X = self._remove_low_variance_features(X)
        
        # Step 8: Handle highly correlated features (optional; tree models don't require this)
        if REMOVE_CORRELATED_FEATURES:
            X = self._remove_correlated_features(X)
        
        if self.use_feature_selection:
            # Step 9: Compute mutual information scores
            self._compute_mutual_info(X, y_encoded)
        
        # Step 10: Analyze and report outliers (but don't remove for tree-based models)
        self._analyze_outliers(X)
        
        # Step 11: Store feature statistics
        self._compute_feature_stats(X)
        
        self.is_fitted = True
        
        n_removed = len(self.manually_dropped_features) + len(self.removed_low_variance_features) + len(self.removed_correlated_features)
        print(f"✅ Done! Shape: {X.shape}, removed {n_removed} features")
        
        return X, y_encoded
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Tuple of (X_processed, y_encoded) or (X_processed, None) if no target.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Remove ID column if present
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])
        
        # Split features and target
        if TARGET_COLUMN in df.columns:
            X = df.drop(columns=[TARGET_COLUMN])
            y = df[TARGET_COLUMN]
            y_encoded = self.label_encoder.transform(y)
        else:
            X = df.copy()
            y_encoded = None

        # Derived features must be created for inference/val/test too (deterministic)
        X = self._create_derived_features(X)
        
        # Remove manually dropped features
        X = X.drop(columns=[col for col in self.manually_dropped_features if col in X.columns])
        
        # Remove low variance features
        X = X.drop(columns=[col for col in self.removed_low_variance_features if col in X.columns])
        
        # Remove correlated features (same ones as in training; list is empty if disabled)
        X = X.drop(columns=[col for col in self.removed_correlated_features if col in X.columns])
        
        return X, y_encoded
    
    def _remove_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID column from DataFrame."""
        if ID_COLUMN in df.columns:
            df = df.drop(columns=[ID_COLUMN])
            print(f"  ✓ Removed ID column: '{ID_COLUMN}'")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            
            # For categorical columns, fill with mode
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
            
            print(f"  ✓ Handled {missing_count} missing values (median/mode imputation)")
        else:
            print(f"  ✓ No missing values found")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            df = df.drop_duplicates()
            print(f"  ✓ Removed {n_duplicates} duplicate rows")
        else:
            print(f"  ✓ No duplicate rows found")
        
        return df
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels to integers."""
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n  Target encoding:")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = (y_encoded == i).sum()
            print(f"    {cls} → {i} (n={count})")
        
        return y_encoded
    
    def _create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific derived features for tree species classification.
        
        These features capture crown shape, vertical distribution, and color-intensity
        relationships that help distinguish between species.
        """
        print(f"\n  Creating derived features...")
        n_original = len(X.columns)
        
        eps = 1e-6

        # Height spread / shape (robust to outliers, helpful for species differences)
        if all(col in X.columns for col in ['height_p75', 'height_p25']):
            X['height_iqr'] = X['height_p75'] - X['height_p25']
        if all(col in X.columns for col in ['height_p90', 'height_p10']):
            X['height_decile_range'] = X['height_p90'] - X['height_p10']
        if all(col in X.columns for col in ['height_p99', 'height_p50']):
            X['height_top_heaviness'] = (X['height_p99'] - X['height_p50']) / (X['height_p50'] + eps)
        if all(col in X.columns for col in ['height_std', 'height_mean']):
            # Stabilized CV-like feature (some rows can have tiny means)
            X['height_std_over_mean'] = X['height_std'] / (X['height_mean'] + eps)

        # Crown shape features (distinguish conifers from deciduous)
        if 'crown_volume_3d' in X.columns and 'crown_area_2d' in X.columns and 'height_range' in X.columns:
            X['crown_compactness'] = X['crown_volume_3d'] / (X['crown_area_2d'] * X['height_range'] + eps)

        if 'crown_volume_3d' in X.columns and 'height_range' in X.columns:
            X['crown_volume_per_height'] = X['crown_volume_3d'] / (X['height_range'] + eps)

        if 'crown_area_2d' in X.columns and 'crown_diameter' in X.columns:
            # Area vs diameter (captures non-circular crowns and segmentation artifacts)
            X['crown_area_over_diameter2'] = X['crown_area_2d'] / ((X['crown_diameter'] ** 2) + eps)
        
        if 'height_range' in X.columns and 'crown_diameter' in X.columns:
            X['crown_elongation'] = X['height_range'] / (X['crown_diameter'] + eps)
        
        # Vertical distribution ratios
        if all(col in X.columns for col in ['density_upper', 'density_middle', 'density_lower']):
            total_density = X['density_lower'] + X['density_middle'] + X['density_upper'] + eps
            X['upper_to_total_density'] = X['density_upper'] / total_density
            X['middle_to_total_density'] = X['density_middle'] / total_density
            X['lower_to_total_density'] = X['density_lower'] / total_density
            X['upper_minus_lower_density'] = X['density_upper'] - X['density_lower']
            X['upper_over_lower_density'] = X['density_upper'] / (X['density_lower'] + eps)
        
        if 'vertical_centroid' in X.columns and 'height_width_ratio' in X.columns:
            X['canopy_shape_index'] = X['vertical_centroid'] * X['height_width_ratio']

        if all(col in X.columns for col in ['vertical_centroid', 'vertical_std']):
            X['vertical_compactness'] = X['vertical_centroid'] / (X['vertical_std'] + eps)

        # Layer intensity contrasts (captures deciduous vs conifer foliage distribution)
        if all(col in X.columns for col in ['intensity_upper_mean', 'intensity_lower_mean']):
            X['intensity_upper_minus_lower'] = X['intensity_upper_mean'] - X['intensity_lower_mean']
            X['intensity_upper_over_lower'] = X['intensity_upper_mean'] / (X['intensity_lower_mean'] + eps)
        if all(col in X.columns for col in ['intensity_upper_mean', 'intensity_middle_mean', 'intensity_lower_mean']):
            X['intensity_layer_range'] = X[['intensity_upper_mean', 'intensity_middle_mean', 'intensity_lower_mean']].max(axis=1) - X[['intensity_upper_mean', 'intensity_middle_mean', 'intensity_lower_mean']].min(axis=1)
        
        # Color-intensity interactions (birch vs. conifers)
        if 'green_mean' in X.columns and 'intensity_mean' in X.columns:
            X['green_intensity_ratio'] = X['green_mean'] / (X['intensity_mean'] + eps)
        
        if all(col in X.columns for col in ['red_mean', 'blue_mean', 'green_mean']):
            X['rgb_balance'] = (X['red_mean'] - X['blue_mean']) / (X['green_mean'] + eps)
            X['red_over_green'] = X['red_mean'] / (X['green_mean'] + eps)
            X['blue_over_green'] = X['blue_mean'] / (X['green_mean'] + eps)
            X['green_over_red'] = X['green_mean'] / (X['red_mean'] + eps)

        # Vegetation index interactions
        if all(col in X.columns for col in ['egi_mean', 'vi_mean']):
            X['egi_minus_vi'] = X['egi_mean'] - X['vi_mean']
            X['egi_over_vi'] = X['egi_mean'] / (X['vi_mean'] + eps)
        
        # Height distribution features
        if all(col in X.columns for col in ['height_p99', 'height_p50', 'height_p25']):
            X['height_quartile_ratio'] = (X['height_p99'] - X['height_p50']) / (X['height_p50'] - X['height_p25'] + eps)
        
        # Penetration-density relationship
        if 'laser_penetration_proxy' in X.columns and 'point_density' in X.columns:
            X['penetration_density_ratio'] = X['laser_penetration_proxy'] / (X['point_density'] + eps)

        # Log features for heavy-tailed distributions (tree models can still benefit)
        for col in ['total_points', 'point_density']:
            if col in X.columns:
                X[f'log1p_{col}'] = np.log1p(np.clip(X[col].values, a_min=0, a_max=None))

        # Intensity min is extremely skewed in your earlier report; stabilize it
        if 'intensity_min' in X.columns:
            X['log1p_intensity_min_pos'] = np.log1p(np.clip(X['intensity_min'].values, a_min=0, a_max=None))
        
        n_created = len(X.columns) - n_original
        print(f"  ✓ Created {n_created} derived features")
        
        return X
    
    def _drop_manual_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features based on domain knowledge.
        
        These are features identified as redundant or uninformative
        based on correlation analysis and feature importance.
        """
        print(f"\n  Dropping pre-defined redundant features...")
        
        to_drop = [col for col in FEATURES_TO_DROP if col in X.columns]
        
        if to_drop:
            X = X.drop(columns=to_drop)
            self.manually_dropped_features = to_drop
            print(f"  ✓ Dropped {len(to_drop)} features based on domain knowledge:")
            for feat in to_drop:
                print(f"      - {feat}")
        else:
            print(f"  ✓ No pre-defined features to drop")
        
        return X
    
    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with very low variance (near-constant).
        """
        print(f"\n  Checking for low-variance features (threshold: {MIN_VARIANCE_THRESHOLD})...")
        
        # Normalize features first to make variance comparable
        X_normalized = (X - X.mean()) / (X.std() + 1e-10)
        
        selector = VarianceThreshold(threshold=MIN_VARIANCE_THRESHOLD)
        selector.fit(X_normalized)
        
        low_var_mask = ~selector.get_support()
        low_var_features = X.columns[low_var_mask].tolist()
        
        if low_var_features:
            X = X.drop(columns=low_var_features)
            self.removed_low_variance_features = low_var_features
            print(f"  ✓ Removed {len(low_var_features)} low-variance features:")
            for feat in low_var_features[:5]:
                print(f"      - {feat}")
            if len(low_var_features) > 5:
                print(f"      ... and {len(low_var_features) - 5} more")
        else:
            print(f"  ✓ No low-variance features found")
        
        return X
    
    def _compute_mutual_info(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Skip mutual info computation for speed."""
        # Skipped for faster preprocessing
        pass
    
    def get_mutual_info_scores(self) -> Dict[str, float]:
        """Return mutual information scores for all features."""
        return self.mutual_info_scores
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Keeps one feature from each highly correlated pair.
        """
        print(f"\n  Checking feature correlations (threshold: {CORRELATION_THRESHOLD})...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > threshold
        # Standard approach: drop the "later" feature in each correlated pair.
        to_drop = [column for column in upper.columns if any(upper[column] > CORRELATION_THRESHOLD)]
        
        self.removed_correlated_features = to_drop
        
        if to_drop:
            X = X.drop(columns=to_drop)
            print(f"  ✓ Removed {len(to_drop)} highly correlated features:")
            for feat in to_drop[:10]:  # Show first 10
                print(f"      - {feat}")
            if len(to_drop) > 10:
                print(f"      ... and {len(to_drop) - 10} more")
        else:
            print(f"  ✓ No highly correlated features found")
        
        return X
    
    def _analyze_outliers(self, X: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze outliers in the dataset using IQR method.
        Note: We don't remove outliers for tree-based models.
        """
        print(f"\n  Analyzing outliers (IQR method, multiplier={OUTLIER_IQR_MULTIPLIER})...")
        
        outlier_info = {}
        features_with_outliers = 0
        
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
            
            outliers = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()
            outlier_pct = outliers / len(X) * 100
            
            if outliers > 0:
                features_with_outliers += 1
                outlier_info[col] = {
                    'count': outliers,
                    'percentage': outlier_pct,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                }
        
        print(f"  ✓ {features_with_outliers}/{len(X.columns)} features have outliers")
        print(f"    (Tree-based models are robust to outliers - not removing)")
        
        # Show top 5 features with most outliers
        if outlier_info:
            sorted_outliers = sorted(outlier_info.items(), key=lambda x: x[1]['percentage'], reverse=True)
            print(f"\n    Top features by outlier %:")
            for feat, info in sorted_outliers[:5]:
                print(f"      - {feat}: {info['percentage']:.1f}%")
        
        return outlier_info
    
    def _compute_feature_stats(self, X: pd.DataFrame) -> None:
        """Compute and store feature statistics."""
        self.feature_stats = {
            'mean': X.mean().to_dict(),
            'std': X.std().to_dict(),
            'min': X.min().to_dict(),
            'max': X.max().to_dict(),
            'skewness': X.skew().to_dict(),
            'kurtosis': X.kurtosis().to_dict(),
        }
        
        # Report highly skewed features
        skewed_features = [col for col, skew in self.feature_stats['skewness'].items() if abs(skew) > 2]
        if skewed_features:
            print(f"\n  ⚠️  {len(skewed_features)} features with high skewness (|skew| > 2):")
            for feat in skewed_features[:5]:
                print(f"      - {feat}: {self.feature_stats['skewness'][feat]:.2f}")
    
    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            y: Encoded target array.
            
        Returns:
            Dictionary mapping class index to weight.
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = dict(zip(classes, weights))
        
        print("\n📊 Class weights (for imbalanced handling):")
        for cls_idx, weight in class_weights.items():
            cls_name = self.label_encoder.inverse_transform([cls_idx])[0]
            print(f"   {cls_name} (class {cls_idx}): {weight:.3f}")
        
        return class_weights
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from encoded labels to original class names."""
        return {i: cls for i, cls in enumerate(self.label_encoder.classes_)}


def apply_adasyn(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    target_classes: List[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply ADASYN oversampling to minority classes.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        target_classes: List of class indices to oversample. 
                       If None, defaults to [1, 2] (cedar, fir).
    
    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    from imblearn.over_sampling import ADASYN
    
    if target_classes is None:
        target_classes = [1, 2]  # cedar and fir by default
    
    class_counts = pd.Series(y_train).value_counts()
    target_count = int(class_counts.median())
    
    # Build sampling strategy for underrepresented classes
    sampling_strategy = {}
    for cls in target_classes:
        if cls in class_counts and class_counts[cls] < target_count:
            sampling_strategy[cls] = target_count
    
    if not sampling_strategy:
        print("  ℹ️  No ADASYN needed - classes already balanced enough")
        return X_train, y_train
    
    print(f"\n🔄 Applying ADASYN oversampling...")
    print(f"   Target classes: {target_classes}")
    print(f"   Target count per class: {target_count}")
    
    try:
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy, 
            random_state=RANDOM_STATE
        )
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame if needed
        if isinstance(X_train, pd.DataFrame):
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        
        print(f"   Original: {len(y_train)} samples")
        print(f"   After ADASYN: {len(y_resampled)} samples (+{len(y_resampled) - len(y_train)})")
        
        # Show new class distribution
        new_counts = pd.Series(y_resampled).value_counts().sort_index()
        print(f"   New class distribution:")
        for cls, count in new_counts.items():
            print(f"     Class {cls}: {count}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        print(f"   ⚠️  ADASYN failed: {e}")
        print(f"   Continuing without oversampling...")
        return X_train, y_train


def select_features_permutation(
    model, 
    X_val: pd.DataFrame, 
    y_val: np.ndarray, 
    threshold: float = 0.0,
    n_repeats: int = 10
) -> List[str]:
    """
    Select features using permutation importance.
    
    Removes features with negative or zero permutation importance.
    
    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.
        threshold: Minimum importance threshold (default 0.0).
        n_repeats: Number of permutation repeats.
    
    Returns:
        List of feature names to keep.
    """
    from sklearn.inspection import permutation_importance
    
    print(f"\n🔍 Computing permutation importance...")
    
    perm = permutation_importance(
        model, X_val, y_val, 
        n_repeats=n_repeats, 
        random_state=RANDOM_STATE, 
        scoring='f1_macro', 
        n_jobs=-1
    )
    
    keep_mask = perm.importances_mean > threshold
    features_to_keep = X_val.columns[keep_mask].tolist()
    features_to_drop = X_val.columns[~keep_mask].tolist()
    
    print(f"   Features to keep: {len(features_to_keep)}")
    print(f"   Features to drop: {len(features_to_drop)}")
    
    if features_to_drop:
        print(f"   Dropped features (importance <= {threshold}):")
        for feat in features_to_drop[:10]:
            idx = list(X_val.columns).index(feat)
            imp = perm.importances_mean[idx]
            print(f"     - {feat}: {imp:.4f}")
        if len(features_to_drop) > 10:
            print(f"     ... and {len(features_to_drop) - 10} more")
    
    return features_to_keep


def create_train_test_split(
    X: pd.DataFrame, 
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Create stratified train-test split.
    
    Args:
        X: Features DataFrame.
        y: Target array.
        test_size: Proportion for test set.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class proportions
    )
    
    print(f"\n✅ Train-test split (stratified):")
    print(f"   Train: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test:  {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    # Verify stratification
    print(f"\n   Class distribution preserved:")
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    
    for cls in range(len(train_dist)):
        print(f"     Class {cls}: Train={train_dist[cls]*100:.1f}%, Test={test_dist[cls]*100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def create_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a stratified train/val/test split at the raw-DataFrame level.

    This prevents leakage when preprocessing (correlation pruning, low-variance pruning,
    feature selection) is fit only on the training set.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in df")

    if val_size <= 0 or test_size <= 0 or (val_size + test_size) >= 0.9:
        raise ValueError(f"Invalid split sizes: val_size={val_size}, test_size={test_size}")

    y = df[TARGET_COLUMN]

    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split remaining into train/val
    val_size_adj = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_adj,
        random_state=random_state,
        stratify=df_trainval[TARGET_COLUMN],
    )

    def _dist(x: pd.DataFrame) -> pd.Series:
        return x[TARGET_COLUMN].value_counts(normalize=True).sort_index()

    print(f"\n✅ Train/val/test split (stratified):")
    print(f"   Train: {len(df_train)} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

    print(f"\n   Class distribution preserved (percent):")
    train_d, val_d, test_d = _dist(df_train), _dist(df_val), _dist(df_test)
    classes = sorted(set(train_d.index) | set(val_d.index) | set(test_d.index))
    for cls in classes:
        print(
            f"     {cls:10s}: "
            f"Train={train_d.get(cls, 0)*100:.1f}%, "
            f"Val={val_d.get(cls, 0)*100:.1f}%, "
            f"Test={test_d.get(cls, 0)*100:.1f}%"
        )

    return df_train, df_val, df_test


if __name__ == "__main__":
    # Test the module
    from data_loader import load_data
    
    df = load_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    class_weights = preprocessor.get_class_weights(y)
    
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

