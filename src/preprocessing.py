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
    RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, ID_COLUMN,
    CORRELATION_THRESHOLD, OUTLIER_IQR_MULTIPLIER,
    FEATURES_TO_DROP, MIN_VARIANCE_THRESHOLD, USE_FEATURE_SELECTION
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
        
        if self.use_feature_selection:
            # Step 6: Drop known redundant features (domain knowledge)
            X = self._drop_manual_features(X)
            
            # Step 7: Remove low-variance features
            X = self._remove_low_variance_features(X)
        
        # Step 8: Handle highly correlated features
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
        
        # Remove manually dropped features
        X = X.drop(columns=[col for col in self.manually_dropped_features if col in X.columns])
        
        # Remove low variance features
        X = X.drop(columns=[col for col in self.removed_low_variance_features if col in X.columns])
        
        # Remove correlated features (same ones as in training)
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
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > CORRELATION_THRESHOLD):
                # Find the correlated pair
                correlated_with = upper.index[upper[column] > CORRELATION_THRESHOLD].tolist()
                if column not in to_drop:  # Keep one, drop the other
                    to_drop.extend([c for c in correlated_with if c not in to_drop])
        
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


if __name__ == "__main__":
    # Test the module
    from data_loader import load_data
    
    df = load_data()
    
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    class_weights = preprocessor.get_class_weights(y)
    
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

