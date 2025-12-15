"""
Data loading and initial inspection module.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from config import DATA_PATH, TARGET_COLUMN, ID_COLUMN, CLASS_NAMES


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the dataset from CSV file.
    
    Args:
        filepath: Path to CSV file. If None, uses default from config.
        
    Returns:
        DataFrame with loaded data.
    """
    if filepath is None:
        filepath = DATA_PATH
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary with dataset information.
    """
    info = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude ID and target
        'n_total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Check for ID column
    if ID_COLUMN in df.columns:
        info['unique_ids'] = df[ID_COLUMN].nunique()
        info['id_column_present'] = True
    else:
        info['id_column_present'] = False
    
    # Check for target column
    if TARGET_COLUMN in df.columns:
        info['target_column_present'] = True
        info['class_distribution'] = df[TARGET_COLUMN].value_counts().to_dict()
        info['n_classes'] = df[TARGET_COLUMN].nunique()
    else:
        info['target_column_present'] = False
    
    return info


def print_data_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive data quality report.
    
    Args:
        df: Input DataFrame.
    """
    info = get_data_info(df)
    
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    print(f"  Total samples:     {info['n_samples']:,}")
    print(f"  Total columns:     {info['n_total_columns']}")
    print(f"  Feature columns:   {info['n_features']}")
    print(f"  Memory usage:      {info['memory_mb']:.2f} MB")
    
    print("\n" + "-"*60)
    print("📋 DATA QUALITY")
    print("-"*60)
    print(f"  Missing values:    {info['missing_values']}")
    print(f"  Duplicate rows:    {info['duplicate_rows']}")
    if info.get('id_column_present'):
        print(f"  Unique IDs:        {info['unique_ids']} (all unique: {info['unique_ids'] == info['n_samples']})")
    
    if info.get('target_column_present'):
        print("\n" + "-"*60)
        print("🎯 CLASS DISTRIBUTION (Target: species)")
        print("-"*60)
        class_dist = info['class_distribution']
        total = sum(class_dist.values())
        
        # Sort by count descending
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)
        
        for cls, count in sorted_classes:
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {cls:10s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Calculate imbalance ratio
        max_count = max(class_dist.values())
        min_count = min(class_dist.values())
        imbalance_ratio = max_count / min_count
        print(f"\n  ⚠️  Imbalance ratio (max/min): {imbalance_ratio:.2f}:1")
    
    print("\n" + "="*60)


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature column names (excluding ID and target).
    
    Args:
        df: Input DataFrame.
        
    Returns:
        List of feature column names.
    """
    exclude_cols = [ID_COLUMN, TARGET_COLUMN]
    return [col for col in df.columns if col not in exclude_cols]


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    
    print(f"✅ Split data: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def check_data_types(df: pd.DataFrame) -> Dict[str, list]:
    """
    Check and categorize column data types.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary with column names grouped by type.
    """
    feature_cols = get_feature_columns(df)
    
    numeric_cols = []
    categorical_cols = []
    other_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            other_cols.append(col)
    
    result = {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'other': other_cols,
    }
    
    print(f"\n📊 Feature types:")
    print(f"  Numeric:     {len(numeric_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")
    print(f"  Other:       {len(other_cols)}")
    
    return result


if __name__ == "__main__":
    # Test the module
    df = load_data()
    print_data_report(df)
    check_data_types(df)
    X, y = split_features_target(df)

