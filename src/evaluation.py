"""
Model evaluation module with metrics and visualization for imbalanced multiclass classification.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings

from config import PLOTS_DIR, CLASS_NAMES, N_CLASSES


class ModelEvaluator:
    """
    Comprehensive evaluation for multiclass classification with imbalanced data.
    """
    
    def __init__(self, label_mapping: Dict[int, str] = None):
        """
        Initialize the evaluator.
        
        Args:
            label_mapping: Dictionary mapping class indices to class names.
        """
        self.label_mapping = label_mapping or {i: name for i, name in enumerate(CLASS_NAMES)}
        self.class_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        self.results = {}
    
    def evaluate_model(
        self, 
        model_name: str,
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for a model.
        
        Args:
            model_name: Name of the model.
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional, for ROC-AUC).
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Macro-averaged metrics (treat all classes equally)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted-averaged metrics (account for class imbalance)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                # One-vs-Rest ROC-AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average='macro'
                )
                metrics['roc_auc_ovr_weighted'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average='weighted'
                )
            except Exception as e:
                warnings.warn(f"Could not compute ROC-AUC: {e}")
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }
        
        return metrics
    
    def print_classification_report(self, model_name: str) -> None:
        """Print detailed classification report for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not evaluated yet.")
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        print(f"\n{'='*60}")
        print(f"📊 CLASSIFICATION REPORT: {model_name.upper()}")
        print('='*60)
        print(classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4,
            zero_division=0
        ))
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all evaluated models.
        
        Returns:
            DataFrame with metrics for all models.
        """
        if not self.results:
            raise ValueError("No models evaluated yet.")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 macro (primary metric for imbalanced data)
        df = df.sort_values('f1_macro', ascending=False)
        
        return df
    
    def print_comparison_summary(self) -> None:
        """Print a summary comparison of all models."""
        df = self.get_comparison_table()
        
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Key metrics for imbalanced classification
        key_metrics = ['Model', 'f1_macro', 'balanced_accuracy', 'precision_macro', 'recall_macro']
        if 'roc_auc_ovr' in df.columns:
            key_metrics.append('roc_auc_ovr')
        
        display_df = df[key_metrics].copy()
        
        # Format numeric columns
        for col in display_df.columns:
            if col != 'Model':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        # Highlight best model
        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['f1_macro']
        print(f"\n🏆 Best Model: {best_model} (F1 Macro: {best_f1:.4f})")
    
    def plot_confusion_matrix(
        self, 
        model_name: str, 
        normalize: bool = True,
        figsize: tuple = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model.
            normalize: If True, normalize by true labels.
            figsize: Figure size.
            save: If True, save the plot.
            
        Returns:
            Matplotlib figure.
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not evaluated yet.")
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / f'confusion_matrix_{model_name}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved confusion matrix to {filename}")
        
        return fig
    
    def plot_all_confusion_matrices(
        self, 
        normalize: bool = True,
        figsize: tuple = (18, 5),
        save: bool = True
    ) -> plt.Figure:
        """Plot confusion matrices for all models side by side."""
        n_models = len(self.results)
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, result) in zip(axes, self.results.items()):
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            cm = confusion_matrix(y_true, y_pred)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
            else:
                fmt = 'd'
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names,
                ax=ax,
                cbar=False
            )
            
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('True', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / 'confusion_matrices_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved confusion matrices comparison to {filename}")
        
        return fig
    
    def plot_roc_curves(
        self, 
        figsize: tuple = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for all models (One-vs-Rest for multiclass).
        
        Args:
            figsize: Figure size.
            save: If True, save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for class_idx, ax in enumerate(axes):
            if class_idx >= len(self.class_names):
                ax.axis('off')
                continue
            
            class_name = self.class_names[class_idx]
            
            for (model_name, result), color in zip(self.results.items(), colors):
                y_true = result['y_true']
                y_proba = result['y_proba']
                
                if y_proba is None:
                    continue
                
                # Binarize for this class
                y_true_binary = (y_true == class_idx).astype(int)
                y_score = y_proba[:, class_idx]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{model_name} (AUC={roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC - {class_name}')
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / 'roc_curves.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved ROC curves to {filename}")
        
        return fig
    
    def plot_precision_recall_curves(
        self, 
        figsize: tuple = (12, 10),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves (more informative for imbalanced data).
        
        Args:
            figsize: Figure size.
            save: If True, save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results)))
        
        for class_idx, ax in enumerate(axes):
            if class_idx >= len(self.class_names):
                ax.axis('off')
                continue
            
            class_name = self.class_names[class_idx]
            
            for (model_name, result), color in zip(self.results.items(), colors):
                y_true = result['y_true']
                y_proba = result['y_proba']
                
                if y_proba is None:
                    continue
                
                # Binarize for this class
                y_true_binary = (y_true == class_idx).astype(int)
                y_score = y_proba[:, class_idx]
                
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
                ap = average_precision_score(y_true_binary, y_score)
                
                ax.plot(recall, precision, color=color, lw=2, 
                       label=f'{model_name} (AP={ap:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve - {class_name}')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Precision-Recall Curves (More Informative for Imbalanced Data)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / 'precision_recall_curves.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved PR curves to {filename}")
        
        return fig
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: int = 20,
        figsize: tuple = (10, 8),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns.
            model_name: Name of the model.
            top_n: Number of top features to show.
            figsize: Figure size.
            save: If True, save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(
            range(len(top_features)), 
            top_features['importance'].values,
            color=colors
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances - {model_name}', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, val in zip(bars, top_features['importance'].values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / f'feature_importance_{model_name}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved feature importance to {filename}")
        
        return fig
    
    def plot_metrics_comparison(
        self,
        figsize: tuple = (12, 6),
        save: bool = True
    ) -> plt.Figure:
        """
        Plot bar chart comparing key metrics across models.
        
        Args:
            figsize: Figure size.
            save: If True, save the plot.
            
        Returns:
            Matplotlib figure.
        """
        df = self.get_comparison_table()
        
        # Key metrics for imbalanced data
        metrics = ['f1_macro', 'balanced_accuracy', 'precision_macro', 'recall_macro']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(df))
        width = 0.2
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            bars = ax.bar(x + offset, df[metric], width, label=metric, color=colors[i])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=45)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], fontsize=11)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filename = PLOTS_DIR / 'metrics_comparison.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Saved metrics comparison to {filename}")
        
        return fig
    
    def get_best_model(self, metric: str = 'f1_macro') -> str:
        """
        Get the name of the best performing model.
        
        Args:
            metric: Metric to use for comparison.
            
        Returns:
            Name of the best model.
        """
        df = self.get_comparison_table()
        best_idx = df[metric].idxmax()
        return df.loc[best_idx, 'Model']


if __name__ == "__main__":
    # Test the module with dummy data
    np.random.seed(42)
    
    # Create dummy predictions
    n_samples = 200
    n_classes = 6
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred_1 = np.random.randint(0, n_classes, n_samples)
    y_pred_2 = np.random.randint(0, n_classes, n_samples)
    y_proba_1 = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_proba_2 = np.random.dirichlet(np.ones(n_classes), n_samples)
    
    # Evaluate
    evaluator = ModelEvaluator()
    
    evaluator.evaluate_model('model_1', y_true, y_pred_1, y_proba_1)
    evaluator.evaluate_model('model_2', y_true, y_pred_2, y_proba_2)
    
    evaluator.print_comparison_summary()
    evaluator.print_classification_report('model_1')

