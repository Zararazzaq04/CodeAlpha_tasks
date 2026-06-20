"""
Model Evaluation Module for Heart Disease Prediction.

This module handles model evaluation with multiple metrics:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix
- Classification Report
- ROC Curve
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Tuple
import joblib

from utils import print_header, print_separator, ensure_directory_exists


class ModelEvaluator:
    """
    A class to handle model evaluation and metrics visualization.

    Attributes:
        reports_dir (str): Directory to save evaluation reports
        results (dict): Dictionary to store evaluation results
    """

    def __init__(self, reports_dir: str = 'reports'):
        """
        Initialize the ModelEvaluator.

        Args:
            reports_dir: Directory to save evaluation reports
        """
        self.reports_dir = reports_dir
        self.plots_dir = os.path.join(reports_dir, 'plots')
        ensure_directory_exists(reports_dir)
        ensure_directory_exists(self.plots_dir)
        self.results = {}

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model and return metrics.

        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: True test labels
            model_name: Name of the model

        Returns:
            Dictionary containing evaluation results
        """
        print_header(f"Evaluating {model_name}")

        y_pred = model.predict(X_test)

        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test)

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        cm = confusion_matrix(y_test, y_pred)

        clf_report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'model_name': model_name,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': clf_report
        }

        self.results[model_name] = results

        print(f"\nMetrics for {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print(f"\nConfusion Matrix:")
        print(cm)

        return results

    def plot_confusion_matrix(self, model_name: str, y_test: pd.Series = None, cm: np.ndarray = None, save: bool = True) -> None:
        """
        Plot confusion matrix for a model.

        Args:
            model_name: Name of the model
            y_test: True labels (optional if cm is provided)
            cm: Pre-computed confusion matrix (optional)
            save: Whether to save the plot
        """
        if cm is None and model_name in self.results:
            cm = self.results[model_name]['confusion_matrix']

        if cm is None:
            raise ValueError("Confusion matrix not found. Run evaluate_model first.")

        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])

        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.5, str(cm[i, j]),
                        ha='center', va='center', fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.reports_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.close()

    def plot_roc_curve(self, models_dict: Dict, X_test: pd.DataFrame, y_test: pd.Series, save: bool = True) -> None:
        """
        Plot ROC curves for all models.

        Args:
            models_dict: Dictionary of model name to model instance
            X_test: Test features
            y_test: True test labels
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 8))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        for idx, (name, model) in enumerate(models_dict.items()):
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)

            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                    label=f'{name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.reports_dir, 'roc_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")

        plt.close()

    def save_classification_report(self, model_name: str, y_test: pd.Series = None, y_pred: np.ndarray = None) -> None:
        """
        Save classification report to a text file.

        Args:
            model_name: Name of the model
            y_test: True labels
            y_pred: Predicted labels
        """
        if model_name in self.results:
            y_test = y_test
            y_pred = self.results[model_name]['y_pred']

        if y_test is None or y_pred is None:
            raise ValueError("Provide y_test and y_pred or run evaluate_model first.")

        report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])

        save_path = os.path.join(self.reports_dir, 'classification_report.txt')

        with open(save_path, 'a') as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Classification Report - {model_name}\n")
            f.write(f"{'=' * 60}\n")
            f.write(report)
            f.write(f"\n{'=' * 60}\n\n")

        print(f"Classification report saved to: {save_path}")

    def generate_all_confusion_matrices(self) -> None:
        """
        Generate confusion matrices for all evaluated models.
        """
        print_header("Generating Confusion Matrices")

        for model_name in self.results.keys():
            self.plot_confusion_matrix(model_name)

    def get_metrics_comparison(self) -> pd.DataFrame:
        """
        Get a comparison table of all model metrics.

        Returns:
            DataFrame comparing metrics across all models
        """
        comparison_data = []

        for model_name, results in self.results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics.get('roc_auc', 'N/A')
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')

        print_header("Model Comparison Table")
        print(comparison_df.to_string())

        return comparison_df

    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, best_score)
        """
        best_model = None
        best_score = -1

        for model_name, results in self.results.items():
            score = results['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name

        print(f"\nBest Model ({metric}): {best_model} with score {best_score:.4f}")
        return best_model, best_score

    def save_evaluation_summary(self) -> None:
        """
        Save a comprehensive evaluation summary to a text file.
        """
        summary_path = os.path.join(self.reports_dir, 'evaluation_summary.txt')

        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("HEART DISEASE PREDICTION - MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            for model_name, results in self.results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 40 + "\n")

                metrics = results['metrics']
                f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall:    {metrics['recall']:.4f}\n")
                f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
                if 'roc_auc' in metrics:
                    f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")

                cm = results['confusion_matrix']
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
                f.write(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")

                f.write("\n" + "=" * 60 + "\n\n")

        print(f"Evaluation summary saved to: {summary_path}")

    def generate_full_evaluation(self, models_dict: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Run full evaluation pipeline for all models.

        Args:
            models_dict: Dictionary of model name to model instance
            X_test: Test features
            y_test: True test labels

        Returns:
            DataFrame comparing all model metrics
        """
        print_header("Full Model Evaluation")

        for name, model in models_dict.items():
            self.evaluate_model(model, X_test, y_test, name)
            self.save_classification_report(name, y_test, self.results[name]['y_pred'])

        self.generate_all_confusion_matrices()
        self.plot_roc_curve(models_dict, X_test, y_test)

        comparison = self.get_metrics_comparison()

        best_model, best_score = self.get_best_model('accuracy')

        self.save_evaluation_summary()

        return comparison


if __name__ == "__main__":
    print("Model Evaluator Module")
    print("Import this module and use with trained models.")