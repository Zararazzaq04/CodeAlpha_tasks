"""
Model Comparison Module for Heart Disease Prediction.

This module handles comprehensive model comparison for:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

Features:
- Metrics comparison
- Visual comparison charts
- Best model selection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Tuple
import joblib

from utils import print_header, ensure_directory_exists


class ModelComparator:
    """
    A class to compare multiple trained models and select the best one.

    Attributes:
        comparison_results (dict): Dictionary to store comparison results
        reports_dir (str): Directory for reports
    """

    def __init__(self, reports_dir: str = 'reports'):
        """
        Initialize the ModelComparator.

        Args:
            reports_dir: Directory to save comparison reports
        """
        self.comparison_results = {}
        self.reports_dir = reports_dir
        self.plots_dir = os.path.join(reports_dir, 'plots')
        ensure_directory_exists(self.reports_dir)
        ensure_directory_exists(self.plots_dir)

    def add_model_results(self, model_name: str, results: Dict[str, Any]) -> None:
        """
        Add results from a model evaluation.

        Args:
            model_name: Name of the model
            results: Dictionary containing evaluation results
        """
        self.comparison_results[model_name] = results

    def compare_models(self, models_dict: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare all models and return a comparison DataFrame.

        Args:
            models_dict: Dictionary of model name to model instance
            X_test: Test features
            y_test: True test labels

        Returns:
            DataFrame comparing all models
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        print_header("Model Comparison Analysis")

        comparison_data = []

        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)

            has_proba = hasattr(model, 'predict_proba')
            if has_proba:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None

            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc if roc_auc is not None else np.nan
            }

            comparison_data.append(metrics)

            cv_score = getattr(model, 'cv_score', None)
            if cv_score is None:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X_test, y_test, cv=5)
                cv_score = cv_scores.mean()

            metrics['CV_Mean'] = cv_score

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')

        self.comparison_df = comparison_df

        print("\nModel Comparison Table:")
        print("=" * 80)
        print(comparison_df.to_string(float_format='%.4f'))
        print("=" * 80)

        return comparison_df

    def plot_metrics_comparison(self, comparison_df: pd.DataFrame = None) -> None:
        """
        Plot bar chart comparing metrics across models.

        Args:
            comparison_df: DataFrame with comparison results
        """
        if comparison_df is None and hasattr(self, 'comparison_df'):
            comparison_df = self.comparison_df

        if comparison_df is None:
            raise ValueError("No comparison data available. Run compare_models first.")

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(comparison_df.index))
        width = 0.15

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                values = comparison_df[metric].fillna(0)
                ax.bar(x + i * width, values, width, label=metric, color=colors[i])

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(comparison_df.index, fontsize=10)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, 'model_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")

        plt.close()

    def plot_radar_chart(self, comparison_df: pd.DataFrame = None) -> None:
        """
        Plot radar chart for model comparison.

        Args:
            comparison_df: DataFrame with comparison results
        """
        if comparison_df is None and hasattr(self, 'comparison_df'):
            comparison_df = self.comparison_df

        if comparison_df is None:
            raise ValueError("No comparison data available.")

        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

        labels = np.array(metrics)
        num_metrics = len(metrics)

        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        for idx, model_name in enumerate(comparison_df.index):
            values = comparison_df.loc[model_name, metrics].fillna(0).values.flatten().tolist()
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, 'radar_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to: {save_path}")

        plt.close()

    def get_best_model(self, metric: str = 'Accuracy') -> Tuple[str, float]:
        """
        Determine the best model based on a specific metric.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, best_score)
        """
        if not hasattr(self, 'comparison_df'):
            raise ValueError("No comparison data. Run compare_models first.")

        if metric not in self.comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {self.comparison_df.columns.tolist()}")

        best_model = self.comparison_df[metric].idxmax()
        best_score = self.comparison_df[metric].max()

        print(f"\nBest Model based on {metric}:")
        print(f"  Model: {best_model}")
        print(f"  Score: {best_score:.4f}")

        return best_model, best_score

    def select_and_save_best_model(self, models_dict: Dict, metric: str = 'Accuracy',
                                    save_path: str = 'models/best_model.pkl') -> Tuple:
        """
        Select and save the best performing model.

        Args:
            models_dict: Dictionary of models
            metric: Metric for selection
            save_path: Path to save the best model

        Returns:
            Tuple of (best_model_name, best_model_instance)
        """
        best_model_name, _ = self.get_best_model(metric)
        best_model = models_dict[best_model_name]

        save_dir = os.path.dirname(save_path)
        ensure_directory_exists(save_dir)

        joblib.dump(best_model, save_path)
        print(f"\nBest model ({best_model_name}) saved to: {save_path}")

        return best_model_name, best_model

    def generate_comparison_report(self) -> None:
        """
        Generate a comprehensive comparison report.
        """
        if not hasattr(self, 'comparison_df'):
            raise ValueError("No comparison data. Run compare_models first.")

        report_path = os.path.join(self.reports_dir, 'model_comparison_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("HEART DISEASE PREDICTION - MODEL COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Performance Metrics:\n")
            f.write("-" * 70 + "\n\n")

            f.write(self.comparison_df.to_string(float_format='%.4f'))
            f.write("\n\n")

            for idx, row in self.comparison_df.iterrows():
                f.write(f"\n{idx}:\n")
                for metric, value in row.items():
                    f.write(f"  {metric}: {value:.4f}\n")

            best_model, best_score = self.get_best_model('Accuracy')
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"RECOMMENDATION: Use {best_model} as the final model\n")
            f.write(f"Best Accuracy Score: {best_score:.4f}\n")
            f.write("=" * 70 + "\n")

        print(f"Comparison report saved to: {report_path}")

    def run_full_comparison(self, models_dict: Dict, X_test: pd.DataFrame,
                            y_test: pd.Series) -> Tuple[pd.DataFrame, str]:
        """
        Run complete model comparison pipeline.

        Args:
            models_dict: Dictionary of models
            X_test: Test features
            y_test: Test labels

        Returns:
            Tuple of (comparison_df, best_model_name)
        """
        comparison_df = self.compare_models(models_dict, X_test, y_test)

        self.plot_metrics_comparison(comparison_df)

        try:
            self.plot_radar_chart(comparison_df)
        except Exception as e:
            print(f"Could not generate radar chart: {e}")

        best_model_name, _ = self.get_best_model('Accuracy')

        self.generate_comparison_report()

        return comparison_df, best_model_name


if __name__ == "__main__":
    print("Model Comparator Module")
    print("Import this module and use with trained models.")
