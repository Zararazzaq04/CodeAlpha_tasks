"""
Data Preprocessing Module for Heart Disease Prediction.

This module handles all data preprocessing tasks including:
- Data loading and inspection
- Missing value handling
- Duplicate removal
- Feature engineering
- Feature scaling
- Train-test splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Dict, Any

from utils import ensure_directory_exists, print_header, print_separator


class DataPreprocessor:
    """
    A class to handle all data preprocessing operations.

    Attributes:
        data_path (str): Path to the dataset
        df (pd.DataFrame): Loaded dataframe
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        scaler (StandardScaler): Feature scaler
        feature_columns (list): List of feature column names
    """

    def __init__(self, data_path: str, plots_dir: str = 'reports/plots'):
        """
        Initialize the DataPreprocessor.

        Args:
            data_path: Path to the CSV data file
            plots_dir: Directory to save plots
        """
        self.data_path = data_path
        self.plots_dir = plots_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        ensure_directory_exists(self.plots_dir)

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.

        Returns:
            Loaded DataFrame
        """
        print_header("Loading Dataset")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully from: {self.data_path}")
            print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")

    def inspect_data(self) -> Dict[str, Any]:
        """
        Perform initial data inspection.

        Returns:
            Dictionary containing inspection results
        """
        print_header("Dataset Inspection")

        inspection_results = {}

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nStatistical Summary:")
        print(self.df.describe())

        print("\nColumn Names:")
        print(self.df.columns.tolist())

        inspection_results['shape'] = self.df.shape
        inspection_results['columns'] = self.df.columns.tolist()
        inspection_results['dtypes'] = self.df.dtypes.to_dict()

        return inspection_results

    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.

        Returns:
            Series with missing value counts
        """
        print_header("Checking Missing Values")

        missing = self.df.isnull().sum()
        print("\nMissing Values per Column:")
        print(missing)

        total_missing = missing.sum()
        print(f"\nTotal Missing Values: {total_missing}")

        if total_missing == 0:
            print("No missing values found in the dataset!")

        return missing

    def check_duplicates(self) -> int:
        """
        Check for duplicate rows in the dataset.

        Returns:
            Number of duplicate rows
        """
        print_header("Checking Duplicates")

        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")

        return duplicates

    def remove_duplicates(self) -> int:
        """
        Remove duplicate rows from the dataset.

        Returns:
            Number of rows removed
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        final_rows = len(self.df)
        rows_removed = initial_rows - final_rows

        if rows_removed > 0:
            print(f"Removed {rows_removed} duplicate rows")
            print(f"Dataset shape after removing duplicates: {self.df.shape}")

        return rows_removed

    def plot_target_distribution(self) -> None:
        """
        Plot and save target class distribution.
        """
        plt.figure(figsize=(10, 6))

        colors = ['#2ecc71', '#e74c3c']
        ax = sns.countplot(data=self.df, x='target', palette=colors)

        plt.title('Target Class Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Target (0 = No Disease, 1 = Disease)', fontsize=12)
        plt.ylabel('Count', fontsize=12)

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'target_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Target distribution plot saved to: {save_path}")

    def plot_correlation_heatmap(self) -> None:
        """
        Plot and save correlation heatmap.
        """
        plt.figure(figsize=(14, 12))

        corr_matrix = self.df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                    center=0, fmt='.2f', square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})

        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'correlation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation heatmap saved to: {save_path}")

    def plot_age_distribution(self) -> None:
        """
        Plot and save age distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(self.df['age'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].set_title('Age Distribution (Histogram)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Age (years)', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)

        sns.boxplot(data=self.df, y='age', ax=axes[1], color='#3498db')
        axes[1].set_title('Age Distribution (Box Plot)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Age (years)', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'age_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Age distribution plot saved to: {save_path}")

    def plot_cholesterol_distribution(self) -> None:
        """
        Plot and save cholesterol distribution.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(self.df['chol'], bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        axes[0].set_title('Cholesterol Distribution (Histogram)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Cholesterol (mg/dl)', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)

        sns.boxplot(data=self.df, y='chol', ax=axes[1], color='#9b59b6')
        axes[1].set_title('Cholesterol Distribution (Box Plot)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Cholesterol (mg/dl)', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'cholesterol_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cholesterol distribution plot saved to: {save_path}")

    def plot_disease_by_gender(self) -> None:
        """
        Plot and save heart disease distribution by gender.
        """
        plt.figure(figsize=(10, 6))

        gender_map = {0: 'Female', 1: 'Male'}
        target_map = {0: 'No Disease', 1: 'Disease'}

        df_temp = self.df.copy()
        df_temp['sex_label'] = df_temp['sex'].map(gender_map)
        df_temp['target_label'] = df_temp['target'].map(target_map)

        colors = ['#2ecc71', '#e74c3c']
        ax = sns.countplot(data=df_temp, x='sex_label', hue='target_label', palette=colors)

        plt.title('Heart Disease Distribution by Gender', fontsize=16, fontweight='bold')
        plt.xlabel('Gender', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Diagnosis', loc='upper right')

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'disease_by_gender.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Disease by gender plot saved to: {save_path}")

    def plot_disease_by_chest_pain(self) -> None:
        """
        Plot and save heart disease distribution by chest pain type.
        """
        plt.figure(figsize=(12, 6))

        cp_map = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        target_map = {0: 'No Disease', 1: 'Disease'}

        df_temp = self.df.copy()
        df_temp['cp_label'] = df_temp['cp'].map(cp_map)
        df_temp['target_label'] = df_temp['target'].map(target_map)

        colors = ['#2ecc71', '#e74c3c']
        ax = sns.countplot(data=df_temp, x='cp_label', hue='target_label', palette=colors)

        plt.title('Heart Disease Distribution by Chest Pain Type', fontsize=16, fontweight='bold')
        plt.xlabel('Chest Pain Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Diagnosis', loc='upper right')
        plt.xticks(rotation=15)

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'disease_by_chest_pain.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Disease by chest pain plot saved to: {save_path}")

    def generate_all_plots(self) -> None:
        """
        Generate and save all EDA plots.
        """
        print_header("Generating EDA Plots")

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_age_distribution()
        self.plot_cholesterol_distribution()
        self.plot_disease_by_gender()
        self.plot_disease_by_chest_pain()

        print("\nAll EDA plots generated successfully!")

    def get_correlation_analysis(self) -> pd.DataFrame:
        """
        Analyze correlations with target variable.

        Returns:
            DataFrame with correlations sorted by absolute value
        """
        print_header("Correlation Analysis")

        target_corr = self.df.corr()['target'].drop('target')
        target_corr_sorted = target_corr.abs().sort_values(ascending=False)

        print("\nFeature correlations with target (sorted by absolute value):")
        for feature in target_corr_sorted.index:
            corr_val = target_corr[feature]
            print(f"  {feature}: {corr_val:+.4f}")

        return target_corr_sorted

    def prepare_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare train-test split of the data.

        Args:
            test_size: Proportion of test data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print_header("Preparing Train-Test Split")

        X = self.df[self.feature_columns]
        y = self.df['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set size: {len(self.X_train)} samples")
        print(f"Test set size: {len(self.X_test)} samples")
        print(f"Test size ratio: {test_size:.0%}")

        print("\nTraining set class distribution:")
        print(self.y_train.value_counts())

        print("\nTest set class distribution:")
        print(self.y_test.value_counts())

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self) -> Tuple:
        """
        Apply feature scaling to training and test data.

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        print_header("Feature Scaling")

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.feature_columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.feature_columns)

        print("Features scaled using StandardScaler")
        print(f"Scaler mean: {self.scaler.mean_}")
        print(f"Scaler variance: {self.scaler.var_}")

        return self.X_train_scaled, self.X_test_scaled

    def get_processed_data(self) -> Dict[str, Any]:
        """
        Get all processed data as a dictionary.

        Returns:
            Dictionary containing all processed data
        """
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'X_train_scaled': self.X_train_scaled if hasattr(self, 'X_train_scaled') else None,
            'X_test_scaled': self.X_test_scaled if hasattr(self, 'X_test_scaled') else None,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }

    def run_full_preprocessing(self) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.

        Returns:
            Dictionary containing all processed data
        """
        self.load_data()
        self.inspect_data()
        self.check_missing_values()
        self.check_duplicates()
        self.remove_duplicates()
        self.get_correlation_analysis()
        self.generate_all_plots()
        self.prepare_train_test_split()
        self.scale_features()

        return self.get_processed_data()


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        data_path='data/heart.csv',
        plots_dir='reports/plots'
    )

    processed_data = preprocessor.run_full_preprocessing()
    print("\nPreprocessing completed successfully!")
