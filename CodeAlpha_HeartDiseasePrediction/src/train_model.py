"""
Model Training Module for Heart Disease Prediction.

This module handles model training and hyperparameter tuning for:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
import joblib
import os
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils import print_header, print_separator, ensure_directory_exists


class ModelTrainer:
    """
    A class to handle model training and hyperparameter tuning.

    Attributes:
        models (dict): Dictionary of model instances
        best_models (dict): Dictionary of best tuned models
        training_results (dict): Dictionary to store training results
    """

    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the ModelTrainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        ensure_directory_exists(models_dir)

        self.models = {}
        self.best_models = {}
        self.training_results = {}

        self._initialize_models()

    def _initialize_models(self) -> None:
        """
        Initialize all classification models with default parameters.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_jobs=1
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=1
            ),
            'SVM': SVC(
                probability=True,
                random_state=42
            )
        }

        print("Initialized models:")
        for name in self.models.keys():
            print(f"  - {name}")

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all models with default parameters.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary containing trained models
        """
        print_header("Training Models with Default Parameters")

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            self.training_results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score
            }

            print(f"  Cross-validation scores: {cv_scores}")
            print(f"  Mean CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")

        return self.training_results

    def tune_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Perform hyperparameter tuning for Random Forest using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        print_header("Tuning Random Forest Classifier")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        print("\nPerforming GridSearchCV...")

        rf_base = RandomForestClassifier(random_state=42, n_jobs=1)
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        self.best_models['Random Forest'] = best_model

        print(f"\nBest Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score:.4f}")

        return best_model, best_params, best_score

    def tune_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Perform hyperparameter tuning for XGBoost using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        print_header("Tuning XGBoost Classifier")

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        print("\nPerforming GridSearchCV...")

        xgb_base = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=1
        )

        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        self.best_models['XGBoost'] = best_model

        print(f"\nBest Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score:.4f}")

        return best_model, best_params, best_score

    def tune_svm(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Perform hyperparameter tuning for SVM using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        print_header("Tuning Support Vector Machine (SVM)")

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        print("\nPerforming GridSearchCV...")

        svm_base = SVC(probability=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=svm_base,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        self.best_models['SVM'] = best_model

        print(f"\nBest Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score:.4f}")

        return best_model, best_params, best_score

    def perform_hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for all applicable models.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary containing tuned models and their parameters
        """
        print_header("Hyperparameter Tuning")

        tuning_results = {}

        rf_model, rf_params, rf_score = self.tune_random_forest(X_train, y_train)
        tuning_results['Random Forest'] = {
            'model': rf_model,
            'best_params': rf_params,
            'best_score': rf_score
        }

        xgb_model, xgb_params, xgb_score = self.tune_xgboost(X_train, y_train)
        tuning_results['XGBoost'] = {
            'model': xgb_model,
            'best_params': xgb_params,
            'best_score': xgb_score
        }

        svm_model, svm_params, svm_score = self.tune_svm(X_train, y_train)
        tuning_results['SVM'] = {
            'model': svm_model,
            'best_params': svm_params,
            'best_score': svm_score
        }

        self.models['Logistic Regression'].fit(X_train, y_train)
        self.best_models['Logistic Regression'] = self.models['Logistic Regression']

        lr_cv_scores = cross_val_score(self.models['Logistic Regression'], X_train, y_train, cv=5, scoring='accuracy')
        tuning_results['Logistic Regression'] = {
            'model': self.models['Logistic Regression'],
            'best_params': {'max_iter': 1000, 'solver': 'lbfgs'},
            'best_score': lr_cv_scores.mean()
        }

        return tuning_results

    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Args:
            model_name: Name of the model to get importance from

        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.best_models.keys())}")

        model = self.best_models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError(f"Model {model_name} does not support feature importance")

        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print(f"\nFeature Importance ({model_name}):")
        print(importance_df.to_string(index=False))

        return importance_df

    def save_model(self, model, filename: str) -> str:
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save
            filename: Name of the file to save

        Returns:
            Path to saved model
        """
        filepath = os.path.join(self.models_dir, filename)
        joblib.dump(model, filepath)
        print(f"Model saved to: {filepath}")
        return filepath

    def load_model(self, filename: str):
        """
        Load a saved model from disk.

        Args:
            filename: Name of the file to load

        Returns:
            Loaded model
        """
        filepath = os.path.join(self.models_dir, filename)
        model = joblib.load(filepath)
        print(f"Model loaded from: {filepath}")
        return model

    def get_all_trained_models(self) -> Dict[str, Any]:
        """
        Get all trained models.

        Returns:
            Dictionary of all models
        """
        return self.best_models


if __name__ == "__main__":
    from preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor(data_path='data/heart.csv')
    preprocessor.load_data()
    preprocessor.check_missing_values()
    preprocessor.check_duplicates()
    preprocessor.remove_duplicates()
    preprocessor.prepare_train_test_split()
    X_train_scaled, X_test_scaled = preprocessor.scale_features()

    trainer = ModelTrainer(models_dir='models')
    trainer.train_all_models(X_train_scaled, preprocessor.y_train)
    tuning_results = trainer.perform_hyperparameter_tuning(X_train_scaled, preprocessor.y_train)

    print("\nTuning Results:")
    for model_name, results in tuning_results.items():
        print(f"{model_name}: {results['best_score']:.4f}")
