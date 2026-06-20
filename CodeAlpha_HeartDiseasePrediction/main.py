#!/usr/bin/env python3
"""
Heart Disease Prediction System - Main Execution Script

This is the main entry point for the Heart Disease Prediction System.
It orchestrates the complete machine learning pipeline including:
- Data preprocessing and EDA
- Model training with hyperparameter tuning
- Model evaluation and comparison
- Best model selection and saving

Author: CodeAlpha Machine Learning Intern
Project: Disease Prediction from Medical Data
"""

import os
import sys
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import DataPreprocessor
from train_model import ModelTrainer
from evaluate_model import ModelEvaluator
from model_comparison import ModelComparator
from utils import print_header, print_separator


class HeartDiseasePredictionPipeline:
    """
    Main pipeline class for heart disease prediction.

    Orchestrates all components of the ML pipeline.
    """

    def __init__(self, data_path: str = 'data/heart.csv',
                 models_dir: str = 'models',
                 reports_dir: str = 'reports'):
        """
        Initialize the pipeline.

        Args:
            data_path: Path to the dataset
            models_dir: Directory for saved models
            reports_dir: Directory for reports and plots
        """
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.processed_data = None
        self.trained_models = None
        self.best_model = None
        self.best_model_name = None

    def run_preprocessing(self) -> dict:
        """
        Run data preprocessing and EDA.

        Returns:
            Dictionary of processed data
        """
        print_header("STEP 1: DATA PREPROCESSING AND EDA")

        preprocessor = DataPreprocessor(
            data_path=self.data_path,
            plots_dir=os.path.join(self.reports_dir, 'plots')
        )

        self.processed_data = preprocessor.run_full_preprocessing()

        self.preprocessor = preprocessor

        return self.processed_data

    def run_training(self) -> dict:
        """
        Run model training and hyperparameter tuning.

        Returns:
            Dictionary of trained models
        """
        print_header("STEP 2: MODEL TRAINING AND TUNING")

        trainer = ModelTrainer(models_dir=self.models_dir)

        X_train = self.processed_data['X_train_scaled']
        y_train = self.processed_data['y_train']

        trainer.train_all_models(X_train, y_train)

        tuning_results = trainer.perform_hyperparameter_tuning(X_train, y_train)

        self.trainer = trainer
        self.trained_models = {name: result['model'] for name, result in tuning_results.items()}

        return self.trained_models

    def run_evaluation(self) -> dict:
        """
        Run model evaluation.

        Returns:
            Dictionary of evaluation results
        """
        print_header("STEP 3: MODEL EVALUATION")

        evaluator = ModelEvaluator(reports_dir=self.reports_dir)

        X_test = self.processed_data['X_test_scaled']
        y_test = self.processed_data['y_test']

        self.evaluation_results = evaluator.generate_full_evaluation(
            self.trained_models, X_test, y_test
        )

        self.evaluator = evaluator

        return self.evaluation_results

    def run_comparison(self) -> tuple:
        """
        Run model comparison and select best model.

        Returns:
            Tuple of (comparison_df, best_model_name)
        """
        print_header("STEP 4: MODEL COMPARISON AND SELECTION")

        comparator = ModelComparator(reports_dir=self.reports_dir)

        X_test = self.processed_data['X_test_scaled']
        y_test = self.processed_data['y_test']

        comparison_df, self.best_model_name = comparator.run_full_comparison(
            self.trained_models, X_test, y_test
        )

        self.comparator = comparator

        return comparison_df, self.best_model_name

    def save_best_model(self) -> str:
        """
        Save the best performing model.

        Returns:
            Path to saved model
        """
        print_header("STEP 5: SAVING BEST MODEL")

        self.best_model = self.trained_models[self.best_model_name]

        model_path = os.path.join(self.models_dir, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"Best model ({self.best_model_name}) saved to: {model_path}")

        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        joblib.dump(self.processed_data['scaler'], scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        return model_path

    def run_demo_prediction(self) -> None:
        """
        Run a demonstration prediction with sample data.
        """
        print_header("STEP 6: DEMONSTRATION PREDICTION")

        from predict import HeartDiseasePredictor

        predictor = HeartDiseasePredictor()
        predictor.set_model(self.best_model)
        predictor.set_scaler(self.processed_data['scaler'])

        sample_patient = {
            'age': 54,
            'sex': 1,
            'cp': 2,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 1,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 1.5,
            'slope': 2,
            'ca': 0,
            'thal': 2
        }

        print("\nSample Patient Data:")
        for key, value in sample_patient.items():
            print(f"  {key}: {value}")

        result = predictor.predict(sample_patient)
        predictor.display_prediction_result(result)

    def run_complete_pipeline(self) -> None:
        """
        Run the complete ML pipeline from start to finish.
        """
        start_time = datetime.now()

        print("\n" + "=" * 70)
        print("  HEART DISEASE PREDICTION SYSTEM")
        print("  CodeAlpha Machine Learning Internship Project")
        print("  Started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70 + "\n")

        self.run_preprocessing()

        self.run_training()

        self.run_evaluation()

        self.run_comparison()

        self.save_best_model()

        self.run_demo_prediction()

        end_time = datetime.now()
        duration = end_time - start_time

        print_header("PIPELINE COMPLETED")
        print(f"Total Execution Time: {duration}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Model saved to: {os.path.join(self.models_dir, 'best_model.pkl')}")

        print("\n" + "=" * 70)
        print("  All outputs saved to respective directories:")
        print(f"  - Models: {self.models_dir}/")
        print(f"  - Reports: {self.reports_dir}/")
        print(f"  - Plots: {os.path.join(self.reports_dir, 'plots')}/")
        print("=" * 70 + "\n")


def main():
    """
    Main entry point for the Heart Disease Prediction System.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))

    pipeline = HeartDiseasePredictionPipeline(
        data_path=os.path.join(project_dir, 'data', 'heart.csv'),
        models_dir=os.path.join(project_dir, 'models'),
        reports_dir=os.path.join(project_dir, 'reports')
    )

    pipeline.run_complete_pipeline()

    print("\nTo run the Streamlit web application, execute:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()