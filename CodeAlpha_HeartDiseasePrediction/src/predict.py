"""
Prediction Module for Heart Disease Prediction.

This module handles making predictions on new patient data:
- Single patient prediction
- Batch predictions
- Risk probability calculation
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Union, List
from sklearn.preprocessing import StandardScaler

from utils import print_header, calculate_risk_category


class HeartDiseasePredictor:
    """
    A class to handle heart disease predictions.

    Attributes:
        model: Trained model for prediction
        scaler: Fitted StandardScaler
        feature_columns (list): List of feature column names
    """

    FEATURE_COLUMNS = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    FEATURE_DESCRIPTIONS = {
        'age': 'Age in years',
        'sex': 'Gender (1 = Male, 0 = Female)',
        'cp': 'Chest Pain Type (0-3)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
        'restecg': 'Resting ECG Results (0-2)',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina (1 = Yes, 0 = No)',
        'oldpeak': 'ST Depression Induced by Exercise',
        'slope': 'Slope of Peak Exercise ST Segment (0-2)',
        'ca': 'Number of Major Vessels (0-4)',
        'thal': 'Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)'
    }

    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model file
            scaler_path: Path to saved scaler file
        """
        self.model = None
        self.scaler = None

        if model_path:
            self.load_model(model_path)

        if scaler_path:
            self.load_scaler(scaler_path)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to model file
        """
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from: {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def load_scaler(self, scaler_path: str) -> None:
        """
        Load a fitted scaler from disk.

        Args:
            scaler_path: Path to scaler file
        """
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded successfully from: {scaler_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
        except Exception as e:
            raise Exception(f"Error loading scaler: {str(e)}")

    def set_model(self, model) -> None:
        """
        Set the model directly.

        Args:
            model: Trained model instance
        """
        self.model = model

    def set_scaler(self, scaler: StandardScaler) -> None:
        """
        Set the scaler directly.

        Args:
            scaler: Fitted StandardScaler instance
        """
        self.scaler = scaler

    def validate_input(self, patient_data: Dict) -> tuple:
        """
        Validate patient input data.

        Args:
            patient_data: Dictionary of patient attributes

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for feature in self.FEATURE_COLUMNS:
            if feature not in patient_data:
                errors.append(f"Missing feature: {feature}")

        validation_rules = {
            'age': (0, 120, 'Age must be between 0 and 120'),
            'sex': (0, 1, 'Sex must be 0 or 1'),
            'cp': (0, 3, 'Chest pain type must be between 0 and 3'),
            'trestbps': (50, 250, 'Resting blood pressure must be between 50 and 250'),
            'chol': (50, 600, 'Cholesterol must be between 50 and 600'),
            'fbs': (0, 1, 'Fasting blood sugar must be 0 or 1'),
            'restecg': (0, 2, 'Resting ECG must be between 0 and 2'),
            'thalach': (50, 250, 'Max heart rate must be between 50 and 250'),
            'exang': (0, 1, 'Exercise induced angina must be 0 or 1'),
            'oldpeak': (0, 10, 'ST depression must be between 0 and 10'),
            'slope': (0, 2, 'Slope must be between 0 and 2'),
            'ca': (0, 4, 'Number of vessels must be between 0 and 4'),
            'thal': (0, 3, 'Thalassemia must be between 0 and 3')
        }

        for feature, (min_val, max_val, error_msg) in validation_rules.items():
            if feature in patient_data:
                value = patient_data[feature]
                if not isinstance(value, (int, float)):
                    errors.append(f"{feature}: Invalid type, must be numeric")
                elif value < min_val or value > max_val:
                    errors.append(f"{feature}: {error_msg}")

        return len(errors) == 0, errors

    def preprocess_input(self, patient_data: Dict) -> np.ndarray:
        """
        Preprocess patient data for prediction.

        Args:
            patient_data: Dictionary of patient attributes

        Returns:
            Preprocessed feature array
        """
        features = []
        for col in self.FEATURE_COLUMNS:
            features.append(patient_data.get(col, 0))

        features_array = np.array(features).reshape(1, -1)

        if self.scaler is not None:
            features_array = self.scaler.transform(features_array)

        return features_array

    def predict(self, patient_data: Dict) -> Dict[str, Union[int, float, str]]:
        """
        Make a prediction for a single patient.

        Args:
            patient_data: Dictionary of patient attributes

        Returns:
            Dictionary containing prediction, probability, and risk category
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        is_valid, errors = self.validate_input(patient_data)
        if not is_valid:
            raise ValueError(f"Invalid input data:\n" + "\n".join(errors))

        processed_data = self.preprocess_input(patient_data)

        prediction = int(self.model.predict(processed_data)[0])

        probability = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(processed_data)[0]
            probability = float(proba[1])
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(processed_data)[0]
            probability = float(1 / (1 + np.exp(-decision)))

        risk_category = calculate_risk_category(probability) if probability is not None else "Unknown"

        result = {
            'prediction': prediction,
            'prediction_label': 'High Risk of Heart Disease' if prediction == 1 else 'Low Risk of Heart Disease',
            'probability': probability,
            'probability_percent': f"{probability * 100:.2f}%" if probability is not None else None,
            'risk_category': risk_category,
            'confidence': f"{max(probability, 1-probability) * 100:.2f}%" if probability is not None else None
        }

        return result

    def predict_batch(self, patients_data: List[Dict]) -> List[Dict[str, Union[int, float, str]]]:
        """
        Make predictions for multiple patients.

        Args:
            patients_data: List of patient attribute dictionaries

        Returns:
            List of prediction results
        """
        results = []
        for i, patient in enumerate(patients_data):
            try:
                result = self.predict(patient)
                result['patient_index'] = i + 1
                results.append(result)
            except Exception as e:
                results.append({
                    'patient_index': i + 1,
                    'error': str(e)
                })

        return results

    def display_prediction_result(self, result: Dict) -> None:
        """
        Display prediction result in a formatted manner.

        Args:
            result: Prediction result dictionary
        """
        print_header("PREDICTION RESULT")
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Risk Probability: {result['probability_percent']}")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Model Confidence: {result['confidence']}")

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for all features.

        Returns:
            Dictionary of feature to description
        """
        return self.FEATURE_DESCRIPTIONS

    def create_sample_input(self) -> Dict[str, Union[int, float]]:
        """
        Create a sample input dictionary for testing.

        Returns:
            Sample patient data dictionary
        """
        return {
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


def interactive_prediction(model_path: str = 'models/best_model.pkl',
                          scaler_path: str = 'models/scaler.pkl') -> None:
    """
    Run interactive prediction in the console.

    Args:
        model_path: Path to saved model
        scaler_path: Path to saved scaler
    """
    print_header("Heart Disease Prediction - Interactive Mode")

    predictor = HeartDiseasePredictor(model_path, scaler_path)

    print("\nEnter patient information:")
    print("-" * 40)

    patient_data = {}

    feature_prompts = {
        'age': 'Age (years): ',
        'sex': 'Sex (1=Male, 0=Female): ',
        'cp': 'Chest Pain Type (0-3): ',
        'trestbps': 'Resting Blood Pressure (mm Hg): ',
        'chol': 'Serum Cholesterol (mg/dl): ',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No): ',
        'restecg': 'Resting ECG Results (0-2): ',
        'thalach': 'Max Heart Rate Achieved: ',
        'exang': 'Exercise Induced Angina (1=Yes, 0=No): ',
        'oldpeak': 'ST Depression: ',
        'slope': 'Slope of Peak ST Segment (0-2): ',
        'ca': 'Number of Major Vessels (0-4): ',
        'thal': 'Thalassemia (0-3): '
    }

    for feature, prompt in feature_prompts.items():
        while True:
            try:
                value = float(input(prompt))
                patient_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    result = predictor.predict(patient_data)
    predictor.display_prediction_result(result)


if __name__ == "__main__":
    predictor = HeartDiseasePredictor()

    sample = predictor.create_sample_input()
    print("Sample input:")
    for key, value in sample.items():
        print(f"  {key}: {value}")