"""
Heart Disease Prediction Web Application

A professional Streamlit application for predicting heart disease
based on patient medical data.

Author: CodeAlpha Machine Learning Intern
Project: Disease Prediction from Medical Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Union
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
    }
    .prediction-negative {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #333333;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .disclaimer strong {
        color: #664d03;
    }
    .stMetric label {
        font-size: 1rem !important;
    }
    .stMetric value {
        font-size: 1.5rem !important;
    }
</style>
"""


def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler."""
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Please run 'python main.py' first to train the model.")
        return None, None

    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at: {scaler_path}")
        st.info("Please run 'python main.py' first to train the model.")
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def create_sidebar_inputs() -> Dict[str, float]:
    """
    Create sidebar input widgets for patient data.

    Returns:
        Dictionary of patient attributes
    """
    st.sidebar.header("Patient Information")
    st.sidebar.markdown("---")

    with st.sidebar.expander("Demographics", expanded=True):
        age = st.slider(
            "Age (years)",
            min_value=20,
            max_value=100,
            value=55,
            help="Patient's age in years"
        )

        sex = st.radio(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Patient's biological sex"
        )

    with st.sidebar.expander("Cardiac Symptoms", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            cp = st.select_slider(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                value=2,
                help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic"
            )

        with col2:
            exang = st.radio(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Chest pain during physical activity"
            )

        oldpeak = st.slider(
            "ST Depression (oldpeak)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest"
        )

    with st.sidebar.expander("Vital Signs", expanded=True):
        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=130,
            help="Resting blood pressure on admission"
        )

        chol = st.slider(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=250,
            help="Serum cholesterol measurement"
        )

        thalach = st.slider(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150,
            help="Maximum heart rate achieved during exercise"
        )

        fbs = st.radio(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Whether fasting blood sugar exceeds 120 mg/dl"
        )

    with st.sidebar.expander("ECG Results", expanded=True):
        restecg = st.select_slider(
            "Resting ECG Results",
            options=[0, 1, 2],
            value=1,
            help="0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy"
        )

        slope = st.select_slider(
            "ST Segment Slope",
            options=[0, 1, 2],
            value=2,
            help="0: Upsloping, 1: Flat, 2: Downsloping"
        )

    with st.sidebar.expander("Additional Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            ca = st.select_slider(
                "Major Vessels",
                options=[0, 1, 2, 3, 4],
                value=0,
                help="Number of major vessels colored by fluoroscopy (0-4)"
            )

        with col2:
            thal = st.select_slider(
                "Thalassemia",
                options=[0, 1, 2, 3],
                value=2,
                help="0: Null, 1: Normal, 2: Fixed Defect, 3: Reversible Defect"
            )

    patient_data = {
        'age': float(age),
        'sex': float(sex),
        'cp': float(cp),
        'trestbps': float(trestbps),
        'chol': float(chol),
        'fbs': float(fbs),
        'restecg': float(restecg),
        'thalach': float(thalach),
        'exang': float(exang),
        'oldpeak': float(oldpeak),
        'slope': float(slope),
        'ca': float(ca),
        'thal': float(thal)
    }

    return patient_data


def preprocess_input(patient_data: Dict, scaler) -> np.ndarray:
    """
    Preprocess patient input data.

    Args:
        patient_data: Dictionary of patient attributes
        scaler: Fitted StandardScaler

    Returns:
        Preprocessed feature array
    """
    feature_order = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    features = np.array([[patient_data[col] for col in feature_order]])

    features_scaled = scaler.transform(features)

    return features_scaled


def make_prediction(model, scaler, patient_data: Dict) -> Dict:
    """
    Make heart disease prediction.

    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        patient_data: Patient attributes dictionary

    Returns:
        Dictionary with prediction results
    """
    processed_data = preprocess_input(patient_data, scaler)

    prediction = int(model.predict(processed_data)[0])

    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(processed_data)[0][1]
    elif hasattr(model, 'decision_function'):
        decision = model.decision_function(processed_data)[0]
        probability = 1 / (1 + np.exp(-decision))
    else:
        probability = 0.5

    confidence = max(probability, 1 - probability) * 100

    risk_category = get_risk_category(probability)

    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'risk_category': risk_category
    }


def get_risk_category(probability: float) -> str:
    """Convert probability to risk category."""
    if probability < 0.20:
        return "Very Low"
    elif probability < 0.40:
        return "Low"
    elif probability < 0.60:
        return "Moderate"
    elif probability < 0.80:
        return "High"
    else:
        return "Very High"


def create_gauge_chart(probability: float) -> go.Figure:
    """
    Create a gauge chart for risk probability.

    Args:
        probability: Disease probability (0-1)

    Returns:
        Plotly gauge figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': "Heart Disease Risk Score",
            'font': {'size': 24, 'color': '#2c3e50'}
        },
        delta={
            'reference': 50,
            'increasing': {'color': "#e74c3c"},
            'decreasing': {'color': "#27ae60"}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#e74c3c" if probability > 0.5 else "#27ae60"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#d5f5e3'},
                {'range': [20, 40], 'color': '#abebc6'},
                {'range': [40, 60], 'color': '#f9e79f'},
                {'range': [60, 80], 'color': '#f5b041'},
                {'range': [80, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        },
        number={'font': {'size': 40}, 'suffix': '%'}
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def display_prediction_result(result: Dict, patient_data: Dict):
    """
    Display the prediction result with visualizations.

    Args:
        result: Prediction result dictionary
        patient_data: Patient input data
    """
    prediction = result['prediction']
    probability = result['probability']
    confidence = result['confidence']
    risk_category = result['risk_category']

    if prediction == 1:
        st.markdown(
            f"""
            <div class="prediction-positive">
                <h1>High Risk of Heart Disease</h1>
                <h3>Risk Probability: {probability * 100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="prediction-negative">
                <h1>Low Risk of Heart Disease</h1>
                <h3>Risk Probability: {probability * 100:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        gauge_fig = create_gauge_chart(probability)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        st.markdown("<h3 style='text-align: center;'>Risk Assessment</h3>", unsafe_allow_html=True)
        st.metric("Risk Category", risk_category)
        st.metric("Model Confidence", f"{confidence:.2f}%")
        st.metric("Prediction", "Disease" if prediction == 1 else "Healthy")

    with col3:
        st.markdown("<h3 style='text-align: center;'>Key Indicators</h3>", unsafe_allow_html=True)
        st.metric("Age", f"{int(patient_data['age'])} years")
        st.metric("Blood Pressure", f"{int(patient_data['trestbps'])} mm Hg")
        st.metric("Cholesterol", f"{int(patient_data['chol'])} mg/dl")

    st.markdown("---")

    st.markdown("#### Patient Input Summary")
    df_summary = pd.DataFrame([patient_data])
    st.dataframe(df_summary.T.rename(columns={0: "Value"}), use_container_width=True)


def display_disclaimer():
    """Display medical disclaimer."""
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Medical Disclaimer:</strong> This tool is for educational and informational purposes only.
            It is not intended to diagnose, treat, cure, or prevent any disease. The predictions made by
            this model should not be considered medical advice. Always consult with a qualified healthcare
            professional for medical decisions. This system should be used as a preliminary screening tool,
            not a substitute for professional medical evaluation.
        </div>
        """,
        unsafe_allow_html=True
    )


def display_about_section():
    """Display information about the model and project."""
    with st.expander("About This Application"):
        st.markdown("""
        ### Heart Disease Prediction System

        This web application uses machine learning to predict the likelihood of heart disease
        based on patient medical data.

        **Technical Details:**
        - **Algorithm:** Ensemble of machine learning models (Logistic Regression, Random Forest, XGBoost)
        - **Training Data:** Heart Disease Dataset from Kaggle (303 unique patients)
        - **Features:** 13 clinical parameters including age, gender, chest pain type, blood pressure, etc.
        - **Performance:** Model achieves high accuracy on the test set

        **Project Information:**
        - **Author:** CodeAlpha Machine Learning Intern
        - **Project:** Disease Prediction from Medical Data

        **How to Use:**
        1. Enter patient information in the sidebar
        2. Click the "Predict Heart Disease" button
        3. View the prediction results and risk assessment

        **Interpretation:**
        - **Risk Score:** Probability of heart disease (0-100%)
        - **Risk Category:** Severity level based on probability
        - **Model Confidence:** How certain the model is about its prediction
        """)


def main():
    """Main function to run the Streamlit app."""
    apply_custom_css()

    st.markdown("<h1 class='main-header'>Heart Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Machine Learning-Based Medical Diagnosis Tool</p>", unsafe_allow_html=True)

    display_disclaimer()

    st.markdown("---")

    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.stop()

    patient_data = create_sidebar_inputs()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "Predict Heart Disease",
            type="primary",
            use_container_width=True
        )

    if predict_button:
        with st.spinner("Analyzing patient data..."):
            result = make_prediction(model, scaler, patient_data)

        st.success("Prediction complete!")
        display_prediction_result(result, patient_data)

    display_about_section()

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
            <p>CodeAlpha Machine Learning Internship Project</p>
            <p>Disease Prediction from Medical Data</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
