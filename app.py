import streamlit as st
import joblib
import numpy as np

# Page Configuration
st.set_page_config(page_title="Breast Cancer AI Diagnostic", layout="wide", page_icon="🎗️")

# Styling: Custom CSS for professional look
st.markdown("""
    <style>
    .main {background-color: #f5f7f9;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white;}
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sidebar
st.sidebar.title("🩺 Medical Dashboard")
st.sidebar.markdown("---")
st.sidebar.info("This AI model is trained to assist in breast cancer detection.")
st.sidebar.write("Developed by: **Aurang Zeb**")
st.sidebar.caption("AI-powered Clinical Assistant v1.0")

# Main Content
st.title("🎗️ Breast Cancer Diagnostic AI")
st.subheader("Patient Clinical Analysis")
st.markdown("---")

# Guidance Section
with st.expander("ℹ️ User Guide: How to proceed"):
    st.write("""
    1. **Input Data:** Enter the 30 specific parameters obtained from the patient's pathology report.
    2. **Accuracy:** Ensure values match the report exactly (up to 4 decimal places).
    3. **Diagnosis:** Click 'Generate Diagnostic Report' to trigger the AI analysis.
    4. **Note:** This tool is for screening purposes only. Always consult with a qualified oncologist for final confirmation.
    """)

# Feature List
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Inputs in Columns
st.subheader("Enter Patient Parameters:")
input_data = []

cols = st.columns(3) 
for i, feature in enumerate(features):
    val = cols[i % 3].number_input(feature.replace('_', ' ').capitalize(), value=0.0, format="%.4f")
    input_data.append(val)

st.markdown("---")

# Prediction Logic
if st.button("Generate Diagnostic Report"):
    features_array = np.array([input_data])
    scaled_data = scaler.transform(features_array)
    prediction = model.predict(scaled_data)
    
    if prediction[0] == 0:
        st.error("🚨 Diagnostic Result: MALIGNANT (Detected)")
        st.write("Recommendation: Immediate consultation with an oncologist is strongly advised.")
    else:
        st.success("✅ Diagnostic Result: BENIGN (Not Detected)")
        st.write("Summary: The AI analysis indicates no immediate signs of malignancy.")