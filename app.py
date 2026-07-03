import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="Bio-Scan AI Lab", layout="wide")

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Header
st.title("🏥 Bio-Scan Diagnostic Laboratory")
st.markdown("### Advanced AI-Powered Clinical Analysis")

# Sidebar
st.sidebar.header("Patient Registration")
p_name = st.sidebar.text_input("Patient Full Name", "John Doe")
p_age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=30)
p_id = st.sidebar.text_input("Patient ID/MRN", "BCA-001")

# Parameters Grid (30 Features)
st.subheader("📋 Clinical Pathology Parameters")
cols = st.columns(3)
inputs = []

# Yahan 30 features ke inputs hain
feature_names = [
    "Mean radius", "Mean texture", "Mean perimeter", "Mean area", "Mean smoothness",
    "Mean compactness", "Mean concavity", "Mean concave points", "Mean symmetry", "Mean fractal dimension",
    "Radius error", "Texture error", "Perimeter error", "Area error", "Smoothness error",
    "Compactness error", "Concavity error", "Concave points error", "Symmetry error", "Fractal dimension error",
    "Worst radius", "Worst texture", "Worst perimeter", "Worst area", "Worst smoothness",
    "Worst compactness", "Worst concavity", "Worst concave points", "Worst symmetry", "Worst fractal dimension"
]

for i, name in enumerate(feature_names):
    val = cols[i % 3].number_input(name, format="%.4f", value=0.0)
    inputs.append(val)

if st.button("Generate Diagnostic Report"):
    # Prediction logic
    data = np.array(inputs).reshape(1, -1)
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    
    st.markdown("---")
    st.subheader("📋 Official Diagnostic Report")
    
    result = "MALIGNANT (Detected)" if prediction[0] == 0 else "BENIGN (Safe)"
    status = "🚨" if prediction[0] == 0 else "✅"
    
    report_content = f"""
    LAB REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    -------------------------------
    Patient Name: {p_name}
    Patient ID  : {p_id}
    Age         : {p_age}
    -------------------------------
    DIAGNOSIS RESULT: {status} {result}
    -------------------------------
    Note: This is an AI-generated report. Please consult a specialist.
    """
    st.info(report_content)
    
    st.download_button("📥 Download Report", report_content, f"Report_{p_name}.txt")
