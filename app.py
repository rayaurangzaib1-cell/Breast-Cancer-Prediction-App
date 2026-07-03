import streamlit as st
import joblib
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="Bio-Scan AI Lab", layout="wide")

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# --- Header Section ---
st.title("🏥 Bio-Scan Diagnostic Laboratory")
st.markdown("### Advanced AI-Powered Clinical Analysis")
st.info("Clinical Assistant v2.0 - Powered by Machine Learning")

# --- Patient Data Input ---
st.sidebar.header("Patient Registration")
p_name = st.sidebar.text_input("Patient Full Name")
p_age = st.sidebar.number_input("Age", min_value=1, max_value=100)
p_id = st.sidebar.text_input("Patient ID/MRN")
lab_name = "Bio-Scan Diagnostic Lab"

# --- Parameters Grid ---
st.subheader("📋 Clinical Pathology Parameters")
cols = st.columns(3)
input_values = []

# Yahan hum 30 parameters ko 3 columns mein distribute kar rahe hain
feature_names = ["Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness", 
                 "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dim",
                 "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
Normally I can help with things like this, but I don't seem to have access to that content. You can try again or ask me for something else.
