"""HeartGuard AI — Streamlit prediction app."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="\U0001FAC0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0E1117; }
.title-text { font-size: 50px; font-weight: bold; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
.subtitle-text { font-size: 20px; color: #FAFAFA; text-align: center; margin-bottom: 40px; }
.stButton>button { background-color: #FF4B4B; color: white; font-size: 20px; border-radius: 10px;
                   height: 50px; width: 100%; border: none; }
.stButton>button:hover { background-color: #FF0000; color: white; }
.result-box-high { background-color: #3d0e0e; border: 2px solid #ff4b4b; padding: 20px;
                   border-radius: 10px; text-align: center; }
.result-box-low  { background-color: #0e3d18; border: 2px solid #00cc44; padding: 20px;
                   border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────
try:
    model = tf.keras.models.load_model("heart_model.keras")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'heart_model.keras' and 'scaler.pkl' are present.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.title("HeartGuard AI")
    st.write("Deep-learning risk predictor for heart disease based on clinical data.")
    st.markdown("---")
    st.write("**Instructions:**")
    st.write("1. Fill in the patient details accurately.")
    st.write("2. Click **Analyze Risk** to get the prediction.")
    st.markdown("---")
    st.caption("Developed for CI Module Assignment")

# ── Header ─────────────────────────────────────────────────────────
st.markdown('<p class="title-text">\U0001FAC0 Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Advanced Deep Learning Analysis</p>', unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────
CP_MAP = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}
EKG_MAP = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
SLOPE_MAP = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THALLIUM_MAP = {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}

with st.container():
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=list(CP_MAP), format_func=CP_MAP.get)

    with col2:
        bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[1, 0],
                           format_func=lambda x: "True" if x == 1 else "False")

    with col3:
        max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
        ex_angina = st.selectbox("Exercise Induced Angina", options=[1, 0],
                                 format_func=lambda x: "Yes" if x == 1 else "No")
        ekg = st.selectbox("Resting EKG Results", options=list(EKG_MAP), format_func=EKG_MAP.get)

    st.markdown("---")
    st.subheader("Advanced Clinical Metrics")
    col4, col5, col6 = st.columns(3)

    with col4:
        oldpeak = st.number_input("ST Depression (Oldpeak)", value=0.0, step=0.1)
    with col5:
        slope = st.selectbox("Slope of ST Segment", options=list(SLOPE_MAP), format_func=SLOPE_MAP.get)
    with col6:
        vessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thallium = st.selectbox("Thallium Stress Test", options=list(THALLIUM_MAP),
                                format_func=THALLIUM_MAP.get)

# ── Prediction ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Risk"):
    input_data = pd.DataFrame({
        "Age": [age], "Sex": [sex], "Chest pain type": [cp], "BP": [bp],
        "Cholesterol": [chol], "FBS over 120": [fbs], "EKG results": [ekg],
        "Max HR": [max_hr], "Exercise angina": [ex_angina], "ST depression": [oldpeak],
        "Slope of ST": [slope], "Number of vessels fluro": [vessels], "Thallium": [thallium],
    })

    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]
    prediction = int(prob > 0.5)
    confidence = prob if prediction == 1 else 1 - prob

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>Analysis Result</h2>", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box-high">
                <h2 style="color: #ff4b4b;">WARNING — HIGH RISK DETECTED</h2>
                <p style="font-size:18px; color:white;">The model predicts a high probability of heart disease.</p>
                <p style="font-size:24px; font-weight:bold; color:white;">Confidence: {confidence*100:.2f}%</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-low">
                <h2 style="color: #00cc44;">LOW RISK DETECTED</h2>
                <p style="font-size:18px; color:white;">The model predicts a low probability of heart disease.</p>
                <p style="font-size:24px; font-weight:bold; color:white;">Confidence: {confidence*100:.2f}%</p>
            </div>""", unsafe_allow_html=True)
