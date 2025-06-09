import os
os.environ["WATCHDOG_OBSERVER_IGNORE_DIRECTORIES"] = "true"

import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the following information to check the risk of diabetes:")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction
if st.button("Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts this person **has diabetes** with a probability of {probability[0][1]:.2%}.")
    else:
        st.success(f"✅ The model predicts this person **does NOT have diabetes** with a probability of {probability[0][0]:.2%}.")

