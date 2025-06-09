# Diabetes Prediction App

**Live Demo**: [Click here to try it out]([https://your-streamlit-app-url](https://diabetespredictor-5bwtpjhefnmc6tsi4fgu4d.streamlit.app/))  

---

## Overview

This project is a Machine Learning web application that predicts whether a person is likely to have diabetes based on key medical indicators. It is designed for
 simplicity and ease of use, making it accessible to both medical professionals and the general public.

This is my **second machine learning project**, and it reflects my ability to take a model from training to deployment with an interactive front end.

---

##  Features

- ✅ Predicts diabetes risk from 8 health indicators
- Powered by a trained **Support Vector Machine (SVM)** classifier
- Built using the **Pima Indians Diabetes Dataset**
- 💡 Fully interactive interface with **Streamlit**
- Model and scaler saved using **Pickle**
- 🌐 Deployed online for public use

---

## Inputs Used

The following parameters are required from the user:

- **Pregnancies**
- **Glucose**
- **Blood Pressure**
- **Skin Thickness**
- **Insulin**
- **BMI (Body Mass Index)**
- **Diabetes Pedigree Function**
- **Age**

The app returns:

- A **binary prediction** (Diabetic / Not Diabetic)
- The model's **confidence level** in percentage

---

## 🧰 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Pickle**
- **Git & GitHub**

---

## 🗂️ Project Structure

├── app.py # Streamlit app script
├── diabetes_model.pkl # Trained ML model
├── scaler.pkl # StandardScaler object
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation
└── diabetes.csv # Dataset used for training
