import streamlit as st
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    sex = st.radio("Sex", ["Male", "Female"])  
    sex = 1 if sex == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )
    cp_map = {
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-anginal Pain":2,
        "Asymptomatic":3
    }
    cp = cp_map[cp]

    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=90, max_value=600, value=200)
    
    restecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
    )
    restecg_map = {
        "Normal":0,
        "ST-T abnormality":1,
        "Left ventricular hypertrophy":2
    }
    restecg = restecg_map[restecg]

    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])  
    exang = 1 if exang == "Yes" else 0
    
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

    slope = st.selectbox(
        "Slope of ST segment",
        ["Upsloping", "Flatsloping", "Downsloping"]
    )
    slope_map = {
        "Upsloping":0,
        "Flatsloping":1,
        "Downsloping":2
    }
    slope = slope_map[slope]

    ca = st.radio(
        "Number of Major Vessels Colored by Fluoroscopy",
        [0,1,2,3],
        horizontal=True
    )

    thal = st.selectbox(
        "Thal",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )
    thal_map = {
        "Normal":0,
        "Fixed Defect":1,
        "Reversible Defect":2
    }
    thal = thal_map[thal]

    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array) 
    prediction = model.predict(X_scaled)
    
    if prediction[0] == 1:
        return "1 (Have Heart Attack)"
    else:
        return "0 (No Heart Attack)"
    
if __name__ == '__main__':
    main()