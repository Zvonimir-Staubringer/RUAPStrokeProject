import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join("model", "stroke_prediction_model.pkl")
ENCODERS_PATH = os.path.join("model", "stroke_encoders.pkl")
FIXED_THRESHOLD = 0.75

@st.cache_resource
def load_model():
    model = None
    encoders = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODERS_PATH):
        try:
            encoders = joblib.load(ENCODERS_PATH)
        except Exception:
            encoders = None
    return model, encoders

model, encoders = load_model()

st.title("Stroke Risk Estimator")
st.write("Enter patient data below and press Predict. The model will estimate the probability of stroke risk based on the input features.")

with st.form("input_form"):
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    hypertension = st.checkbox("Hypertension", value=False)
    heart_disease = st.checkbox("Heart disease", value=False)
    avg_glucose_level = st.number_input("Avg glucose level", min_value=0.0, value=100.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)

    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    ever_married = st.selectbox("Ever married", ["No", "Yes"])
    work_type = st.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence type", ["Rural", "Urban"])
    smoking_status = st.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict")

# feature column order used in training CSV (excluding target)
feature_columns = [
    "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
    "gender_Male", "gender_Other", "ever_married_Yes",
    "work_type_Never_worked", "work_type_Private", "work_type_Self-employed", "work_type_children",
    "Residence_type_Urban",
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"
]

def build_feature_row():
    row = {
        "age": float(age),
        "hypertension": bool(hypertension),
        "heart_disease": bool(heart_disease),
        "avg_glucose_level": float(avg_glucose_level),
        "bmi": float(bmi),
        "gender_Male": gender == "Male",
        "gender_Other": gender == "Other",
        "ever_married_Yes": ever_married == "Yes",
        "work_type_Never_worked": work_type == "Never_worked",
        "work_type_Private": work_type == "Private",
        "work_type_Self-employed": work_type == "Self-employed",
        "work_type_children": work_type == "children",
        "Residence_type_Urban": residence == "Urban",
        "smoking_status_formerly smoked": smoking_status == "formerly smoked",
        "smoking_status_never smoked": smoking_status == "never smoked",
        "smoking_status_smokes": smoking_status == "smokes"
    }
    return pd.DataFrame([[row[c] for c in feature_columns]], columns=feature_columns)

if submit:
    if model is None:
        st.error(f"Model not found at {MODEL_PATH}.")
    else:
        X = build_feature_row()
        X = X.astype({c: int for c in X.select_dtypes(include=['bool']).columns})
        try:
            probs = model.predict_proba(X)[:, 1]
            st.session_state['stroke_prob'] = float(probs[0])
        except Exception:
            preds = model.predict(X)
            st.session_state['stroke_prob'] = None
            st.write("Predicted class (no probability available):", int(preds[0]))

# Display results using fixed threshold
if 'stroke_prob' in st.session_state and st.session_state['stroke_prob'] is not None:
    stroke_prob = st.session_state['stroke_prob']
    threshold = FIXED_THRESHOLD
    predicted = int(stroke_prob >= threshold)

    st.subheader("Prediction")
    st.write(f"Predicted stroke risk probability: **{stroke_prob:.3f}**")
    st.write(f"Using fixed threshold = **{threshold:.2f}** → Predicted class: **{predicted}**")
    if predicted == 1:
        st.warning("Model indicates elevated stroke risk — consider further clinical evaluation.")
    else:
        st.success("Model indicates low stroke risk with given inputs.")

    if st.button("Clear prediction"):
        del st.session_state['stroke_prob']
        st.experimental_rerun()