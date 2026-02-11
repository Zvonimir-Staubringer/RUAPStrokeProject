import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

MODEL_PATH = os.path.join("model", "stroke_prediction_model.pkl")
ENCODERS_PATH = os.path.join("model", "stroke_encoders.pkl")
DATA_PATH = "stroke_prepared_with_outliers.csv"
EDA_DIR = os.path.join("model", "EDAoutput")
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

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None

model, encoders = load_model()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose page", ["Predict", "Dataset & Model"])

# Shared feature columns
feature_columns = [
    "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
    "gender_Male", "gender_Other", "ever_married_Yes",
    "work_type_Never_worked", "work_type_Private", "work_type_Self-employed", "work_type_children",
    "Residence_type_Urban",
    "smoking_status_formerly smoked", "smoking_status_never smoked", "smoking_status_smokes"
]

def build_feature_row_from_inputs(age, hypertension, heart_disease, avg_glucose_level, bmi,
                                  gender, ever_married, work_type, residence, smoking_status):
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

def show_eda_image(filename, title=None, interpretation=None):
    path = os.path.join(EDA_DIR, filename)
    if os.path.exists(path):
        if title:
            st.subheader(title)
        st.image(path, use_column_width=True)
        if interpretation:
            st.markdown(f"**Interpretation:** {interpretation}")
        st.write("---")
    else:
        st.info(f"EDA image not found: {filename}")

if page == "Predict":
    st.title("Stroke Risk Estimator")
    st.markdown("""
    This application uses **Logistic Regression model** trained on stroke patient data.
    
    The model takes patient information as input and predicts the **probability of stroke risk**.
    """)
    st.write("Enter patient data below and press Predict. The app will then display the predicted stroke risk probability.")

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

    if submit:
        if model is None:
            st.error(f"Model not found at `{MODEL_PATH}`. Please make sure the trained model is available.")
        else:
            X = build_feature_row_from_inputs(age, hypertension, heart_disease, avg_glucose_level, bmi,
                                              gender, ever_married, work_type, residence, smoking_status)
            X = X.astype({c: int for c in X.select_dtypes(include=['bool']).columns})
            try:
                probs = model.predict_proba(X)[:, 1]
                st.session_state['stroke_prob'] = float(probs[0])
            except Exception:
                preds = model.predict(X)
                st.session_state['stroke_prob'] = None
                st.write("Predicted class (no probability available):", int(preds[0]))

    # Display result using fixed threshold (if available)
    if 'stroke_prob' in st.session_state and st.session_state['stroke_prob'] is not None:
        stroke_prob = st.session_state['stroke_prob']
        predicted = int(stroke_prob >= FIXED_THRESHOLD)

        st.subheader("Prediction")
        st.write(f"Predicted stroke risk probability: **{stroke_prob:.3f}**")
        st.write(f"Using fixed threshold = **{FIXED_THRESHOLD:.2f}** → Predicted class: **{predicted}**")
        if predicted == 1:
            st.warning("Model indicates elevated stroke risk for this given patient — consider further clinical evaluation.")
        else:
            st.success("Model indicates low stroke risk for this given patient.")

        if st.button("Clear prediction"):
            del st.session_state['stroke_prob']

if page == "Dataset & Model":
    st.title("Dataset & Model Information")

    st.header("Dataset")
    if df is None:
        st.warning(f"Dataset not found at `{DATA_PATH}`.")
    else:
        st.write(f"Rows: **{len(df):,}**, Columns: **{df.shape[1]}**")
        if 'stroke' in df.columns:
            vc = df['stroke'].value_counts()
            st.write("Class distribution (stroke):")
            st.write(vc.to_frame(name="count"))
            st.write(f"Percent with stroke: **{(df['stroke'].mean()*100):.2f}%**")
        st.subheader("Preview")
        st.dataframe(df.head(10))
        st.subheader("Numeric summary")
        st.dataframe(df.describe().T)

    st.header("Exploratory Data Analysis (graphs)")
    if not os.path.isdir(EDA_DIR):
        st.info(f"No EDA images found in `{EDA_DIR}`. Run `DescriptiveStatistics.py` to generate them.")
    else:
        # Provide prioritized displays with short interpretations
        show_eda_image("correlation_matrix.png",
                       "Correlation Matrix",
                       "Shows pairwise Spearman correlations between numeric features and stroke. Look for features with stronger positive correlation with stroke to identify potential predictors.")

        show_eda_image("age_histogram.png",
                       "Age Distribution",
                       "Histogram of ages in the dataset. If skewed toward older ages, age may be a strong risk factor.")

        show_eda_image("age_boxplot.png",
                       "Age Boxplot",
                       "Boxplot shows median and spread; check for outliers and skew.")

        show_eda_image("bmi_histogram.png",
                       "BMI Distribution",
                       "Distribution of BMI values; elevated BMI can correlate with stroke risk in some cohorts.")

        show_eda_image("bmi_boxplot.png",
                       "BMI Boxplot",
                       "Boxplot highlights outliers and spread in BMI data; missing BMI values were previously dropped in EDA.")

        # Categorical counts and interpretations
        show_eda_image("hypertension_countplot.png",
                       "Hypertension Count",
                       "Counts of patients with/without hypertension. Hypertension is a known stroke risk factor — observe its prevalence.")

        show_eda_image("heart_disease_countplot.png",
                       "Heart Disease Count",
                       "Counts of heart disease presence. Presence of heart disease increases stroke risk; note proportions.")

        show_eda_image("ever_married_countplot.png",
                       "Marital Status Distribution",
                       "Shows distribution of marital status categories — useful for demographic context, not causal by itself.")

        show_eda_image("smoking_status_countplot.png",
                       "Smoking Status Distribution",
                       "Distribution of smoking categories; smoking is a risk factor for cardiovascular events including stroke.")

        # Stroke-specific visualizations
        show_eda_image("age_distribution_stroke_patients.png",
                       "Age Distribution of Stroke Patients",
                       "KDE of ages for stroke patients — helps compare with overall age distribution to see if stroke skews older.")

        show_eda_image("bmi_distribution_stroke_patients.png",
                       "BMI Distribution of Stroke Patients",
                       "KDE of BMI for stroke patients — compare with overall BMI to spot differences.")

        # If there are other files, list them
        other_files = [f for f in os.listdir(EDA_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        shown = {"correlation_matrix.png", "age_histogram.png", "age_boxplot.png", "bmi_histogram.png",
                 "bmi_boxplot.png", "hypertension_countplot.png", "heart_disease_countplot.png",
                 "ever_married_countplot.png", "smoking_status_countplot.png",
                 "age_distribution_stroke_patients.png", "bmi_distribution_stroke_patients.png"}
        extras = [f for f in other_files if f not in shown]
        if extras:
            st.subheader("Other EDA images")
            for fn in extras:
                show_eda_image(fn, title=fn, interpretation="Additional EDA image.")

    st.header("Model")
    if model is None:
        st.warning(f"Model not found at `{MODEL_PATH}`.")
    else:
        st.write("Model class:", type(model).__name__)
        st.write(f"Fixed decision threshold used in app: **{FIXED_THRESHOLD:.2f}**")
        # Feature importance or coefficients
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_, dtype=float)
            if coef.ndim == 1 or coef.shape[0] == 1:
                importances = np.abs(coef.ravel())
            else:
                importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = None

        if importances is not None:
            fi = pd.DataFrame({
                "feature": feature_columns,
                "importance": importances
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            st.subheader("Feature importances / coefficient magnitudes")
            st.dataframe(fi)
            st.bar_chart(fi.set_index("feature")["importance"])
        else:
            st.info("Model does not expose feature importances or coefficients.")

        # If dataset available, report model performance on full dataset (informational only)
        if df is not None and 'stroke' in df.columns:
            X_full = df.drop(columns=['stroke'])
            y_full = df['stroke']
            try:
                # ensure types match expected (booleans -> ints)
                X_full = X_full.astype({c: int for c in X_full.select_dtypes(include=['bool']).columns})
            except Exception:
                pass
            try:
                probs_full = model.predict_proba(X_full)[:, 1]
                auc = roc_auc_score(y_full, probs_full)
                st.write(f"ROC AUC on full dataset (informational): **{auc:.3f}**")
                preds_full = (probs_full >= FIXED_THRESHOLD).astype(int)
                cm = confusion_matrix(y_full, preds_full)
                st.subheader("Confusion matrix (app threshold, full dataset)")
                st.write(cm)
                st.subheader("Classification report (app threshold, full dataset)")
                st.text(classification_report(y_full, preds_full, digits=3))
            except Exception:
                st.info("Could not compute full-dataset evaluation (model may not support `predict_proba` or shapes differ).")