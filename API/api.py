
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Stroke Prediction API")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "stroke_prediction_model.pkl")
ENC_PATH = os.path.join(BASE_DIR, "model", "stroke_encoders.pkl")
CSV_PATH = os.path.join(BASE_DIR, "stroke_prepared_with_outliers.csv")

model = None
model_load_error = None
try:
	model = joblib.load(MODEL_PATH)
except Exception as e:
	model_load_error = str(e)

encoders = None
if os.path.exists(ENC_PATH):
	try:
		encoders = joblib.load(ENC_PATH)
	except Exception:
		encoders = None

# Determine expected feature names (try model metadata first, then CSV header)
expected_features = None
if model is not None and hasattr(model, "feature_names_in_"):
	try:
		expected_features = list(model.feature_names_in_)
	except Exception:
		expected_features = None

if expected_features is None and os.path.exists(CSV_PATH):
	try:
		expected_features = [c for c in pd.read_csv(CSV_PATH, nrows=0).columns if c != "stroke"]
	except Exception:
		expected_features = None


class PredictRequest(BaseModel):
	data: dict


@app.get("/")
def root():
	return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
	if model is None:
		raise HTTPException(status_code=500, detail=f"Model not loaded: {model_load_error}")
	if expected_features is None:
		raise HTTPException(status_code=500, detail="Expected feature list not available on server.")

	input_data = req.data
	missing = [f for f in expected_features if f not in input_data]
	if missing:
		raise HTTPException(status_code=400, detail={"missing_fields": missing})

	# Build a single-row DataFrame with proper ordering
	row = {feat: input_data.get(feat) for feat in expected_features}
	df = pd.DataFrame([row], columns=expected_features)

	# Convert boolean-like values to numeric
	for col in df.columns:
		if df[col].dtype == 'bool':
			df[col] = df[col].astype(int)
		# try to coerce strings 'True'/'False' to booleans
		if df[col].dtype == object:
			df[col] = df[col].replace({"True": 1, "False": 0})

	# Attempt prediction
	try:
		proba = None
		if hasattr(model, "predict_proba"):
			proba = float(model.predict_proba(df)[0][1])
		pred = int(model.predict(df)[0])
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

	return {"stroke_prob": proba, "prediction": pred}
