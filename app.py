
import joblib
import pandas as pd
from fastapi import FastAPI

# Load saved model
artifact = joblib.load("heart_disease_model.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

# Create FastAPI application
app = FastAPI(title="Heart Disease Prediction API")


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.post("/predict")
def predict(patient_data: dict):

    patient_df = pd.DataFrame([patient_data])

    probability = model.predict_proba(patient_df)[0, 1]

    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(probability), 4),
        "threshold": threshold
    }
