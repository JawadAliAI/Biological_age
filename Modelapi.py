from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------
# Load the scaler
# ---------------------------------------------------
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# ---------------------------------------------------
# Initialize FastAPI with metadata
# ---------------------------------------------------
app = FastAPI(
    title="Biological Age Prediction API",
    description="This API predicts the Biological Age of a Patient using clinical input values.",
    version="1.0.0"
)

# ---------------------------------------------------
# Input schema (no restrictions, user can enter any float)
# ---------------------------------------------------
class InputData(BaseModel):
    albumin_gl: float
    creatinine_umoll: float
    glucose_mmoll: float
    crp_mgdl: float
    lymphocyte_percent: float
    mcv_fl: float
    rdw_percent: float
    alkp_ul: float
    wbc_10_9l: float
    chronic_age: float

# ---------------------------------------------------
# Root endpoint
# ---------------------------------------------------
@app.get("/")
def root():
    return {"status": "API is running 🚀"}

# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to numpy array in correct order (same as training!)
        features = np.array([[ 
            data.albumin_gl,
            data.creatinine_umoll,
            data.glucose_mmoll,
            data.crp_mgdl,
            data.lymphocyte_percent,
            data.mcv_fl,
            data.rdw_percent,
            data.alkp_ul,
            data.wbc_10_9l,
            data.chronic_age
        ]])

        # Scale the features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]  # single value

        return {
            "predicted_biological_age": float(prediction),
            "status": "Success"
        }
    except Exception as e:
        return {"error": str(e)}
