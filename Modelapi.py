from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ---------------------------------------------------
# Load your trained model
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
    description="This API predicts the **Biological Age of a Patient** using clinical input values.",
    version="1.0.0"
)

# Define input schema
class InputData(BaseModel):
    Albumin_gL: float
    Creatinine_umolL: float
    Glucose_mmolL: float
    CRP_mg_dL: float
    Lymphocyte_percent: float
    MCV_fL: float
    RDW_percent: float
    ALKP_U_L: float
    WBC_10_9_L: float
    ChronicAge: float

# Root endpoint
@app.get("/")
def root():
    return {"status": "API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to numpy array in correct order
        features = np.array([[ 
            data.Albumin_gL,
            data.Creatinine_umolL,
            data.Glucose_mmolL,
            data.CRP_mg_dL,
            data.Lymphocyte_percent,
            data.MCV_fL,
            data.RDW_percent,
            data.ALKP_U_L,
            data.WBC_10_9_L,
            data.ChronicAge
        ]])

        # Scale the features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)

        return {
            "predicted_biological_age": prediction.tolist()
        }
    except Exception as e:
        return {"error": str(e)}

