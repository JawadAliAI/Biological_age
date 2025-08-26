from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI with metadata
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

        # Predict
        prediction = model.predict(features)

        return {
            "input": data.dict(),
            "predicted_biological_age": prediction.tolist()
        }
    except Exception as e:
        return {"error": str(e)}
