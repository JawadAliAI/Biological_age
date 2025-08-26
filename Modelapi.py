from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd
import joblib
import traceback

# Load the trained model
model_path = "model.pkl"
model = joblib.load(model_path)

app = FastAPI(
    title="Biological Age Prediction API",
    description="This API predicts the **Biological Age of a Patient** using clinical input values.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model with dataset-based input structure
class UserInput(BaseModel):
    Albumin_gL: Annotated[float, Field(gt=0, description="Quantity of Albumin in g/L")]
    Creatinine_umolL: Annotated[float, Field(gt=0, description="Quantity of Creatinine in umol/L")]
    Glucose_mmolL: Annotated[float, Field(gt=0, description="Quantity of Glucose in mmol/L")]
    CRP_mg_dL: Annotated[float, Field(gt=0, description="C-Reactive Protein in mg/dL")]
    Lymphocyte_percent: Annotated[float, Field(gt=0, description="Lymphocyte percentage (%)")]
    MCV_fL: Annotated[float, Field(gt=0, description="Mean Corpuscular Volume (fL)")]
    RDW_percent: Annotated[float, Field(gt=0, description="Red Cell Distribution Width (%)")]
    ALKP_U_L: Annotated[float, Field(gt=0, description="Alkaline Phosphatase in U/L")]
    WBC_10_9_L: Annotated[float, Field(gt=0, description="White Blood Cells (10^9/L)")]
    ChronicAge: Annotated[float, Field(gt=0, description="Chronological Age of Patient")]

@app.get("/")
def root():
    return {"status": "API is running 🚀", "message": "Go to /docs to test the prediction API"}

@app.post("/predict")
def predict_premium(data: UserInput):
    try:
        # Convert input into DataFrame (to match training data format)
        input_df = pd.DataFrame([data.dict()])

        # Make prediction
        prediction_value = float(model.predict(input_df)[0])

        return JSONResponse(
            status_code=200,
            content={
                "message": "Prediction successful ✅",
                "Predicted Biological Age of Patient": prediction_value,
                "input_received": data.dict()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )
