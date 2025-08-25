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

# Enable CORS (so frontend apps can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model with user input fields
class UserInput(BaseModel):
    age: Annotated[int, Field(gt=0, description="Age of the Patient")]
    albumin_gL: Annotated[float, Field(gt=0, description="Quantity of Albumin in gL")]
    creat_umol: Annotated[float, Field(gt=0, description="Quantity of Creatinine in umol")]
    glucose_mmol: Annotated[float, Field(gt=0, description="Quantity of Glucose in mmol")]
    lncrp: Annotated[float, Field(gt=0, description="Log of CRP")]
    lymph: Annotated[float, Field(gt=0, description="Lymphocytes")]
    mcv: Annotated[float, Field(gt=0, description="Mean Corpuscular Volume")]
    rdw: Annotated[float, Field(gt=0, description="Red Cell Distribution Width")]
    alp: Annotated[float, Field(gt=0, description="Alkaline Phosphatase")]
    wbc: Annotated[float, Field(gt=0, description="White Blood Cell Count")]

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
                "Predicted Biological Age of Patient": prediction_value
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

