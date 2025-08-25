from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"status": "API is running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    try:
        X = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
this work perfectikly but i want the out put of my code like this one 
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
import joblib
import traceback

from fastapi.middleware.cors import CORSMiddleware

# Path to your saved pickle file
model_path = "xgb_model_reg.pkl"  

# Load the model
model = joblib.load(model_path)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific domains
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

#pydantic Model to validate data
class UserInput(BaseModel):
    age: Annotated[int, Field(gt=0,description='Age of the Patient')]
    albumin_gL: Annotated[float, Field(gt=0,description='Quantity of Albumin in gL')]
    creat_umol: Annotated[float, Field(gt=0,description='Quantity of Creatnine in umol')]
    glucose_mmol: Annotated[float, Field(gt=0,description='Qunatity of Glucose in mmol')]
    lncrp:Annotated[float, Field(gt=0,description='Log of Crp')]
    lymph: Annotated[float, Field(gt=0,description='lym ph')]
    mcv: Annotated[float, Field(gt=0,description='mcv')]
    rdw: Annotated[float, Field(gt=0,description='rdw')]
    alp: Annotated[float, Field(gt=0,description='alp')]
    wbc: Annotated[float, Field(gt=0,description='white blood cell')]

@app.post('/predict')
def predict_premium(data: UserInput):
    try:
        input_df = pd.DataFrame(
            [
                {
                    'age': data.age,
                    'albumin_gL': data.albumin_gL,
                    'creat_umol': data.creat_umol,
                    'glucose_mmol': data.glucose_mmol,
                    'lncrp': data.lncrp,
                    'lymph': data.lymph,
                    'mcv': data.mcv,
                    'rdw': data.rdw,
                    'alp': data.alp,
                    'wbc': data.wbc
                }
            ]
        )

        prediction_value = float(model.predict(input_df)[0])  # <-- FIXED HERE

        return JSONResponse(
            status_code=200,
            content={"Predicted Biological Age of Patient": prediction_value}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
