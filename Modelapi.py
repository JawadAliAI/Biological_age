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
