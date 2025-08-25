from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the input format (schema)
class InputData(BaseModel):
    features: list[float]  # enforce list of numbers

# Create an endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input into numpy array
        X = np.array(data.features).reshape(1, -1)

        # Get prediction
        prediction = model.predict(X)

        # Return as JSON
        return {"prediction": prediction.tolist()}
    
    except Exception as e:
        # Return detailed error if something fails
        return {"error": str(e)}
