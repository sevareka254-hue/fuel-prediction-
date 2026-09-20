from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")


@app.get("/")
def home():
    return {
        "message": "Fleet Fuel Prediction API is running successfully."
    }


@app.post("/predict")
def predict(data: dict):

    input_data = pd.DataFrame([data])

    prediction = model.predict(input_data)

    return {
        "predicted_liters": float(prediction[0])
    }