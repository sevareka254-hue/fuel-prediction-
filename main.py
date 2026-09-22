from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


app = FastAPI(
    title="Fleet Fuel Prediction API",
    description="API for predicting vehicle fuel consumption",
    version="1.0"
)




model = joblib.load("model.pkl")




class PredictionInput(BaseModel):

    

    distance_km: float = Field(
        gt=0,
        description="Distance in kilometers"
    )

    duration_min: float = Field(
        gt=0,
        description="Trip duration in minutes"
    )

    vehicle_year: int = Field(
        ge=2016,
        le=2025
    )

    seats: int = Field(
        gt=0
    )

    nominal_l_per_100km: float = Field(
        gt=0
    )

    allowed_load_kg: float = Field(
        ge=0
    )

    passengers: int = Field(
        ge=0
    )

    load_kg: float = Field(
        ge=0
    )

    ac_used: int = Field(
        ge=0,
        le=1
    )

    urban_share: float = Field(
        ge=0,
        le=1
    )

    highway_share: float = Field(
        ge=0,
        le=1
    )


   

    vehicle_type: str

    make: str

    model: str

    fuel_type: str

    traffic_band: str

    weather: str




@app.get("/")
def home():

    return {
        "message": "Fleet Fuel Prediction API is running successfully."
    }



@app.post("/predict")
def predict(data: PredictionInput):

    

    input_data = pd.DataFrame([
        data.model_dump()
    ])




    prediction = model.predict(input_data)


    

    predicted_liters = max(
        0,
        float(prediction[0])
    )


    return {
        "predicted_liters": predicted_liters
    }
