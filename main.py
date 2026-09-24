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

    route_km: float = Field(
        gt=0,
        description="Route distance in kilometers"
    )

    fuel_price: float = Field(
        gt=0,
        description="Fuel price"
    )

    passengers: int = Field(
        ge=0,
        description="Number of passengers"
    )

    load_kg: float = Field(
        ge=0,
        description="Vehicle load in kilograms"
    )

    ac_used: int = Field(
        ge=0,
        le=1,
        description="Air conditioning: 0 or 1"
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

    vehicle_type: str

    make: str

    model: str

    fuel_type: str

    traffic_band: str


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


    baseline_prediction = (
        data.route_km *
        data.nominal_l_per_100km
    ) / 100



    prediction = model.predict(input_data)

    ml_prediction = max(
        0,
        float(prediction[0])
    )



    if baseline_prediction <= ml_prediction:

        selected_prediction = baseline_prediction
        selected_method = "baseline"

    else:

        selected_prediction = ml_prediction
        selected_method = "linear_regression"


  

    return {

        "baseline_prediction_liters": round(
            baseline_prediction,
            2
        ),

        "ml_prediction_liters": round(
            ml_prediction,
            2
        ),

        "predicted_liters": round(
            selected_prediction,
            2
        ),

        "selected_method": selected_method
    }
