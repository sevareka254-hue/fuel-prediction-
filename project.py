import pandas as pd
import joblib

df = pd.read_csv("fleet_fuel_training_clean.csv")

print(df.shape)

print(df.head())
print(df.tail())

print(df.columns)

print(df["actual_liters"].describe())

import matplotlib.pyplot as plt

plt.hist(df["actual_liters"], bins=30)
plt.xlabel("Actual Fuel (Liters)")
plt.ylabel("Number of Trips")
plt.title("Distribution of Actual Fuel Consumption")
plt.show()

import matplotlib.pyplot as plt

plt.scatter(df["distance_km"], df["actual_liters"])
plt.xlabel("Distance (km)")
plt.ylabel("Actual Fuel (liters)")
plt.title("Distance vs Actual Fuel Consumption")
plt.show()

y = df["actual_liters"]

features = [
    "distance_km",
    "duration_min",
    "vehicle_type",
    "make",
    "model",
    "vehicle_year",
    "seats",
    "fuel_type",
    "nominal_l_per_100km",
    "allowed_load_kg",
    "passengers",
    "load_kg",
    "traffic_band",
    "weather",
    "ac_used",
    "urban_share",
    "highway_share"
]

X = df[features]
y = df["actual_liters"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

categorical_features = [
    "vehicle_type",
    "make",
    "model",
    "fuel_type",
    "traffic_band",
    "weather"
]

numerical_features = [
    "distance_km",
    "duration_min",
    "vehicle_year",
    "seats",
    "nominal_l_per_100km",
    "allowed_load_kg",
    "passengers",
    "load_kg",
    "ac_used",
    "urban_share",
    "highway_share"
]

X = df[categorical_features + numerical_features]
y = df["actual_liters"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

model.fit(X_train, y_train)

import os
joblib.dump(model, r"C:\Users\Office\Desktop\model\model.pkl")

 y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)

from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("MAPE:", mape, "%")

baseline_pred = (
    X_test["distance_km"]
    * X_test["nominal_l_per_100km"]
    / 100
)

baseline_mae = mean_absolute_error(y_test, baseline_pred)

baseline_mape = (
    mean_absolute_percentage_error(y_test, baseline_pred) * 100
)

print("Baseline MAE:", baseline_mae)
print("Baseline MAPE:", baseline_mape)
