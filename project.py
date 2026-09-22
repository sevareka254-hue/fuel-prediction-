import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error




df = pd.read_csv("fleet_fuel_training_clean.csv")




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

features = categorical_features + numerical_features


X = df[features]
y = df["actual_liters"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)




categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
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




y_pred = model.predict(X_test)


y_pred = y_pred.clip(min=0)

mae = mean_absolute_error(y_test, y_pred)

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
) * 100


print("MAE:", mae)
print("MAPE:", mape, "%")




joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")
