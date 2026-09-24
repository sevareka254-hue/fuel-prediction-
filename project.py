import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


reservations = pd.read_csv("seed_reservations.csv")
vehicles = pd.read_csv("seed_vehicles.csv")



vehicle_columns = [
    "vehicle_id",
    "make",
    "model",
    "vehicle_year",
    "seats",
    "fuel_type",
    "nominal_l_per_100km",
    "allowed_load_kg"
]

df = reservations.merge(
    vehicles[vehicle_columns],
    on="vehicle_id",
    how="left"
)


print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns.tolist())



df = df[
    (df["status"] == "completed") &
    (df["actual_fuel_liters"].notna())
].copy()


print("\nRows used:", len(df))



df["baseline_fuel_liters"] = (
    df["route_km"] *
    df["nominal_l_per_100km"]
) / 100


categorical_features = [
    "vehicle_type",
    "make",
    "model",
    "fuel_type",
    "traffic_band"
]

numerical_features = [
    "route_km",
    "fuel_price",
    "passengers",
    "load_kg",
    "ac_used",
    "vehicle_year",
    "seats",
    "nominal_l_per_100km",
    "allowed_load_kg"
]

features = categorical_features + numerical_features


X = df[features]
y = df["actual_fuel_liters"]


print("\nFeatures shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
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

\

y_pred = model.predict(X_test)

y_pred = y_pred.clip(min=0)



mae = mean_absolute_error(
    y_test,
    y_pred
)

mape = mean_absolute_percentage_error(
    y_test,
    y_pred
) * 100




baseline_test = df.loc[
    X_test.index,
    "baseline_fuel_liters"
]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_test
)

baseline_mape = mean_absolute_percentage_error(
    y_test,
    baseline_test
) * 100




print("\n==============================")
print("BASELINE")
print("==============================")

print("Baseline MAE:", baseline_mae)
print("Baseline MAPE:", baseline_mape, "%")


print("\n==============================")
print("ML MODEL")
print("==============================")

print("ML MAE:", mae)
print("ML MAPE:", mape, "%")




joblib.dump(
    model,
    "model.pkl"
)

print("\nModel saved successfully as model.pkl")
