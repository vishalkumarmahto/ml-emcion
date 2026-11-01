# import os
# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.ensemble import IsolationForest


# # --------------------------------
# # 1. LOAD TRAINING DATA
# # --------------------------------

# # DATA_PATH = "data/training_data.csv"
# # MODEL_PATH = "models"

# # os.makedirs(MODEL_PATH, exist_ok=True)

# # df = pd.read_csv(DATA_PATH)
# from pathlib import Path

# # Project directory
# BASE_DIR = Path(__file__).resolve().parent

# # Correct dataset paths
# DATA_PATH = BASE_DIR / "ml-service" / "data" / "training_data.csv"

# TELEMETRY_PATH = BASE_DIR / "ml-service" / "data" / "telemetry.csv"

# # Save trained models inside ml-service/models
# MODEL_PATH = BASE_DIR / "ml-service" / "models"

# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # Verify paths
# print("Training data:", DATA_PATH)
# print("Telemetry data:", TELEMETRY_PATH)

# print("Training file exists:", DATA_PATH.exists())
# print("Telemetry file exists:", TELEMETRY_PATH.exists())

# # Load datasets
# df = pd.read_csv(DATA_PATH)
# # telemetry = pd.read_csv(TELEMETRY_PATH)

# print("Training dataset shape:", df.shape)
# # print("Telemetry dataset shape:", telemetry.shape)

# # Remove invalid records
# df = df.dropna()

# df = df[
#     (df["good_qty"] > 0) &
#     (df["energy_kwh_per_unit"] > 0) &
#     (df["availability"] >= 0) &
#     (df["availability"] <= 1)
# ]

# # --------------------------------
# # 2. FEATURES AND TARGET
# # --------------------------------

# features = [
#     "line_id",
#     "product_name",
#     "production_rate",
#     "availability",
#     "avg_power_kw",
#     "operating_minutes"
# ]

# target = "energy_kwh_per_unit"

# X = df[features]
# y = df[target]

# categorical = ["line_id", "product_name"]

# numerical = [
#     "production_rate",
#     "availability",
#     "avg_power_kw",
#     "operating_minutes"
# ]

# # --------------------------------
# # 3. PREPROCESSING
# # --------------------------------

# preprocessor = ColumnTransformer(
#     transformers=[
#         (
#             "categorical",
#             OneHotEncoder(handle_unknown="ignore"),
#             categorical
#         ),
#         (
#             "numerical",
#             "passthrough",
#             numerical
#         )
#     ]
# )

# # --------------------------------
# # 4. RANDOM FOREST MODEL
# # --------------------------------

# energy_model = Pipeline([
#     ("preprocessor", preprocessor),

#     ("regressor", RandomForestRegressor(
#         n_estimators=200,
#         max_depth=10,
#         min_samples_leaf=2,
#         random_state=42,
#         n_jobs=-1
#     ))
# ])

# # --------------------------------
# # 5. TRAIN AND TEST
# # --------------------------------

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42
# )

# energy_model.fit(X_train, y_train)

# predictions = energy_model.predict(X_test)

# mae = mean_absolute_error(y_test, predictions)
# r2 = r2_score(y_test, predictions)

# print("ENERGY MODEL TRAINED")
# print("MAE:", round(mae, 4))
# print("R2 Score:", round(r2, 4))

# # --------------------------------
# # 6. SAVE MODEL
# # --------------------------------

# joblib.dump(
#     energy_model,
#     f"{MODEL_PATH}/energy_model.pkl"
# )

# print("Energy model saved successfully!")


# # --------------------------------
# # 7. ANOMALY DETECTION MODEL
# # --------------------------------

# telemetry = pd.read_csv("data/telemetry.csv")

# anomaly_features = [
#     "power_kw",
#     "energy_delta_kwh",
#     "production_rate",
#     "availability",
#     "scrap_rate"
# ]

# telemetry = telemetry.dropna(subset=anomaly_features)

# anomaly_model = IsolationForest(
#     n_estimators=200,
#     contamination=0.03,
#     random_state=42
# )

# anomaly_model.fit(telemetry[anomaly_features])

# joblib.dump(
#     anomaly_model,
#     f"{MODEL_PATH}/anomaly_model.pkl"
# )

# print("Anomaly model trained and saved!")


import os
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "ml-service" / "data"
MODEL_PATH = BASE_DIR / "ml-service" / "models"

DATA_PATH = DATA_DIR / "training_data.csv"
TELEMETRY_PATH = DATA_DIR / "telemetry.csv"
MODEL_PATH = BASE_DIR/"ml-service"/"models"

MODEL_PATH.mkdir(parents=True, exist_ok=True)

print("\n========== EMICOM ML TRAINING ==========\n")

print("Training data:", DATA_PATH)
print("Telemetry data:", TELEMETRY_PATH)

# Check files
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Missing training dataset: {DATA_PATH}")

if not TELEMETRY_PATH.exists():
    raise FileNotFoundError(f"Missing telemetry dataset: {TELEMETRY_PATH}")


# ==========================================
# 2. LOAD DATASETS
# ==========================================

df = pd.read_csv(DATA_PATH)
telemetry = pd.read_csv(TELEMETRY_PATH)

print("\nTraining dataset shape:", df.shape)
print("Telemetry dataset shape:", telemetry.shape)


# ==========================================
# 3. ENERGY DATA CLEANING
# ==========================================

required_energy_columns = [
    "line_id",
    "product_name",
    "production_rate",
    "availability",
    "avg_power_kw",
    "operating_minutes",
    "good_qty",
    "energy_kwh_per_unit"
]

missing = set(required_energy_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing energy columns: {missing}")

df = df.dropna(subset=required_energy_columns)

df = df[
    (df["good_qty"] > 0) &
    (df["energy_kwh_per_unit"] > 0) &
    (df["availability"] >= 0) &
    (df["availability"] <= 1)
].copy()

if len(df) < 10:
    raise ValueError("Not enough valid energy records to train the model.")

print("Valid energy records:", len(df))


# ==========================================
# 4. FEATURES AND TARGET
# ==========================================

features = [
    "line_id",
    "product_name",
    "production_rate",
    "availability",
    "avg_power_kw",
    "operating_minutes"
]

target = "energy_kwh_per_unit"

X = df[features]
y = df[target]

categorical = [
    "line_id",
    "product_name"
]

numerical = [
    "production_rate",
    "availability",
    "avg_power_kw",
    "operating_minutes"
]


# ==========================================
# 5. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "numerical",
            "passthrough",
            numerical
        )
    ]
)


# ==========================================
# 6. RANDOM FOREST ENERGY MODEL
# ==========================================

energy_model = Pipeline([
    ("preprocessor", preprocessor),

    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])


# ==========================================
# 7. TRAIN AND EVALUATE ENERGY MODEL
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

energy_model.fit(X_train, y_train)

predictions = energy_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n========== ENERGY MODEL RESULTS ==========")

print("Model: Random Forest Regressor")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 4))
print("R2 Score:", round(r2, 4))


# ==========================================
# 8. SAVE ENERGY MODEL
# ==========================================

energy_model_path = MODEL_PATH / "energy_model.pkl"

joblib.dump(
    energy_model,
    energy_model_path
)

print("\nEnergy model saved:", energy_model_path)


# ==========================================
# 9. ANOMALY DATA CLEANING
# ==========================================

anomaly_features = [
    "power_kw",
    "energy_delta_kwh",
    "production_rate",
    "availability",
    "scrap_rate"
]

missing = set(anomaly_features) - set(telemetry.columns)

if missing:
    raise ValueError(f"Missing telemetry columns: {missing}")

telemetry = telemetry.dropna(
    subset=anomaly_features
).copy()

# Ensure numerical values
for column in anomaly_features:
    telemetry[column] = pd.to_numeric(
        telemetry[column],
        errors="coerce"
    )

telemetry = telemetry.dropna(
    subset=anomaly_features
)

if len(telemetry) < 10:
    raise ValueError("Not enough valid telemetry records to train anomaly model.")

print("\nValid telemetry records:", len(telemetry))


# ==========================================
# 10. ISOLATION FOREST ANOMALY MODEL
# ==========================================

anomaly_model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

anomaly_model.fit(
    telemetry[anomaly_features]
)

print("\n========== ANOMALY MODEL RESULTS ==========")

print("Model: Isolation Forest")
print("Training records:", len(telemetry))


# ==========================================
# 11. SAVE ANOMALY MODEL
# ==========================================

anomaly_model_path = MODEL_PATH / "anomaly_model.pkl"

joblib.dump(
    anomaly_model,
    anomaly_model_path
)

print("\nAnomaly model saved:", anomaly_model_path)


# ==========================================
# 12. TRAINING COMPLETED
# ==========================================

print("\n==========================================")
print("ALL ML MODELS TRAINED SUCCESSFULLY!")
print("==========================================")

print("\nSaved Models:")

print("- energy_model.pkl")
print("- anomaly_model.pkl")

print("\nModel directory:", MODEL_PATH)
