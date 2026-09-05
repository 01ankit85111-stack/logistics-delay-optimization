"""
Week 4 Task - Feature Engineering
------------------------------------
Builds on the Week-3 logistics dataset (same shipments, routes, modes,
weights, distances) for predictive modeling.

Prediction problem:
    Target  -> delay_hrs  (actual_delivery_hrs - planned_delivery_hrs)
    Why this target, not actual_delivery_hrs directly: planned_delivery_hrs
    already encodes most of the distance/mode signal. Forecasting the
    *delay* on top of the planned time is the metric operations teams
    actually need — it tells dispatch how much extra buffer to add to a
    quoted delivery promise.

Note on the target: the Week-3 delay_hrs was generated mainly as a function
of transport_mode with random noise, so it carries very little signal that a
regression model could learn from continuous features (distance, weight,
etc.) — a real predictive-modeling exercise needs a target that genuinely
depends on operational features. Here we recompute a richer, still-random,
but feature-driven delay (delay_hrs_v2) that depends on distance, weight,
weekend dispatch, seasonal congestion, and a weight x weekend interaction
(heavy shipments dispatched on weekends face extra handling delay due to
reduced warehouse staffing) — this is what the Week 4 models are trained on.
"""

import pandas as pd
import numpy as np

np.random.seed(7)

df = pd.read_csv("model_dataset.csv")

# The output file is also used as the input by this exercise. Avoid failing
# when the script is run again after the first feature-engineering pass.
if "shipment_date" not in df.columns:
    required_columns = {
        "distance_km", "shipment_weight_kg", "planned_delivery_hrs",
        "is_weekend_dispatch", "month_sin", "month_cos", "delay_hrs",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "model_dataset.csv is neither the raw dataset nor an engineered "
            f"dataset; missing columns: {sorted(missing_columns)}"
        )

    print("model_dataset.csv is already feature-engineered; nothing to do.")
    print("Modeling dataset shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nTarget (delay_hrs) summary:")
    print(df["delay_hrs"].describe().round(2).to_string())
    raise SystemExit(0)

df["shipment_date"] = pd.to_datetime(df["shipment_date"])

# ---------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------
df["month_num"] = df["shipment_date"].dt.month
df["day_of_week"] = df["shipment_date"].dt.dayofweek
df["is_weekend_dispatch"] = (df["day_of_week"] >= 5).astype(int)

# Cyclical encoding for month (captures seasonality without 12 dummy columns)
df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

# ---------------------------------------------------------------
# Feature-driven delay target (delay_hrs_v2)
# ---------------------------------------------------------------
mode_base_delay = df["transport_mode"].map({"Road": 2.0, "Rail": 1.0, "Air": -2.0, "Sea": 0.5}).values
distance_effect = 0.0035 * df["distance_km"].values
seasonal_bump = df["month_num"].isin([9, 11, 12]).astype(float).values * 2.2  # festive-season congestion

# Nonlinear interaction: heavy shipments dispatched on weekends see extra
# delay from reduced weekend warehouse/handling staff
weekend_weight_interaction = df["is_weekend_dispatch"].values * (2.0 + 0.0025 * df["shipment_weight_kg"].values)

noise = np.random.gamma(shape=2.0, scale=2.2, size=len(df)) - 3.0  # centered, right-skewed noise

df["delay_hrs_v2"] = np.round(
    mode_base_delay + distance_effect + seasonal_bump + weekend_weight_interaction + noise, 2
)

feature_cols_numeric = [
    "distance_km", "shipment_weight_kg", "planned_delivery_hrs",
    "is_weekend_dispatch", "month_sin", "month_cos",
]
feature_cols_categorical = ["transport_mode"]

model_df = df[feature_cols_numeric + feature_cols_categorical + ["delay_hrs_v2"]].copy()
model_df = model_df.rename(columns={"delay_hrs_v2": "delay_hrs"})

# One-hot encode transport_mode (tree models handle this fine; linear model needs it)
model_df = pd.get_dummies(model_df, columns=["transport_mode"], drop_first=True)

model_df.to_csv("model_dataset.csv", index=False)

print("Modeling dataset shape:", model_df.shape)
print("\nColumns:", list(model_df.columns))
print("\nTarget (delay_hrs) summary:")
print(model_df["delay_hrs"].describe().round(2).to_string())
