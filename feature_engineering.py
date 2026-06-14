import os
import pandas as pd
import numpy as np
from datetime import datetime



# =====================================================
# Load Cleaned Data
# =====================================================

df = pd.read_csv("data/processed/cleaned_data.csv")
df["event_timestamp"] = pd.Timestamp.now()
print("=" * 60)
print("FEATURE ENGINEERING STARTED")
print("=" * 60)

print(f"Original Shape: {df.shape}")

# =====================================================
# Convert Timestamp
# =====================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# =====================================================
# Time-Based Features
# =====================================================

df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

# =====================================================
# Weekend Feature
# Assuming:
# 0 = Monday
# 6 = Sunday
# =====================================================

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# =====================================================
# Night Driving Feature
# =====================================================

df["is_night"] = (
    (df["hour_of_day"] >= 20) |
    (df["hour_of_day"] <= 5)
).astype(int)

# =====================================================
# Speed Ratio
# =====================================================

df["speed_ratio"] = (
    df["avg_speed_kmph"] /
    df["speed_limit_kmph"]
)

# Replace infinities if any
df["speed_ratio"] = (
    df["speed_ratio"]
    .replace([np.inf, -np.inf], 0)
)

# =====================================================
# Traffic Density
# =====================================================

df["traffic_density"] = (
    df["vehicle_count_per_hr"] /
    df["lane_count"]
)

df["traffic_density"] = (
    df["traffic_density"]
    .replace([np.inf, -np.inf], 0)
)
# =====================================================
# Latitude Longitutde Efficiency
# =====================================================

df.drop(
    columns=["latitude", "longitude"],
    inplace=True,
    errors="ignore"
)

# =====================================================
# Signal Efficiency
# =====================================================

df["signal_efficiency"] = (
    df["green_duration_s"] /
    df["cycle_time_s"]
)

df["signal_efficiency"] = (
    df["signal_efficiency"]
    .replace([np.inf, -np.inf], 0)
)

# =====================================================
# Convert Boolean Columns to Integers
# =====================================================

bool_cols = [
    "has_signal",
    "is_peak",
    "peak",
    "accident_occurred"
]

for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# =====================================================
# Drop Data Leakage Columns
# =====================================================

leakage_cols = [
    "severity",
    "vehicles_involved",
    "cause",
    "veh_count_at_accident"
]

df.drop(
    columns=leakage_cols,
    inplace=True,
    errors="ignore"
)

# =====================================================
# Drop Identifier Columns
# =====================================================

identifier_cols = [
    "location_id",
    "latitude",
    "longitude"
]

df.drop(
    columns=identifier_cols,
    inplace=True,
    errors="ignore"
)

# =====================================================
# Optional:
# Drop Timestamp After Extracting Features
# =====================================================

df.drop(
    columns=["timestamp"],
    inplace=True,
    errors="ignore"
)

# =====================================================
# Final Validation
# =====================================================

print("\nFinal Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nColumns:")
print(df.columns.tolist())

# =====================================================
# Save Feature Engineered Data
# =====================================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/feature_engineered_data.csv"
)

df.insert(
    0,
    "traffic_record_id",
    range(len(df))
)

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print(f"Saved To: {output_path}")
print("=" * 60)