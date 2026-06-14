import pandas as pd
import numpy as np
import os

# Load data
df = pd.read_csv("G:/AI_Spry/Internship_Project/Smart_Trafiic_Accident_Project/Project_dataset.csv")

print("Original Shape:", df.shape)

# --------------------------------------------------
# 1. Remove Duplicate Rows
# --------------------------------------------------
df.drop_duplicates(inplace=True)

# --------------------------------------------------
# 2. Convert Timestamp
# --------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------------------------------
# 3. Handle Missing Values
# --------------------------------------------------

# Numeric columns
numeric_cols = [
    "vehicle_count_per_hr",
    "avg_speed_kmph",
    "green_duration_s",
    "red_duration_s",
    "yellow_duration_s",
    "cycle_time_s",
    "violations_count"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Categorical columns
categorical_cols = [
    "lighting",
    "weather",
    "signal_status",
    "peak"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# --------------------------------------------------
# 4. Accident-specific Columns
# --------------------------------------------------

accident_cols = [
    "severity",
    "vehicles_involved",
    "cause"
]

for col in accident_cols:
    if col in df.columns:
        df[col] = df[col].fillna("No Accident")

# --------------------------------------------------
# 5. Quality Flags
# --------------------------------------------------

quality_cols = [
    "traffic_data_quality_flag",
    "signal_data_quality_flag"
]

for col in quality_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# --------------------------------------------------
# 6. Remove Impossible Values
# --------------------------------------------------

df = df[df["avg_speed_kmph"] >= 0]
df = df[df["vehicle_count_per_hr"] >= 0]

# --------------------------------------------------
# 7. Check Missing Values
# --------------------------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 8. Save Cleaned Data
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/cleaned_data.csv",
    index=False
)

print("\nCleaned Shape:", df.shape)
print("Cleaning Completed Successfully!")