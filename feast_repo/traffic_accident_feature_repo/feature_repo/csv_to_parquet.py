import pandas as pd

df = pd.read_csv("data/traffic_features.csv")

df.to_parquet(
    "data/traffic_features.parquet",
    index=False
)

print("Parquet file created successfully!")