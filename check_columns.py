import pandas as pd

df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)

print(df.columns.tolist())