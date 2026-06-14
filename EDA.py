import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# Load Data
# =====================================================

df = pd.read_csv("data/processed/cleaned_data.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Shape: {df.shape}")

print("\nColumn Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include='all'))

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("eda_reports", exist_ok=True)

# =====================================================
# Missing Values Analysis
# =====================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()
print(missing)

missing.to_csv("eda_reports/missing_values.csv")

# =====================================================
# Target Variable Distribution
# =====================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 60)

print(df["accident_occurred"].value_counts())

print(
    df["accident_occurred"]
    .value_counts(normalize=True)
    * 100
)

plt.figure(figsize=(6, 4))
sns.countplot(x="accident_occurred", data=df)
plt.title("Accident Occurrence Distribution")
plt.savefig("eda_reports/target_distribution.png")
plt.close()

# =====================================================
# Numerical Columns
# =====================================================

numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical Columns:")
print(list(numerical_cols))

# =====================================================
# Histograms
# =====================================================

for col in numerical_cols:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(f"Distribution of {col}")

    plt.tight_layout()

    plt.savefig(
        f"eda_reports/{col}_histogram.png"
    )

    plt.close()

# =====================================================
# Boxplots (Outlier Detection)
# =====================================================

for col in numerical_cols:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[col]
    )

    plt.title(f"Boxplot of {col}")

    plt.tight_layout()

    plt.savefig(
        f"eda_reports/{col}_boxplot.png"
    )

    plt.close()

# =====================================================
# Correlation Matrix
# =====================================================

plt.figure(figsize=(16, 10))

corr_matrix = df[numerical_cols].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "eda_reports/correlation_matrix.png"
)

plt.close()

# =====================================================
# Categorical Features vs Target
# =====================================================

categorical_cols = df.select_dtypes(
    include=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_cols))

for col in categorical_cols:

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x=col,
        hue="accident_occurred"
    )

    plt.xticks(rotation=45)

    plt.title(
        f"{col} vs Accident Occurred"
    )

    plt.tight_layout()

    plt.savefig(
        f"eda_reports/{col}_vs_target.png"
    )

    plt.close()

# =====================================================
# Accident Analysis
# =====================================================

accidents = df[
    df["accident_occurred"] == 1
]

if len(accidents) > 0:

    plt.figure(figsize=(12, 5))

    sns.countplot(
        data=accidents,
        x="hour_of_day"
    )

    plt.title(
        "Accidents by Hour of Day"
    )

    plt.tight_layout()

    plt.savefig(
        "eda_reports/accidents_by_hour.png"
    )

    plt.close()

# =====================================================
# Peak Hours Analysis
# =====================================================

plt.figure(figsize=(8, 4))

sns.countplot(
    data=df,
    x="is_peak",
    hue="accident_occurred"
)

plt.title(
    "Peak Hours vs Accident Occurrence"
)

plt.tight_layout()

plt.savefig(
    "eda_reports/peak_vs_accident.png"
)

plt.close()

# =====================================================
# Top Correlations with Target
# =====================================================

if "accident_occurred" in numerical_cols:

    corr_target = (
        corr_matrix["accident_occurred"]
        .sort_values(ascending=False)
    )

    print("\n" + "=" * 60)
    print("CORRELATION WITH TARGET")
    print("=" * 60)

    print(corr_target)

    corr_target.to_csv(
        "eda_reports/target_correlations.csv"
    )

# =====================================================
# Save Clean EDA Summary
# =====================================================

with open(
    "eda_reports/eda_summary.txt",
    "w"
) as f:

    f.write(
        f"Dataset Shape: {df.shape}\n\n"
    )

    f.write(
        "Missing Values:\n"
    )

    f.write(
        str(df.isnull().sum())
    )

    f.write("\n\n")

    f.write(
        "Target Distribution:\n"
    )

    f.write(
        str(
            df["accident_occurred"]
            .value_counts()
        )
    )

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("All reports saved in 'eda_reports' folder")
print("=" * 60)