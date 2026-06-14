import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# =====================================================
# Load Data
# =====================================================

print("=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)

df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)

print(f"Dataset Shape: {df.shape}")

# =====================================================
# Split Features & Target
# =====================================================

X = df.drop(
    "accident_occurred",
    axis=1
)

y = df["accident_occurred"]

# =====================================================
# Encode Categorical Columns
# =====================================================

label_encoders = {}

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

    label_encoders[col] = le

print("\nCategorical Columns Encoded:")
print(list(categorical_cols))

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =====================================================
# SMOTE
# =====================================================

print("\nBefore SMOTE:")
print(y_train.value_counts())

smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# =====================================================
# Random Forest
# =====================================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

# =====================================================
# XGBoost
# =====================================================

print("Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_smote,
    y_train_smote
)

xgb_pred = xgb_model.predict(
    X_test
)

xgb_prob = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_f1 = f1_score(
    y_test,
    xgb_pred
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)

# =====================================================
# Compare Models
# =====================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Random Forest F1 : {rf_f1:.4f}")
print(f"Random Forest AUC: {rf_auc:.4f}")

print()

print(f"XGBoost F1 : {xgb_f1:.4f}")
print(f"XGBoost AUC: {xgb_auc:.4f}")

# =====================================================
# Select Best Model
# =====================================================

if xgb_f1 > rf_f1:

    best_model = xgb_model
    best_name = "XGBoost"

    best_pred = xgb_pred
    best_prob = xgb_prob

else:

    best_model = rf_model
    best_name = "RandomForest"

    best_pred = rf_pred
    best_prob = rf_prob

print("\nBest Model Selected:", best_name)

# =====================================================
# Final Evaluation
# =====================================================

report = classification_report(
    y_test,
    best_pred
)

matrix = confusion_matrix(
    y_test,
    best_pred
)

auc = roc_auc_score(
    y_test,
    best_prob
)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)

print("\nROC AUC:", auc)

# =====================================================
# Save Artifacts
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

# =====================================================
# Save Evaluation Report
# =====================================================

with open(
    "models/model_report.txt",
    "w"
) as f:

    f.write(
        f"Best Model: {best_name}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(report)

    f.write("\n\n")

    f.write(
        "Confusion Matrix\n"
    )

    f.write(str(matrix))

    f.write("\n\n")

    f.write(
        f"ROC AUC: {auc}"
    )

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print(f"Best Model Saved: {best_name}")
print("Location: models/best_model.pkl")
print("=" * 60)