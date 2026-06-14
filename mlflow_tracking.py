import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# =====================================================
# MLFLOW EXPERIMENT
# =====================================================

mlflow.set_experiment(
    "Smart_Traffic_Accident_Prediction"
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "data/processed/feature_engineered_data.csv"
)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop(
    "accident_occurred",
    axis=1
)

y = df["accident_occurred"]

# =====================================================
# LABEL ENCODING
# =====================================================

for col in X.select_dtypes(
    include=["object"]
).columns:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# =====================================================
# SMOTE
# =====================================================

smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# =====================================================
# START MLFLOW RUN
# =====================================================

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train_smote,
        y_train_smote
    )

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    # ==========================================
    # Log Parameters
    # ==========================================

    mlflow.log_param(
        "n_estimators",
        300
    )

    mlflow.log_param(
        "max_depth",
        10
    )

    mlflow.log_param(
        "sampling_strategy",
        0.5
    )

    # ==========================================
    # Log Metrics
    # ==========================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.log_metric(
        "roc_auc",
        auc
    )

    # ==========================================
    # Log Model
    # ==========================================

    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )

    print("\n===== MODEL METRICS =====")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)
    print("ROC AUC  :", auc)

print("\nMLflow Tracking Completed Successfully!")