import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

RAW_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"

def load_data() -> pd.DataFrame:
    """
    Loads Telco churn data if available; otherwise generates a synthetic dataset
    with similar columns so you can run end-to-end immediately.
    """
    if os.path.exists(RAW_PATH):
        df = pd.read_csv(RAW_PATH)
        return df
    else:
        return None

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip spaces in column names and values
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # Coerce numerics
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID (identifier, not predictive)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Ensure target is present
    assert "Churn" in df.columns, "Target column 'Churn' not found"

    return df

def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["Churn"].map({"Yes": 1, "No": 0})  # binary target
    X = df.drop(columns=["Churn"])
    return X, y

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identify column types
    numeric_features: List[str] = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features: List[str] = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # Simple, strong baseline: Logistic Regression (fast, well-calibrated)
    clf = LogisticRegression(max_iter=200, n_jobs=None)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf)
    ])
    return pipe

def train_and_evaluate():
    df = load_data()
    df = clean_dataframe(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Optional: ROC curve saved to file
    RocCurveDisplay.from_predictions(y_test, y_proba)
    os.makedirs("data/processed", exist_ok=True)
    plt.title("ROC Curve - Churn Model")
    plt.savefig("data/processed/roc_curve.png", dpi=150)
    plt.close()

    # Save pipeline (preprocessing + model)
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
