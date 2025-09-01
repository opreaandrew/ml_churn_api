import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

TARGET_COL = "Churn"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "customerID" in df:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_preprocessor(df: pd.DataFrame):
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET_COL])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ])
    return pre, X, y


def fit_transform(pre, X):
    return pre.fit_transform(X)
