import os
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _wait_for(path: str, timeout: int = 60, interval: float = 1.0):
    """Poll for a file to appear before proceeding.
    Returns True if found, False if timed out.
    """
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            return True
        time.sleep(interval)
    return False


def main():
    x_path = os.environ.get("X_PATH", "/data/X.npy")
    y_path = os.environ.get("Y_PATH", "/data/y.npy")
    model_out = os.environ.get("MODEL_OUT", "/models/churn_model.pkl")

    timeout = int(os.environ.get("WAIT_TIMEOUT", "90"))
    if not _wait_for(x_path, timeout=timeout) or not _wait_for(y_path, timeout=timeout):
        raise FileNotFoundError(f"Timed out waiting for training data: {x_path}, {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved -> {model_out} | Val ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
