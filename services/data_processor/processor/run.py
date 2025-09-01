import os
import joblib
import numpy as np
from .preprocess import load_raw, basic_clean, build_preprocessor, fit_transform, TARGET_COL


def main():
    raw_csv = os.environ.get("RAW_CSV", "/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    pre_out = os.environ.get("PREPROC_OUT", "/models/preprocessor.pkl")
    x_out = os.environ.get("X_OUT", "/data/X.npy")
    y_out = os.environ.get("Y_OUT", "/data/y.npy")

    df = load_raw(raw_csv)
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"Missing target {TARGET_COL}")
    df = basic_clean(df)
    pre, X, y = build_preprocessor(df)
    Xt = fit_transform(pre, X)

    os.makedirs(os.path.dirname(pre_out), exist_ok=True)
    os.makedirs(os.path.dirname(x_out), exist_ok=True)

    joblib.dump(pre, pre_out)
    np.save(x_out, Xt)
    np.save(y_out, y.values)
    print(f"Saved preprocessor -> {pre_out}, X -> {x_out}, y -> {y_out}, shape={Xt.shape}")


if __name__ == "__main__":
    main()
