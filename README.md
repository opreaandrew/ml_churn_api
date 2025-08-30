# Churn Prediction Mini Project

Educational, end-to-end machine learning pipeline for the Telco Customer Churn problem. The focus is learning ML engineering practices hands-on.

## Learning Objectives

I am using this repo to practice and internalize:
- Data ingestion & lightweight cleaning (`load_data`, `clean_dataframe` in `src/train.py`)
- Target / feature separation and label encoding (`split_features_target`)
- Building reproducible preprocessing + model pipelines with scikit-learn (`build_pipeline`)
- Handling mixed numeric / categorical features (imputation, scaling, one-hot encoding)
- Model training, evaluation metrics (classification report, ROC AUC) and artifact persistence (`train_and_evaluate`)
- Basic experiment structure (clear `data/raw`, `data/processed`, `models/`)
- Reproducible execution via pinned dependencies (`requirements.txt`)
- Readability, small, composable functions
- Preparing for future extension (API inference endpoint, CI tests, simple MLOps hooks)

## Repository Structure

```
requirements.txt
src/
  train.py
data/
  raw/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
  processed/
models/
tests/
```

Key script: `src/train.py` (functions: `load_data`, `clean_dataframe`, `split_features_target`, `build_pipeline`, `train_and_evaluate`).

## Dataset Source & Attribution

The raw file `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` is the public Telco Customer Churn sample dataset (originally distributed by IBM sample datasets and widely mirrored, e.g. on Kaggle). I did not create this dataset, and I assert no ownership or legal claim over it. It is included solely for educational, non-commercial demonstration.

## Installation & Usage

```bash
python -m venv .venv
source .venv/bin/activate  # or Windows equivalent
pip install -r requirements.txt
python src/train.py
```

Outputs:
- Trained pipeline: `models/churn_pipeline.pkl`
- ROC curve image: `data/processed/roc_curve.png`
- Console metrics (classification report, ROC AUC)

## Model Pipeline

Steps inside `build_pipeline`:
1. Separate numeric vs categorical columns.
2. Numeric: median imputation + standard scaling.
3. Categorical: most-frequent imputation + one-hot (ignore unseen).
4. Logistic Regression baseline classifier.

## Future Extensions (Planned Exploration)

- Add inference script / lightweight API
- Persist feature schema
- Basic unit tests in `tests/`
- Hyperparameter search & model comparison
- Monitoring drift / simple metadata logging

## Disclaimer

This project is for learning. No guarantees of correctness, completeness, performance, or fitness for production use. Dataset remains property of its original source.
