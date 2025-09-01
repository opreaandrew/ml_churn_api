# Churn Prediction Mini Project

Tiny end-to-end churn project: train a model, wrap it in a FastAPI service, poke it via a little interactive UI, and keep everything lightweight + readable. Goal is hands-on learning.

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

## Repository Structure (Current State)

```
requirements.txt        # Dependencies
src/
  train.py              # Training script (data cleaning, pipeline, evaluation)
app/
  api.py                # FastAPI app (HTML demo + /predict)
  schemas.py            # Pydantic request/response models
data/
  raw/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
  processed/            # Derived artifacts
models/
  churn_pipeline.pkl    # Persisted sklearn Pipeline
tests/                  # (placeholder for future tests)
```

Core training functions live in `src/train.py`: `load_data`, `clean_dataframe`, `split_features_target`, `build_pipeline`, `train_and_evaluate`.

## Dataset Source & Attribution

The raw file `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` is the public Telco Customer Churn sample dataset (originally distributed by IBM sample datasets and widely mirrored, e.g. on Kaggle). I did not create this dataset, and I assert no ownership or legal claim over it. It is included solely for educational, non-commercial demonstration.

## Install & Train

```bash
python -m venv .venv
source .venv/bin/activate  # or Windows equivalent
pip install -r requirements.txt
python src/train.py
```

Outputs after training:
- `models/churn_pipeline.pkl` (persisted pipeline)
- `data/processed/roc_curve.png`
- Metrics printed (classification report + ROC AUC)

## Run the API

Once you have a trained `churn_pipeline.pkl` (or you drop one in manually):

```bash
uvicorn app.api:app --reload
```

Then visit: http://127.0.0.1:8000/

What you get:
- Root `/` serves a minimal HTML demo page.
- Highlighted churn probability badge updates live when you submit.
- Quick example buttons auto-fill common high/low risk scenarios.
- OpenAPI docs live at `/docs` (Swagger) and `/redoc`.

### Endpoints

| Method | Path      | Description |
|--------|-----------|-------------|
| GET    | `/`       | HTML demo page (form + examples) |
| GET    | `/health` | Simple health probe (`{"status":"ok"}`) |
| POST   | `/predict`| Batch predict churn probabilities |

### Prediction Request Schema

`POST /predict` expects JSON shaped like:

```json
{
  "records": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": null,
      "InternetService": "Fiber optic",
      "OnlineSecurity": null,
      "OnlineBackup": null,
      "DeviceProtection": null,
      "TechSupport": null,
      "StreamingTV": null,
      "StreamingMovies": null,
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 70.35,
      "TotalCharges": 840.50
    }
  ]
}
```

All fields are optional in the Pydantic model; missing values get imputed by the pipeline. `SeniorCitizen` must be 0 or 1 if provided.

### Prediction Response

```json
{
  "probabilities": [0.2315],
  "predictions": [0]
}
```

`probabilities[i]` is the churn probability for `records[i]`; `predictions[i]` is 1 if probability ≥ 0.5 else 0 (simple threshold baseline).

## Model Pipeline (Under the Hood)

Steps inside `build_pipeline`:
1. Separate numeric vs categorical columns.
2. Numeric: median imputation + standard scaling.
3. Categorical: most-frequent imputation + one-hot (ignore unseen).
4. Logistic Regression baseline classifier.

## Disclaimer

Purely educational. No guarantees (accuracy, robustness, reliability). Dataset belongs to original provider; I claim no rights over it. Use responsibly.
