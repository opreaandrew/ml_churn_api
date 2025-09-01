# Churn Prediction Mini Project

End-to-end churn prediction demo now fully split into **three containers** (data processing, model training, inference API). Old single-script / monolithic setup was removed to keep the repo lean.

## Learning Focus

Hands-on practice with:
- Separating concerns across containers (ETL vs training vs serving)
- Portable feature engineering (shared preprocessor artifact)
- Reproducible model training (deterministic split / params)
- Lightweight inference service (FastAPI + preloaded artifacts)
- Docker Compose orchestration (volumes, dependency ordering, network)
- Minimizing image bloat via per-stage dependency sets

## Repo Structure (Multi-Container)

```
docker-compose.yml
services/
  data_processor/
    Dockerfile
    requirements.txt
    processor/
      preprocess.py
      run.py
  model_trainer/
    Dockerfile
    requirements.txt
    trainer/
      train.py
  prediction_api/
    Dockerfile
    requirements.txt
    api/
      main.py
      schemas.py
data/
  raw/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
models/                # (populated by containers: preprocessor.pkl, churn_model.pkl)
```

## Run the Whole Pipeline

```bash
docker compose up --build
```

Sequence:
1. data_processor: reads raw CSV -> fits preprocessor -> saves `preprocessor.pkl`, `X.npy`, `y.npy`.
2. model_trainer: trains RandomForest -> saves `churn_model.pkl`.
3. prediction_api: serves `/`, `/predict`, `/docs` using shared artifacts.

Visit: http://localhost:8000 for demo UI / docs.

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

## Under the Hood

Preprocessor (data_processor):
- Numeric: median impute + StandardScaler
- Categorical: most_frequent impute + OneHotEncoder(handle_unknown=ignore)

Trainer:
- RandomForestClassifier (200 trees, balanced class weight)
- Metric printed: ROC AUC (validation split)

Inference:
- Loads preprocessor + RF model
- Transforms incoming records consistently with training
- Threshold 0.5 -> binary prediction

## TODO / Improvement Ideas

Retraining & Data Refresh (refine & implement):
- Simple: run `data_processor` + `model_trainer` then restart API.
- Safer versioned rollout: produce `preprocessor_vX.pkl`, `X_vX.npy`, `y_vX.npy`, `churn_model_vX.pkl`; smoke test a temp API container pointing at new artifacts; flip env vars or symlinks; restart only API (fast cutover, near-zero downtime); retain previous version for rollback.
- Potential hot-reload endpoint to swap artifacts in-memory without restart.
- Append model metadata JSON (version, timestamp, ROC AUC, params hash) for traceability.

Other ideas:
- Healthcheck scripts + readiness gate (wait for artifacts before starting API)
- Basic monitoring: log prediction probability distribution & input drift stats
- Batch scoring job container (reuse preprocessor + model)
- CI pipeline: lint, type check, unit + integration tests per service
- Add tests for schema validation & round-trip predict
- Add alternative model (e.g. LightGBM / XGBoost) behind feature flag
- Implement graceful shutdown & concurrency tuning for API
- Security: basic rate limiting / auth token for `/predict`
---

## Multi-Container Architecture

This repo includes a dockerized pipeline split into three services:

| Service | Role | Outputs | Image Focus |
|---------|------|---------|-------------|
| data_processor | Read raw CSV, clean + fit preprocessor | `/data/X.npy`, `/data/y.npy`, `/models/preprocessor.pkl` | Minimal ETL + sklearn preprocessing |
| model_trainer | Train model from preprocessed arrays | `/models/churn_model.pkl` | Training only (no pandas) |
| prediction_api | Serve FastAPI inference + demo UI | HTTP responses | Inference only |

Shared volumes:
- `data_volume` : intermediate arrays (X/y)
- `model_volume`: preprocessor + final model

Compose file: `docker-compose.yml` orchestrates startup order (trainer waits for processor, API waits for trainer).

### Why Split?
- Clear separation of concerns (data prep vs training vs serving)
- Leaner images (faster cold starts for API)
- Reusable preprocessor as a single source of feature engineering truth
- Easier to swap models (only rebuild trainer + model volume)

## Dataset Source & Attribution

The raw file `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` is the public Telco Customer Churn sample dataset (originally distributed by IBM sample datasets and widely mirrored, e.g. on Kaggle). I did not create this dataset, and I assert no ownership or legal claim over it. It is included solely for educational, non-commercial demonstration.

## Disclaimer

Purely educational. No guarantees (accuracy, robustness, reliability). Dataset belongs to original provider; I claim no rights over it. Use responsibly.
