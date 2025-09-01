import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from app.schemas import PredictionRequest, PredictionResponse
import pandas as pd

MODEL_PATH = "models/churn_pipeline.pkl"

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Lazy load so app starts even if model missing (better error msg)
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    pipeline = joblib.load(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Churn Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            .container { max-width: 820px; }
            .panel { margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
            button { padding: 8px 14px; margin: 6px 6px 0 0; cursor: pointer; }
            select, input { margin: 4px 0 12px 0; padding: 6px; width: 220px; }
            .row { display: flex; flex-wrap: wrap; gap: 32px; }
            .col { flex: 1 1 240px; min-width: 240px; }
            pre { background:#f7f7f7; padding:12px; border-radius:6px; overflow:auto; }
            #probBox { font-size: 2.2rem; font-weight: 600; padding: 18px 26px; border-radius: 12px; display:inline-block; margin-top:10px; }
            .badge-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; display:block; margin-bottom:4px; opacity:0.7; }
        </style>
        <script>
            async function callPredict(record) {
                const payload = { records: [record] };
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                const proba = data.probabilities[0];
                updateProbability(proba);
                document.getElementById("rawResult").innerText = JSON.stringify(data, null, 2);
            }

            function updateProbability(p) {
                const box = document.getElementById("probBox");
                const pct = (p * 100).toFixed(2) + "%";
                box.innerText = pct;
                // Color scale green (0) -> yellow (0.5) -> red (1)
                const r = p < 0.5 ? Math.round(510 * p) : 255;
                const g = p < 0.5 ? 255 : Math.round(510 * (1 - p));
                box.style.background = `rgba(${r},${g},80,0.18)`;
                box.style.color = `rgb(${r},${g},40)`;
                box.title = "Churn probability";
            }

            function submitForm() {
                const record = {
                    gender: document.getElementById("gender").value,
                    SeniorCitizen: parseInt(document.getElementById("senior").value),
                    Partner: document.getElementById("partner").value,
                    Dependents: document.getElementById("dependents").value,
                    tenure: parseInt(document.getElementById("tenure").value),
                    PhoneService: document.getElementById("phone").value,
                    InternetService: document.getElementById("internet").value,
                    Contract: document.getElementById("contract").value,
                    PaperlessBilling: document.getElementById("paperless").value,
                    PaymentMethod: document.getElementById("payment").value,
                    MonthlyCharges: parseFloat(document.getElementById("charges").value),
                    TotalCharges: parseFloat(document.getElementById("totalCharges").value)
                };
                callPredict(record);
            }

            function loadExample(rec) {
                for (const [k,v] of Object.entries(rec)) {
                    const el = document.getElementById(kMap[k] || k);
                    if (!el) continue;
                    el.value = v;
                }
                callPredict(rec);
            }

            const kMap = {}; // simple passthrough mapping (kept for future)
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Churn Prediction API Demo</h1>
            <p>Interactive form + one‑click examples. Full schema in <a href="/docs" target="_blank">OpenAPI docs</a>.</p>

            <div class="panel">
                <h3>Highlighted Prediction</h3>
                <span class="badge-label">Churn Probability</span>
                <div id="probBox">--%</div>
            </div>

            <div class="panel">
                <h3>Customize Input</h3>
                <div class="row">
                    <div class="col">
                        <label>Gender</label><br/>
                        <select id="gender"><option>Female</option><option>Male</option></select><br/>
                        <label>SeniorCitizen (0/1)</label><br/>
                        <select id="senior"><option value="0">0</option><option value="1">1</option></select><br/>
                        <label>Partner</label><br/>
                        <select id="partner"><option>Yes</option><option>No</option></select><br/>
                        <label>Dependents</label><br/>
                        <select id="dependents"><option>No</option><option>Yes</option></select><br/>
                        <label>Tenure (months)</label><br/>
                        <input id="tenure" type="number" min="0" value="12"/><br/>
                        <label>Monthly Charges</label><br/>
                        <input id="charges" type="number" step="0.01" value="70.35"/><br/>
                    </div>
                    <div class="col">
                        <label>PhoneService</label><br/>
                        <select id="phone"><option>Yes</option><option>No</option></select><br/>
                        <label>InternetService</label><br/>
                        <select id="internet">
                            <option>Fiber optic</option>
                            <option>DSL</option>
                            <option>None</option>
                        </select><br/>
                        <label>Contract</label><br/>
                        <select id="contract">
                            <option>Month-to-month</option>
                            <option>One year</option>
                            <option>Two year</option>
                        </select><br/>
                        <label>PaperlessBilling</label><br/>
                        <select id="paperless"><option>Yes</option><option>No</option></select><br/>
                        <label>PaymentMethod</label><br/>
                        <select id="payment">
                            <option>Electronic check</option>
                            <option>Mailed check</option>
                            <option>Bank transfer (automatic)</option>
                            <option>Credit card (automatic)</option>
                        </select><br/>
                        <label>TotalCharges</label><br/>
                        <input id="totalCharges" type="number" step="0.01" value="840.50"/><br/>
                    </div>
                </div>
                <button onclick="submitForm()">Predict</button>
            </div>

            <div class="panel">
                <h3>Quick Examples</h3>
                <button onclick='loadExample({"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":5,"PhoneService":"Yes","InternetService":"Fiber optic","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":89.10,"TotalCharges":445.50})'>High Risk-ish</button>
                <button onclick='loadExample({"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"Yes","tenure":60,"PhoneService":"Yes","InternetService":"DSL","Contract":"Two year","PaperlessBilling":"No","PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":49.95,"TotalCharges":2997.00})'>Likely Low Risk</button>
            </div>

            <div class="panel">
                <h3>Raw API Response</h3>
                <pre id="rawResult">{ "probabilities": [], "predictions": [] }</pre>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    rows = [r.model_dump() for r in req.records]
    X = pd.DataFrame(rows)

    # Make sure all expected training columns exist (fill missing with NA)
    try:
        pre = pipeline.named_steps["preprocess"]
        expected_cols = []
        for name, trans, cols in pre.transformers:
            expected_cols.extend(cols)
        missing = [c for c in expected_cols if c not in X.columns]
        for c in missing:
            X[c] = pd.NA
        # Reorder
        X = X[expected_cols]
    except Exception:
        pass  # Fallback: attempt prediction directly

    proba = pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return PredictionResponse(
        probabilities=[float(p) for p in proba],
        predictions=[int(x) for x in preds]
    )
