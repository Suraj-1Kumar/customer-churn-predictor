"""
api.py  —  FastAPI Churn Prediction Endpoint
─────────────────────────────────────────────
Loads the best model trained on Telco Customer Churn data and exposes:

  GET  /              → welcome message
  GET  /health        → liveness check (used by Render)
  GET  /model/info    → AUC, model type, feature count
  POST /predict       → single customer churn probability
  POST /predict/batch → batch predictions (up to 500 customers)

Run locally:
    uvicorn src.api:app --reload --port 8000

Swagger UI (interactive docs):
    http://localhost:8000/docs
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────
# Load model + metadata at startup
# ─────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/best_model.joblib")
META_PATH  = Path("models/model_meta.json")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found at models/best_model.joblib — run src/train.py first."
    )

model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

FEATURE_NAMES: list = meta.get("feature_names", [])
print(f"[API] Model loaded  : {meta.get('model_type')}")
print(f"[API] AUC           : {meta.get('best_auc')}")
print(f"[API] Features      : {len(FEATURE_NAMES)}")


# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn Predictor",
    description = (
        "Predicts the probability that a telecom customer will churn. "
        "Trained on the Telco Customer Churn dataset (7,043 customers) "
        "using LightGBM with SMOTE resampling. "
        "Tracked with MLflow · Containerised with Docker · Deployed on Render."
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve dashboard UI from /static folder
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────
# Pydantic input schema
# ─────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    """
    All 19 raw input features for one Telco customer.
    Pydantic validates types and allowed values automatically.
    Bad inputs return a 422 error before the model ever runs.
    """
    # Numeric
    tenure          : int            = Field(..., ge=0, le=72,  description="Months with the company (0–72)")
    MonthlyCharges  : float          = Field(..., ge=0, le=200, description="Monthly bill in USD")
    TotalCharges    : Optional[float]= Field(None,              description="Total spend — null = new customer")

    # Demographics
    gender          : str = Field(..., description="Male | Female")
    SeniorCitizen   : int = Field(..., ge=0, le=1, description="1 = senior, 0 = not")
    Partner         : str = Field(..., description="Yes | No")
    Dependents      : str = Field(..., description="Yes | No")

    # Phone
    PhoneService    : str = Field(..., description="Yes | No")
    MultipleLines   : str = Field(..., description="Yes | No | No phone service")

    # Internet
    InternetService : str = Field(..., description="DSL | Fiber optic | No")
    OnlineSecurity  : str = Field(..., description="Yes | No | No internet service")
    OnlineBackup    : str = Field(..., description="Yes | No | No internet service")
    DeviceProtection: str = Field(..., description="Yes | No | No internet service")
    TechSupport     : str = Field(..., description="Yes | No | No internet service")
    StreamingTV     : str = Field(..., description="Yes | No | No internet service")
    StreamingMovies : str = Field(..., description="Yes | No | No internet service")

    # Account
    Contract        : str = Field(..., description="Month-to-month | One year | Two year")
    PaperlessBilling: str = Field(..., description="Yes | No")
    PaymentMethod   : str = Field(..., description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in {"Male", "Female"}:
            raise ValueError("gender must be 'Male' or 'Female'")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        if v not in {"DSL", "Fiber optic", "No"}:
            raise ValueError("InternetService must be 'DSL', 'Fiber optic', or 'No'")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in {"Month-to-month", "One year", "Two year"}:
            raise ValueError("Contract must be 'Month-to-month', 'One year', or 'Two year'")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment(cls, v):
        valid = {
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        }
        if v not in valid:
            raise ValueError(f"PaymentMethod must be one of: {valid}")
        return v


# ─────────────────────────────────────────────────────────────
# Pydantic output schemas
# ─────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    churn_probability : float = Field(..., description="Probability of churn 0.0–1.0")
    churn_prediction  : int   = Field(..., description="1 = will churn, 0 = will stay")
    risk_tier         : str   = Field(..., description="low | medium | high | critical")
    model_type        : str
    latency_ms        : float


class BatchResult(BaseModel):
    predictions : List[PredictionResult]
    total       : int
    churn_count : int
    churn_rate  : float


# ─────────────────────────────────────────────────────────────
# Feature engineering — mirrors preprocess.py exactly
# ─────────────────────────────────────────────────────────────
def build_feature_vector(c: CustomerFeatures) -> np.ndarray:
    """
    Replicates the same feature engineering from preprocess.py
    for a single live inference request.

    Column order must exactly match what the model was trained on.
    The order is defined by FEATURE_NAMES in model_meta.json:
      tenure, MonthlyCharges, TotalCharges, ChargesPerTenure,
      TenureChargesInteraction, ServiceCount, gender, SeniorCitizen,
      Partner, Dependents, PhoneService, MultipleLines,
      OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
      StreamingTV, StreamingMovies, PaperlessBilling,
      IsNewCustomer, HasMultipleServices,
      InternetService_Fiber optic, InternetService_No,
      Contract_One year, Contract_Two year,
      PaymentMethod_Credit card (automatic),
      PaymentMethod_Electronic check, PaymentMethod_Mailed check
    """
    tenure  = c.tenure
    monthly = c.MonthlyCharges
    total   = c.TotalCharges if c.TotalCharges is not None else (tenure * monthly)

    # Derived features
    charges_per_tenure         = monthly / (tenure + 1)
    tenure_charges_interaction = tenure * monthly
    is_new_customer            = int(tenure <= 6)

    # Service count
    svc = {"Yes": 1, "No": 0, "No internet service": 0}
    service_count = sum([
        svc.get(c.OnlineSecurity,   0),
        svc.get(c.OnlineBackup,     0),
        svc.get(c.DeviceProtection, 0),
        svc.get(c.TechSupport,      0),
        svc.get(c.StreamingTV,      0),
        svc.get(c.StreamingMovies,  0),
    ])
    has_multiple_services = int(service_count >= 3)

    # Binary encodings
    yn = {"Yes": 1, "No": 0}
    gender_enc  = 1 if c.gender == "Female" else 0
    partner_enc = yn.get(c.Partner, 0)
    dep_enc     = yn.get(c.Dependents, 0)
    phone_enc   = yn.get(c.PhoneService, 0)
    paper_enc   = yn.get(c.PaperlessBilling, 0)
    multi_enc   = 1 if c.MultipleLines == "Yes" else 0

    # Service binary encodings
    sec_enc  = svc.get(c.OnlineSecurity,   0)
    back_enc = svc.get(c.OnlineBackup,     0)
    dev_enc  = svc.get(c.DeviceProtection, 0)
    tech_enc = svc.get(c.TechSupport,      0)
    tv_enc   = svc.get(c.StreamingTV,      0)
    mov_enc  = svc.get(c.StreamingMovies,  0)

    # One-hot: InternetService (DSL = baseline/dropped)
    inet_fiber = int(c.InternetService == "Fiber optic")
    inet_no    = int(c.InternetService == "No")

    # One-hot: Contract (Month-to-month = baseline/dropped)
    contract_one = int(c.Contract == "One year")
    contract_two = int(c.Contract == "Two year")

    # One-hot: PaymentMethod (Bank transfer automatic = baseline/dropped)
    pay_credit = int(c.PaymentMethod == "Credit card (automatic)")
    pay_echeck = int(c.PaymentMethod == "Electronic check")
    pay_mail   = int(c.PaymentMethod == "Mailed check")

    # Assemble in exact training column order
    row = np.array([
        tenure, monthly, total,
        charges_per_tenure, tenure_charges_interaction, service_count,
        gender_enc, c.SeniorCitizen,
        partner_enc, dep_enc, phone_enc, multi_enc,
        sec_enc, back_enc, dev_enc, tech_enc, tv_enc, mov_enc,
        paper_enc,
        is_new_customer, has_multiple_services,
        inet_fiber, inet_no,
        contract_one, contract_two,
        pay_credit, pay_echeck, pay_mail,
    ], dtype=float)

    return row.reshape(1, -1)


def get_risk_tier(prob: float) -> str:
    """
    Converts raw probability into a business action tier.
      low      (<25%) → no action needed
      medium   (<50%) → add to watch list
      high     (<75%) → proactive outreach
      critical (≥75%) → immediate retention offer
    """
    if prob < 0.25: return "low"
    if prob < 0.50: return "medium"
    if prob < 0.75: return "high"
    return "critical"


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Serve the dashboard UI."""
    ui_path = Path("static/index.html")
    if ui_path.exists():
        return FileResponse(ui_path)
    return {
        "message"   : "Customer Churn Predictor API — live",
        "docs"      : "/docs",
        "health"    : "/health",
        "model_info": "/model/info",
    }


@app.get("/health")
def health():
    """Liveness check — Render pings this every 30s to confirm service is up."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info():
    """Returns metadata about the currently deployed model."""
    return {
        "model_type"   : meta.get("model_type"),
        "best_auc"     : meta.get("best_auc"),
        "best_recall"  : meta.get("best_recall"),
        "best_f1"      : meta.get("best_f1"),
        "feature_count": len(FEATURE_NAMES),
        "features"     : FEATURE_NAMES,
        "params"       : meta.get("params", {}),
    }


@app.post("/predict", response_model=PredictionResult)
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    Send raw Telco feature values — engineering happens server-side.
    """
    t0 = time.perf_counter()
    try:
        X    = build_feature_vector(customer)
        prob = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResult(
        churn_probability = round(prob, 4),
        churn_prediction  = int(prob >= 0.5),
        risk_tier         = get_risk_tier(prob),
        model_type        = meta.get("model_type", "unknown"),
        latency_ms        = round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchResult)
def predict_batch(customers: List[CustomerFeatures]):
    """
    Batch churn prediction for up to 500 customers.
    Returns individual predictions + aggregate churn rate.
    """
    if len(customers) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 500 customers per request."
        )

    predictions = []
    for c in customers:
        X    = build_feature_vector(c)
        prob = float(model.predict_proba(X)[0, 1])
        predictions.append(PredictionResult(
            churn_probability = round(prob, 4),
            churn_prediction  = int(prob >= 0.5),
            risk_tier         = get_risk_tier(prob),
            model_type        = meta.get("model_type", "unknown"),
            latency_ms        = 0.0,
        ))

    churn_count = sum(p.churn_prediction for p in predictions)
    return BatchResult(
        predictions = predictions,
        total       = len(predictions),
        churn_count = churn_count,
        churn_rate  = round(churn_count / len(predictions), 4),
    )


# ─────────────────────────────────────────────────────────────
# Dev server
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)