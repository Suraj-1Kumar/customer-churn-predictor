# ChurnSight — Customer Churn Predictor

<div align="center">

![ChurnSight Banner](https://img.shields.io/badge/ChurnSight-ML%20Pipeline-00d4ff?style=for-the-badge&logo=python&logoColor=black)

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-00e5a0?style=flat-square)](https://customer-churn-predictor.onrender.com)
[![API Docs](https://img.shields.io/badge/📖%20API%20Docs-Swagger-00d4ff?style=flat-square)](https://customer-churn-predictor.onrender.com/docs)
[![Model Info](https://img.shields.io/badge/🤖%20Model%20Info-JSON-a855f7?style=flat-square)](https://customer-churn-predictor.onrender.com/model/info)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.84-brightgreen?style=flat-square)

**End-to-end ML pipeline that predicts which telecom customers will churn.**  
EDA → Feature Engineering → XGBoost/LightGBM → MLflow → FastAPI → Docker → Render

</div>

---

## 🎯 Resume Bullet

> *"Engineered end-to-end churn prediction pipeline (7K Telco customers) with LightGBM achieving 0.84 AUC and 0.64 Recall; applied SMOTE for class imbalance, tracked 14 experiments via MLflow, built interactive dashboard with FastAPI, and deployed as a live REST API on Render."*

---

## 🖥️ Live Demo

**Dashboard →** `https://customer-churn-predictor-mymg.onrender.com`  
**Swagger UI →** `https://customer-churn-predictor-mymg.onrender.com/docs`

> ⚠️ Hosted on Render free tier — may take 30s to wake up on first visit.

---

## 📊 Results

| Metric | Score | What it means |
|---|---|---|
| **AUC** | **0.8431** | Ranks churners above non-churners 84% of the time |
| **Recall** | **0.6364** | Catches 64% of customers who would actually leave |
| **F1** | **0.6079** | Balanced precision-recall score |
| **Precision** | **0.5819** | 58% of flagged customers truly at risk |

**Winner:** LightGBM (`lgb_run_05`) — beat XGBoost across all 14 experiments.

---

## 🏗️ Architecture

```
Telco CSV (7,043 customers)
    │
    ▼
┌─────────────────────────────┐
│   EDA + Feature Engineering │  Pandas · 5 derived features · SMOTE
│   preprocess.py             │  stratify split · StandardScaler
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   14 MLflow Experiments     │  7 × XGBoost + 7 × LightGBM
│   train.py                  │  AUC · Recall · F1 · Precision tracked
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   SHAP Explainability        │  TreeExplainer · Bar + Beeswarm plots
│   explain.py                 │  Top churn drivers identified
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   FastAPI REST Endpoint      │  /predict · /predict/batch · /model/info
│   src/api.py                 │  Pydantic validation · <5ms latency
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Dashboard UI               │  Dark theme · Live gauge · Risk tiers
│   static/index.html          │  Served directly by FastAPI
└─────────────┬───────────────┘
              │
              ▼
    Docker → Render (live HTTPS)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| Preprocessing | Scikit-learn, imbalanced-learn (SMOTE) |
| Modeling | XGBoost, LightGBM |
| Experiment Tracking | MLflow (14 runs, unified experiment) |
| Explainability | SHAP (TreeExplainer, bar + beeswarm) |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Container | Docker (python:3.11-slim) |
| Deployment | Render (free tier, auto-HTTPS) |

---

## 📁 Project Structure

```
customer-churn-predictor/
├── src/
│   ├── preprocess.py      # EDA · feature engineering · SMOTE pipeline
│   ├── train.py           # 14 MLflow experiments · model selection
│   ├── explain.py         # SHAP explainability plots
│   └── api.py             # FastAPI endpoint · Pydantic validation
├── static/
│   └── index.html         # Interactive dashboard UI
├── models/
│   ├── best_model.joblib  # Saved best model (LightGBM)
│   └── model_meta.json    # AUC, params, feature names
├── notebooks/
│   └── eda_prototyping.py # Exploratory analysis
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

**1. Clone and install**
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-predictor.git
cd customer-churn-predictor
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Add the dataset**

Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it at:
```
data/Telco-Customer-Churn.csv
```

**3. Run the full pipeline**
```bash
python src/preprocess.py    # EDA + feature engineering
python src/train.py         # 14 MLflow experiments
```

**4. Start the API + dashboard**
```bash
$env:PYTHONPATH = "."                                    # Windows PowerShell
python -m uvicorn src.api:app --reload --port 8000
```

Open `http://localhost:8000` — dashboard loads instantly.

**5. View MLflow experiments**
```bash
mlflow ui --port 5000
# → http://localhost:5000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model metadata + AUC |
| `POST` | `/predict` | Single customer prediction |
| `POST` | `/predict/batch` | Batch predictions (max 500) |
| `GET` | `/docs` | Interactive Swagger UI |

**Example — high risk customer:**
```bash
curl -X POST https://customer-churn-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8247,
  "churn_prediction": 1,
  "risk_tier": "critical",
  "model_type": "LightGBM",
  "latency_ms": 3.2
}
```

---

## 🔍 Key Engineering Decisions

**Why SMOTE after split, not before?**  
Applying SMOTE before splitting lets synthetic samples leak into the test set → falsely optimistic metrics. SMOTE runs on training data only. Test set stays at real-world 26.5% churn rate.

**Why AUC over Accuracy?**  
73.5% of customers don't churn. A model that always predicts "No" scores 73.5% accuracy for free. AUC measures ranking quality regardless of class balance — it's the honest metric here.

**Why Recall over Precision as the business metric?**  
Missing a churner costs their lifetime value (~$1,000+). Wrongly flagging a loyal customer costs a retention coupon (~$10). High recall is worth the tradeoff.

**Why one MLflow experiment for both models?**  
XGBoost and LightGBM tracked in a single experiment → side-by-side comparison in the UI. Separate experiments make head-to-head analysis manual.

---

## 📈 MLflow Experiment Leaderboard

| Run | Model | AUC | Recall | F1 |
|---|---|---|---|---|
| lgb_run_05 | **LightGBM ✓** | **0.8431** | 0.6364 | 0.6079 |
| lgb_run_01 | LightGBM | 0.8411 | 0.6417 | 0.6091 |
| lgb_run_04 | LightGBM | 0.8407 | 0.6471 | 0.6080 |
| xgb_run_01 | XGBoost  | 0.8371 | 0.6070 | 0.5958 |
| xgb_run_06 | XGBoost  | 0.8364 | 0.6016 | 0.5898 |

*14 total runs. Full comparison: `mlflow ui --port 5000`*

---

## 🐳 Docker

```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
# → http://localhost:8000
```

---

## 📄 License

MIT — free to use, fork, and build on.

---

<div align="center">
Built as a DS portfolio project · Dataset: Telco Customer Churn (IBM / Kaggle)
</div>
