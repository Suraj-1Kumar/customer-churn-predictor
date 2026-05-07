"""
train.py
────────
Runs 14 MLflow experiments (7 XGBoost + 7 LightGBM) in ONE unified
experiment so you can compare them side-by-side in the MLflow UI.

Improvements over baseline:
  ✓ Single data load — shared across all 14 runs (no duplicate loading)
  ✓ stratify=y in split — handled inside preprocess.py
  ✓ SMOTE — handled inside preprocess.py
  ✓ AUC + Recall tracked (not just accuracy)
  ✓ One MLflow experiment, model_type tag for filtering
  ✓ Safe GPU detection — no crash on CPU-only machines
  ✓ SHAP plots on best model
  ✓ Best model saved to models/ for FastAPI to load

Usage:
    python src/train.py
    mlflow ui --port 5000   ← view all 14 runs
"""

import warnings
warnings.filterwarnings("ignore")

import json
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import shap
import xgboost as xgb
import lightgbm as lgb
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from preprocess import get_processed_splits


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_PATH       = 'data/Telco-Customer-Churn.csv'
EXPERIMENT_NAME = "Churn_Prediction_Comparison"   # ONE experiment, both models
MODELS_DIR      = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# GPU detection — safe fallback to CPU
# ─────────────────────────────────────────────────────────────
def get_device() -> str:
    """
    Checks if an NVIDIA GPU is available.
    Returns 'cuda' if yes, 'cpu' if no.

    WHY: Hardcoding device='cuda' crashes on CPU-only machines
    (Colab free tier, CI pipelines, Render deployment).
    This makes the code portable without any manual changes.
    """
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
        print("[Device] GPU detected → using cuda")
        return 'cuda'
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[Device] No GPU found → using cpu")
        return 'cpu'


DEVICE = get_device()


# ─────────────────────────────────────────────────────────────
# Hyperparameter grids — 7 configs per model = 14 total runs
# ─────────────────────────────────────────────────────────────
XGB_GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.80, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.80, "colsample_bytree": 0.8},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.70, "colsample_bytree": 0.7},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.20, "subsample": 0.90, "colsample_bytree": 0.9},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.80, "colsample_bytree": 0.8},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.70, "colsample_bytree": 0.7},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85},
]

LGB_GRID = [
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "num_leaves": 31,  "subsample": 0.8},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "num_leaves": 63,  "subsample": 0.8},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "num_leaves": 127, "subsample": 0.7},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.20, "num_leaves": 31,  "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.10, "num_leaves": 15,  "subsample": 0.8},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "num_leaves": 63,  "subsample": 0.7},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08, "num_leaves": 95,  "subsample": 0.85},
]


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_proba: np.ndarray) -> dict:
    """
    WHY these metrics for churn?

    AUC (primary) — ranks churners above non-churners. Not affected
                    by class imbalance. Industry standard for churn.

    Recall        — % of actual churners caught. The most business-critical
                    metric. Missing a churner = losing their lifetime value.
                    You want this HIGH even if precision drops.

    F1            — balances precision and recall. Good for overall quality.

    Precision     — of customers you flagged, how many actually churned.
                    Matters when intervention cost is high (e.g. $50 coupon).

    Accuracy      — included for reporting, but misleading on imbalanced data.
    """
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "auc":       round(roc_auc_score(y_true, y_proba), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, params: dict):
    """
    Trains one XGBoost configuration.
    tree_method='hist' is the fastest algorithm and works on both
    CPU and GPU — no separate code path needed.
    """
    model = xgb.XGBClassifier(
        **params,
        tree_method  = 'hist',
        device       = DEVICE,
        eval_metric  = 'auc',
        random_state = 42,
        n_jobs       = -1,
        verbosity    = 0,
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    elapsed = round(time.time() - t0, 2)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    metrics["train_time_s"] = elapsed
    return model, metrics


def train_lightgbm(X_train, y_train, X_test, y_test, params: dict):
    """
    Trains one LightGBM configuration.
    early_stopping(50) stops training if AUC doesn't improve for
    50 rounds — prevents overfitting and saves time.
    """
    model = lgb.LGBMClassifier(
        **params,
        random_state = 42,
        n_jobs       = -1,
        verbose      = -1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set  = [(X_test, y_test)],
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    elapsed = round(time.time() - t0, 2)

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_proba)
    metrics["train_time_s"] = elapsed
    return model, metrics


# ─────────────────────────────────────────────────────────────
# SHAP explainability
# ─────────────────────────────────────────────────────────────
def generate_shap_plots(model, X_test: np.ndarray, feature_names: list, model_name: str):
    """
    Plot 1 — Bar chart  : Which features matter most overall?
    Plot 2 — Beeswarm   : How does each feature's value affect churn direction?
                          Red = high value, Blue = low value
                          Right of centre = pushed toward churn

    WHY 300 samples? Enough for reliable importance estimates,
    fast enough not to time out in Colab.
    """
    Path("data/shap_plots").mkdir(parents=True, exist_ok=True)

    sample      = X_test[:300]
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
    ax.set_title(f"Feature Importance (SHAP) — {model_name}", fontsize=13)
    plt.tight_layout()
    bar_path = f"data/shap_plots/shap_bar_{model_name}.png"
    plt.savefig(bar_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13)
    plt.tight_layout()
    bee_path = f"data/shap_plots/shap_beeswarm_{model_name}.png"
    plt.savefig(bee_path, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"[SHAP] Saved → {bar_path}")
    print(f"[SHAP] Saved → {bee_path}")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:10]


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────
def run_all_experiments():

    # ── 1. Load data ONCE — shared across all 14 runs ─────────
    # Old code loaded inside each train function = 14x disk reads
    # + inconsistent splits. Now: one load, one split, shared everywhere.
    print("=" * 58)
    print("  Loading and preprocessing data...")
    print("=" * 58)

    splits        = get_processed_splits(DATA_PATH)
    X_train       = splits["X_train"]
    y_train       = splits["y_train"]
    X_test        = splits["X_test"]
    y_test        = splits["y_test"]
    feature_names = splits["feature_names"]

    # ── 2. One experiment for both models ─────────────────────
    # Filter in MLflow UI by clicking the model_type tag column
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_auc   = 0.0
    best_model = None
    best_meta  = {}

    print(f"\n{'=' * 58}")
    print(f"  Running 14 experiments → '{EXPERIMENT_NAME}'")
    print(f"{'=' * 58}\n")

    # ── 3. XGBoost: 7 runs ────────────────────────────────────
    for i, params in enumerate(XGB_GRID, 1):
        run_name = f"xgb_run_{i:02d}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("dataset", "Telco-Customer-Churn")
            mlflow.log_params(params)
            mlflow.log_param("smote", True)
            mlflow.log_param("stratify", True)

            model, metrics = train_xgboost(X_train, y_train, X_test, y_test, params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_test, model.predict_proba(X_test))
            mlflow.sklearn.log_model(model, "model", signature=signature)

            print(
                f"  {run_name} | AUC={metrics['auc']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"t={metrics['train_time_s']}s"
            )

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                best_model = model
                best_meta  = {
                    "model_type":     "XGBoost",
                    "best_auc":       metrics["auc"],
                    "best_recall":    metrics["recall"],
                    "best_f1":        metrics["f1"],
                    "best_precision": metrics["precision"],
                    "params":         params,
                    "feature_names":  feature_names,
                }

    # ── 4. LightGBM: 7 runs ───────────────────────────────────
    for i, params in enumerate(LGB_GRID, 1):
        run_name = f"lgb_run_{i:02d}"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("dataset", "Telco-Customer-Churn")
            mlflow.log_params(params)
            mlflow.log_param("smote", True)
            mlflow.log_param("stratify", True)

            model, metrics = train_lightgbm(X_train, y_train, X_test, y_test, params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_test, model.predict_proba(X_test))
            mlflow.sklearn.log_model(model, "model", signature=signature)

            print(
                f"  {run_name} | AUC={metrics['auc']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"t={metrics['train_time_s']}s"
            )

            if metrics["auc"] > best_auc:
                best_auc   = metrics["auc"]
                best_model = model
                best_meta  = {
                    "model_type":     "LightGBM",
                    "best_auc":       metrics["auc"],
                    "best_recall":    metrics["recall"],
                    "best_f1":        metrics["f1"],
                    "best_precision": metrics["precision"],
                    "params":         params,
                    "feature_names":  feature_names,
                }

    # ── 5. Summary ────────────────────────────────────────────
    print(f"\n{'=' * 58}")
    print(f"  WINNER : {best_meta['model_type']}")
    print(f"  AUC    : {best_meta['best_auc']:.4f}   ← primary metric")
    print(f"  Recall : {best_meta['best_recall']:.4f}   ← % churners caught")
    print(f"  F1     : {best_meta['best_f1']:.4f}")
    print(f"  Prec.  : {best_meta['best_precision']:.4f}")
    print(f"{'=' * 58}\n")

    # ── 6. SHAP on best model ─────────────────────────────────
    print("[SHAP] Generating plots for best model...")
    top_features = generate_shap_plots(
        best_model, X_test, feature_names, best_meta["model_type"]
    )
    print("\n  Top 5 churn drivers:")
    for feat, val in top_features[:5]:
        print(f"    {feat:<40} mean|SHAP|={val:.4f}")

    # ── 7. Save best model for FastAPI ────────────────────────
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)

    print(f"\n✓ Model saved  → models/best_model.joblib")
    print(f"✓ Meta saved   → models/model_meta.json")
    print(f"\n  To view all runs: mlflow ui --port 5000")

    return best_model, best_meta


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all_experiments()