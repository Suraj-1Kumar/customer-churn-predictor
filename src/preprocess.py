"""
preprocess.py
─────────────
Handles everything before the model sees the data:
  1. Load & clean raw Telco CSV
  2. Feature engineering (raw + 5 derived features)
  3. Encode categoricals (binary map + one-hot)
  4. Train/test split with stratify=y
  5. Scale numerics (fit on train only — no leakage)
  6. SMOTE oversampling (training set only)

Usage:
    from preprocess import get_processed_splits
    splits = get_processed_splits('data/Telco-Customer-Churn.csv')

Quick test:
    python src/preprocess.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load & Clean
# ─────────────────────────────────────────────────────────────
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the Telco CSV and fixes two known data quality issues.

    Issue 1 — TotalCharges is object dtype:
        New customers with tenure=0 have a blank space " " instead
        of a number. pd.to_numeric(..., errors='coerce') converts
        those to NaN, which our imputer handles later.
        We do NOT dropna here — imputing is better than losing rows.

    Issue 2 — customerID has zero predictive value:
        It's a row identifier. Keeping it would let the model memorize
        individual customers instead of learning patterns.
    """
    df = pd.read_csv(filepath)
    print(f"[Load] Raw shape       : {df.shape}")

    # Fix TotalCharges: " " (blank string) → NaN → median imputed later
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop row identifier
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    missing = df.isnull().sum()
    if missing.any():
        print(f"[Load] Missing values  :\n{missing[missing > 0]}")
        print(f"       Will be filled by median imputer in pipeline")

    print(f"[Load] Cleaned shape   : {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 5 derived features + encodes all categoricals.

    WHY derive features at all?
        Raw columns tell you WHAT happened.
        Derived features tell you WHAT IT MEANS.
        e.g. MonthlyCharges=$80 means different risk for a 1-month
        customer vs a 48-month customer. ChargesPerTenure captures that.

    Derived features:
        ChargesPerTenure        → spend efficiency (high = paying a lot for short time)
        TenureChargesInteraction→ loyalty-value signal (long tenure × high spend)
        IsNewCustomer           → first 6 months = peak churn window (binary flag)
        ServiceCount            → number of add-on services subscribed
        HasMultipleServices     → 3+ services = high switching cost → churns less
    """
    df = df.copy()

    # ── Derived Feature 1 ─────────────────────────────────────
    # Monthly spend per tenure month — normalizes charges by loyalty
    # High value = paying a lot relative to how long they've stayed
    df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

    # ── Derived Feature 2 ─────────────────────────────────────
    # Tenure × MonthlyCharges = proxy for total committed value
    # A 60-month customer paying $90/month is much more valuable
    # (and more loyal) than a 2-month customer paying the same
    df['TenureChargesInteraction'] = df['tenure'] * df['MonthlyCharges']

    # ── Derived Feature 3 ─────────────────────────────────────
    # Hard binary flag — churn rate spikes in months 1-6
    # Tree models love hard boundaries like this
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)

    # ── Derived Feature 4 & 5 ─────────────────────────────────
    # Count add-on services — more services = higher switching cost
    # "No internet service" means they couldn't subscribe, counts as 0
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})

    existing_services = [c for c in service_cols if c in df.columns]
    df['ServiceCount']        = df[existing_services].sum(axis=1)
    df['HasMultipleServices'] = (df['ServiceCount'] >= 3).astype(int)

    # ── Encode Target ──────────────────────────────────────────
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].astype(str).str.strip().map({'Yes': 1, 'No': 0})

    # ── Binary Yes/No columns ─────────────────────────────────
    # str.strip() handles any accidental trailing spaces in the CSV
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().map({'Yes': 1, 'No': 0})

    # ── Gender ────────────────────────────────────────────────
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})

    # ── MultipleLines: 3-value column ─────────────────────────
    # "No phone service" → 0 (same as No for our purposes)
    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].map({
            'Yes': 1, 'No': 0, 'No phone service': 0
        })

    # ── One-Hot Encode nominal categoricals ───────────────────
    # These have no natural order, so we can't use 0/1/2 encoding.
    # drop_first=True removes one column per group to avoid the
    # dummy variable trap (perfect multicollinearity).
    ohe_cols = ['InternetService', 'Contract', 'PaymentMethod']
    existing_ohe = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_ohe, drop_first=True, dtype=int)

    # ── Safety: keep only numeric ─────────────────────────────
    # Catches any column we forgot to encode
    df = df.select_dtypes(include=[np.number])

    print(f"[FeatEng] Shape after engineering : {df.shape}")
    print(f"[FeatEng] Features (excl. target) : {df.shape[1] - 1}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Sklearn Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────
def build_preprocessor(numeric_cols: list, passthrough_cols: list) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
      - Imputes then scales numeric columns
      - Passes binary/OHE columns through unchanged

    WHY a Pipeline object instead of manual transforms?
      1. Fit on train only, apply to test → zero data leakage
      2. Same object used at inference time in FastAPI → consistency
      3. Can be saved with joblib alongside the model
    """
    numeric_pipeline = Pipeline([
        # Median imputation: robust to outliers vs mean
        ("imputer", SimpleImputer(strategy="median")),
        # StandardScaler: mean=0, std=1
        # XGBoost/LightGBM don't need scaling (tree splits are scale-invariant)
        # but it's good practice for portability and future model additions
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  numeric_pipeline,  numeric_cols),
            ("pass", "passthrough",     passthrough_cols),
        ],
        remainder             = "drop",
        verbose_feature_names_out = False,
    )
    return preprocessor


# ─────────────────────────────────────────────────────────────
# STEP 4 — Master function: full pipeline in one call
# ─────────────────────────────────────────────────────────────
def get_processed_splits(
    filepath: str,
    test_size: float    = 0.2,
    random_state: int   = 42,
) -> dict:
    """
    Runs the complete preprocessing pipeline end-to-end.

    Returns a dict the training script uses directly:
    {
      X_train, y_train  → SMOTE-balanced, scaled training data
      X_test,  y_test   → untouched held-out test set (raw class distribution)
      feature_names     → column names after all transforms
      preprocessor      → fitted ColumnTransformer (for FastAPI inference)
      X_test_raw        → unscaled X_test (for SHAP value display)
    }

    Critical ordering — always in this sequence:
      Split first → fit preprocessor on train only
      → SMOTE on train only → test set never touched by fit()
    """

    # 1. Load and clean
    df = load_and_clean_data(filepath)

    # 2. Feature engineer
    df = feature_engineering(df)

    # 3. Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    print(f"\n[Split] Class distribution:")
    print(f"        Retained : {(y==0).sum():,}  ({(y==0).mean():.1%})")
    print(f"        Churned  : {(y==1).sum():,}  ({(y==1).mean():.1%})")

    # 4. Train/test split
    # stratify=y is critical — without it, a bad random seed could
    # give you 40% churn in test and 20% in train → unreliable evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,           # ← preserves churn ratio in both sets
    )
    print(f"\n[Split] Train : {X_train.shape[0]:,} | Test : {X_test.shape[0]:,}")

    # 5. Identify column types for the preprocessor
    numeric_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'ChargesPerTenure', 'TenureChargesInteraction', 'ServiceCount'
    ]
    numeric_cols     = [c for c in numeric_cols if c in X.columns]
    passthrough_cols = [c for c in X.columns if c not in numeric_cols]

    # 6. Fit preprocessor on TRAIN only, transform both
    preprocessor  = build_preprocessor(numeric_cols, passthrough_cols)
    X_train_proc  = preprocessor.fit_transform(X_train)  # fit + transform
    X_test_proc   = preprocessor.transform(X_test)       # transform only — no fit

    feature_names = preprocessor.get_feature_names_out().tolist()
    print(f"[Preprocess] Feature count : {len(feature_names)}")

    # 7. SMOTE — only on training data
    # SMOTE (Synthetic Minority Oversampling TEchnique):
    #   Finds each minority-class sample (churned customer),
    #   looks at its k nearest neighbors, and creates new synthetic
    #   samples by interpolating between them in feature space.
    #
    # Result: balanced training set → model must learn churn patterns
    # not just predict the majority class.
    #
    # NEVER apply to test data — it must reflect the real world distribution
    # so your evaluation metrics are honest.
    print(f"\n[SMOTE] Before → {X_train_proc.shape[0]:,} rows | "
          f"Churn rate: {y_train.mean():.1%}")

    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    print(f"[SMOTE] After  → {X_train_res.shape[0]:,} rows | "
          f"Churn rate: {y_train_res.mean():.1%}")

    return {
        "X_train":       X_train_res,
        "y_train":       y_train_res,
        "X_test":        X_test_proc,
        "y_test":        y_test,
        "feature_names": feature_names,
        "preprocessor":  preprocessor,
        "X_test_raw":    X_test,       # unscaled, for SHAP display
    }


# ─────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to verify
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    splits = get_processed_splits('data/Telco-Customer-Churn.csv')

    print("\n── Final verification ────────────────────────────")
    print(f"X_train shape      : {splits['X_train'].shape}")
    print(f"X_test  shape      : {splits['X_test'].shape}")
    print(f"y_train churn rate : {splits['y_train'].mean():.1%}  (should be ~50% after SMOTE)")
    print(f"y_test  churn rate : {splits['y_test'].mean():.1%}   (should be ~26%, real distribution)")
    print(f"\nAll features ({len(splits['feature_names'])}):")
    for f in splits['feature_names']:
        print(f"  {f}")