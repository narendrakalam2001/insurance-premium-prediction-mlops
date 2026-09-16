# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Insurance Premium ML System
# Dataset: Kaggle Playground Series S4E12 (Premium Amount target)
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns (post-normalization, snake_case) ─────────
REQUIRED_COLS = [
    "age", "gender", "annual_income", "marital_status", "number_of_dependents",
    "education_level", "occupation", "health_score", "location", "policy_type",
    "previous_claims", "vehicle_age", "credit_score", "insurance_duration",
    "policy_start_date", "customer_feedback", "smoking_status",
    "exercise_frequency", "property_type", "premium_amount",
]

NUMERIC_COLS = [
    "age", "annual_income", "number_of_dependents", "health_score",
    "previous_claims", "vehicle_age", "credit_score", "insurance_duration",
]

CATEGORICAL_COLS = [
    "gender", "marital_status", "education_level", "occupation", "location",
    "policy_type", "customer_feedback", "smoking_status", "exercise_frequency",
    "property_type",
]


# ============================================================
# DATA VALIDATION + CLEANING
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names → snake_case ──────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
        .str.strip("_")
    )

    # ── Drop id column if present ─────────────────────────────
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Normalize categorical text ────────────────────────────
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    # ── Parse date column ──────────────────────────────────────
    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"], errors="coerce")

    # ── Target validation ────────────────────────────────────
    df = df[df["premium_amount"].notna()]
    if (df["premium_amount"] <= 0).any():
        n_bad = int((df["premium_amount"] <= 0).sum())
        logger.warning("Dropping %d rows with non-positive premium_amount", n_bad)
        df = df[df["premium_amount"] > 0]

    # ── Missing value report ─────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset contains %d missing values — imputing in validate_input_data", nulls)

    # ── Impute numeric columns with median ────────────────────
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # ── Impute categorical columns with mode / 'Unknown' ───────
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # ── Impute date column with median date ───────────────────
    if df["policy_start_date"].isnull().any():
        median_date = df["policy_start_date"].median()
        df["policy_start_date"] = df["policy_start_date"].fillna(median_date)

    # ── Minimum size check ───────────────────────────────────
    if df.shape[0] < 500:
        raise ValueError("Dataset too small for training (< 500 rows)")

    # ── Deduplication ────────────────────────────────────────
    before = len(df)
    dedup_cols = [c for c in df.columns if c != "policy_start_date"]
    df.drop_duplicates(subset=dedup_cols, ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info("Data validation passed  |  shape=%s  |  mean_premium=%.2f",
                df.shape, df["premium_amount"].mean())

    return df.reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Actuarial-grade feature engineering for insurance premium pricing,
    based on health, credit, driving, and lifestyle risk signals.
    """
    df = df.copy()

    # ── Normalize categorical text (safe if called standalone) ─
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ── Binary risk encodes ───────────────────────────────────
    df["is_smoker"] = (df["smoking_status"] == "Yes").astype(int)
    df["is_male"]   = (df["gender"] == "Male").astype(int)
    df["is_senior"] = (df["age"] >= 60).astype(int)

    df["low_health_score_flag"]  = (df["health_score"] < 30).astype(int)
    df["low_credit_score_flag"]  = (df["credit_score"] < 500).astype(int)
    df["high_claims_flag"]       = (df["previous_claims"] >= 3).astype(int)
    df["old_vehicle_flag"]       = (df["vehicle_age"] >= 15).astype(int)
    df["has_dependents"]         = (df["number_of_dependents"] > 0).astype(int)

    # ── Policy tenure features from policy_start_date ──────────
    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"], errors="coerce")
    reference_date = pd.Timestamp("2025-01-01")
    df["policy_age_days"]   = (reference_date - df["policy_start_date"]).dt.days.clip(lower=0)
    df["policy_start_year"] = df["policy_start_date"].dt.year.astype(float)
    df["policy_start_month"] = df["policy_start_date"].dt.month.astype(float)

    # ── High-signal interaction features ───────────────────────
    # Smoking + poor health compounds risk; credit + income relates to
    # both claim propensity and ability to pay premiums on time.
    df["smoker_x_low_health"]   = df["is_smoker"] * df["low_health_score_flag"]
    df["smoker_x_claims"]       = df["is_smoker"] * df["previous_claims"]
    df["age_x_claims"]          = df["age"] * df["previous_claims"]
    df["credit_x_income"]       = df["credit_score"] * np.log1p(df["annual_income"])
    df["health_x_age"]          = (100 - df["health_score"]) * df["age"] / 100.0

    # ── Composite actuarial risk score (heuristic, not the model) ─
    df["risk_score"] = (
        df["is_smoker"] * 2.5
        + df["low_health_score_flag"] * 2.0
        + df["low_credit_score_flag"] * 1.5
        + df["high_claims_flag"] * 3.0
        + df["old_vehicle_flag"] * 1.0
        + df["is_senior"] * 1.0
        + df["previous_claims"] * 0.5
    )

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 12):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'premium_amount' and raw date column.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    exclude = {"premium_amount", "policy_start_date"}

    for col in df.columns:
        if col in exclude:
            continue

        dtype_name  = df[col].dtype.name
        n_unique    = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            # Text categoricals always need encoding, regardless of
            # cardinality — passthrough (bin_cols) would leak raw strings.
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols
