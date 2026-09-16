# ============================================================
# PREDICTION SERVICE — Insurance Premium Prediction ML System
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.premium_engine import quote_policyholder
from src.data_loader    import add_engineered_features

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict → returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering.
    """
    df = pd.DataFrame([input_data])

    # ── Normalize keys to snake_case (mirrors validate_input_data) ─
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
        .str.strip("_")
    )

    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"], errors="coerce")

    df = add_engineered_features(df)
    df = df.drop(columns=["policy_start_date"])
    return df


# ============================================================
# PREDICT — single policyholder
# ============================================================

def predict_policyholder(model, input_data: dict, margin: float = 0.0) -> dict:
    """
    Full prediction flow for one policyholder:
      1. Feature engineering
      2. Model predicted annual premium
      3. Premium engine scoring (rules + ML tiering)

    Returns structured premium quote dict.
    """
    df = prepare_features(input_data)

    try:
        premium = float(model.predict(df)[0])
        premium = max(premium, 0.0)
    except Exception as e:
        logger.error("predict failed: %s", e)
        premium = 0.0

    row = df.iloc[0].to_dict()

    result = quote_policyholder(row, premium)
    result["prediction_interval_90pct"] = {
        "low":  round(max(0.0, premium - margin), 2),
        "high": round(premium + margin, 2),
    }

    logger.info(
        "Prediction  |  premium=%.2f  tier=%s  decision=%s",
        result["predicted_annual_premium"],
        result["premium_tier"],
        result["decision"]
    )

    return result
