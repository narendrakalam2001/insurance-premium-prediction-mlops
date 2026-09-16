# ============================================================
# PREMIUM ENGINE — Insurance Premium Prediction ML System
# ============================================================
# Actuarial-grade 3-tier pricing decision system:
#   STANDARD   → predicted premium within normal band, auto-quote
#   ELEVATED   → predicted premium moderately high, flagged for review
#   HIGH_RISK  → predicted premium very high, requires underwriter sign-off
#
# Decision hierarchy:
#   1. Hard business rules (override / annotate ML prediction)
#   2. ML predicted premium + premium tier bands
# ============================================================

import pandas as pd
import logging

from src.config import (
    PREMIUM_TIERS, LOW_HEALTH_SCORE_THRESHOLD, LOW_CREDIT_SCORE_THRESHOLD,
    HIGH_PREVIOUS_CLAIMS_THRESHOLD, OLD_VEHICLE_AGE_THRESHOLD, SENIOR_AGE_THRESHOLD
)

logger = logging.getLogger(__name__)


# ============================================================
# PREMIUM TIER — predicted premium → STANDARD / ELEVATED / HIGH_RISK
# ============================================================

def get_premium_tier(predicted_premium: float) -> str:
    for tier, (low, high) in PREMIUM_TIERS.items():
        if low <= predicted_premium < high:
            return tier
    return "HIGH_RISK"


# ============================================================
# PREMIUM ENGINE — row-level decisions (used during training eval)
# ============================================================

def premium_engine(policy_df: pd.DataFrame, predicted_premiums) -> list:
    """
    For each policyholder row → returns decision string.

    Rule priority:
      1. 3+ previous claims + low health score  → HIGH_RISK_RULE (compounding risk)
      2. Low credit score + old vehicle          → REVIEW_CREDIT_VEHICLE
      3. Senior + smoker                          → REVIEW_SENIOR_SMOKER
      4. predicted premium → tier band            → ML-driven tier
    """
    decisions = []

    for idx, (_, row) in enumerate(policy_df.iterrows()):

        premium = predicted_premiums[idx]

        previous_claims = row.get("previous_claims", 0)
        health_score    = row.get("health_score", 100)
        credit_score    = row.get("credit_score", 700)
        vehicle_age     = row.get("vehicle_age", 0)
        age             = row.get("age", 30)
        is_smoker       = row.get("is_smoker", 1 if row.get("smoking_status") == "Yes" else 0)

        # ── Hard rule: many prior claims + poor health ────────
        if previous_claims >= HIGH_PREVIOUS_CLAIMS_THRESHOLD and health_score < LOW_HEALTH_SCORE_THRESHOLD:
            decisions.append("HIGH_RISK_RULE")
            continue

        # ── Soft rule: low credit + old vehicle ────────────────
        if credit_score < LOW_CREDIT_SCORE_THRESHOLD and vehicle_age >= OLD_VEHICLE_AGE_THRESHOLD:
            decisions.append("REVIEW_CREDIT_VEHICLE")
            continue

        # ── Soft rule: senior smoker ───────────────────────────
        if age >= SENIOR_AGE_THRESHOLD and is_smoker:
            decisions.append("REVIEW_SENIOR_SMOKER")
            continue

        # ── ML-driven tier ──────────────────────────────────────
        decisions.append(get_premium_tier(premium))

    return decisions


# ============================================================
# PREMIUM SCORING — single policyholder (for API)
# ============================================================

def quote_policyholder(row: dict, predicted_premium: float) -> dict:
    """
    Returns structured premium quote output for a single policyholder.
    Used by FastAPI prediction endpoint.
    """
    tier = get_premium_tier(predicted_premium)

    rule_triggered = None

    previous_claims = row.get("previous_claims", 0)
    health_score    = row.get("health_score", 100)
    credit_score    = row.get("credit_score", 700)
    vehicle_age     = row.get("vehicle_age", 0)
    age             = row.get("age", 30)
    is_smoker       = row.get("is_smoker", 0)

    if previous_claims >= HIGH_PREVIOUS_CLAIMS_THRESHOLD and health_score < LOW_HEALTH_SCORE_THRESHOLD:
        decision       = "MANUAL_UNDERWRITING"
        rule_triggered = "HIGH_CLAIMS_LOW_HEALTH"

    elif credit_score < LOW_CREDIT_SCORE_THRESHOLD and vehicle_age >= OLD_VEHICLE_AGE_THRESHOLD:
        decision       = "MANUAL_UNDERWRITING"
        rule_triggered = "LOW_CREDIT_OLD_VEHICLE"

    elif age >= SENIOR_AGE_THRESHOLD and is_smoker:
        decision       = "MANUAL_UNDERWRITING"
        rule_triggered = "SENIOR_SMOKER"

    elif tier == "HIGH_RISK":
        decision = "MANUAL_UNDERWRITING"

    elif tier == "ELEVATED":
        decision = "AUTO_QUOTE_WITH_LOADING"

    else:
        decision = "AUTO_QUOTE"

    return {
        "predicted_annual_premium": round(float(predicted_premium), 2),
        "premium_tier":             tier,
        "decision":                 decision,
        "rule_triggered":           rule_triggered,
    }
