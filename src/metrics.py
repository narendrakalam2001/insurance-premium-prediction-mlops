# ============================================================
# METRICS — Insurance Premium Prediction ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


# ============================================================
# CORE REGRESSION METRICS
# ============================================================

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def rmsle(y_true, y_pred) -> float:
    """
    Root Mean Squared Log Error — the official evaluation metric for the
    Kaggle Playground Series S4E12 competition this dataset is based on.
    Penalizes under-prediction of large premiums less harshly than plain
    RMSE, and is undefined for negative predictions, so predictions are
    floored at 0 before applying log1p.
    """
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error. Guards against division by ~0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def adjusted_r2(r2_value: float, n: int, p: int) -> float:
    """Adjusted R² — penalizes extra predictors. n=rows, p=n_features."""
    if n - p - 1 <= 0:
        return r2_value
    return float(1 - (1 - r2_value) * (n - 1) / (n - p - 1))


# ============================================================
# PSI — Population Stability Index (unchanged — distribution-agnostic)
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → no significant shift (stable)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH `expected` and `actual` using those SAME edges
      3. Compare bin proportions
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)

        bin_edges  = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# RESIDUAL DIAGNOSTICS
# ============================================================

def residual_stats(y_true, y_pred) -> dict:
    """
    Summary statistics of (y_true - y_pred) residuals.
    Useful for checking heteroscedasticity / bias.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "mean_residual":   float(np.mean(resid)),
        "std_residual":    float(np.std(resid)),
        "median_residual": float(np.median(resid)),
        "max_underpredict": float(resid.max()),   # y_true >> y_pred
        "max_overpredict":  float(-resid.min()),  # y_pred >> y_true
    }


# ============================================================
# TOP-K ERROR CONCENTRATION (analogue of recall@k / lift@k)
# ============================================================

def error_concentration_at_k(y_true, y_pred, k: float = 0.10) -> float:
    """
    Fraction of total absolute error contributed by the top-k%
    highest-charge policyholders. High concentration → model
    under-serves the most expensive (high-risk) segment, which
    matters most for actuarial solvency.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred)})
    df["abs_err"] = (df["y"] - df["p"]).abs()
    df = df.sort_values("y", ascending=False)
    top_n = max(1, int(len(df) * k))
    top_err   = df.iloc[:top_n]["abs_err"].sum()
    total_err = df["abs_err"].sum() + 1e-9
    return float(top_err / total_err)


# ============================================================
# COST-SENSITIVE EVALUATION (actuarial-grade)
# ============================================================

def cost_sensitive_evaluation(
    y_true,
    y_pred,
    underpricing_loss_mult: float = 1.0,
    overpricing_churn_cost: float = 0.15,
) -> dict:
    """
    Actuarial cost framing of prediction errors:

      UNDER-PRICING (predicted premium < actual claim cost):
          insurer collects less premium than the risk actually costs
          → direct underwriting loss = sum(shortfall) * underpricing_loss_mult

      OVER-PRICING (predicted premium > actual claim cost):
          policyholder is quoted more than their risk warrants
          → churn / competitiveness risk, modeled as a fraction of
            the excess quoted amount = sum(excess) * overpricing_churn_cost
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    under_mask = y_pred < y_true
    over_mask  = y_pred > y_true

    underpricing_shortfall = float((y_true[under_mask] - y_pred[under_mask]).sum())
    overpricing_excess     = float((y_pred[over_mask]  - y_true[over_mask]).sum())

    underwriting_loss = underpricing_shortfall * underpricing_loss_mult
    churn_risk_cost    = overpricing_excess * overpricing_churn_cost

    total_business_cost = underwriting_loss + churn_risk_cost

    result = {
        "underpriced_count":       int(under_mask.sum()),
        "overpriced_count":        int(over_mask.sum()),
        "underpricing_shortfall":  round(underpricing_shortfall, 2),
        "overpricing_excess":      round(overpricing_excess, 2),
        "estimated_underwriting_loss": round(underwriting_loss, 2),
        "estimated_churn_risk_cost":   round(churn_risk_cost, 2),
        "total_estimated_business_cost": round(total_business_cost, 2),
    }

    logger.info(
        "Cost eval  |  underpriced=%d  overpriced=%d  underwriting_loss=%.2f  churn_cost=%.2f  total=%.2f",
        result["underpriced_count"], result["overpriced_count"],
        result["estimated_underwriting_loss"], result["estimated_churn_risk_cost"],
        result["total_estimated_business_cost"]
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(X_ref: "pd.DataFrame", X_new: "pd.DataFrame", top_n: int = 10):
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel   = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)
