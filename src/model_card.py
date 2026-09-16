# ============================================================
# MODEL CARD — Insurance Premium Prediction ML System
# ============================================================
# Builds and saves a structured model card (Google Model Cards style).
#
# Contains:
#   - Model metadata (name, version, date)
#   - Dataset splits info
#   - All regression evaluation metrics
#   - Premium tier decision summary
#   - Cost evaluation result
#   - Feature importances / SHAP top features
#   - Prediction interval (replaces classification threshold/calibration)
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# BUILD MODEL CARD
# ============================================================

def build_model_card(
    selected_name:    str,
    train_fit_size:   int,
    cal_size:         int,
    test_size:        int,
    mean_charges_train: float,
    metrics:          dict,          # test_rmse, test_mae, test_r2, test_mape
    prediction_interval: dict,
    cost_result:      dict,
    decision_counts:  dict,
    feature_order:    list,
    cat_indices:      list,
    selector_k:       int,
    version:          str = "v1",
    fi_dict:          Optional[dict] = None,
    shap_dict:        Optional[dict] = None,
) -> dict:
    """
    Assembles the full model card dictionary.
    """

    card = {

        # ── Identity ──────────────────────────────────────────
        "model_version":    version,
        "model_name":       selected_name,
        "trained_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":          "Insurance Premium Prediction System",

        # ── Dataset info ──────────────────────────────────────
        "dataset": {
            "train_fit_size":     train_fit_size,
            "calibration_size":   cal_size,
            "test_size":          test_size,
            "mean_charges_train": round(float(mean_charges_train), 2),
        },

        # ── Evaluation metrics ────────────────────────────────
        "metrics": {
            "test_rmse":  round(float(metrics.get("test_rmse", 0)), 4),
            "test_mae":   round(float(metrics.get("test_mae",  0)), 4),
            "test_r2":    round(float(metrics.get("test_r2",   0)), 4),
            "test_mape":  round(float(metrics.get("test_mape", 0)), 4),
            "test_rmsle": round(float(metrics.get("test_rmsle", 0)), 4),
        },

        # ── Prediction interval (regression analogue of threshold) ─
        "prediction_interval": prediction_interval,

        # ── Premium tier decisions ─────────────────────────────
        "premium_decisions": decision_counts,

        # ── Cost evaluation ───────────────────────────────────
        "cost_evaluation": cost_result,

        # ── Pipeline config ───────────────────────────────────
        "pipeline_config": {
            "feature_order":       feature_order,
            "categorical_indices": cat_indices,
            "selector_k":          selector_k,
        },
    }

    if fi_dict is not None:
        card["feature_importances"] = fi_dict

    if shap_dict is not None:
        card["shap_top_features"] = shap_dict

    return card


# ============================================================
# SAVE MODEL CARD
# ============================================================

def save_model_card(card: dict, model_dir: str, selected_name: str, version: str = "v1") -> str:
    os.makedirs(model_dir, exist_ok=True)

    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")

    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    logger.info("Model card saved → %s", card_path)

    return card_path


# ============================================================
# LOAD MODEL CARD
# ============================================================

def load_model_card(model_dir: str, selected_name: str, version: str = "v1") -> dict:
    card_path = os.path.join(model_dir, f"model_card_{selected_name}_{version}.json")

    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Model card not found: {card_path}")

    with open(card_path) as f:
        card = json.load(f)

    logger.info("Model card loaded ← %s", card_path)

    return card
