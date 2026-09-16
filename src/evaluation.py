# ============================================================
# EVALUATION — Insurance Premium Prediction ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from typing import Dict, List, Optional, Tuple

from sklearn.model_selection   import cross_val_score, KFold
from sklearn.inspection        import permutation_importance
from sklearn.pipeline          import Pipeline

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR
from src.metrics import rmse, mae, r2, mape, rmsle, psi, residual_stats, error_concentration_at_k, cost_sensitive_evaluation

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT
# ============================================================

def safe_predict(pipe, X):
    try:
        return pipe.predict(X)
    except Exception as e:
        logger.error("predict failed: %s", e)
        return None


# ============================================================
# EVALUATE ALL MODELS — summary table
# ============================================================

def evaluate_models(
    pipelines:    Dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
) -> pd.DataFrame:

    rows = []

    for name, pipe in pipelines.items():

        logger.info("Evaluating: %s", name)

        # ── Train / test predictions ───────────────────────────
        y_pred_train = safe_predict(pipe, X_train)
        y_pred_test  = safe_predict(pipe, X_test)

        train_r2 = r2(y_train, y_pred_train) if y_pred_train is not None else None
        test_r2  = r2(y_test,  y_pred_test)  if y_pred_test  is not None else None

        test_rmse  = rmse(y_test, y_pred_test)
        test_mae   = mae(y_test,  y_pred_test)
        test_mape  = mape(y_test, y_pred_test)
        test_rmsle = rmsle(y_test, y_pred_test)

        gap = abs((train_r2 or 0) - (test_r2 or 0))

        # ── CV stability (RMSE) ─────────────────────────────────
        # NeuralNet is skipped: cross_val_score refits MLPRegressor 3x more
        # on top of the RandomizedSearchCV fits already done, and MLP
        # already carries its own internal early_stopping validation split
        # — the extra refits added significant time/memory for little
        # additional signal and were the direct cause of an OOM crash here.
        if name == "NeuralNet":
            cv_mean_rmse, cv_std_rmse = None, None
        else:
            try:
                cv_scores = cross_val_score(
                    pipe, X_train, y_train,
                    scoring = "neg_root_mean_squared_error",
                    cv      = KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs  = N_JOBS
                )
                cv_mean_rmse = float(-cv_scores.mean())
                cv_std_rmse  = float(cv_scores.std())
            except Exception:
                cv_mean_rmse = None
                cv_std_rmse = None

        rows.append({
            "model":          name,
            "train_r2":       round(train_r2, 4) if train_r2 is not None else None,
            "test_r2":        round(test_r2,  4) if test_r2  is not None else None,
            "train_test_gap": round(gap, 4),
            "cv_mean_rmse":   round(cv_mean_rmse, 4) if cv_mean_rmse is not None else None,
            "cv_std_rmse":    round(cv_std_rmse,  4) if cv_std_rmse  is not None else None,
            "test_rmse":      round(test_rmse, 4),
            "test_mae":       round(test_mae,  4),
            "test_mape":      round(test_mape, 4),
            "test_rmsle":     round(test_rmsle, 4),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["test_rmse"], ascending=True)
        .reset_index(drop=True)
    )

    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    return summary


# ============================================================
# SELECT BEST MODEL — with generalization filter
# ============================================================

def select_best_model(
    summary:      pd.DataFrame,
    pipelines:    Dict,
    scaled_pipes: Dict,
    unscaled_pipes: Dict
) -> Tuple[str, object]:

    def _filter(df, gen_gap, cv_std_ratio, min_r2):
        # cv_std_ratio expressed relative to cv_mean_rmse to stay scale-free
        mask = (
            df["train_test_gap"].notna() &
            df["cv_std_rmse"].notna() &
            df["cv_mean_rmse"].notna() &
            (df["train_test_gap"] <= gen_gap) &
            ((df["cv_std_rmse"] / (df["cv_mean_rmse"] + 1e-9)) <= cv_std_ratio) &
            (df["test_r2"] >= min_r2)
        )
        return df[mask].copy()

    thresholds = [
        (0.05, 0.15, 0.85),
        (0.08, 0.25, 0.80),
        (0.12, 0.35, 0.70),
        (0.30, 1.00, 0.00),
    ]

    candidates = pd.DataFrame()
    for gg, cvs, mr2 in thresholds:
        candidates = _filter(summary, gg, cvs, mr2)
        if not candidates.empty:
            logger.info("Candidates found with gen_gap<=%.2f cv_std_ratio<=%.2f min_r2>=%.2f", gg, cvs, mr2)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        candidates    = candidates.sort_values(
            ["test_rmse", "test_r2"], ascending=[True, False]
        ).reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    selected_pipe = scaled_pipes.get(selected_name) or unscaled_pipes.get(selected_name)
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# PREDICTION INTERVALS  (residual-quantile — replaces probability
# calibration, which does not apply to regression targets)
# ============================================================

def compute_prediction_intervals(selected_pipe, X_cal, y_cal, confidence: float = 0.90) -> dict:
    """
    Holdout-based residual-quantile prediction interval.
    Uses a calibration split (no leakage from train/test) to estimate
    a symmetric error margin such that ~confidence% of true charges
    fall within [pred - margin, pred + margin].
    """
    try:
        y_pred_cal = selected_pipe.predict(X_cal)
        residuals  = np.abs(np.asarray(y_cal) - y_pred_cal)
        margin     = float(np.quantile(residuals, confidence))
        logger.info("Prediction interval  |  confidence=%.2f  margin=$%.2f", confidence, margin)
        return {"confidence": confidence, "margin": round(margin, 2)}
    except Exception as e:
        logger.exception("Prediction interval computation failed: %s", e)
        return {"confidence": confidence, "margin": None}


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(selected_pipe, X_train, y_train, top_k: int = 20):

    reg   = selected_pipe.named_steps["regressor"]
    names = _get_feature_names(selected_pipe, X_train)

    if hasattr(reg, "feature_importances_"):
        imp   = np.asarray(reg.feature_importances_)
        names = names if names and len(names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        fi    = pd.Series(imp, index=names).sort_values(ascending=False).head(top_k)
        return fi

    if hasattr(reg, "coef_"):
        coef  = np.abs(np.ravel(reg.coef_))
        names = names if names and len(names) == len(coef) else [f"f{i}" for i in range(len(coef))]
        fi    = pd.Series(coef, index=names).sort_values(ascending=False).head(top_k)
        return fi

    logger.info("Using permutation importance (slow fallback)...")
    try:
        r_imp = permutation_importance(
            selected_pipe, X_train, y_train,
            n_repeats=10, scoring="r2", n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        idx = np.argsort(r_imp.importances_mean)[::-1][:top_k]
        fi  = pd.Series(r_imp.importances_mean[idx],
                        index=[f"f{i}" for i in idx])
        return fi
    except Exception as e:
        logger.exception("Permutation importance failed: %s", e)
        return None


def _get_feature_names(pipe, X_sample):
    """
    Extracts clean feature names after preprocessor + selector.
    Strips sklearn prefixes like 'ord__', 'skewed__', 'bin__' etc.
    """
    try:
        pre       = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())

        clean_names = []
        for n in raw_names:
            if "__" in n:
                clean_names.append(n.split("__", 1)[1])
            else:
                clean_names.append(n)

        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask        = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]

        return clean_names

    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(selected_pipe, X_train, X_explain, max_explain: int = 200):

    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    reg  = selected_pipe.named_steps["regressor"]
    pre  = selected_pipe.named_steps["preprocessor"]
    sel  = selected_pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train)
        X_tr  = sel.transform(X_pre) if sel is not None else X_pre
        X_ex  = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex  = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(selected_pipe, X_train)

        tree_types = ("RandomForestRegressor", "ExtraTreesRegressor",
                      "GradientBoostingRegressor", "XGBRegressor",
                      "LGBMRegressor", "CatBoostRegressor", "DecisionTreeRegressor",
                      "AdaBoostRegressor")

        if type(reg).__name__ in tree_types:
            explainer = shap.TreeExplainer(reg)
            try:
                vals = np.array(explainer(X_ex).values)
            except Exception:
                vals = np.array(explainer.shap_values(X_ex))

        else:
            bg_sample = shap.sample(X_tr, min(200, len(X_tr)))
            explainer = shap.KernelExplainer(reg.predict, bg_sample)
            vals      = np.array(explainer.shap_values(X_ex, nsamples=100))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        mean_abs   = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]

        fi_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        try:
            shap.summary_plot(vals, features=X_ex, feature_names=feat_names, show=False)
        except Exception:
            pass

        logger.info("SHAP done  |  explainer=%s", type(explainer).__name__)

        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


# ============================================================
# SAVE MODEL
# ============================================================

def save_challenger_artifact(
    selected_name:   str,
    pipe,
    model_card:      dict,
    version:         str = "v1",
    model_card_path: str = ""
):
    """
    Saves the trained CHALLENGER pipeline as .joblib — does NOT touch the
    registry (latest_model.json). The registry is only updated by
    run_challenger_comparison() in model_loader.py, and only when the
    challenger genuinely beats the current champion on all gates.
    """
    model_path = os.path.join(MODEL_DIR, f"premium_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    logger.info("Challenger model saved → %s (registry untouched until gates pass)", model_path)

    return model_path


def save_model_and_card(
    selected_name:   str,
    pipe,
    model_card:      dict,
    version:         str = "v1",
    model_card_path: str = ""
):
    """
    DEPRECATED for use inside the training pipeline's main flow — kept only
    for direct/manual model promotion (e.g. first-time setup scripts).
    """
    model_path = os.path.join(MODEL_DIR, f"premium_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    registry = {
        "model_name":      os.path.basename(model_path),
        "model_card_path": model_card_path or ""
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Model saved   → %s", model_path)
    logger.info("Registry      → premium_models/latest_model.json")

    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:      str,
    selected_name: str,
    pipe,
    model_card:    dict,
    X_train_sample: "pd.DataFrame" = None
):
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        tracking_dir = os.path.abspath("mlruns")
        os.makedirs(tracking_dir, exist_ok=True)
        db_path = os.path.join(tracking_dir, "mlflow.db").replace(os.sep, "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment("insurance_premium_experiments")

        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):

            mlflow.log_param("model_name", selected_name)
            mlflow.log_param("selector_k", model_card.get("pipeline_config", {}).get("selector_k", "?"))

            mlflow.log_metric("test_rmse", float(metrics_dict.get("test_rmse", 0)))
            mlflow.log_metric("test_mae",  float(metrics_dict.get("test_mae",  0)))
            mlflow.log_metric("test_r2",   float(metrics_dict.get("test_r2",   0)))
            mlflow.log_metric("test_mape", float(metrics_dict.get("test_mape", 0)))

            try:
                log_kwargs = {
                    "sk_model":         pipe,
                    "name":             "model",
                    "pip_requirements": [
                        "scikit-learn==1.3.2",
                        "lightgbm",
                        "xgboost",
                        "pandas",
                        "numpy",
                    ],
                    # MLflow 3.x scans pickled sklearn pipelines for "untrusted"
                    # types before saving. Our pipeline contains a custom
                    # transformer (src.preprocessing.Clipper) and
                    # sklearn's mutual_info_regression callable, both of
                    # which are safe (part of this codebase / sklearn itself)
                    # so we explicitly trust them to avoid a silent log_model
                    # failure.
                    "skops_trusted_types": [
                        "src.preprocessing.Clipper",
                        "sklearn.feature_selection._univariate_selection.f_regression",
                    ],
                }
                if X_train_sample is not None:
                    log_kwargs["input_example"] = X_train_sample.iloc[:5].astype(float, errors="ignore")

                mlflow.sklearn.log_model(**log_kwargs)

            except Exception as e:
                logger.warning("mlflow log_model failed: %s", e)

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.exception("MLflow logging failed: %s", e)
