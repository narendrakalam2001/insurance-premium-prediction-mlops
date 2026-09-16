# ============================================================
# TRAINING PIPELINE — Insurance Premium Prediction ML System
# Dataset: Kaggle Playground Series S4E12 (Premium Amount target)
# ============================================================

import os
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
import matplotlib.pyplot as plt
import seaborn as sns

from src.config         import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR, SELECT_K, MAX_TUNING_SAMPLE_SIZE
from src.data_loader    import validate_input_data, add_engineered_features, detect_feature_types
from src.preprocessing  import build_preprocessors, safe_k
from src.model_tuning   import scaled_models, unscaled_models, tune_models, train_mlp_pipeline
from src.evaluation     import (evaluate_models, select_best_model, compute_prediction_intervals,
                                 compute_feature_importance, compute_shap,
                                 save_challenger_artifact,
                                 mlflow_log_run, safe_predict)
from src.leakage_check  import detect_leakage
from src.model_card     import build_model_card, save_model_card
from src.model_loader   import run_challenger_comparison
from src.metrics        import rmse, mae, r2, mape, rmsle, psi, cost_sensitive_evaluation
from src.premium_engine import premium_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default dataset location — override with env var INSURANCE_DATA_PATH
DEFAULT_DATA_PATH = r"D:\Data Science Datasets\playground-series-s4e12\train.csv"


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training():

    # ─────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    DATA_PATH = os.getenv("INSURANCE_DATA_PATH", DEFAULT_DATA_PATH)

    logger.info("Loading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = validate_input_data(df)

    # ─────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(df)
    df = df.drop(columns=["policy_start_date"])  # raw date dropped after feature extraction

    # ─────────────────────────────────────────────────────────
    # 3. FEATURE TYPE DETECTION
    # ─────────────────────────────────────────────────────────
    ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=12)

    X = df.drop(columns=["premium_amount"])
    y = df["premium_amount"]

    logger.info("Dataset  |  shape=%s  |  mean_premium=%.2f", df.shape, y.mean())

    # ─────────────────────────────────────────────────────────
    # 4. TRAIN / TEST / CALIBRATION SPLIT
    # ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    logger.info("Split  |  train_fit=%d  cal=%d  test=%d",
                len(X_train_fit), len(X_cal), len(X_test))

    # ── Subsample the tuning set for RandomizedSearchCV ────────
    # 128K rows x up to 200 trees x 3-fold CV x 10 search candidates x
    # multiple models was too memory-heavy for this environment (silent
    # OOM kill on RandomForest). 40K rows is still a large, statistically
    # stable sample for hyperparameter selection. Final evaluation always
    # uses the full, untouched X_test/y_test — this only affects what the
    # models are FIT on, not what they're SCORED on.
    if len(X_train_fit) > MAX_TUNING_SAMPLE_SIZE:
        tune_idx = X_train_fit.sample(n=MAX_TUNING_SAMPLE_SIZE, random_state=RANDOM_STATE).index
        X_tune, y_tune = X_train_fit.loc[tune_idx], y_train_fit.loc[tune_idx]
        logger.info("Tuning sample capped: %d -> %d rows", len(X_train_fit), len(X_tune))
    else:
        X_tune, y_tune = X_train_fit, y_train_fit

    # ─────────────────────────────────────────────────────────
    # 5. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    leak_warnings = detect_leakage(X_train_fit, y_train_fit)
    if leak_warnings:
        for w in leak_warnings:
            logger.warning("LEAKAGE: %s", w)
    else:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 6. BUILD PREPROCESSORS
    # ─────────────────────────────────────────────────────────
    pre_scaled, pre_unscaled, cat_indices, feature_order = build_preprocessors(
        ord_cols, cont_cols, bin_cols, X_train_fit
    )

    k_safe = safe_k(SELECT_K, pre_scaled, X_train_fit)

    # ─────────────────────────────────────────────────────────
    # 7. TUNE SCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning scaled models ...")
    scaled_pipelines = tune_models(
        scaled_models, pre_scaled, cat_indices,
        X_tune, y_tune, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 8. TUNE UNSCALED MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning unscaled models ...")
    unscaled_pipelines = tune_models(
        unscaled_models, pre_unscaled, cat_indices,
        X_tune, y_tune, selector_k=k_safe
    )

    # ─────────────────────────────────────────────────────────
    # 9. TRAIN NEURAL NETWORK (separately)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    mlp_pipe = train_mlp_pipeline(X_tune, y_tune, pre_scaled, cat_indices)

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines, "NeuralNet": mlp_pipe}

    # ─────────────────────────────────────────────────────────
    # 10. EVALUATE ALL MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating all models ...")
    summary = evaluate_models(all_pipelines, X_tune, y_tune, X_test, y_test)

    print("\n" + "=" * 60)
    print("ALL MODELS SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # ─────────────────────────────────────────────────────────
    # 11. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, scaled_pipelines, unscaled_pipelines
    )

    # ─────────────────────────────────────────────────────────
    # 12. DETAILED EVALUATION — SELECTED MODEL
    # ─────────────────────────────────────────────────────────
    y_pred_sel = safe_predict(selected_pipe, X_test)

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {selected_name}")
    print("=" * 60)
    print(f"RMSE = {rmse(y_test, y_pred_sel):.2f}   MAE = {mae(y_test, y_pred_sel):.2f}   "
          f"R2 = {r2(y_test, y_pred_sel):.4f}   MAPE = {mape(y_test, y_pred_sel):.2f}%   "
          f"RMSLE = {rmsle(y_test, y_pred_sel):.4f}")

    os.makedirs(os.path.join("docs", "plots"), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred_sel, alpha=0.3, s=8, color="steelblue")
    lims = [min(y_test.min(), y_pred_sel.min()), max(y_test.max(), y_pred_sel.max())]
    plt.plot(lims, lims, "r--", lw=1)
    plt.xlabel("Actual Premium ($)")
    plt.ylabel("Predicted Premium ($)")
    plt.title(f"{selected_name} — Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "predicted_vs_actual.png"))
    plt.close()

    residuals = y_test.values - y_pred_sel
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(y_pred_sel, residuals, alpha=0.3, s=8, color="darkorange")
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xlabel("Predicted Premium ($)")
    ax[0].set_ylabel("Residual")
    ax[0].set_title("Residuals vs Predicted")
    sns.histplot(residuals, bins=60, kde=True, ax=ax[1], color="seagreen")
    ax[1].set_title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join("docs", "plots", "residual_analysis.png"))
    plt.close()

    # ─────────────────────────────────────────────────────────
    # 13. PREMIUM ENGINE DECISIONS
    # ─────────────────────────────────────────────────────────
    decisions = premium_engine(X_test, y_pred_sel)
    decision_counts = pd.Series(decisions).value_counts()

    print("\nPREMIUM ENGINE DECISIONS")
    print(decision_counts)

    # ─────────────────────────────────────────────────────────
    # 14. COST-SENSITIVE EVALUATION
    # ─────────────────────────────────────────────────────────
    cost_result = cost_sensitive_evaluation(y_test, y_pred_sel)
    print("\nCOST EVALUATION")
    for k, v in cost_result.items():
        print(f"  {k}: {v}")

    # ─────────────────────────────────────────────────────────
    # 15. PREDICTION INTERVAL (regression analogue of calibration)
    # ─────────────────────────────────────────────────────────
    pred_interval = compute_prediction_intervals(selected_pipe, X_cal, y_cal, confidence=0.90)

    # ─────────────────────────────────────────────────────────
    # 16. REPEATED CV STABILITY CHECK (subsampled — dataset is large)
    # ─────────────────────────────────────────────────────────
    try:
        cv_sample_idx = X_train_fit.sample(n=min(20000, len(X_train_fit)), random_state=RANDOM_STATE).index
        rkf        = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        rep_scores = cross_val_score(
            selected_pipe, X_train_fit.loc[cv_sample_idx], y_train_fit.loc[cv_sample_idx],
            scoring = "neg_root_mean_squared_error",
            cv      = rkf,
            n_jobs  = 1
        )
        logger.info("Repeated CV (subsampled 20k)  |  mean_rmse=%.4f  std=%.4f",
                    -rep_scores.mean(), rep_scores.std())
    except Exception as e:
        logger.warning("Repeated CV failed: %s", e)

    # ─────────────────────────────────────────────────────────
    # 17. FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────
    fi = compute_feature_importance(selected_pipe, X_train_fit, y_train_fit)
    if fi is not None:
        print("\nTOP FEATURE IMPORTANCES")
        print(fi.head(10))

    # ─────────────────────────────────────────────────────────
    # 18. SHAP EXPLAINABILITY (sampled — large dataset)
    # ─────────────────────────────────────────────────────────
    shap_result = compute_shap(selected_pipe, X_train_fit.sample(n=min(5000, len(X_train_fit)), random_state=RANDOM_STATE), X_test.head(200))

    # ─────────────────────────────────────────────────────────
    # 19. PSI — FEATURE DRIFT
    # ─────────────────────────────────────────────────────────
    psi_scores = {}
    for col in cont_cols:
        if col in X_train_fit.columns and col in X_test.columns:
            psi_scores[col] = psi(X_train_fit[col].values, X_test[col].values)

    psi_df = pd.Series(psi_scores).sort_values(ascending=False)
    psi_df.reset_index().rename(
        columns={"index": "feature", 0: "drift_score"}
    ).to_csv(os.path.join(MODEL_DIR, "feature_drift_report.csv"), index=False)

    print("\nTOP PSI (train vs test)")
    print(psi_df.head(10))

    # ─────────────────────────────────────────────────────────
    # 20. SAVE MONITOR SCORES
    # ─────────────────────────────────────────────────────────
    final_preds     = y_pred_sel
    final_decisions = premium_engine(X_test, final_preds)

    monitor_df = pd.DataFrame({
        "predicted_premium": final_preds,
        "actual_premium":    y_test.values,
        "decision":          final_decisions,
        "premium_tier":      [d if d in ("STANDARD", "ELEVATED", "HIGH_RISK") else "REVIEW"
                               for d in final_decisions],
    })
    monitor_df.to_csv(os.path.join(MODEL_DIR, "monitor_scores.csv"), index=False)
    logger.info("Monitor scores saved")

    # ─────────────────────────────────────────────────────────
    # 21. MODEL CARD
    # ─────────────────────────────────────────────────────────
    model_card = build_model_card(
        selected_name       = selected_name,
        train_fit_size      = int(len(X_train_fit)),
        cal_size            = int(len(X_cal)),
        test_size           = int(len(X_test)),
        mean_charges_train  = float(y_train_fit.mean()),
        metrics = {
            "test_rmse":  float(rmse(y_test, y_pred_sel)),
            "test_mae":   float(mae(y_test,  y_pred_sel)),
            "test_r2":    float(r2(y_test,   y_pred_sel)),
            "test_mape":  float(mape(y_test, y_pred_sel)),
            "test_rmsle": float(rmsle(y_test, y_pred_sel)),
        },
        prediction_interval = pred_interval,
        cost_result         = cost_result,
        decision_counts     = decision_counts.to_dict(),
        feature_order       = feature_order,
        cat_indices         = cat_indices,
        selector_k          = k_safe,
        fi_dict             = fi.head(20).to_dict() if fi is not None else None,
        shap_dict           = shap_result.get("shap_top", {}) if shap_result is not None else None,
    )

    import time as _time
    _run_ts   = _time.strftime("%Y%m%d_%H%M%S")
    card_path = save_model_card(model_card, MODEL_DIR, selected_name, version=f"v1_{_run_ts}")

    # ─────────────────────────────────────────────────────────
    # 22. SAVE MODEL
    # ─────────────────────────────────────────────────────────
    model_path = save_challenger_artifact(selected_name, selected_pipe, model_card, model_card_path=card_path)

    # ─────────────────────────────────────────────────────────
    # 23. CHALLENGER MODEL COMPARISON
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running Champion vs Challenger comparison ...")

    train_r2_sel = r2(y_tune, safe_predict(selected_pipe, X_tune))
    test_r2_sel  = r2(y_test, y_pred_sel)

    challenger_result = run_challenger_comparison(
        challenger_name       = selected_name,
        challenger_rmse       = float(rmse(y_test, y_pred_sel)),
        challenger_r2         = float(test_r2_sel),
        challenger_gap        = float(abs(train_r2_sel - test_r2_sel)),
        challenger_model_path = model_path,
        challenger_margin     = float(pred_interval.get("margin") or 0.0),
        challenger_card_path  = card_path,
    )

    print("\n" + "=" * 60)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"Reason: {challenger_result['reason']}")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────
    # 24. MLFLOW
    # ─────────────────────────────────────────────────────────
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{selected_name}"
    mlflow_log_run(run_name, selected_name, selected_pipe, model_card, X_train_sample=X_train_fit)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE  |  Best model: {selected_name}")
    print(f"Challenger status : {challenger_result['decision']}")
    print("=" * 60)

    return selected_name, selected_pipe, model_card


if __name__ == "__main__":
    start = time.time()
    name, model, card = run_training()
    logger.info("Finished in %.1fs", time.time() - start)
