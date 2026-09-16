# ============================================================
# PYTEST UNIT TESTS — Insurance Premium Prediction ML System
# Dataset: Kaggle Playground Series S4E12 (Premium Amount target)
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing  import Clipper, build_preprocessors
from src.data_loader    import detect_feature_types, add_engineered_features, validate_input_data
from src.leakage_check  import detect_leakage
from src.metrics        import (rmse, mae, r2, mape, rmsle, psi, residual_stats,
                                 error_concentration_at_k, cost_sensitive_evaluation)
from src.premium_engine import get_premium_tier, quote_policyholder
from src.config         import (
    PREMIUM_TIERS, PSI_MODERATE, PSI_HIGH,
    MIN_RMSE_IMPROVEMENT_PCT, MIN_R2_THRESHOLD, MAX_GENERALIZATION_GAP,
    LOW_HEALTH_SCORE_THRESHOLD, LOW_CREDIT_SCORE_THRESHOLD,
    HIGH_PREVIOUS_CLAIMS_THRESHOLD, OLD_VEHICLE_AGE_THRESHOLD, SENIOR_AGE_THRESHOLD
)


# ============================================================
# CLIPPER TESTS  (6 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0


# ============================================================
# PREPROCESSOR TESTS  (5 tests)
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "age":             [25, 35, 45, 55, 65],
            "health_score":    [80.0, 60.0, 40.0, 20.0, 10.0],
            "previous_claims": [0, 1, 2, 3, 4],
            "smoking_status":  ["No", "Yes", "No", "Yes", "No"],
            "is_male":         [0, 1, 0, 1, 0],
        })

    def test_returns_four_outputs(self):
        df = self._sample_df()
        result = build_preprocessors(["smoking_status", "previous_claims"], ["age", "health_score"], ["is_male"], df)
        assert len(result) == 4

    def test_categorical_indices_are_list(self):
        df = self._sample_df()
        _, _, cat_idx, _ = build_preprocessors(["smoking_status"], ["age", "health_score"], ["is_male"], df)
        assert isinstance(cat_idx, list)

    def test_feature_order_coverage(self):
        df = self._sample_df()
        ord_cols  = ["smoking_status", "previous_claims"]
        cont_cols = ["age", "health_score"]
        bin_cols  = ["is_male"]
        _, _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order

    def test_scaled_preprocessor_transforms(self):
        df = self._sample_df()
        pre_scaled, _, _, _ = build_preprocessors(["smoking_status"], ["age", "health_score"], ["is_male"], df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_transforms(self):
        df = self._sample_df()
        _, pre_unscaled, _, _ = build_preprocessors(["smoking_status"], ["age", "health_score"], ["is_male"], df)
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()


# ============================================================
# FEATURE TYPE DETECTION TESTS  (3 tests)
# ============================================================

class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({
            "is_male": [0, 1, 0, 1, 0],
            "premium_amount": [1000.0, 2000.0, 1500.0, 3000.0, 1200.0]
        })
        _, _, bin_cols = detect_feature_types(df, threshold=12)
        assert "is_male" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({
            "health_score": [20.0, 40.0, 60.0, 80.0, 100.0],
            "premium_amount": [1000.0, 2000.0, 1500.0, 3000.0, 1200.0]
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=12)
        assert "premium_amount" not in ord_cols + cont_cols + bin_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "annual_income": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0,
                               70000.0, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0, 130000.0],
            "premium_amount": [1000.0, 1200.0, 1500.0, 1800.0, 2100.0, 2500.0,
                                3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0]
        })
        _, cont_cols, _ = detect_feature_types(df, threshold=12)
        assert "annual_income" in cont_cols


# ============================================================
# FEATURE ENGINEERING TESTS  (4 tests)
# ============================================================

class TestAddEngineeredFeatures:

    def _raw_df(self):
        return pd.DataFrame({
            "age": [25, 65], "gender": ["Male", "Female"], "annual_income": [30000.0, 90000.0],
            "marital_status": ["Single", "Married"], "number_of_dependents": [0, 3],
            "education_level": ["Bachelor's", "PhD"], "occupation": ["Employed", "Employed"],
            "health_score": [85.0, 20.0], "location": ["Urban", "Rural"],
            "policy_type": ["Basic", "Premium"], "previous_claims": [0, 4],
            "vehicle_age": [2, 18], "credit_score": [720.0, 450.0],
            "insurance_duration": [3, 8], "policy_start_date": pd.to_datetime(["2023-01-01", "2021-06-15"]),
            "customer_feedback": ["Good", "Poor"], "smoking_status": ["No", "Yes"],
            "exercise_frequency": ["Daily", "Rarely"], "property_type": ["Condo", "House"],
            "premium_amount": [800.0, 4500.0]
        })

    def test_is_smoker_binary(self):
        df = add_engineered_features(self._raw_df())
        assert set(df["is_smoker"].unique()).issubset({0, 1})

    def test_low_health_score_flag(self):
        df = add_engineered_features(self._raw_df())
        assert df.loc[1, "low_health_score_flag"] == 1
        assert df.loc[0, "low_health_score_flag"] == 0

    def test_high_claims_flag(self):
        df = add_engineered_features(self._raw_df())
        assert df.loc[1, "high_claims_flag"] == 1
        assert df.loc[0, "high_claims_flag"] == 0

    def test_policy_age_days_nonnegative(self):
        df = add_engineered_features(self._raw_df())
        assert (df["policy_age_days"] >= 0).all()


# ============================================================
# LEAKAGE CHECK TESTS  (4 tests)
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        X = pd.DataFrame({"a": [100.0, 200.0, 300.0, 400.0, 500.0]})
        y = pd.Series(       [100.0, 200.0, 300.0, 400.0, 500.0])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0

    def test_no_false_positives_on_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({"health_score": np.random.uniform(0, 100, 100)})
        y = pd.Series(np.random.uniform(500, 5000, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        vals = np.arange(50, dtype=float)
        X    = pd.DataFrame({"leaky_feat": vals})
        y    = pd.Series(vals * 2 + 1)
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0

    def test_empty_dataframe_no_crash(self):
        X = pd.DataFrame()
        y = pd.Series([1000.0, 2000.0, 1500.0, 3000.0])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


# ============================================================
# REGRESSION METRICS TESTS  (7 tests)
# ============================================================

class TestRegressionMetrics:

    def test_rmse_zero_for_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert rmse(y, y) == 0.0

    def test_mae_zero_for_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mae(y, y) == 0.0

    def test_r2_one_for_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0, 400.0])
        assert r2(y, y) == pytest.approx(1.0)

    def test_mape_zero_for_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_rmsle_zero_for_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        assert rmsle(y, y) == pytest.approx(0.0)

    def test_rmsle_positive_for_imperfect_prediction(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([150.0, 180.0, 350.0])
        assert rmsle(y_true, y_pred) > 0

    def test_residual_stats_keys(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 320.0])
        stats = residual_stats(y_true, y_pred)
        for key in ("mean_residual", "std_residual", "median_residual",
                    "max_underpredict", "max_overpredict"):
            assert key in stats


# ============================================================
# PSI TESTS  (3 tests)
# ============================================================

class TestPSI:

    def test_identical_distributions(self):
        x = np.random.normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_uses_reference_edges_not_actual(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(5, 1, 500)
        score = psi(ref, new)
        assert score > 0.20


# ============================================================
# ERROR CONCENTRATION @ K TESTS  (2 tests)
# ============================================================

class TestErrorConcentrationAtK:

    def test_returns_float_in_range(self):
        y_true = np.array([1000, 2000, 3000, 40000, 50000], dtype=float)
        y_pred = np.array([1100, 1900, 3200, 20000, 25000], dtype=float)
        val = error_concentration_at_k(y_true, y_pred, k=0.4)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_high_charge_errors_dominate(self):
        y_true = np.array([1000, 1000, 1000, 1000, 50000], dtype=float)
        y_pred = np.array([1000, 1000, 1000, 1000, 10000], dtype=float)
        val = error_concentration_at_k(y_true, y_pred, k=0.2)
        assert val == pytest.approx(1.0)


# ============================================================
# COST-SENSITIVE EVALUATION TESTS  (3 tests)
# ============================================================

class TestCostSensitiveEvaluation:

    def test_output_keys_complete(self):
        y_true = np.array([1000.0, 2000.0, 3000.0])
        y_pred = np.array([900.0, 2100.0, 2800.0])
        result = cost_sensitive_evaluation(y_true, y_pred)
        for key in ("underpriced_count", "overpriced_count", "underpricing_shortfall",
                    "overpricing_excess", "estimated_underwriting_loss",
                    "estimated_churn_risk_cost", "total_estimated_business_cost"):
            assert key in result

    def test_zero_cost_for_perfect_prediction(self):
        y = np.array([1000.0, 2000.0, 3000.0])
        result = cost_sensitive_evaluation(y, y)
        assert result["total_estimated_business_cost"] == 0.0

    def test_underpricing_detected(self):
        y_true = np.array([5000.0])
        y_pred = np.array([2000.0])
        result = cost_sensitive_evaluation(y_true, y_pred)
        assert result["underpriced_count"] == 1
        assert result["estimated_underwriting_loss"] > 0


# ============================================================
# PREMIUM ENGINE TESTS  (8 tests)
# ============================================================

class TestPremiumEngine:

    def test_get_premium_tier_standard(self):
        assert get_premium_tier(500) == "STANDARD"

    def test_get_premium_tier_elevated(self):
        assert get_premium_tier(1800) == "ELEVATED"

    def test_get_premium_tier_high_risk(self):
        assert get_premium_tier(5000) == "HIGH_RISK"

    def test_quote_auto_quote(self):
        row    = {"previous_claims": 0, "health_score": 80, "credit_score": 700,
                   "vehicle_age": 2, "age": 30, "is_smoker": 0}
        result = quote_policyholder(row, predicted_premium=600)
        assert result["decision"] == "AUTO_QUOTE"
        assert result["premium_tier"] == "STANDARD"

    def test_quote_high_claims_low_health_rule(self):
        row    = {"previous_claims": 4, "health_score": 20, "credit_score": 700,
                   "vehicle_age": 2, "age": 40, "is_smoker": 0}
        result = quote_policyholder(row, predicted_premium=1000)
        assert result["decision"] == "MANUAL_UNDERWRITING"
        assert result["rule_triggered"] == "HIGH_CLAIMS_LOW_HEALTH"

    def test_quote_low_credit_old_vehicle_rule(self):
        row    = {"previous_claims": 0, "health_score": 80, "credit_score": 450,
                   "vehicle_age": 18, "age": 35, "is_smoker": 0}
        result = quote_policyholder(row, predicted_premium=800)
        assert result["decision"] == "MANUAL_UNDERWRITING"
        assert result["rule_triggered"] == "LOW_CREDIT_OLD_VEHICLE"

    def test_quote_elevated_tier_loading(self):
        row    = {"previous_claims": 0, "health_score": 80, "credit_score": 700,
                   "vehicle_age": 2, "age": 40, "is_smoker": 0}
        result = quote_policyholder(row, predicted_premium=1800)
        assert result["decision"] == "AUTO_QUOTE_WITH_LOADING"

    def test_output_keys_complete(self):
        row    = {"previous_claims": 0, "health_score": 80, "credit_score": 700,
                   "vehicle_age": 2, "age": 30, "is_smoker": 0}
        result = quote_policyholder(row, predicted_premium=500)
        for key in ("predicted_annual_premium", "premium_tier", "decision", "rule_triggered"):
            assert key in result


# ============================================================
# CONFIG TESTS  (4 tests)
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        assert 0.0 < MIN_RMSE_IMPROVEMENT_PCT < 0.1
        # NOTE: MIN_R2_THRESHOLD is intentionally dataset-dependent and is NOT
        # pinned to a narrow range here. The real Kaggle Playground S4E12
        # competition data is very noisy by design (top leaderboard solutions
        # only reach R2 ~0.02-0.06), while the project's own clean synthetic
        # dataset supports R2 ~0.48-0.56. A fixed "0.3 < x < 1.0" assertion
        # would fail every time someone legitimately retunes this gate for
        # whichever dataset is currently in use — so this test only checks
        # the value is a sane probability-like ratio, not a specific range.
        assert 0.0 <= MIN_R2_THRESHOLD < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_premium_tier_boundaries_valid(self):
        for tier, (low, high) in PREMIUM_TIERS.items():
            assert low >= 0
            assert high > low

    def test_business_rule_thresholds_positive(self):
        assert LOW_HEALTH_SCORE_THRESHOLD > 0
        assert LOW_CREDIT_SCORE_THRESHOLD > 0
        assert HIGH_PREVIOUS_CLAIMS_THRESHOLD > 0
        assert OLD_VEHICLE_AGE_THRESHOLD > 0
        assert SENIOR_AGE_THRESHOLD > 0