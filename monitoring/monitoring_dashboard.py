# ============================================================
# MONITORING DASHBOARD — Insurance Premium Prediction ML System
# ============================================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from datetime import date, timedelta

st.set_page_config(page_title="Insurance Premium Dashboard", layout="wide")
st.title("🏥 Insurance Premium Monitoring Dashboard")

API_URL = os.getenv(
    "INSURANCE_API_URL",
    "http://127.0.0.1:8000"
) + "/predict"

PSI_MODERATE = 0.10
PSI_HIGH     = 0.20

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(_SCRIPT_DIR)
except Exception:
    BASE_DIR = os.getcwd()

MONITOR_PATH    = os.path.join(BASE_DIR, "premium_models", "monitor_scores.csv")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "prediction_logs.csv")
PSI_PATH        = os.path.join(BASE_DIR, "premium_models", "feature_drift_report.csv")
CHALLENGER_PATH = os.path.join(BASE_DIR, "premium_models", "challenger_log.json")


# ============================================================
# SIDEBAR — LIVE PREDICTION
# ============================================================

st.sidebar.header("🔮 Quote a Policyholder")

age               = st.sidebar.number_input("Age", value=35, min_value=18, max_value=64)
gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
annual_income     = st.sidebar.number_input("Annual Income", value=55000.0, min_value=0.0)
marital_status    = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
dependents        = st.sidebar.number_input("Number of Dependents", value=1, min_value=0, max_value=10, step=1)
education         = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
occupation        = st.sidebar.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
health_score      = st.sidebar.slider("Health Score", 0.0, 100.0, 60.0)
location          = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
policy_type       = st.sidebar.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
previous_claims   = st.sidebar.number_input("Previous Claims", value=0, min_value=0, max_value=15, step=1)
vehicle_age       = st.sidebar.number_input("Vehicle Age", value=5, min_value=0, max_value=25, step=1)
credit_score      = st.sidebar.slider("Credit Score", 300, 850, 680)
insurance_duration = st.sidebar.number_input("Insurance Duration (yrs)", value=3, min_value=1, max_value=10, step=1)
policy_start_date = st.sidebar.date_input("Policy Start Date", value=date.today() - timedelta(days=180))
customer_feedback = st.sidebar.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
smoking_status    = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
exercise_freq     = st.sidebar.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"])
property_type     = st.sidebar.selectbox("Property Type", ["House", "Apartment", "Condo"])

if st.sidebar.button("Get Premium Quote"):
    payload = {
        "age": age, "gender": gender, "annual_income": annual_income,
        "marital_status": marital_status, "number_of_dependents": dependents,
        "education_level": education, "occupation": occupation,
        "health_score": health_score, "location": location, "policy_type": policy_type,
        "previous_claims": previous_claims, "vehicle_age": vehicle_age,
        "credit_score": credit_score, "insurance_duration": insurance_duration,
        "policy_start_date": policy_start_date.isoformat(),
        "customer_feedback": customer_feedback, "smoking_status": smoking_status,
        "exercise_frequency": exercise_freq, "property_type": property_type,
    }
    with st.sidebar:
        with st.spinner("Calling API... First request may take 30-60s (Render cold start)"):
            try:
                response = requests.post(API_URL, json=payload, timeout=90)
                if response.status_code == 200:
                    result   = response.json()
                    decision = result["decision"]
                    color    = {
                        "AUTO_QUOTE": "green",
                        "AUTO_QUOTE_WITH_LOADING": "orange",
                        "MANUAL_UNDERWRITING": "red"
                    }.get(decision, "gray")
                    st.success("Quote received!")
                    st.markdown(f"**Predicted Annual Premium:** `${result['predicted_annual_premium']:.2f}`")
                    st.markdown(f"**Premium Tier:** `{result['premium_tier']}`")
                    st.markdown(f"<h3 style='color:{color}'>Decision: {decision}</h3>", unsafe_allow_html=True)
                    if result.get("rule_triggered"):
                        st.warning(f"Rule: {result['rule_triggered']}")
                    pi = result.get("prediction_interval_90pct")
                    if pi:
                        st.caption(f"90% interval: ${pi['low']:.2f} – ${pi['high']:.2f}")
                else:
                    st.error(f"API error: HTTP {response.status_code}")
                    st.code(response.text[:300])
            except requests.exceptions.Timeout:
                st.warning("Request timed out (90s). Render is waking up. Wait 30s and try again.")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ============================================================
# SECTION 1 — REAL-TIME MONITORING ALERTS
# ============================================================

st.markdown("---")
st.subheader("🚨 Real-Time Monitoring Alerts")

alerts_found = False

if os.path.exists(MONITOR_PATH):
    df_monitor = pd.read_csv(MONITOR_PATH)
    if "predicted_premium" in df_monitor.columns:
        avg_premium = df_monitor["predicted_premium"].mean()
        st.caption(f"Rolling mean predicted premium: ${avg_premium:,.2f}")
    if "decision" in df_monitor.columns:
        manual_rate = (df_monitor["decision"].astype(str).str.contains("MANUAL|REVIEW|HIGH_RISK")).mean()
        if manual_rate > 0.25:
            st.warning(f"🟡 HIGH MANUAL-UNDERWRITING RATE: {manual_rate:.1%} (expected < 25%).")
            alerts_found = True

if os.path.exists(PSI_PATH):
    df_psi_alert = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi_alert.columns and len(df_psi_alert) > 0:
        max_psi     = df_psi_alert["drift_score"].max()
        top_feature = df_psi_alert.iloc[0]["feature"] if "feature" in df_psi_alert.columns else "unknown"
        if max_psi >= PSI_HIGH:
            st.error(f"🔴 CRITICAL DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Retrain now.")
            alerts_found = True
        elif max_psi >= PSI_MODERATE:
            st.warning(f"🟡 MODERATE DRIFT: PSI={max_psi:.4f} on '{top_feature}'. Monitor.")
            alerts_found = True

if not alerts_found:
    st.success("All systems normal — no alerts triggered")


# ============================================================
# SECTION 2 — CHAMPION vs CHALLENGER HISTORY
# ============================================================

st.markdown("---")
st.subheader("🏆 Champion vs Challenger History")

if os.path.exists(CHALLENGER_PATH):
    with open(CHALLENGER_PATH) as f:
        challenger_log = json.load(f)
    if challenger_log:
        latest         = challenger_log[-1]
        decision_color = "green" if latest["decision"] == "PROMOTED" else "red"
        icon           = "✅" if latest["decision"] == "PROMOTED" else "❌"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Challenger", latest.get("challenger_name", "—"))
        with col2:
            st.metric("Challenger RMSE",   latest.get("challenger_rmse", "—"))
        with col3:
            st.metric("Champion RMSE",     latest.get("champion_rmse", "—") or "First Run")
        with col4:
            st.metric("Challenger R²",     latest.get("challenger_r2", "—"))

        st.markdown(
            f"<h4 style='color:{decision_color}'>{icon} Decision: {latest['decision']} — {latest.get('reason', '')}</h4>",
            unsafe_allow_html=True
        )

        if latest.get("gates"):
            g = latest["gates"]
            gcol1, gcol2, gcol3 = st.columns(3)
            gcol1.metric("RMSE Gate", "✅ Pass" if g.get("rmse_improvement_passed") else "❌ Fail")
            gcol2.metric("R² Gate",   "✅ Pass" if g.get("r2_threshold_passed")     else "❌ Fail")
            gcol3.metric("Gap Gate",  "✅ Pass" if g.get("gap_passed")              else "❌ Fail")

        if len(challenger_log) > 1:
            with st.expander("View full challenger history"):
                history_df   = pd.DataFrame(challenger_log)
                display_cols = [c for c in ["evaluated_at", "challenger_name", "challenger_rmse",
                                             "champion_name", "champion_rmse", "decision", "reason"]
                                if c in history_df.columns]
                st.dataframe(history_df[display_cols], width="stretch")
else:
    st.info("No challenger log found.")


# ============================================================
# SECTION 3 — KPI METRICS + CHARTS
# ============================================================

st.markdown("---")
st.subheader("📊 Model Performance KPIs")

if os.path.exists(MONITOR_PATH):
    df = pd.read_csv(MONITOR_PATH)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if "predicted_premium" in df.columns:
            st.metric("Avg Predicted Premium", f"${df['predicted_premium'].mean():,.0f}")
    with col2:
        if "premium_tier" in df.columns:
            st.metric("Standard Rate",  f"{(df['premium_tier']=='STANDARD').mean():.1%}")
    with col3:
        if "premium_tier" in df.columns:
            st.metric("Elevated Rate",  f"{(df['premium_tier']=='ELEVATED').mean():.1%}")
    with col4:
        if "premium_tier" in df.columns:
            st.metric("High-Risk Rate", f"{(df['premium_tier']=='HIGH_RISK').mean():.1%}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Predicted Premium Distribution")
        if "predicted_premium" in df.columns:
            fig, ax = plt.subplots()
            ax.hist(df["predicted_premium"], bins=50, alpha=0.7, color="steelblue")
            ax.axvline(1200, color="orange", linestyle="--", label="STANDARD/ELEVATED")
            ax.axvline(2500, color="red",    linestyle="--", label="ELEVATED/HIGH_RISK")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
    with col_b:
        st.subheader("Decision Distribution")
        if "decision" in df.columns:
            st.bar_chart(df["decision"].value_counts())

    if "premium_tier" in df.columns:
        st.subheader("Premium Tier Breakdown")
        st.bar_chart(df["premium_tier"].value_counts())

    st.subheader("Prediction Error (vs actual premium)")
    if "predicted_premium" in df.columns and "actual_premium" in df.columns:
        err = (df["predicted_premium"] - df["actual_premium"])
        st.write(err.describe().to_frame(name="residual").T.round(2))
else:
    st.warning("No monitor scores found.")


# ============================================================
# SECTION 4 — PSI DRIFT REPORT
# ============================================================

st.markdown("---")
st.subheader("📉 Feature Drift Report (PSI)")

if os.path.exists(PSI_PATH):
    df_psi = pd.read_csv(PSI_PATH)
    if "drift_score" in df_psi.columns and len(df_psi) > 0:
        def _psi_flag(val):
            if val >= PSI_HIGH:       return "🔴 CRITICAL"
            elif val >= PSI_MODERATE: return "🟡 MODERATE"
            return "🟢 OK"
        df_psi["status"] = df_psi["drift_score"].apply(_psi_flag)
        st.dataframe(df_psi.head(10), width="stretch")

        fig, ax = plt.subplots(figsize=(12, 4))
        colors  = [
            "#E74C3C" if v >= PSI_HIGH else "#F39C12" if v >= PSI_MODERATE else "#2ECC71"
            for v in df_psi["drift_score"].head(10)
        ]
        ax.barh(df_psi["feature"].head(10)[::-1],
                df_psi["drift_score"].head(10)[::-1],
                color=colors[::-1])
        ax.axvline(PSI_MODERATE, color="orange", linestyle="--", label=f"Moderate ({PSI_MODERATE})")
        ax.axvline(PSI_HIGH,     color="red",    linestyle="--", label=f"Critical ({PSI_HIGH})")
        ax.set_title("Feature PSI Drift Scores (Top 10)")
        ax.set_xlabel("PSI Score")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("PSI file found but 'drift_score' column missing/empty.")
else:
    st.warning("Feature drift report not found.")
    st.caption(f"Path checked: `{PSI_PATH}`")


# ============================================================
# SECTION 5 — RECENT PREDICTIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Recent Predictions")

if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    st.dataframe(log_df.tail(20), width="stretch")
else:
    st.info(
        "Prediction logs are written by the Render API — not available on Streamlit Cloud. "
        "Run locally to see logs."
    )