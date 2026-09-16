"""
Synthetic data generator matching the schema of Kaggle Playground Series S4E12
("Regression with an Insurance Dataset" — target: Premium Amount).

WHY THIS EXISTS:
The real competition train.csv (~1.2M rows) is only downloadable via the Kaggle
API, which requires authenticated access to kaggle.com — not reachable from this
sandboxed build environment. To keep local development/testing unblocked, this
script generates a LARGE, SCHEMA-IDENTICAL synthetic dataset with realistic,
non-trivial feature-target relationships + noise + missingness, so the full
pipeline (data_loader -> preprocessing -> model_tuning -> ... ) can be built,
run, and unit-tested end-to-end.

TO USE THE REAL DATA: download train.csv from
https://www.kaggle.com/competitions/playground-series-s4e12/data and point
INSURANCE_DATA_PATH at it. Column names below match the real competition
schema exactly, so no code changes are required.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 200_000

# ── Categorical domains (match real competition schema) ──────
genders           = ["Male", "Female"]
marital_status    = ["Single", "Married", "Divorced"]
education_levels  = ["High School", "Bachelor's", "Master's", "PhD"]
occupations       = ["Employed", "Self-Employed", "Unemployed"]
locations         = ["Urban", "Suburban", "Rural"]
policy_types      = ["Basic", "Comprehensive", "Premium"]
customer_feedback = ["Poor", "Average", "Good"]
smoking_status    = ["Yes", "No"]
exercise_freq     = ["Rarely", "Monthly", "Weekly", "Daily"]
property_types    = ["House", "Apartment", "Condo"]

df = pd.DataFrame({
    "id":                     np.arange(1, N + 1),
    "Age":                    rng.integers(18, 65, N),
    "Gender":                 rng.choice(genders, N),
    "Annual Income":          np.round(np.clip(rng.lognormal(mean=10.5, sigma=0.6, size=N), 1000, 500000), 2),
    "Marital Status":         rng.choice(marital_status, N, p=[0.35, 0.50, 0.15]),
    "Number of Dependents":   rng.integers(0, 6, N),
    "Education Level":        rng.choice(education_levels, N, p=[0.30, 0.40, 0.22, 0.08]),
    "Occupation":             rng.choice(occupations, N, p=[0.65, 0.20, 0.15]),
    "Health Score":           np.round(np.clip(rng.normal(50, 20, N), 0, 100), 2),
    "Location":               rng.choice(locations, N, p=[0.45, 0.35, 0.20]),
    "Policy Type":            rng.choice(policy_types, N, p=[0.40, 0.35, 0.25]),
    "Previous Claims":        rng.poisson(0.6, N),
    "Vehicle Age":            rng.integers(0, 20, N),
    "Credit Score":           np.round(np.clip(rng.normal(650, 100, N), 300, 850), 0),
    "Insurance Duration":     rng.integers(1, 10, N),
    "Policy Start Date":      pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1825, N), unit="D"),
    "Customer Feedback":      rng.choice(customer_feedback, N, p=[0.20, 0.45, 0.35]),
    "Smoking Status":         rng.choice(smoking_status, N, p=[0.22, 0.78]),
    "Exercise Frequency":     rng.choice(exercise_freq, N, p=[0.25, 0.25, 0.30, 0.20]),
    "Property Type":          rng.choice(property_types, N, p=[0.45, 0.35, 0.20]),
})

# ── Realistic latent premium formula + noise (mirrors actuarial pricing logic) ─
policy_mult = df["Policy Type"].map({"Basic": 0.85, "Comprehensive": 1.15, "Premium": 1.55})
smoker_mult = df["Smoking Status"].map({"Yes": 1.35, "No": 1.0})
feedback_mult = df["Customer Feedback"].map({"Poor": 1.15, "Average": 1.0, "Good": 0.92})
exercise_mult = df["Exercise Frequency"].map({"Rarely": 1.12, "Monthly": 1.03, "Weekly": 0.95, "Daily": 0.88})

base = (
    300
    + df["Age"] * 3.2
    + (100 - df["Health Score"]) * 4.5
    + df["Previous Claims"] * 180
    + (850 - df["Credit Score"]) * 0.9
    + df["Vehicle Age"] * 6.0
    + df["Number of Dependents"] * 25
    + df["Annual Income"] * 0.0018
    + df["Insurance Duration"] * 8
)

premium = base * policy_mult * smoker_mult * feedback_mult * exercise_mult
premium = premium * rng.lognormal(mean=0, sigma=0.28, size=N)   # multiplicative noise
premium = np.clip(premium, 20, 50000)

df["Premium Amount"] = np.round(premium, 2)

# ── Inject realistic missingness (real competition data has NaNs) ─────
def inject_missing(col, frac):
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

for col, frac in [
    ("Age", 0.02), ("Annual Income", 0.05), ("Marital Status", 0.02),
    ("Number of Dependents", 0.04), ("Occupation", 0.03), ("Health Score", 0.03),
    ("Previous Claims", 0.03), ("Vehicle Age", 0.01), ("Credit Score", 0.05),
    ("Insurance Duration", 0.01), ("Customer Feedback", 0.03),
]:
    inject_missing(col, frac)

df.to_csv("data/insurance_premium.csv", index=False)
print("Wrote data/insurance_premium.csv", df.shape)
print(df.isnull().sum()[df.isnull().sum() > 0])
