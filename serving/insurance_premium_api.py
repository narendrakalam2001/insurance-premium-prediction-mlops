# ============================================================
# INSURANCE PREMIUM API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging
import time
import os
import json

from src.model_loader             import load_latest_model
from services.prediction_service  import predict_policyholder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Insurance Premium Prediction API")

# ── Load model on startup ─────────────────────────────────────
try:
    model, margin = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model  = None
    margin = 0.0


# ============================================================
# INPUT SCHEMA
# ============================================================

class PolicyholderInput(BaseModel):
    age:                   int
    gender:                str    # "Male" / "Female"
    annual_income:         float
    marital_status:        str    # "Single" / "Married" / "Divorced"
    number_of_dependents:  int
    education_level:       str    # "High School" / "Bachelor's" / "Master's" / "PhD"
    occupation:            str    # "Employed" / "Self-Employed" / "Unemployed"
    health_score:          float  # 0-100
    location:              str    # "Urban" / "Suburban" / "Rural"
    policy_type:           str    # "Basic" / "Comprehensive" / "Premium"
    previous_claims:       int
    vehicle_age:           int
    credit_score:          float  # 300-850
    insurance_duration:    int    # years
    policy_start_date:     str    # "YYYY-MM-DD"
    customer_feedback:     str    # "Poor" / "Average" / "Good"
    smoking_status:        str    # "Yes" / "No"
    exercise_frequency:    str    # "Rarely" / "Monthly" / "Weekly" / "Daily"
    property_type:         str    # "House" / "Apartment" / "Condo"


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Insurance Premium Prediction API is live 🚀",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    registry_path = "premium_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


# ── Prediction endpoint ───────────────────────────────────────

@app.post("/predict")
def predict(policyholder: PolicyholderInput):

    start      = time.time()
    input_data = policyholder.dict()

    result = predict_policyholder(model, input_data, margin)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":                time.time(),
        "age":                      input_data["age"],
        "health_score":             input_data["health_score"],
        "previous_claims":          input_data["previous_claims"],
        "smoking_status":           input_data["smoking_status"],
        "predicted_annual_premium": result["predicted_annual_premium"],
        "premium_tier":             result["premium_tier"],
        "decision":                 result["decision"],
        "rule_triggered":           result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result
