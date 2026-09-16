# ============================================================
# POLICYHOLDER SIMULATOR — Insurance Premium Prediction ML System
# ============================================================

import requests
import random
import time
import os
from datetime import date, timedelta

API_URL = os.getenv("INSURANCE_API_URL", "http://127.0.0.1:8000") + "/predict"

GENDERS       = ["Male", "Female"]
MARITAL       = ["Single", "Married", "Divorced"]
EDUCATION     = ["High School", "Bachelor's", "Master's", "PhD"]
OCCUPATIONS   = ["Employed", "Self-Employed", "Unemployed"]
LOCATIONS     = ["Urban", "Suburban", "Rural"]
POLICY_TYPES  = ["Basic", "Comprehensive", "Premium"]
FEEDBACK      = ["Poor", "Average", "Good"]
EXERCISE      = ["Rarely", "Monthly", "Weekly", "Daily"]
PROPERTY      = ["House", "Apartment", "Condo"]


def _random_start_date():
    days_ago = random.randint(0, 1825)
    return (date.today() - timedelta(days=days_ago)).isoformat()


def generate_policyholder(scenario: str = "random") -> dict:
    """
    Scenarios:
        random     — mixed realistic applicants
        high_risk  — smoker + low health score + many prior claims
        low_risk   — non-smoker, healthy, good credit, no claims
    """

    if scenario == "high_risk":
        return {
            "age": random.randint(45, 64), "gender": random.choice(GENDERS),
            "annual_income": round(random.uniform(15000, 60000), 2),
            "marital_status": random.choice(MARITAL), "number_of_dependents": random.randint(2, 5),
            "education_level": random.choice(EDUCATION), "occupation": random.choice(OCCUPATIONS),
            "health_score": round(random.uniform(5, 28), 1), "location": random.choice(LOCATIONS),
            "policy_type": random.choice(POLICY_TYPES), "previous_claims": random.randint(3, 8),
            "vehicle_age": random.randint(15, 20), "credit_score": round(random.uniform(300, 480), 0),
            "insurance_duration": random.randint(1, 9), "policy_start_date": _random_start_date(),
            "customer_feedback": "Poor", "smoking_status": "Yes",
            "exercise_frequency": "Rarely", "property_type": random.choice(PROPERTY),
        }

    elif scenario == "low_risk":
        return {
            "age": random.randint(22, 40), "gender": random.choice(GENDERS),
            "annual_income": round(random.uniform(50000, 150000), 2),
            "marital_status": random.choice(MARITAL), "number_of_dependents": random.randint(0, 1),
            "education_level": random.choice(EDUCATION), "occupation": "Employed",
            "health_score": round(random.uniform(75, 100), 1), "location": random.choice(LOCATIONS),
            "policy_type": random.choice(POLICY_TYPES), "previous_claims": 0,
            "vehicle_age": random.randint(0, 5), "credit_score": round(random.uniform(720, 850), 0),
            "insurance_duration": random.randint(1, 9), "policy_start_date": _random_start_date(),
            "customer_feedback": "Good", "smoking_status": "No",
            "exercise_frequency": random.choice(["Weekly", "Daily"]), "property_type": random.choice(PROPERTY),
        }

    else:  # random
        return {
            "age": random.randint(18, 64), "gender": random.choice(GENDERS),
            "annual_income": round(random.uniform(1000, 300000), 2),
            "marital_status": random.choice(MARITAL), "number_of_dependents": random.randint(0, 5),
            "education_level": random.choice(EDUCATION), "occupation": random.choice(OCCUPATIONS),
            "health_score": round(random.uniform(0, 100), 1), "location": random.choice(LOCATIONS),
            "policy_type": random.choice(POLICY_TYPES), "previous_claims": random.randint(0, 6),
            "vehicle_age": random.randint(0, 19), "credit_score": round(random.uniform(300, 850), 0),
            "insurance_duration": random.randint(1, 9), "policy_start_date": _random_start_date(),
            "customer_feedback": random.choice(FEEDBACK),
            "smoking_status": random.choices(["Yes", "No"], weights=[0.22, 0.78])[0],
            "exercise_frequency": random.choice(EXERCISE), "property_type": random.choice(PROPERTY),
        }


def send_policyholder(policyholder: dict, idx: int):
    try:
        response = requests.post(API_URL, json=policyholder, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  Age={policyholder['age']}  "
                  f"Health={policyholder['health_score']}  "
                  f"Claims={policyholder['previous_claims']}  "
                  f"Smoker={policyholder['smoking_status']}  "
                  f"→  premium=${result['predicted_annual_premium']:.2f}  "
                  f"tier={result['premium_tier']}  "
                  f"decision={result['decision']}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")
    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


def simulate_applications(n: int = 20, scenario: str = "random"):
    print(f"\nSimulating {n} policy applications  |  scenario={scenario}\n" + "-" * 60)
    for i in range(n):
        policyholder = generate_policyholder(scenario)
        send_policyholder(policyholder, i)
        time.sleep(0.5)
    print("-" * 60 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_applications(20, scenario="random")
