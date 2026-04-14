from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


def load_model(model_path: str | Path = MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    return joblib.load(model_path)


def predict_single(input_data: dict) -> dict:
    """
    단일 고객 데이터 예측
    """
    model = load_model()
    df = pd.DataFrame([input_data])

    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]
    else:
        proba = None

    return {
        "prediction": int(pred),
        "churn_probability": None if proba is None else float(proba),
    }


if __name__ == "__main__":
    sample_input = {
        "age": 29,
        "gender": "Female",
        "subscription_type": "Basic",
        "watch_hours": 50,
        "last_login_days": 42,
        "region": "Asia",
        "device": "Mobile",
        "monthly_fee": 9.99,
        "payment_method": "Credit Card",
        "number_of_profiles": 2,
        "avg_watch_time_per_day": 1.6,
        "favorite_genre": "Drama",
        "inactive_user_flag": 1,
        "high_watch_user_flag": 0,
        "premium_user_flag": 0,
        "profiles_per_fee": 2 / 9.99,
    }

    result = predict_single(sample_input)
    print(result)