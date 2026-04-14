from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import joblib

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

st.set_page_config(page_title="Netflix Customer Churn Prediction", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def build_input_df(
    age: int,
    gender: str,
    subscription_type: str,
    watch_hours: float,
    last_login_days: int,
    region: str,
    device: str,
    monthly_fee: float,
    payment_method: str,
    number_of_profiles: int,
    avg_watch_time_per_day: float,
    favorite_genre: str,
) -> pd.DataFrame:
    inactive_user_flag = int(last_login_days >= 30)
    high_watch_user_flag = 0
    premium_user_flag = int(subscription_type.lower() == "premium")
    profiles_per_fee = number_of_profiles / (monthly_fee + 1e-6)

    data = {
        "age": [age],
        "gender": [gender],
        "subscription_type": [subscription_type],
        "watch_hours": [watch_hours],
        "last_login_days": [last_login_days],
        "region": [region],
        "device": [device],
        "monthly_fee": [monthly_fee],
        "payment_method": [payment_method],
        "number_of_profiles": [number_of_profiles],
        "avg_watch_time_per_day": [avg_watch_time_per_day],
        "favorite_genre": [favorite_genre],
        "inactive_user_flag": [inactive_user_flag],
        "high_watch_user_flag": [high_watch_user_flag],
        "premium_user_flag": [premium_user_flag],
        "profiles_per_fee": [profiles_per_fee],
    }
    return pd.DataFrame(data)


st.title("Netflix Customer Churn Prediction")
st.write("고객 정보를 입력하면 이탈 여부와 이탈 확률을 예측합니다.")

model = load_model()

if model is None:
    st.warning("학습된 모델이 없습니다. 먼저 `python src/train.py` 를 실행하세요.")
else:
    age = st.number_input("Age", min_value=10, max_value=100, value=29)
    gender = st.selectbox("Gender", ["Male", "Female"])
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    watch_hours = st.number_input("Watch Hours", min_value=0.0, value=50.0, step=1.0)
    last_login_days = st.number_input("Last Login Days", min_value=0, value=10, step=1)
    region = st.selectbox("Region", ["Asia", "Europe", "North America", "South America", "Africa"])
    device = st.selectbox("Device", ["Mobile", "Tablet", "TV", "Laptop"])
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=9.99, step=0.01)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Crypto", "Gift Card", "PayPal"])
    number_of_profiles = st.number_input("Number of Profiles", min_value=1, max_value=10, value=2, step=1)
    avg_watch_time_per_day = st.number_input("Avg Watch Time Per Day", min_value=0.0, value=1.5, step=0.1)
    favorite_genre = st.selectbox("Favorite Genre", ["Drama", "Action", "Comedy", "Romance", "Sci-Fi", "Documentary"])

    if st.button("예측하기"):
        input_df = build_input_df(
            age=age,
            gender=gender,
            subscription_type=subscription_type,
            watch_hours=watch_hours,
            last_login_days=last_login_days,
            region=region,
            device=device,
            monthly_fee=monthly_fee,
            payment_method=payment_method,
            number_of_profiles=number_of_profiles,
            avg_watch_time_per_day=avg_watch_time_per_day,
            favorite_genre=favorite_genre,
        )

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            churn_proba = float(model.predict_proba(input_df)[0][1])
        else:
            churn_proba = None

        st.subheader("예측 결과")
        st.write(f"예측 클래스: {'이탈' if prediction == 1 else '유지'}")

        if churn_proba is not None:
            st.write(f"이탈 확률: {churn_proba:.2%}")

        st.subheader("입력 데이터")
        st.dataframe(input_df)