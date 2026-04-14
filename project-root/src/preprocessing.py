from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    CSV 파일을 읽어 DataFrame으로 반환한다.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    return pd.read_csv(csv_path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    간단한 파생변수를 추가한다.
    """
    df = df.copy()

    if "last_login_days" in df.columns:
        df["inactive_user_flag"] = (df["last_login_days"] >= 30).astype(int)

    if "watch_hours" in df.columns:
        df["high_watch_user_flag"] = (df["watch_hours"] >= df["watch_hours"].median()).astype(int)

    if "subscription_type" in df.columns:
        df["premium_user_flag"] = (df["subscription_type"].astype(str).str.lower() == "premium").astype(int)

    if "number_of_profiles" in df.columns and "monthly_fee" in df.columns:
        df["profiles_per_fee"] = df["number_of_profiles"] / (df["monthly_fee"] + 1e-6)

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    target과 feature를 분리한다.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"타깃 컬럼 '{TARGET_COLUMN}' 이 없습니다.")

    x = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    y = df[TARGET_COLUMN]

    if ID_COLUMN in x.columns:
        x = x.drop(columns=[ID_COLUMN])

    return x, y


def get_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    수치형/범주형 컬럼 분리
    """
    numeric_features = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """
    전처리기 생성
    """
    numeric_features, categorical_features = get_feature_types(x)

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor