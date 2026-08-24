"""Data loading and feature engineering for the banking fraud project.

Mirrors notebooks/01_eda.ipynb and notebooks/02_data_preparation.ipynb so
the dashboard and the notebooks stay consistent.
"""

import numpy as np
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """Load creditcard.csv and deduplicate exact-duplicate rows (see EDA
    finding: 1,081 exact duplicates, 19 of them fraud). Deduplication happens
    before any split so a duplicate pair can never straddle train/test."""
    df = pd.read_csv(path)
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day (fraud rate spikes overnight, per EDA) and log_amount
    (Amount is heavily right-skewed). V1-V28 are left untouched — the EDA
    showed the fraud signal lives in their extreme values."""
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["log_amount"] = np.log1p(df["Amount"])
    return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """80/20 split by Time (already sorted) — train on the earlier period,
    test on the most recent one, matching real fraud-detection deployment."""
    cutoff_time = df["Time"].quantile(1 - test_frac)
    train_df = df[df["Time"] < cutoff_time].copy()
    test_df = df[df["Time"] >= cutoff_time].copy()
    return train_df, test_df, cutoff_time


def build_model_dataset(raw_path: str):
    """Full pipeline: raw CSV -> deduplicated, feature-engineered,
    chronologically split train/test sets."""
    df = load_raw(raw_path)
    df = engineer_features(df)
    train_df, test_df, cutoff_time = chronological_split(df)
    return df, train_df, test_df, cutoff_time


def get_xy(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y
