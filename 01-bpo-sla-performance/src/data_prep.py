"""Data loading and feature engineering for the BPO SLA project.

Mirrors the logic in notebooks/01_eda.ipynb and notebooks/02_data_preparation.ipynb
so the dashboard and the notebooks stay consistent. See those notebooks for the
reasoning behind each decision (leakage handling, chronological split, etc.).
"""

import pandas as pd

CATEGORICAL_COLS = ["agent_id", "product_id", "lang_id", "day_of_week"]
NUMERIC_COLS = ["calls_handled", "agent_prior_pass_rate", "agent_prior_avg_calls"]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw call_metrics_dataset.csv, handling the European CSV format
    (';' separator, ',' decimal, '.' thousands)."""
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "agent_id"]).reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features and leakage-safe agent history features, then
    drop avg_aht (which determines std_pass almost by construction — see the
    EDA notebook's data leakage finding)."""
    df = df.copy()

    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_rollout_phase"] = (df["date"].dt.month == 7).astype(int)

    df["agent_prior_pass_rate"] = (
        df.groupby("agent_id")["std_pass"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["agent_prior_avg_calls"] = (
        df.groupby("agent_id")["calls_handled"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )

    global_median_calls = df["calls_handled"].median()
    df["agent_prior_pass_rate"] = df["agent_prior_pass_rate"].fillna(0.5)
    df["agent_prior_avg_calls"] = df["agent_prior_avg_calls"].fillna(global_median_calls)

    return df.drop(columns=["avg_aht"])


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS, drop_first=True)


def chronological_split(df_encoded: pd.DataFrame, test_frac: float = 0.2):
    """80/20 split by calendar date (not random) — train on the earlier
    period, test on the most recent one, matching the forward-prediction
    use case described in 02_data_preparation.ipynb."""
    unique_dates = sorted(df_encoded["date"].unique())
    cutoff_date = unique_dates[int(len(unique_dates) * (1 - test_frac))]
    train_df = df_encoded[df_encoded["date"] < cutoff_date].copy()
    test_df = df_encoded[df_encoded["date"] >= cutoff_date].copy()
    return train_df, test_df, cutoff_date


def build_model_dataset(raw_path: str):
    """Full pipeline: raw CSV -> engineered, encoded, chronologically split
    train/test sets ready for modeling."""
    df_raw = load_raw(raw_path)
    df_feat = engineer_features(df_raw)
    df_encoded = encode_categoricals(df_feat)
    train_df, test_df, cutoff_date = chronological_split(df_encoded)
    return df_raw, train_df, test_df, cutoff_date


def get_xy(df: pd.DataFrame):
    """Split a processed dataframe into features (X) and the sla_risk target
    (1 = incumple SLA), dropping non-feature columns."""
    X = df.drop(columns=["date", "std_pass"])
    y = 1 - df["std_pass"]
    return X, y
