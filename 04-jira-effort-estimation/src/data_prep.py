"""Data loading and feature engineering for the Jira effort estimation project.

Mirrors notebooks/01_eda.ipynb and notebooks/02_data_preparation.ipynb so the
dashboard and the notebooks stay consistent.
"""

import os

import numpy as np
import pandas as pd

PROJECTS = [
    "appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
    "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
    "springxd", "talenddataquality", "talendesb", "titanium", "usergrid",
]


def load_and_combine(data_dir: str) -> pd.DataFrame:
    """Loads only the 16 known raw project CSVs — not processed_train.csv /
    processed_test.csv / embedding files that notebooks also write to this
    same directory."""
    dfs = []
    for project_name in PROJECTS:
        f = os.path.join(data_dir, f"{project_name}.csv")
        d = pd.read_csv(f)
        d["project"] = project_name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["description"] = df["description"].fillna("")
    df["combined_text"] = (df["title"] + ". " + df["description"]).str.strip()
    df["log_storypoint"] = np.log1p(df["storypoint"])
    df["issue_num"] = df["issuekey"].str.extract(r"-(\d+)$").astype(int)
    return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Sorts by issue_num within each project before splitting — row order in
    the raw CSVs does not perfectly match creation order (see 02_data_preparation.ipynb,
    up to 13.8% out-of-order rows in titanium)."""
    train_parts, test_parts = [], []
    for _, group in df.groupby("project", sort=False):
        group = group.sort_values("issue_num")
        cutoff = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def build_model_dataset(data_dir: str):
    df = load_and_combine(data_dir)
    df = engineer_features(df)
    train_df, test_df = chronological_split(df)
    return df, train_df, test_df
