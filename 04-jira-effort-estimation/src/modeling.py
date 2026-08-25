"""Embedding + regression logic for the Jira effort estimation project.

Mirrors notebooks/02_data_preparation.ipynb and notebooks/03_modeling.ipynb.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .data_prep import PROJECTS

RANDOM_STATE = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(embedder: SentenceTransformer, texts) -> np.ndarray:
    return embedder.encode(list(texts), batch_size=64, show_progress_bar=False)


def project_dummies(series: pd.Series) -> pd.DataFrame:
    """One-hot encodes `project` against the fixed PROJECTS list (not
    pd.get_dummies directly) so train/test/live-input columns always align,
    even if a given split doesn't contain every project."""
    return pd.DataFrame(
        {f"proj_{p}": (series == p).astype(int) for p in PROJECTS},
        index=series.index,
    )


def build_pooled_model(train_embeddings: np.ndarray, train_df: pd.DataFrame) -> Ridge:
    X = np.hstack([train_embeddings, project_dummies(train_df["project"]).values])
    y = train_df["log_storypoint"].values
    return Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X, y)


def build_per_project_models(train_embeddings: np.ndarray, train_df: pd.DataFrame) -> dict:
    models = {}
    for project in train_df["project"].unique():
        mask = (train_df["project"] == project).values
        if mask.sum() < 5:
            continue
        models[project] = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(
            train_embeddings[mask], train_df.loc[mask, "log_storypoint"],
        )
    return models


def predict_pooled(model: Ridge, embeddings: np.ndarray, projects: pd.Series) -> np.ndarray:
    X = np.hstack([embeddings, project_dummies(projects).values])
    return np.expm1(model.predict(X))


def evaluate_by_project(test_df: pd.DataFrame, pooled_pred: np.ndarray, per_project_models: dict, test_embeddings: np.ndarray) -> pd.DataFrame:
    test_df = test_df.copy()
    test_df["pooled_pred"] = pooled_pred

    rows = []
    for project, group in test_df.groupby("project"):
        mae_pooled = mean_absolute_error(group["storypoint"], group["pooled_pred"])
        row = {"project": project, "n_test": len(group), "mae_pooled": mae_pooled}
        if project in per_project_models:
            mask = (test_df["project"] == project).values
            pred = np.expm1(per_project_models[project].predict(test_embeddings[mask]))
            row["mae_por_proyecto"] = mean_absolute_error(group["storypoint"], pred)
        rows.append(row)
    return pd.DataFrame(rows).set_index("project")
