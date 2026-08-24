"""Model training and cost-based evaluation for the BPO SLA project.

Mirrors notebooks/03_modeling.ipynb and notebooks/04_evaluation.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data_prep import NUMERIC_COLS

RANDOM_STATE = 42


def build_logreg() -> Pipeline:
    preprocess = ColumnTransformer([("scale", StandardScaler(), NUMERIC_COLS)], remainder="passthrough")
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])


def build_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )


def train_models(X_train, y_train):
    logreg = build_logreg().fit(X_train, y_train)
    xgb = build_xgb().fit(X_train, y_train)
    return logreg, xgb


def total_cost(y_true, proba, threshold, cost_fn, cost_fp):
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * cost_fp + fn * cost_fn, fp, fn


def cost_curve(y_true, proba, cost_fn, cost_fp, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, proba, t, cost_fn, cost_fp)[0] for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


def optimal_threshold(y_true, proba, cost_fn, cost_fp):
    curve = cost_curve(y_true, proba, cost_fn, cost_fp)
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), float(best["cost"])
