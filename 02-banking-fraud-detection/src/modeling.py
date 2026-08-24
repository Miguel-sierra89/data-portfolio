"""Model training and cost-based evaluation for the banking fraud project.

Mirrors notebooks/03_modeling.ipynb and notebooks/04_evaluation.ipynb.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42


def build_logreg() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def build_xgb(**kwargs) -> XGBClassifier:
    params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", random_state=RANDOM_STATE,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


def train_all_models(X_train, y_train):
    """Trains the four models compared in 03_modeling.ipynb and returns
    fitted estimators plus the SMOTE-resampled XGBoost."""
    logreg = build_logreg().fit(X_train, y_train)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb_cw = build_xgb(scale_pos_weight=neg / pos).fit(X_train, y_train)

    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    xgb_smote = build_xgb().fit(X_res, y_res)

    iso = IsolationForest(
        n_estimators=200, contamination=y_train.mean(), random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train)

    return {
        "Logistic Regression": logreg,
        "XGBoost + class_weight": xgb_cw,
        "XGBoost + SMOTE": xgb_smote,
        "Isolation Forest": iso,
    }


def score(model, X, model_name: str) -> np.ndarray:
    """Returns a risk score in [roughly comparable ranges] for any of the
    trained models, handling Isolation Forest's different API."""
    if model_name == "Isolation Forest":
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]


def total_cost(y_true, proba, amounts, threshold, admin_cost):
    pred = (proba >= threshold).astype(int)
    y_true = np.asarray(y_true)
    fn_mask = (y_true == 1) & (pred == 0)
    flagged_mask = pred == 1
    cost = amounts[fn_mask].sum() + admin_cost * flagged_mask.sum()
    return cost, int(fn_mask.sum()), int(flagged_mask.sum())


def cost_curve(y_true, proba, amounts, admin_cost, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_true, proba, amounts, t, admin_cost)[0] for t in thresholds]
    return pd.DataFrame({"threshold": thresholds, "cost": costs})


def optimal_threshold(y_true, proba, amounts, admin_cost):
    curve = cost_curve(y_true, proba, amounts, admin_cost)
    best = curve.loc[curve["cost"].idxmin()]
    return float(best["threshold"]), float(best["cost"])
