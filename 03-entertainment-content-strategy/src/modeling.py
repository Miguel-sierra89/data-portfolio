"""Recommender and renewal-classification logic for the Netflix content
strategy project. Mirrors notebooks/03_modeling.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
CAT_COLS = ["rating", "primary_genre", "primary_country_bucket"]
NUM_COLS = ["release_year", "n_genres", "n_countries", "has_director", "description_len"]


def build_recommender(titles: pd.DataFrame):
    soup = titles["content_soup"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(soup)
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)
    title_to_idx = pd.Series(titles.index, index=titles["title"]).drop_duplicates()
    return similarity, title_to_idx


def recommend(titles: pd.DataFrame, similarity, title_to_idx, title: str, n: int = 5) -> pd.DataFrame:
    if title not in title_to_idx:
        raise ValueError(f'"{title}" no está en el catálogo')
    idx = title_to_idx[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    result_idx = [i for i, _ in scores]
    result = titles.loc[result_idx, ["title", "type", "listed_in", "release_year"]].copy()
    result["similitud"] = [round(float(s), 3) for _, s in scores]
    return result.reset_index(drop=True)


def build_features(tv_shows: pd.DataFrame):
    top_countries = tv_shows["primary_country"].value_counts().head(10).index.tolist()
    tv_shows = tv_shows.copy()
    tv_shows["primary_country_bucket"] = tv_shows["primary_country"].where(
        tv_shows["primary_country"].isin(top_countries), "Other",
    )
    X = pd.get_dummies(tv_shows[CAT_COLS + NUM_COLS], columns=CAT_COLS, drop_first=True)
    y = tv_shows["renewed"]
    return X, y


def train_models(X_train, y_train):
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE).fit(X_train, y_train)

    logreg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ]).fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    return {"Dummy (mayoría)": dummy, "Logistic Regression": logreg, "XGBoost": xgb}
