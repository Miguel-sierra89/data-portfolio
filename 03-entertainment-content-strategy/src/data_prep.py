"""Data loading and feature engineering for the Netflix content strategy project.

Mirrors notebooks/01_eda.ipynb and notebooks/02_data_preparation.ipynb so the
dashboard and the notebooks stay consistent.
"""

import numpy as np
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fix_rating_duration_bug(df: pd.DataFrame) -> pd.DataFrame:
    """3 rows have a duration value ('74 min', etc.) shifted into `rating`,
    with `duration` left null — a column-shift bug in the raw CSV."""
    df = df.copy()
    bug_mask = df["rating"].astype(str).str.contains("min", na=False)
    df.loc[bug_mask, "duration"] = df.loc[bug_mask, "rating"]
    df.loc[bug_mask, "rating"] = np.nan
    return df


def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["director", "cast", "country", "rating"]:
        df[col] = df[col].fillna("Unknown")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), format="%B %d, %Y", errors="coerce")

    df["duration_min"] = np.where(
        df["type"] == "Movie",
        pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0], errors="coerce"),
        np.nan,
    )
    df["n_seasons"] = np.where(
        df["type"] == "TV Show",
        pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0], errors="coerce"),
        np.nan,
    )

    def clean_list_field(text):
        if text == "Unknown":
            return ""
        return text.replace(",", " ").replace("  ", " ")

    df["content_soup"] = (
        df["description"].fillna("") + " " +
        (df["listed_in"] + " ") * 2 +
        df["cast"].apply(clean_list_field) + " " +
        df["director"].apply(clean_list_field)
    ).str.lower()

    return df


def build_tv_shows(df: pd.DataFrame) -> pd.DataFrame:
    """Filters to TV shows and builds the renewal target + features. See
    02_data_preparation.ipynb for why `renewed` (2+ seasons) replaces a
    fabricated 'success' proxy — it's a real business outcome."""
    tv = df[df["type"] == "TV Show"].copy()
    tv["renewed"] = (tv["n_seasons"] > 1).astype(int)

    tv["primary_country"] = tv["country"].str.split(", ").str[0]
    tv["primary_genre"] = tv["listed_in"].str.split(", ").str[0]
    tv["n_genres"] = tv["listed_in"].str.split(", ").apply(len)
    tv["n_countries"] = tv["country"].apply(lambda s: 0 if s == "Unknown" else len(s.split(", ")))
    tv["has_director"] = (tv["director"] != "Unknown").astype(int)
    tv["description_len"] = tv["description"].str.len()

    return tv


def build_model_dataset(raw_path: str):
    """Full pipeline: raw CSV -> cleaned catalog + TV-shows-only subset."""
    df = load_raw(raw_path)
    df = fix_rating_duration_bug(df)
    df = fill_nulls(df)
    df = engineer_features(df)
    tv_shows = build_tv_shows(df)
    return df, tv_shows
