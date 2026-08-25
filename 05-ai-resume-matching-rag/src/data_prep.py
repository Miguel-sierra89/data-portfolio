"""Data loading and feature engineering for the resume matching RAG project.

Mirrors notebooks/01_eda.ipynb and notebooks/02_data_preparation.ipynb so the
FastAPI service and the notebooks stay consistent.
"""

import ast

import pandas as pd

LIST_LIKE_COLS = [
    "skills", "educational_institution_name", "degree_names", "passing_years",
    "educational_results", "result_types", "major_field_of_studies",
    "professional_company_names", "company_urls", "start_dates", "end_dates",
    "related_skils_in_job", "positions", "locations",
    "extra_curricular_activity_types", "extra_curricular_organization_names",
    "extra_curricular_organization_links", "role_positions", "languages",
    "proficiency_levels", "certification_providers", "certification_skills",
    "online_links", "issue_dates", "expiry_dates",
]

CANDIDATE_COLS = [
    "candidate_id", "address", "career_objective", "skills",
    "educational_institution_name", "degree_names", "major_field_of_studies",
    "educational_results", "professional_company_names", "positions",
    "related_skils_in_job", "candidate_past_responsibilities", "languages",
    "proficiency_levels", "certification_providers", "certification_skills",
]
JOB_COLS = [
    "job_id", "job_position_name", "educationaL_requirements",
    "experiencere_requirement", "age_requirement", "job_responsibilities",
    "skills_required",
]


def clean_list_field(raw) -> str:
    """Parses a "['A', 'B']"-style string (possibly nested, possibly full of
    None) into plain comma-separated text. ast.literal_eval is safe here —
    the content is Python literals (lists/None/strings), not arbitrary code."""
    if raw is None or isinstance(raw, float):
        return ""
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return str(raw).strip()

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        elif x is not None:
            yield str(x)

    parts = [p.strip() for p in flatten(parsed) if p and p.strip().upper() not in ("N/A", "NONE")]
    return ", ".join(parts)


def join_nonempty(*parts) -> str:
    clean = []
    for p in parts:
        if p is None or isinstance(p, float):
            continue
        s = str(p).strip()
        if s:
            clean.append(s)
    return " ".join(clean)


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("﻿", "") for c in df.columns]
    return df.rename(columns={
        "responsibilities": "candidate_past_responsibilities",
        "responsibilities.1": "job_responsibilities",
    })


def build_dataset(raw_path: str):
    """Full pipeline: raw CSV -> deduplicated candidates/jobs tables with
    combined text profiles ready to embed."""
    df = load_raw(raw_path)

    for col in LIST_LIKE_COLS:
        df[col] = df[col].apply(clean_list_field)

    candidate_fp = (
        df["career_objective"].fillna("") + "|" + df["skills"] + "|" +
        df["educational_institution_name"] + "|" + df["degree_names"] + "|" +
        df["professional_company_names"]
    )
    df["candidate_id"] = candidate_fp.factorize()[0]
    df["job_id"] = df["job_position_name"].factorize()[0]

    candidates = df[CANDIDATE_COLS].drop_duplicates(subset="candidate_id").reset_index(drop=True)
    jobs = df[JOB_COLS].drop_duplicates(subset="job_id").reset_index(drop=True)
    pairs = df[["candidate_id", "job_id", "matched_score"]].copy()

    candidates["candidate_profile"] = candidates.apply(
        lambda r: join_nonempty(
            r["career_objective"], r["skills"], r["degree_names"], r["major_field_of_studies"],
            r["professional_company_names"], r["positions"], r["related_skils_in_job"],
            r["candidate_past_responsibilities"], r["languages"], r["certification_skills"],
        ), axis=1,
    )
    jobs["job_profile"] = jobs.apply(
        lambda r: join_nonempty(
            r["job_position_name"], r["educationaL_requirements"], r["experiencere_requirement"],
            r["skills_required"], r["job_responsibilities"],
        ), axis=1,
    )

    return candidates, jobs, pairs
