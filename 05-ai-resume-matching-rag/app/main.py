"""FastAPI service — AI Resume Matching (RAG).

Servicio mínimo que expone el pipeline de matching semántico construido en
notebooks/03_modeling.ipynb: dado un job_id existente o una descripción de
vacante en texto libre, devuelve los candidatos más afines por similitud de
embeddings, junto con una explicación por overlap de skills.

Ejecutar (desde 05-ai-resume-matching-rag/):

    uvicorn app.main:app --reload

Docs interactivas en http://127.0.0.1:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import build_dataset  # noqa: E402
from src.matching import (  # noqa: E402
    build_candidate_index, embed_text, explain_match,
    load_embedder, load_precomputed_index, search,
)

DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "resume_data.csv"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not RAW_PATH.exists():
        raise RuntimeError(
            f"No se encontró {RAW_PATH}. Descargá el dataset (ver README del proyecto) "
            "y colocalo como data/resume_data.csv antes de levantar el servicio."
        )

    candidates, jobs, _pairs = build_dataset(str(RAW_PATH))
    embedder = load_embedder()

    index, embeddings = load_precomputed_index(DATA_DIR, candidates)
    if index is None:
        index, embeddings = build_candidate_index(embedder, candidates)

    state["candidates"] = candidates
    state["jobs"] = jobs
    state["embedder"] = embedder
    state["index"] = index
    state["embeddings"] = embeddings

    yield
    state.clear()


app = FastAPI(
    title="AI Resume Matching (RAG)",
    description=(
        "Matching semántico candidato-vacante sobre embeddings + FAISS, con explicabilidad "
        "por overlap de skills. Ver notebooks/03_modeling.ipynb para la validación contra "
        "matched_score y las limitaciones documentadas."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class JobSummary(BaseModel):
    job_id: int
    job_position_name: str


class CandidateMatch(BaseModel):
    candidate_id: int
    similarity: float = Field(..., description="Similitud de coseno con la vacante (0-1)")
    skills: str
    positions: str
    skills_matched: list[str] = Field(..., description="Skills del candidato que aparecen textualmente en la vacante")


class TextMatchRequest(BaseModel):
    job_description: str = Field(..., min_length=10, description="Descripción de la vacante en texto libre")
    k: int = Field(5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok", "candidates": len(state["candidates"]), "jobs": len(state["jobs"])}


@app.get("/jobs", response_model=list[JobSummary])
def list_jobs():
    return state["jobs"][["job_id", "job_position_name"]].to_dict(orient="records")


def _record(row: pd.Series) -> dict:
    """Standard JSON doesn't allow NaN — Starlette's JSONResponse renders with
    allow_nan=False, so nulls from missing CV/job fields (e.g. age_requirement)
    must become None, not float('nan'), before returning."""
    return row.where(pd.notnull(row), None).to_dict()


@app.get("/jobs/{job_id}")
def get_job(job_id: int):
    jobs = state["jobs"]
    row = jobs[jobs["job_id"] == job_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"job_id {job_id} no existe")
    return _record(row.iloc[0])


@app.get("/candidates/{candidate_id}")
def get_candidate(candidate_id: int):
    candidates = state["candidates"]
    row = candidates[candidates["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"candidate_id {candidate_id} no existe")
    return _record(row.iloc[0])


def _matches_from_query(query_embedding, k: int, job_text: str) -> list[CandidateMatch]:
    candidates = state["candidates"]
    scores, idxs = search(state["index"], query_embedding, k)
    results = []
    for score, idx in zip(scores, idxs):
        row = candidates.iloc[idx]
        results.append(CandidateMatch(
            candidate_id=int(row["candidate_id"]),
            similarity=round(float(score), 4),
            skills=row["skills"],
            positions=row["positions"],
            skills_matched=explain_match(row["skills"], job_text),
        ))
    return results


@app.post("/match/job/{job_id}", response_model=list[CandidateMatch])
def match_by_job_id(job_id: int, k: int = 5):
    jobs = state["jobs"]
    row = jobs[jobs["job_id"] == job_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"job_id {job_id} no existe")
    # Las vacantes no tienen su embedding en el índice de candidatos (ese índice
    # es solo de CVs); se re-embebe el perfil de la vacante on-the-fly — barato,
    # un solo texto corto por request.
    query_embedding = embed_text(state["embedder"], row.iloc[0]["job_profile"])
    return _matches_from_query(query_embedding, k, row.iloc[0]["job_profile"])


@app.post("/match/text", response_model=list[CandidateMatch])
def match_by_text(request: TextMatchRequest):
    query_embedding = embed_text(state["embedder"], request.job_description)
    return _matches_from_query(query_embedding, request.k, request.job_description)
