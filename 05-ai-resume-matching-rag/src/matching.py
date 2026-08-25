"""Embedding + FAISS matching logic for the resume matching RAG project.

Mirrors notebooks/03_modeling.ipynb.
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def build_candidate_index(embedder: SentenceTransformer, candidates: pd.DataFrame):
    embeddings = embedder.encode(
        candidates["candidate_profile"].fillna("").tolist(), normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def load_precomputed_index(data_dir: Path, candidates: pd.DataFrame):
    """Reuses the FAISS index + embeddings 03_modeling.ipynb already saved to
    disk, if present and consistent with the current candidate table —
    avoids re-embedding 344 profiles (and reloading the model) on every
    API restart."""
    index_path = data_dir / "candidates.faiss"
    emb_path = data_dir / "candidate_embeddings.npy"
    if not (index_path.exists() and emb_path.exists()):
        return None, None
    embeddings = np.load(emb_path)
    if embeddings.shape[0] != len(candidates):
        return None, None
    index = faiss.read_index(str(index_path))
    return index, embeddings


def embed_text(embedder: SentenceTransformer, text: str) -> np.ndarray:
    return embedder.encode([text], normalize_embeddings=True).astype("float32")


def search(index: faiss.Index, query_embedding: np.ndarray, k: int):
    scores, idxs = index.search(query_embedding, k)
    return scores[0], idxs[0]


def explain_match(candidate_skills: str, job_text: str) -> list[str]:
    """Deterministic, no-API-key explainability: skills the candidate lists
    that appear verbatim in the job text. See 03_modeling.ipynb for why this
    stands in for an LLM-based justification (avoids requiring a paid API
    key to run this service)."""
    if pd.isnull(candidate_skills) or pd.isnull(job_text):
        return []
    job_lower = str(job_text).lower()
    return [
        skill.strip() for skill in str(candidate_skills).split(", ")
        if len(skill.strip()) > 2 and skill.strip().lower() in job_lower
    ]
