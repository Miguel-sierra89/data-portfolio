"""Dashboard — AI Resume Matching (RAG).

Demo interactiva del motor de matching semántico construido en
notebooks/03_modeling.ipynb y expuesto como servicio en app/main.py (FastAPI).
Esta versión en Streamlit corre la misma lógica (mismo embedder, mismo índice
FAISS, misma explicabilidad por overlap de skills) sin necesitar levantar el
servicio — pensada para poder probarse desde el navegador. Ejecutar con:

    streamlit run dashboard/app.py

desde la carpeta 05-ai-resume-matching-rag/.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import build_dataset  # noqa: E402
from src.matching import (  # noqa: E402
    build_candidate_index, embed_text, explain_match,
    load_embedder, load_precomputed_index, search,
)

DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "resume_data.csv"

st.set_page_config(page_title="AI Resume Matching (RAG)", page_icon="🤖", layout="wide")


@st.cache_data
def load_data():
    return build_dataset(str(RAW_PATH))


@st.cache_resource
def get_embedder():
    return load_embedder()


if not RAW_PATH.exists():
    st.error(
        f"No se encontró `{RAW_PATH}`. Ver el README del proyecto para el link de descarga."
    )
    st.stop()

with st.spinner("Cargando datos y modelo de embeddings..."):
    candidates, jobs, pairs = load_data()
    embedder = get_embedder()
    index, embeddings = load_precomputed_index(DATA_DIR, candidates)

if index is None:
    with st.spinner("No se encontró un índice precomputado — generando embeddings (puede tardar unos segundos)..."):
        index, embeddings = build_candidate_index(embedder, candidates)

st.title("🤖 IA Aplicada — Matching de Candidatos con RAG")
st.caption(
    "¿Cómo automatizamos el matching semántico entre vacantes y CVs, con explicabilidad "
    "del ranking? Escribí una descripción de vacante y el motor devuelve los candidatos "
    "más afines por similitud de embeddings."
)

c1, c2 = st.columns(2)
c1.metric("Candidatos indexados", f"{len(candidates):,}")
c2.metric("Vacantes de referencia", f"{len(jobs):,}")

tab_buscador, tab_vacantes, tab_contexto = st.tabs([
    "🔎 Buscador de candidatos",
    "📋 Vacantes de referencia",
    "ℹ️ Sobre este proyecto",
])

# ---------------------------------------------------------------------------
# Tab 1 — Buscador en vivo
# ---------------------------------------------------------------------------
with tab_buscador:
    st.markdown(
        "Pegá una descripción de vacante en texto libre (o dejá el ejemplo precargado "
        "de una vacante real del dataset) y el motor busca los candidatos más afines por "
        "similitud de coseno sobre embeddings del perfil completo del CV."
    )

    ejemplo = jobs.iloc[0]["job_profile"] if len(jobs) else ""
    job_text = st.text_area("Descripción de la vacante", value=ejemplo, height=140)
    k = st.slider("Cantidad de candidatos a mostrar", 1, 20, 5)

    if st.button("Buscar candidatos"):
        if not job_text or len(job_text.strip()) < 10:
            st.warning("Escribí una descripción de al menos 10 caracteres.")
        else:
            query_embedding = embed_text(embedder, job_text)
            scores, idxs = search(index, query_embedding, k)
            rows = []
            for score, idx in zip(scores, idxs):
                row = candidates.iloc[idx]
                rows.append({
                    "candidate_id": int(row["candidate_id"]),
                    "similarity": round(float(score), 4),
                    "skills": row["skills"],
                    "positions": row["positions"],
                    "skills_matched": ", ".join(explain_match(row["skills"], job_text)) or "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2 — Vacantes de referencia
# ---------------------------------------------------------------------------
with tab_vacantes:
    st.caption(
        "Vacantes reales del dataset — copiá el texto de `job_profile` de alguna en el "
        "buscador de arriba para probar el matching con un caso real."
    )
    st.dataframe(
        jobs[["job_id", "job_position_name", "job_profile"]],
        use_container_width=True, hide_index=True,
    )

# ---------------------------------------------------------------------------
# Tab 3 — Contexto del proyecto
# ---------------------------------------------------------------------------
with tab_contexto:
    st.markdown(
        """
Este motor de matching es el mismo que corre en el servicio FastAPI del proyecto
(`app/main.py`, endpoints `/match/job/{job_id}` y `/match/text`) — acá se expone en
Streamlit para poder probarlo desde el navegador sin levantar el servicio.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Búsqueda:** FAISS (`IndexFlatIP`, similitud de coseno sobre embeddings normalizados).
- **Explicabilidad:** overlap textual de skills del candidato contra el texto de la
  vacante — determinístico, sin depender de un LLM ni de una API key de pago (ver
  `notebooks/03_modeling.ipynb` para por qué se descartó esa opción).
- **Validación cuantitativa:** Spearman 0.30 contra el `matched_score` de referencia
  del dataset — un pool de 344 candidatos, no un sistema en producción. Limitaciones
  completas en el [README del proyecto](../).
        """
    )
