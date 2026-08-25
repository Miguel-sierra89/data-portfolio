"""Dashboard ejecutivo — Jira Effort Estimation.

Storytelling del proyecto para una audiencia no técnica: resume los hallazgos
de los 3 notebooks (EDA, preparación, modelado) en una sola app interactiva,
incluyendo un estimador de story points en vivo. Ejecutar con:

    streamlit run dashboard/app.py

desde la carpeta 04-jira-effort-estimation/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import PROJECTS, build_model_dataset  # noqa: E402
from src.modeling import (  # noqa: E402
    build_per_project_models, build_pooled_model, embed_texts,
    evaluate_by_project, load_embedder, predict_pooled,
)

DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Jira Effort Estimation", page_icon="🧩", layout="wide")


@st.cache_data
def load_data():
    return build_model_dataset(str(DATA_DIR))


@st.cache_resource
def get_embedder():
    return load_embedder()


def _load_precomputed(name: str, expected_issuekeys: pd.Series):
    """Reuses the embeddings 02_data_preparation.ipynb already saved to disk
    (generating them live takes several minutes for 23K issues on CPU) —
    only if the saved processed_*.csv row order still matches the split
    src.data_prep produces right now."""
    npy_path = DATA_DIR / f"{name}_embeddings.npy"
    csv_path = DATA_DIR / f"processed_{name}.csv"
    if not (npy_path.exists() and csv_path.exists()):
        return None
    saved_keys = pd.read_csv(csv_path)["issuekey"]
    if not saved_keys.equals(expected_issuekeys.reset_index(drop=True)):
        return None
    return np.load(npy_path)


@st.cache_data
def get_embeddings(_embedder, texts: tuple):
    return embed_texts(_embedder, texts)


@st.cache_resource
def get_models(train_embeddings: np.ndarray, train_df: pd.DataFrame):
    pooled = build_pooled_model(train_embeddings, train_df)
    per_project = build_per_project_models(train_embeddings, train_df)
    return pooled, per_project


# ---------------------------------------------------------------------------
# Carga de datos, embeddings y modelos
# ---------------------------------------------------------------------------

if not all((DATA_DIR / f"{p}.csv").exists() for p in PROJECTS):
    st.error(
        f"No se encontraron los 16 CSV de proyectos en `{DATA_DIR}`.\n\n"
        "Ver el README del proyecto para el link de descarga."
    )
    st.stop()

with st.spinner("Cargando datos y modelos..."):
    df, train_df, test_df = load_data()
    embedder = get_embedder()

    train_embeddings = _load_precomputed("train", train_df["issuekey"])
    test_embeddings = _load_precomputed("test", test_df["issuekey"])

if train_embeddings is None or test_embeddings is None:
    st.info(
        "No se encontraron embeddings pre-calculados (o no coinciden con el split actual) — "
        "generándolos ahora. Esto tarda varios minutos la primera vez (~23K issues en CPU); "
        "correr `02_data_preparation.ipynb` primero deja esto instantáneo."
    )
    with st.spinner("Generando embeddings..."):
        train_embeddings = get_embeddings(embedder, tuple(train_df["combined_text"]))
        test_embeddings = get_embeddings(embedder, tuple(test_df["combined_text"]))

with st.spinner("Entrenando modelos..."):
    pooled_model, per_project_models = get_models(train_embeddings, train_df)

pooled_pred = predict_pooled(pooled_model, test_embeddings, test_df["project"])
mae_pooled_global = mean_absolute_error(test_df["storypoint"], pooled_pred)
rmse_pooled_global = mean_squared_error(test_df["storypoint"], pooled_pred) ** 0.5
bias_pooled_global = (pooled_pred - test_df["storypoint"]).mean()

comparison = evaluate_by_project(test_df, pooled_pred, per_project_models, test_embeddings)

st.title("🧩 Jira — Estimación de Esfuerzo en Equipos Ágiles")
st.caption(
    "¿Cómo automatizamos la estimación de story points a partir del texto del issue, "
    "para reducir la subestimación sistemática que rompe la planificación de sprints?"
)

tab_resumen, tab_analista, tab_modelo, tab_estimador = st.tabs([
    "📋 Resumen ejecutivo",
    "📊 Análisis descriptivo",
    "🤖 Modelo: pooled vs. por proyecto",
    "✍️ Estimador en vivo",
])

# ---------------------------------------------------------------------------
# Tab 1 — Resumen ejecutivo
# ---------------------------------------------------------------------------
with tab_resumen:
    st.warning(
        "⚠️ **Dataset sustituto:** el Public Jira Dataset original (Zenodo) pesa 13.8 TB "
        "y requiere login — inviable para este proyecto. Se usa en su lugar el dataset "
        "académico de 23.313 issues / 16 proyectos que originó esa línea de investigación "
        "(Choetkiertikul et al., IEEE TSE 2018). Al no traer timestamps de flujo, el "
        "análisis de cycle time/lead time del alcance original no es posible acá."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Issues totales", f"{len(df):,}")
    c2.metric("Proyectos", df["project"].nunique())
    c3.metric("MAE del modelo (pooled)", f"{mae_pooled_global:.2f} pts")
    c4.metric("Bias (pred - real)", f"{bias_pooled_global:+.2f} pts")

    st.markdown("### Hallazgos clave")
    st.markdown(
        """
- **Las 16 escalas de story points no son comparables entre sí** — de 5 valores
  posibles (`usergrid`, Fibonacci estricto) a 79 (`datamanagement`, casi continua).
- **Ni un modelo *pooled* ni 16 modelos *por proyecto* ganan de forma universal**
  — pooled gana en proyectos grandes con escalas idiosincráticas, separar gana en
  proyectos chicos con escalas Fibonacci limpias. Ver la pestaña de modelo.
- **El modelo hereda el sesgo que el proyecto busca corregir:** tiende a subestimar
  levemente, igual que los equipos humanos — un bias negativo, no solo error.
- **MAE ≈ 3 story points** — un punto de partida razonable para asistir la
  estimación humana, no para reemplazarla.
        """
    )

    with st.expander("Metodología completa (notebooks)"):
        st.markdown(
            """
1. `01_eda.ipynb` — Data Understanding: nulos, heterogeneidad de escalas entre proyectos, outliers.
2. `02_data_preparation.ipynb` — embeddings de texto, split cronológico corregido por proyecto.
3. `03_modeling.ipynb` — comparación pooled vs. por proyecto.
            """
        )

# ---------------------------------------------------------------------------
# Tab 2 — Análisis descriptivo (lo que sí es posible con este dataset)
# ---------------------------------------------------------------------------
with tab_analista:
    st.caption(
        "El dataset no trae historial de flujo, así que el análisis descriptivo se "
        "limita a lo que sí está disponible: distribución de story points y "
        "características del texto por proyecto."
    )

    col1, col2 = st.columns(2)

    with col1:
        counts = df["project"].value_counts().sort_values()
        fig = px.bar(
            x=counts.values, y=counts.index, orientation="h",
            title="Issues por proyecto", labels={"x": "# issues", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        scale = df.groupby("project")["storypoint"].nunique().sort_values()
        fig = px.bar(
            x=scale.values, y=scale.index, orientation="h",
            title="Cardinalidad de la escala de story points por proyecto",
            labels={"x": "# valores distintos de storypoint", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        df, x="project", y="storypoint", log_y=True,
        title="Distribución de storypoint por proyecto (escala log)",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 — Modelo: pooled vs. por proyecto
# ---------------------------------------------------------------------------
with tab_modelo:
    st.markdown(
        f"**Modelo pooled global:** MAE = {mae_pooled_global:.2f} pts, "
        f"RMSE = {rmse_pooled_global:.2f} pts, bias = {bias_pooled_global:+.2f} pts."
    )

    comparison_display = comparison.copy()
    if "mae_por_proyecto" in comparison_display.columns:
        comparison_display["gana"] = np.where(
            comparison_display["mae_pooled"] < comparison_display["mae_por_proyecto"],
            "pooled", "por proyecto",
        )
    st.dataframe(comparison_display.sort_values("mae_pooled").round(2), use_container_width=True)

    if "mae_por_proyecto" in comparison.columns:
        plot_df = comparison.reset_index().melt(
            id_vars="project", value_vars=["mae_pooled", "mae_por_proyecto"],
            var_name="enfoque", value_name="MAE",
        )
        fig = px.bar(
            plot_df, x="project", y="MAE", color="enfoque", barmode="group",
            title="MAE por proyecto: pooled vs. modelos separados",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "El resultado mixto (ni un enfoque gana siempre) conecta directamente con la "
        "heterogeneidad de escalas: pooling ayuda cuando la escala de un proyecto es "
        "idiosincrática y hay poco volumen; separar ayuda cuando la escala es estándar "
        "y consistente."
    )

# ---------------------------------------------------------------------------
# Tab 4 — Estimador en vivo
# ---------------------------------------------------------------------------
with tab_estimador:
    st.markdown(
        "Escribí un título y (opcionalmente) una descripción de issue, elegí un "
        "proyecto de referencia, y el modelo pooled estima los story points."
    )

    title_input = st.text_input("Título del issue", "Fix memory leak in connection pool under high load")
    desc_input = st.text_area("Descripción (opcional)", "")
    project_input = st.selectbox("Proyecto de referencia (opcional)", ["(genérico, sin proyecto)"] + PROJECTS)

    if st.button("Estimar story points"):
        combined = (title_input + ". " + desc_input).strip()
        live_embedding = embed_texts(embedder, [combined])
        project_series = pd.Series([
            project_input if project_input != "(genérico, sin proyecto)" else "__none__"
        ])
        pred = predict_pooled(pooled_model, live_embedding, project_series)[0]
        st.success(f"**Estimación: {pred:.1f} story points**")
        st.caption(
            "Referencia: MAE del modelo en test ≈ "
            f"{mae_pooled_global:.1f} pts — tratar como punto de partida, no como estimación final."
        )
