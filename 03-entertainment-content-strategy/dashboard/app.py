"""Dashboard ejecutivo — Entertainment Content Strategy.

Storytelling del proyecto para una audiencia no técnica: resume los hallazgos
de los 3 notebooks (EDA, preparación, modelado) en una sola app interactiva.
Ejecutar con:

    streamlit run dashboard/app.py

desde la carpeta 03-entertainment-content-strategy/.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import build_model_dataset  # noqa: E402
from src.modeling import build_features, build_recommender, recommend, train_models  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv"

st.set_page_config(page_title="Netflix Content Strategy", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    return build_model_dataset(str(DATA_PATH))


@st.cache_resource
def get_recommender(titles: pd.DataFrame):
    return build_recommender(titles)


@st.cache_resource
def get_trained_models(tv_shows: pd.DataFrame):
    X, y = build_features(tv_shows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    models = train_models(X_train, y_train)
    return models, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        f"No se encontró el dataset en `{DATA_PATH}`.\n\n"
        "Descargalo desde https://www.kaggle.com/datasets/shivamb/netflix-shows "
        "y colocá el CSV en `03-entertainment-content-strategy/data/netflix_titles.csv`."
    )
    st.stop()

with st.spinner("Cargando catálogo y entrenando modelos..."):
    titles, tv_shows = load_data()
    similarity, title_to_idx = get_recommender(titles)
    models, X_train, X_test, y_train, y_test = get_trained_models(tv_shows)

st.title("🎬 Entretenimiento — Estrategia de Contenido")
st.caption(
    "¿Qué contenido debería adquirir o producir la plataforma para maximizar engagement, "
    "e identificar brechas de catálogo por género/país/rating?"
)

tab_resumen, tab_analista, tab_recomendador, tab_modelo = st.tabs([
    "📋 Resumen ejecutivo",
    "📊 Análisis descriptivo",
    "🔍 Recomendador",
    "🤖 Modelo de renovación",
])

# ---------------------------------------------------------------------------
# Tab 1 — Resumen ejecutivo
# ---------------------------------------------------------------------------
with tab_resumen:
    st.warning(
        "⚠️ **Este dataset es pura metadata de catálogo — no trae vistas, ratings "
        "de usuarios ni retención.** El recomendador content-based no depende de "
        "eso, pero el modelo de renovación es un proxy débil de \"éxito\", no una "
        "medición real de audiencia. Ver el detalle en cada pestaña."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Títulos en catálogo", f"{len(titles):,}")
    c2.metric("Movies / TV Shows", f"{(titles['type']=='Movie').mean()*100:.0f}% / {(titles['type']=='TV Show').mean()*100:.0f}%")
    c3.metric("Países representados", titles["country"].str.split(", ").explode().nunique())
    c4.metric("Tasa de renovación (series)", f"{tv_shows['renewed'].mean()*100:.1f}%")

    st.markdown("### Hallazgos clave")
    st.markdown(
        """
- **Concentración fuerte en EE. UU.** (41.9% de los títulos) y en pocos géneros
  (*International Movies*, *Dramas*) — primera brecha de catálogo evidente.
- **Bug de datos corregido:** 3 títulos tenían la duración filtrada en la columna
  de rating por un corrimiento de columna en el CSV original.
- **Sin métrica de audiencia en el dataset:** en vez de inventar un proxy de
  "éxito" para todo el catálogo, el modelo predictivo se acotó a algo real y
  verificable — si una serie fue **renovada** para una segunda temporada.
- **El modelo de renovación tiene desempeño modesto pero real** (ROC-AUC ≈ 0.72)
  — muy por debajo de los otros proyectos del portafolio, consistente con que la
  metadata de catálogo es un proxy débil de una decisión que depende de datos de
  audiencia que no están disponibles acá.
- **País y género pesan más que la descripción o el director** en qué tan
  probable es que una serie se renueve.
        """
    )

    with st.expander("Metodología completa (notebooks)"):
        st.markdown(
            """
1. `01_eda.ipynb` — Data Understanding: nulos, bug de datos, cardinalidad de campos multivaluados.
2. `02_data_preparation.ipynb` — limpieza, `content_soup`, redefinición honesta del target de éxito.
3. `03_modeling.ipynb` — recomendador content-based (TF-IDF + coseno) y modelo de renovación.
            """
        )

# ---------------------------------------------------------------------------
# Tab 2 — Análisis descriptivo (enfoque analista)
# ---------------------------------------------------------------------------
with tab_analista:
    col1, col2 = st.columns(2)

    with col1:
        countries = titles["country"].str.split(", ").explode()
        countries = countries[countries != "Unknown"]
        top_countries = countries.value_counts().head(10).sort_values()
        fig = px.bar(
            x=top_countries.values, y=top_countries.index, orientation="h",
            title="Top 10 países (títulos, no exclusivo)",
            labels={"x": "# títulos", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        genres = titles["listed_in"].str.split(", ").explode()
        top_genres = genres.value_counts().head(10).sort_values()
        fig = px.bar(
            x=top_genres.values, y=top_genres.index, orientation="h",
            title="Top 10 géneros (títulos, no exclusivo)",
            labels={"x": "# títulos", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        by_year = titles["date_added"].dt.year.value_counts().sort_index()
        fig = px.bar(
            x=by_year.index, y=by_year.values,
            title="Títulos incorporados al catálogo por año",
            labels={"x": "Año", "y": "# títulos"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        rating_dist = titles["rating"].value_counts().head(10)
        fig = px.pie(values=rating_dist.values, names=rating_dist.index, title="Distribución por rating")
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Los conteos por país/género no suman al total de títulos — ambos campos son "
        "multivaluados (un título puede tener varios países o géneros)."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Recomendador
# ---------------------------------------------------------------------------
with tab_recomendador:
    st.markdown(
        "Recomendador **content-based**: sugiere títulos similares por sinopsis, "
        "género y reparto (TF-IDF + similitud de coseno). No usa datos de audiencia — "
        "no sabe qué mira la gente en conjunto, solo qué tan parecido es el contenido."
    )

    title_options = sorted(titles["title"].tolist())
    default_idx = title_options.index("Ganglands") if "Ganglands" in title_options else 0
    selected_title = st.selectbox("Elegí un título", title_options, index=default_idx)
    n_results = st.slider("Cantidad de recomendaciones", 3, 15, 5)

    results = recommend(titles, similarity, title_to_idx, selected_title, n=n_results)
    st.dataframe(results, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 4 — Modelo de renovación (enfoque científico)
# ---------------------------------------------------------------------------
with tab_modelo:
    st.markdown(
        "¿Se renueva una serie para una segunda temporada? Features disponibles "
        "*antes* de que eso se sepa (año, rating, país, género, cantidad de "
        "géneros/países, si tiene director acreditado, largo de la sinopsis) — "
        "nunca `n_seasons`/`duration`, que son la fuente literal de la etiqueta."
    )

    def summarize(name, model):
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        return {
            "modelo": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }

    results_df = pd.DataFrame([summarize(name, m) for name, m in models.items()]).set_index("modelo").round(3)
    st.dataframe(results_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for name in ["Logistic Regression", "XGBoost"]:
            proba = models[name].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Curva ROC", xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        importances = (
            pd.Series(models["XGBoost"].feature_importances_, index=X_train.columns)
            .sort_values(ascending=False)
            .head(10)
            .iloc[::-1]
        )
        fig = px.bar(
            importances, orientation="h",
            title="Top 10 features — XGBoost",
            labels={"value": "Importancia (gain)", "index": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Por qué el split es aleatorio y no cronológico:** el target `renewed` tiene "
        "un sesgo de madurez — los títulos recientes tuvieron menos tiempo para "
        "renovarse. Ese sesgo está en todo el dataset, no solo en el \"futuro\", así "
        "que un split por año lo hubiera concentrado en el test en vez de evitarlo."
    )
