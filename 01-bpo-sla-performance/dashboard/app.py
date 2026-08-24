"""Dashboard ejecutivo — BPO SLA Performance.

Storytelling del proyecto para una audiencia no técnica: resume los hallazgos
de los 4 notebooks (EDA, preparación, modelado, evaluación de costo) en una
sola app interactiva. Ejecutar con:

    streamlit run dashboard/app.py

desde la carpeta 01-bpo-sla-performance/.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import build_model_dataset, get_xy  # noqa: E402
from src.modeling import build_logreg, build_xgb, cost_curve, optimal_threshold, total_cost  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "call_metrics_dataset.csv"

st.set_page_config(page_title="BPO SLA Performance", page_icon="📞", layout="wide")


@st.cache_data
def load_data():
    return build_model_dataset(str(DATA_PATH))


@st.cache_resource
def get_trained_models(train_df: pd.DataFrame):
    X_train, y_train = get_xy(train_df)
    logreg = build_logreg().fit(X_train, y_train)
    xgb = build_xgb().fit(X_train, y_train)
    return logreg, xgb


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        f"No se encontró el dataset en `{DATA_PATH}`.\n\n"
        "Descargalo desde https://www.kaggle.com/datasets/unifrancouni/call-center-metrics-dataset "
        "y colocá el CSV en `01-bpo-sla-performance/data/call_metrics_dataset.csv`."
    )
    st.stop()

df_raw, train_df, test_df, cutoff_date = load_data()
logreg, xgb = get_trained_models(train_df)

X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)
logreg_proba = logreg.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

df_jul = df_raw[df_raw["date"].dt.month == 7].copy()
df_jul["dow"] = df_jul["date"].dt.day_name()

st.title("📞 BPO — Performance y Cumplimiento de SLA")
st.caption(
    "¿Cómo reducimos el % de tickets que incumplen SLA y qué factores lo predicen, "
    "para optimizar la dotación de personal?"
)

tab_resumen, tab_analista, tab_modelo, tab_costo = st.tabs([
    "📋 Resumen ejecutivo",
    "📊 Análisis descriptivo",
    "🤖 Modelo predictivo",
    "💰 Simulador de costo",
])

# ---------------------------------------------------------------------------
# Tab 1 — Resumen ejecutivo
# ---------------------------------------------------------------------------
with tab_resumen:
    st.warning(
        "⚠️ **Dataset pequeño (270 filas).** Las métricas de este dashboard son "
        "consistentes y muy buenas, pero con tan pocos datos eso también es "
        "consistente con sobreajuste. Tratar los resultados como una demostración "
        "de metodología, no como una predicción lista para producción — ver el "
        "detalle en la pestaña *Modelo predictivo*."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros analizados", f"{len(df_raw):,}")
    c2.metric("Tasa de cumplimiento SLA", f"{df_raw['std_pass'].mean() * 100:.1f}%")
    c3.metric("Agentes", df_raw["agent_id"].nunique())
    c4.metric("Rango de fechas", f"{df_raw['date'].min().date()} → {df_raw['date'].max().date()}")

    st.markdown("### Hallazgos clave")
    st.markdown(
        """
- **Se detectó y corrigió un data leakage:** `avg_aht` casi determina por sí sola el
  cumplimiento de SLA, así que se excluyó del modelo y se reemplazó por el historial
  real de desempeño de cada agente (ver notebook de EDA y de preparación).
- **El modelo (XGBoost) predice el riesgo de incumplimiento con ROC-AUC ≈ 0.99** en el
  período de test, muy por encima de un baseline que nunca detecta incumplimientos.
- **Traducido a costo de negocio, el modelo reduce el costo total en ~98%** frente a no
  usar ningún sistema de alerta, bajo los supuestos de costo (editables) de la pestaña
  *Simulador de costo*.
- El riesgo de incumplimiento está más ligado al **volumen del día y al producto/cola**
  que al historial del agente — un hallazgo que corrigió una hipótesis inicial.
        """
    )

    with st.expander("Metodología completa (notebooks)"):
        st.markdown(
            """
1. `01_eda.ipynb` — Data Understanding: nulos, outliers, cardinalidad, data leakage.
2. `02_data_preparation.ipynb` — feature engineering seguro (sin leakage) y split cronológico.
3. `03_modeling.ipynb` — Logistic Regression (baseline) vs XGBoost (principal).
4. `04_evaluation.ipynb` — umbral de decisión óptimo según costo de negocio.
            """
        )

# ---------------------------------------------------------------------------
# Tab 2 — Análisis descriptivo (enfoque analista)
# ---------------------------------------------------------------------------
with tab_analista:
    st.markdown(
        "Vista descriptiva restringida a **julio 2020**, el único período con los "
        "10 agentes operando en paralelo (en enero solo operaba 1 agente piloto — "
        "ver el EDA para el detalle)."
    )

    col1, col2 = st.columns(2)

    with col1:
        agent_rate = (
            df_jul.groupby("agent_id")["std_pass"].mean().sort_values().reset_index()
        )
        fig = px.bar(
            agent_rate, x="std_pass", y="agent_id", orientation="h",
            title="Tasa de cumplimiento por agente",
            labels={"std_pass": "% cumple SLA", "agent_id": "Agente"},
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        product_rate = (
            df_jul.groupby("product_id")["std_pass"].mean().sort_values().reset_index()
        )
        fig = px.bar(
            product_rate, x="std_pass", y="product_id", orientation="h",
            title="Tasa de cumplimiento por producto",
            labels={"std_pass": "% cumple SLA", "product_id": "Producto"},
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_rate = df_jul.groupby("dow")["std_pass"].mean().reindex(dow_order).reset_index()
        fig = px.bar(
            dow_rate, x="dow", y="std_pass",
            title="Tasa de cumplimiento por día de semana",
            labels={"std_pass": "% cumple SLA", "dow": "Día"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df_jul, x="std_pass", y="calls_handled",
            title="Volumen de llamadas vs. cumplimiento",
            labels={"std_pass": "0 = incumple, 1 = cumple", "calls_handled": "Llamadas atendidas"},
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 — Modelo predictivo (enfoque científico)
# ---------------------------------------------------------------------------
with tab_modelo:
    st.markdown(
        "Comparación de modelos sobre el set de test (predicción hacia adelante en "
        f"el tiempo, desde el {cutoff_date.date()})."
    )

    dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]

    def summarize(name, y_pred, proba):
        return {
            "modelo": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision (en riesgo)": precision_score(y_test, y_pred, zero_division=0),
            "recall (en riesgo)": recall_score(y_test, y_pred, zero_division=0),
            "f1 (en riesgo)": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
        }

    results = pd.DataFrame([
        summarize("Dummy (mayoría)", dummy_pred, dummy_proba),
        summarize("Logistic Regression", logreg.predict(X_test), logreg_proba),
        summarize("XGBoost", xgb.predict(X_test), xgb_proba),
    ]).set_index("modelo").round(3)

    st.dataframe(results, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for name, proba in [("Logistic Regression", logreg_proba), ("XGBoost", xgb_proba)]:
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash", color="gray")))
        fig.update_layout(title="Curva ROC", xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        importances = (
            pd.Series(xgb.feature_importances_, index=X_train.columns)
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
        "Un chequeo de sanidad importante: el modelo **no usa `avg_aht`** (la métrica "
        "que define el propio SLA) como feature — ver el hallazgo de data leakage en "
        "el EDA. Las features usadas son desempeño histórico del agente, volumen del "
        "día, producto, idioma y variables temporales."
    )

# ---------------------------------------------------------------------------
# Tab 4 — Simulador de costo (evaluación de negocio)
# ---------------------------------------------------------------------------
with tab_costo:
    st.markdown(
        "El umbral de decisión por defecto (0.5) es arbitrario desde la perspectiva "
        "de negocio. Ajustá los costos de cada tipo de error para ver cuál es el "
        "umbral que realmente conviene."
    )

    col_a, col_b = st.columns(2)
    cost_fn = col_a.slider(
        "Costo de un incumplimiento no detectado (falso negativo)", 1, 500, 150, step=5,
        help="Penalidad contractual + reputación cuando el modelo dice 'cumple' pero el agente incumple.",
    )
    cost_fp = col_b.slider(
        "Costo de una alerta innecesaria (falso positivo)", 1, 200, 25, step=5,
        help="Costo de reasignar personal cuando el modelo alerta pero el agente sí cumplía.",
    )

    st.caption(
        "⚠️ Estos valores son **ilustrativos** — el dataset no trae costos reales. "
        "Reemplazalos por cifras del negocio antes de usar el umbral recomendado en producción."
    )

    curve = cost_curve(y_test, xgb_proba, cost_fn, cost_fp)
    opt_threshold, opt_cost = optimal_threshold(y_test, xgb_proba, cost_fn, cost_fp)
    default_cost, _, _ = total_cost(y_test, xgb_proba, 0.5, cost_fn, cost_fp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["cost"], mode="lines", name="Costo total"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Default (0.5)")
    fig.add_vline(x=opt_threshold, line_dash="dash", line_color="green", annotation_text=f"Óptimo ({opt_threshold:.2f})")
    fig.update_layout(
        title="Costo total vs. umbral de decisión",
        xaxis_title="Umbral de decisión",
        yaxis_title=f"Costo total en test (FN={cost_fn}, FP={cost_fp})",
    )
    st.plotly_chart(fig, use_container_width=True)

    n_risk = int(y_test.sum())
    n_ok = int((y_test == 0).sum())
    cost_never = n_risk * cost_fn
    cost_always = n_ok * cost_fp

    comparison = pd.DataFrame({
        "estrategia": [
            "Sin modelo — nunca alertar",
            "Sin modelo — alertar siempre",
            "XGBoost — umbral default (0.5)",
            f"XGBoost — umbral óptimo ({opt_threshold:.2f})",
        ],
        "costo_total": [cost_never, cost_always, default_cost, opt_cost],
    })
    comparison["ahorro_%"] = ((cost_never - comparison["costo_total"]) / cost_never * 100).round(1)

    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.success(
        f"**Recomendación:** con estos costos, el umbral óptimo es **{opt_threshold:.2f}** "
        f"(costo total: {opt_cost:.0f}), un ahorro del **{comparison['ahorro_%'].iloc[-1]:.1f}%** "
        "frente a no usar ningún sistema de alerta."
    )
