"""Dashboard ejecutivo — Banking Fraud Detection.

Storytelling del proyecto para una audiencia no técnica: resume los hallazgos
de los 4 notebooks (EDA, preparación, modelado, evaluación de costo) en una
sola app interactiva. Ejecutar con:

    streamlit run dashboard/app.py

desde la carpeta 02-banking-fraud-detection/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import build_model_dataset, get_xy  # noqa: E402
from src.modeling import cost_curve, optimal_threshold, score, total_cost, train_all_models  # noqa: E402

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

st.set_page_config(page_title="Banking Fraud Detection", page_icon="💳", layout="wide")


@st.cache_data
def load_data():
    return build_model_dataset(str(DATA_PATH))


@st.cache_resource
def get_trained_models(train_df: pd.DataFrame):
    X_train, y_train = get_xy(train_df)
    return train_all_models(X_train, y_train)


@st.cache_data
def get_shap_importance(_model, X_test: pd.DataFrame, sample_size: int = 2000):
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(sample)
    return pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Carga de datos y modelos
# ---------------------------------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        f"No se encontró el dataset en `{DATA_PATH}`.\n\n"
        "Descargalo desde https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
        "y colocá el CSV en `02-banking-fraud-detection/data/creditcard.csv`."
    )
    st.stop()

with st.spinner("Cargando datos y entrenando modelos (una sola vez por sesión)..."):
    df, train_df, test_df, cutoff_time = load_data()
    models = get_trained_models(train_df)

X_train, y_train = get_xy(train_df)
X_test, y_test = get_xy(test_df)
amounts_test = test_df["Amount"].values

proba = {name: score(m, X_test, name) for name, m in models.items()}

st.title("💳 Banking — Detección de Fraude en Transacciones")
st.caption(
    "¿Qué transacciones son fraudulentas, y cuál es el umbral de decisión que minimiza "
    "el costo total (fraude no detectado vs. fricción por falso positivo)?"
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
    st.info(
        "ℹ️ Dataset real (Credit Card Fraud Detection, ULB) con las variables originales "
        "anonimizadas por PCA (`V1`–`V28`). El modelo puede decir *cuánto* pesa cada "
        "componente en el riesgo, pero no *qué* variable de negocio representa."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transacciones (deduplicadas)", f"{len(df):,}")
    c2.metric("Tasa de fraude", f"{df['Class'].mean() * 100:.3f}%")
    c3.metric("Fraudes totales", f"{df['Class'].sum():,}")
    c4.metric("Período cubierto", "~48 horas")

    st.markdown("### Hallazgos clave")
    st.markdown(
        """
- **Desbalance extremo (0.17% fraude):** accuracy queda descartada de entrada — todo el
  proyecto usa **PR-AUC** como métrica principal.
- **XGBoost + SMOTE gana la comparación de modelos** (PR-AUC ≈ 0.79), por encima de
  class weights (0.78) y muy por encima de un enfoque no supervisado (Isolation Forest,
  0.04) — ver la pestaña *Modelo predictivo*.
- **El costo de un fraude no detectado es real, no ilustrativo:** se usó el monto real
  de cada transacción (`Amount`), no un número inventado — solo el costo de investigar
  una alerta es un supuesto ajustable en la pestaña *Simulador de costo*.
- **Hallazgo contraintuitivo:** el umbral óptimo de negocio resultó más alto que el
  0.5 por defecto — subirlo filtra falsas alarmas sin perder fraude adicional, porque
  cada alerta (acierto o error) cuesta lo mismo en revisión.
- **\"Alertar siempre\" es la peor estrategia posible** — investigar cada transacción
  sale mucho más caro que el fraude total que se busca prevenir.
        """
    )

    with st.expander("Metodología completa (notebooks)"):
        st.markdown(
            """
1. `01_eda.ipynb` — Data Understanding: duplicados, desbalance, outliers en componentes PCA.
2. `02_data_preparation.ipynb` — dedup antes del split, features temporales, split cronológico.
3. `03_modeling.ipynb` — Logistic Regression, XGBoost (class weights y SMOTE), Isolation Forest.
4. `04_evaluation.ipynb` — umbral óptimo según costo real de negocio.
            """
        )

# ---------------------------------------------------------------------------
# Tab 2 — Análisis descriptivo (enfoque analista)
# ---------------------------------------------------------------------------
with tab_analista:
    col1, col2 = st.columns(2)

    with col1:
        class_counts = df["Class"].value_counts().rename({0: "Legítima", 1: "Fraude"})
        fig = px.bar(
            x=class_counts.index, y=class_counts.values, log_y=True,
            title="Distribución de clases (escala log)",
            labels={"x": "", "y": "# transacciones"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        rate_by_hour = df.assign(hour=(df["Time"] % 86400) // 3600).groupby("hour")["Class"].mean()
        fig = px.bar(
            x=rate_by_hour.index, y=rate_by_hour.values,
            title="Tasa de fraude por hora del día",
            labels={"x": "Hora del día", "y": "% fraude"},
        )
        fig.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        sample_plot = df.sample(min(30000, len(df)), random_state=42)
        fig = px.box(
            sample_plot, x="Class", y="Amount", log_y=True,
            title="Monto por clase (muestra de 30k, escala log)",
            labels={"Class": "0 = legítima, 1 = fraude"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        v_cols = [c for c in df.columns if c.startswith("V")]
        corr = df[v_cols + ["Class"]].corr()["Class"].drop("Class").sort_values()
        top = pd.concat([corr.head(4), corr.tail(4)])
        fig = px.bar(
            x=top.values, y=top.index, orientation="h",
            title="Componentes PCA más correlacionadas con fraude",
            labels={"x": "Correlación con Class", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Nota: se eliminaron 1.081 filas duplicadas exactas del dataset original antes de "
        "cualquier análisis (19 de ellas fraude) — ver `01_eda.ipynb` y `02_data_preparation.ipynb`."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Modelo predictivo (enfoque científico)
# ---------------------------------------------------------------------------
with tab_modelo:
    st.markdown(
        f"Comparación de modelos sobre el set de test (predicción hacia adelante en el "
        f"tiempo, desde `Time = {cutoff_time:.0f}s`)."
    )

    results = pd.DataFrame({
        "modelo": list(models.keys()),
        "pr_auc": [average_precision_score(y_test, proba[name]) for name in models],
        "roc_auc": [roc_auc_score(y_test, proba[name]) for name in models],
    }).set_index("modelo").sort_values("pr_auc", ascending=False).round(4)

    st.dataframe(results, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        for name in models:
            precision, recall, _ = precision_recall_curve(y_test, proba[name])
            ap = average_precision_score(y_test, proba[name])
            fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AP={ap:.3f})"))
        fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="gray", annotation_text="Azar")
        fig.update_layout(title="Curva Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        importances = get_shap_importance(models["XGBoost + SMOTE"], X_test).head(10).iloc[::-1]
        fig = px.bar(
            x=importances.values, y=importances.index, orientation="h",
            title="Top 10 features — SHAP (XGBoost + SMOTE)",
            labels={"x": "Importancia SHAP media (|valor|)", "y": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Isolation Forest** (no supervisado) queda muy por detrás en PR-AUC — esperable, "
        "ya que no usa las etiquetas de fraude para entrenar. Su valor real no es ganar esta "
        "comparación, sino poder detectar patrones de fraude *nuevos*, no vistos en el "
        "historial etiquetado."
    )

# ---------------------------------------------------------------------------
# Tab 4 — Simulador de costo (evaluación de negocio)
# ---------------------------------------------------------------------------
with tab_costo:
    best_model_name = "XGBoost + SMOTE"
    best_proba = proba[best_model_name]

    st.markdown(
        f"Umbral de decisión para **{best_model_name}** (el modelo ganador). El costo de un "
        "fraude no detectado es el **monto real** de la transacción — solo el costo de "
        "investigar una alerta es ajustable abajo."
    )

    admin_cost = st.slider(
        "Costo de investigar una transacción marcada (ADMIN_COST)", 1, 200, 10, step=1,
        help="Costo de tiempo de analista / fricción con el cliente al revisar CUALQUIER transacción marcada, sea acierto o falsa alarma.",
    )
    st.caption(
        "⚠️ `ADMIN_COST` es ilustrativo — el dataset no trae ese costo real. El monto "
        "de fraude perdido (`Amount`) sí es real."
    )

    curve = cost_curve(y_test, best_proba, amounts_test, admin_cost)
    opt_threshold, opt_cost = optimal_threshold(y_test, best_proba, amounts_test, admin_cost)
    default_cost, _, _ = total_cost(y_test, best_proba, amounts_test, 0.5, admin_cost)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["cost"], mode="lines", name="Costo total"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Default (0.5)")
    fig.add_vline(x=opt_threshold, line_dash="dash", line_color="green", annotation_text=f"Óptimo ({opt_threshold:.2f})")
    fig.update_layout(
        title="Costo total vs. umbral de decisión",
        xaxis_title="Umbral de decisión",
        yaxis_title=f"Costo total en test (ADMIN_COST={admin_cost})",
    )
    st.plotly_chart(fig, use_container_width=True)

    cost_never = amounts_test[y_test.values == 1].sum()
    cost_always = admin_cost * len(y_test)

    comparison = pd.DataFrame({
        "estrategia": [
            "Sin modelo — nunca alertar",
            "Sin modelo — alertar siempre",
            f"{best_model_name} — umbral default (0.5)",
            f"{best_model_name} — umbral óptimo ({opt_threshold:.2f})",
        ],
        "costo_total": [cost_never, cost_always, default_cost, opt_cost],
    })
    comparison["ahorro_%"] = ((cost_never - comparison["costo_total"]) / cost_never * 100).round(1)

    st.dataframe(
        comparison.style.format({"costo_total": "${:,.2f}", "ahorro_%": "{:.1f}%"}),
        use_container_width=True, hide_index=True,
    )

    st.success(
        f"**Recomendación:** con `ADMIN_COST = {admin_cost}`, el umbral óptimo es "
        f"**{opt_threshold:.2f}** (costo total: ${opt_cost:,.2f}), un ahorro del "
        f"**{comparison['ahorro_%'].iloc[-1]:.1f}%** frente a nunca alertar."
    )
