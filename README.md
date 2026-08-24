# Data Portfolio — Analista de Datos & Científico de Datos

Portafolio de 5 proyectos end-to-end que demuestran el ciclo completo de trabajo
con datos: desde la pregunta de negocio hasta un modelo validado y comunicado a
una audiencia no técnica.

Cada proyecto sigue el mismo framework metodológico (CRISP-DM adaptado) y
separa explícitamente dos roles dentro del mismo flujo de trabajo:

- **Enfoque analista (descriptivo/diagnóstico):** entender qué pasó y por qué,
  mediante EDA, segmentación y dashboards.
- **Enfoque científico de datos (predictivo/prescriptivo):** modelar, predecir
  y prescribir una acción, con validación estadística y métricas atadas al
  costo de negocio.

## Framework metodológico (aplicado en todos los proyectos)

1. **Business Understanding** — la pregunta se formula como una decisión de
   negocio, no como una tarea técnica.
2. **Data Understanding** — EDA riguroso: nulos, outliers, cardinalidad y
   detección temprana de data leakage.
3. **Data Preparation** — limpieza, feature engineering, manejo de
   desbalance de clases.
4. **Modeling** — separación clara entre el análisis descriptivo/diagnóstico
   y el modelo predictivo/prescriptivo.
5. **Evaluation** — métricas alineadas al costo de negocio (no solo
   accuracy).
6. **Deployment/Storytelling** — dashboard (Streamlit) + README ejecutivo
   para una audiencia no técnica.

## Stack técnico

Python · pandas · scikit-learn · XGBoost · SHAP · sentence-transformers ·
FAISS · Streamlit · FastAPI

## Proyectos

| # | Proyecto | Dominio | Pregunta de negocio | Enfoque analista | Enfoque científico |
|---|----------|---------|----------------------|-------------------|---------------------|
| 1 | [BPO — SLA Performance](01-bpo-sla-performance/) | Call Center / Workforce Planning | ¿Cómo reducimos el % de tickets que incumplen SLA y cómo optimizamos la dotación de personal? | Volumen por canal/hora, tasa de incumplimiento por agente/cola, teoría de colas | Clasificación de riesgo de incumplimiento de SLA |
| 2 | [Banca — Detección de Fraude](02-banking-fraud-detection/) | Fintech / Riesgo | ¿Qué transacciones son fraudulentas y cuál es el umbral de decisión que minimiza el costo total? | Distribución de clases, análisis costo-beneficio del umbral | Clasificación con desbalance extremo (SMOTE, class weights, Isolation Forest), SHAP sobre PCA |
| 3 | [Entretenimiento — Estrategia de Contenido](03-entertainment-content-strategy/) | Streaming / Producto | ¿Qué contenido debería adquirir o producir la plataforma para maximizar engagement? | EDA de brechas de catálogo por género/país/rating | Recomendador content-based + modelo de probabilidad de éxito |
| 4 | [Jira — Estimación de Esfuerzo](04-jira-effort-estimation/) | Ingeniería / Agile | ¿Cómo automatizamos la estimación de story points para dejar de romper la planificación de sprints? | Cycle time, lead time, burndown, cuellos de botella | NLP (embeddings + regresión) para predecir story points |
| 5 | [IA Aplicada — Matching de Candidatos con RAG](05-ai-resume-matching-rag/) | RRHH / IA Aplicada | ¿Cómo automatizamos el matching semántico entre vacantes y CVs con explicabilidad del ranking? | — | Embeddings + FAISS + LLM para extracción de skills y justificación del ranking (servicio FastAPI) |

## Cómo navegar este repo

Cada carpeta de proyecto (`0N-nombre-proyecto/`) es autocontenida y tiene:

```
0N-nombre-proyecto/
├── README.md       # contexto, pregunta, dataset, metodología, resultados, limitaciones, cómo correr
├── data/            # vacío en el repo — cada README indica cómo descargar el dataset
├── notebooks/        # EDA y experimentación
├── src/              # código productivizado (limpieza, features, modelos)
└── dashboard/ o app/  # Streamlit o FastAPI
```

Empieza por el `README.md` de cada proyecto — está escrito para poder leerse
de forma independiente, como si fuera un informe ejecutivo.

## Setup general

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Cada proyecto puede tener dependencias adicionales específicas documentadas
en su propio README.

## Estado del portafolio

Los 5 proyectos están inicializados con su contexto de negocio, pregunta,
dataset y metodología definidos. Los resultados cuantificados se irán
completando a medida que se ejecute el análisis y modelado de cada uno.
