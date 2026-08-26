# Data Portfolio — Analista de Datos & Científico de Datos

Portafolio de 5 proyectos end-to-end que demuestran el ciclo completo de trabajo
con datos: desde la pregunta de negocio hasta un modelo validado y comunicado a
una audiencia no técnica.

## Sobre mí

Soy Miguel Alejandro Sierra Jaramillo. Estoy formándome en data a mi manera:
armando proyectos propios, de punta a punta, para vincularme de verdad con
este mundo — no solo para completar un curso. Lo que más me mueve es la idea
de ayudar a que los negocios tomen mejores decisiones con lo que ya tienen en
sus datos, para que puedan progresar.

[LinkedIn](https://www.linkedin.com/in/miguel-sierra-a20a18a0/) · [GitHub](https://github.com/Miguel-sierra89) · miguelalejandrosierra@gmail.com

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

| # | Proyecto | Dominio | Pregunta de negocio | Enfoque científico | Resultado clave | Dashboard |
|---|----------|---------|----------------------|---------------------|------------------|-----------|
| 1 | [BPO — SLA Performance](01-bpo-sla-performance/) | Call Center / Workforce Planning | ¿Cómo reducimos el % de tickets que incumplen SLA y cómo optimizamos la dotación de personal? | XGBoost para riesgo de incumplimiento + umbral óptimo por costo de negocio | ROC-AUC 0.99, ~98% de reducción de costo vs. no usar modelo (dataset chico, ver limitaciones) | [en vivo](https://miguel-bpo-sla.streamlit.app/) |
| 2 | [Banca — Detección de Fraude](02-banking-fraud-detection/) | Fintech / Riesgo | ¿Qué transacciones son fraudulentas y cuál es el umbral de decisión que minimiza el costo total? | XGBoost + SMOTE vs. class weights vs. Isolation Forest, matriz de costo real (monto real de fraude) | PR-AUC 0.791, 60.6% de reducción de costo — "alertar siempre" resultó ser peor que no hacer nada | [en vivo](https://miguel-banking-frauddetection.streamlit.app/) |
| 3 | [Entretenimiento — Estrategia de Contenido](03-entertainment-content-strategy/) | Streaming / Producto | ¿Qué contenido debería adquirir o producir la plataforma para maximizar engagement? | Recomendador content-based (TF-IDF + coseno) + modelo de renovación de series | ROC-AUC 0.72 — modesto pero honesto: sin datos de audiencia, "éxito" se redefinió como renovación real | [en vivo](https://miguel-entertainment-content-strategy.streamlit.app/) |
| 4 | [Jira — Estimación de Esfuerzo](04-jira-effort-estimation/) | Ingeniería / Agile | ¿Cómo automatizamos la estimación de story points para dejar de romper la planificación de sprints? | Embeddings + regresión (Ridge), pooled vs. 16 modelos por proyecto | MAE ≈ 3 story points; ni pooled ni por-proyecto gana siempre — depende de la escala de cada equipo | [en vivo](https://miguel-jira-effort-estimation.streamlit.app/) |
| 5 | [IA Aplicada — Matching de Candidatos con RAG](05-ai-resume-matching-rag/) | RRHH / IA Aplicada | ¿Cómo automatizamos el matching semántico entre vacantes y CVs con explicabilidad del ranking? | Embeddings + FAISS, explicabilidad por overlap de skills (sin LLM, por diseño), servicio FastAPI | Spearman 0.30 vs. referencia externa; `POST /match/text` matchea vacantes en texto libre en vivo | [en vivo](https://ai-resume-matching-rag.streamlit.app/) |

Los 5 dashboards están desplegados en Streamlit Community Cloud (capa
gratuita) — si nadie los visitó en un rato, tardan unos segundos en
"despertar" la primera vez que se abren. El proyecto 5 despliega una demo
del motor de matching en Streamlit; el servicio FastAPI en sí (`app/main.py`)
se corre local — ver su README para el porqué.

Los 5 datasets planeados originalmente no siempre coincidieron con lo disponible en la práctica — dos casos concretos quedaron documentados como decisiones explícitas, no como atajos silenciosos: en **project 04** el dataset de Zenodo pesaba 13.8 TB y requería login, así que se sustituyó por el dataset académico de 23.313 issues que originó esa línea de investigación; en **project 05** el CSV descargado resultó ser pares candidato-vacante con `matched_score` ya calculado, en vez de CVs sueltos — un punto de partida mejor, que cambió el diseño del pipeline. Cada README de proyecto explica el porqué.

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

**Los 5 proyectos están completos de punta a punta:** EDA → preparación de
datos → modelado → evaluación (donde aplicaba una matriz de costo real) →
storytelling (dashboard en Streamlit o servicio FastAPI) → README con
resultados y limitaciones reales, no plantillas pendientes.

Algunos denominadores comunes entre los 5, más allá de los números de cada
tabla:

- **Cada proyecto encontró al menos un hallazgo real de calidad de datos**
  que cambió el diseño del pipeline (leakage en project 01, duplicados en
  project 02, un bug de columnas corridas en project 03, un dataset
  reemplazado en project 04, campos serializados como listas de Python en
  project 05) — documentados como decisiones explícitas, no escondidos.
- **Ninguna métrica se presenta sin su contexto de negocio:** accuracy no
  aparece como métrica principal en ningún proyecto con desbalance de
  clases; los umbrales de decisión se optimizan contra costo, no se dejan
  en el 0.5 por defecto.
- **Las limitaciones son específicas de cada proyecto, no genéricas** — cada
  README dice explícitamente qué no se puede afirmar con ese dataset.
