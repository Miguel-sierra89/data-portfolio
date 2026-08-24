# 01 — BPO: Análisis de Performance y Cumplimiento de SLA

## Contexto de negocio

En un centro de contacto (BPO/Call Center), el incumplimiento del SLA
(Service Level Agreement — típicamente % de llamadas/tickets atendidos dentro
de un tiempo objetivo) tiene un costo directo: penalidades contractuales con
el cliente, pérdida de reputación y fuga de clientes finales insatisfechos.
El equipo de Workforce Planning necesita entender qué combinación de factores
(agente, canal, franja horaria, volumen de llamadas) empuja el incumplimiento,
para poder anticiparlo y ajustar la dotación de personal antes de que ocurra,
en lugar de reaccionar después del hecho.

## Pregunta / problema a resolver

¿Cómo reducimos el porcentaje de tickets/llamadas que incumplen el SLA, y qué
factores (agente, canal, hora del día, volumen) predicen mejor ese
incumplimiento, de forma que el equipo de workforce planning pueda optimizar
la dotación de personal por turno?

## Dataset

- **Nombre:** Call Center Metrics Dataset
- **Fuente:** Kaggle (dataset real, anonimizado)
- **Link:** https://www.kaggle.com/datasets/unifrancouni/call-center-metrics-dataset
- **Tamaño:** 270 filas × 7 columnas — una fila por combinación
  agente × fecha × producto × idioma, con volumen (`calls_handled`), tiempo
  promedio de manejo (`avg_aht`) y el flag de cumplimiento (`std_pass`).
  Cubre enero 2020 (27 filas, 1 agente piloto) y julio 2020 (243 filas, los
  10 agentes en paralelo).

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/` como `call_metrics_dataset.csv`.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio (ver notebooks en
`notebooks/`, en orden):

1. **Business Understanding:** se redefinió el target como `sla_risk` (riesgo
   de *incumplir*, en vez de "cumple") para que el recall del modelo se
   alinee directamente con lo que le importa al negocio: cuántos
   incumplimientos reales se detectan a tiempo.
2. **Data Understanding** (`01_eda.ipynb`): sin nulos ni duplicados. Se
   detectó **data leakage**: `avg_aht` determina casi por completo
   `std_pass` (es la métrica contra la que se mide el propio SLA), por lo
   que se excluyó como feature. También se identificó que la aparente
   "inconsistencia de granularidad" entre enero y julio era en realidad la
   expansión de un piloto de 1 agente al equipo completo de 10.
3. **Data Preparation** (`02_data_preparation.ipynb`): reemplazo de
   `avg_aht` por features históricas del agente calculadas solo con datos
   *anteriores* a cada fila (`agent_prior_pass_rate`, `agent_prior_avg_calls`,
   vía ventana expandiente), one-hot encoding de categóricas, y **split
   train/test cronológico** (no aleatorio) para evaluar predicción hacia
   adelante en el tiempo.
4. **Modeling** (`03_modeling.ipynb`):
   - *Enfoque analista:* tasa de incumplimiento por agente, producto,
     idioma y día de semana; volumen vs. cumplimiento (`01_eda.ipynb`
     sección 11, replicado en el dashboard).
   - *Enfoque científico:* Logistic Regression (baseline) vs. XGBoost
     (principal), comparados contra un baseline trivial (DummyClassifier) y
     validados con Stratified K-Fold sobre train.
5. **Evaluation** (`04_evaluation.ipynb`): el umbral de decisión se optimizó
   contra un costo de negocio explícito (costo de un incumplimiento no
   detectado vs. costo de una alerta innecesaria), no por accuracy ni por el
   umbral por defecto de 0.5.
6. **Deployment/Storytelling** (`dashboard/app.py`): dashboard en Streamlit
   con 4 vistas (resumen ejecutivo, análisis descriptivo, modelo predictivo,
   simulador de costo interactivo) para el equipo de workforce planning.

## Resultados clave (cuantificados)

- **XGBoost predice el riesgo de incumplimiento con ROC-AUC = 0.99 y
  PR-AUC = 0.99** en el set de test (predicción hacia adelante en el
  tiempo), superando a Logistic Regression (ROC-AUC 0.97) y al baseline
  trivial (ROC-AUC 0.50, 0% recall por construcción).
- **Recall de 97–100% en la clase "en riesgo"** con precision de 89–90% en
  el umbral óptimo — el modelo detecta prácticamente todos los
  incumplimientos reales del período de test.
- **El umbral óptimo de negocio es 0.44, no el 0.5 por defecto**, bajo
  supuestos de costo ilustrativos (incumplimiento no detectado = 150,
  alerta innecesaria = 25). En ese umbral, el costo total se reduce **~98%**
  frente a no usar ningún sistema de alerta.
- **Hallazgo que corrigió una hipótesis inicial:** se esperaba que el
  historial del agente dominara la importancia de features; en cambio,
  `calls_handled` (volumen del día) y `product_id` encabezan el ranking —
  el riesgo está más ligado al volumen y al producto/cola que al agente.

## Limitaciones

- **Dataset muy pequeño** (171 filas de train, 99 de test). Las métricas
  del modelo son inusualmente altas para ese tamaño de muestra — es tan
  consistente con señal real como con sobreajuste o con que el dataset sea
  sintético/simplificado. No debe tratarse como una predicción lista para
  producción sin validar contra más meses de datos.
- **Costos de negocio ilustrativos:** `COST_FN` y `COST_FP` usados en la
  evaluación (y ajustables en el dashboard) son valores de ejemplo, no
  cifras reales de penalidad contractual ni de costo operativo — deben
  reemplazarse antes de fijar un umbral en producción.
- **El agente piloto (agente 3) no tiene filas en el período de test**
  (las últimas fechas de julio), así que su desempeño específico no puede
  validarse en el holdout, aunque sí aporta historial de entrenamiento.
- **Sin columnas de canal ni de hora del día** (a diferencia del alcance
  inicial planteado) — el dataset real está agregado a nivel diario por
  agente/producto/idioma, así que el análisis de teoría de colas y de
  franjas horarias no fue posible con este dataset.

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 01-bpo-sla-performance/data/call_metrics_dataset.csv

# 2. Correr los notebooks en orden (cada uno alimenta al siguiente)
jupyter notebook 01-bpo-sla-performance/notebooks/
#   01_eda.ipynb -> 02_data_preparation.ipynb -> 03_modeling.ipynb -> 04_evaluation.ipynb

# 3. Correr el dashboard
streamlit run 01-bpo-sla-performance/dashboard/app.py
```
