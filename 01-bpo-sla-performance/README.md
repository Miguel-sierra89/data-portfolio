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
- **Tamaño:** por confirmar tras la descarga (pendiente de EDA inicial)

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/`.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio:

1. **Business Understanding:** formalizar el incumplimiento de SLA como
   variable objetivo binaria (cumple/no cumple) y definir el costo de negocio
   asociado a cada tipo de error.
2. **Data Understanding:** EDA de nulos, outliers en tiempos de espera/manejo,
   cardinalidad de agentes/colas/canales, y verificación de que no existan
   variables que filtren información posterior al momento de la decisión
   (data leakage, ej. duración total de la llamada si el SLA se define sobre
   el tiempo de espera).
3. **Data Preparation:** limpieza de timestamps, creación de variables
   temporales (hora, día de semana, franja pico), agregaciones por
   agente/cola/canal, y evaluación de desbalance de clases en la variable de
   incumplimiento.
4. **Modeling:**
   - *Enfoque analista:* dashboard descriptivo de volumen por canal y hora,
     tasa de incumplimiento por agente y por cola, y una aproximación de
     teoría de colas básica (Erlang C / relación entre volumen, dotación y
     tiempo de espera esperado).
   - *Enfoque científico:* modelo de clasificación (regresión logística como
     baseline, XGBoost como modelo principal) para predecir el riesgo de
     incumplimiento de SLA por ticket/llamada.
5. **Evaluation:** métricas de clasificación (precision, recall, F1, AUC-ROC)
   interpretadas junto al costo real de sub-dotación (SLA incumplido) vs.
   sobre-dotación (costo de personal ocioso), no solo accuracy.
6. **Deployment/Storytelling:** dashboard en Streamlit para el equipo de
   workforce planning + este README como resumen ejecutivo.

## Resultados clave (cuantificados)

_Pendiente — se completará una vez ejecutado el análisis y el modelado._

## Limitaciones

_Pendiente — se documentarán las limitaciones del dataset y del modelo tras
el EDA (p. ej. representatividad temporal, granularidad de canales, sesgos
en la asignación de agentes)._

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 01-bpo-sla-performance/data/

# 2. Explorar el EDA
jupyter notebook 01-bpo-sla-performance/notebooks/

# 3. Correr el dashboard (una vez desarrollado)
streamlit run 01-bpo-sla-performance/dashboard/app.py
```
