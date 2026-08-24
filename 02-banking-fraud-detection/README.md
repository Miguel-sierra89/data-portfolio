# 02 — Banca: Detección de Fraude en Transacciones

## Contexto de negocio

Las entidades financieras enfrentan un trade-off constante en la detección de
fraude: un modelo demasiado permisivo deja pasar transacciones fraudulentas
(pérdida directa de dinero y confianza), mientras que un modelo demasiado
estricto genera fricción a clientes legítimos (bloqueos, llamadas de
verificación, abandono de producto). La decisión de negocio no es solo
"¿es fraude o no?", sino dónde fijar el umbral de decisión que minimiza el
costo total combinado de ambos tipos de error.

## Pregunta / problema a resolver

¿Qué transacciones son fraudulentas y cuál es el umbral de decisión óptimo
que minimiza el costo total del negocio, considerando explícitamente el costo
de un fraude no detectado (falso negativo) frente al costo de fricción de un
falso positivo?

## Dataset

- **Nombre:** Credit Card Fraud Detection
- **Fuente:** Université Libre de Bruxelles (ULB) vía Kaggle (dataset real)
- **Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Tamaño:** ~284.807 transacciones, 492 fraudes (~0.17% — desbalance
  extremo), 30 variables (28 componentes PCA + Time + Amount)

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/`.

## Metodología

1. **Business Understanding:** definir explícitamente el costo asociado a un
   falso negativo (fraude no detectado) vs. un falso positivo (fricción a
   cliente legítimo), como insumo para la elección del umbral de decisión.
2. **Data Understanding:** EDA de la distribución de clases (desbalance
   extremo ~0.17% de fraude), análisis de las variables PCA (V1-V28) y las
   variables originales (Time, Amount), verificación de ausencia de nulos y
   de posibles fugas de información temporal.
3. **Data Preparation:** escalado de `Amount` y `Time`, y evaluación de
   distintas estrategias de manejo de desbalance: SMOTE (oversampling
   sintético), class weights, y undersampling, comparando su efecto sobre el
   modelo final.
4. **Modeling:**
   - *Enfoque analista:* análisis de la distribución de clases y del
     costo-beneficio de distintos umbrales de decisión sobre el score del
     modelo.
   - *Enfoque científico:* modelos de clasificación supervisada (regresión
     logística, XGBoost) con manejo de desbalance extremo, complementados con
     un enfoque no supervisado (Isolation Forest) para detección de anomalías;
     interpretabilidad de los componentes PCA mediante SHAP.
5. **Evaluation:** curva Precision-Recall (más informativa que ROC-AUC bajo
   desbalance extremo), y análisis de costo total esperado por umbral de
   decisión — nunca solo accuracy, que es engañosa en este contexto.
6. **Deployment/Storytelling:** dashboard en Streamlit que permita simular el
   umbral de decisión y visualizar su impacto en el costo total + este README
   como resumen ejecutivo.

## Resultados clave (cuantificados)

_Pendiente — se completará una vez ejecutado el análisis y el modelado._

## Limitaciones

_Pendiente — se documentarán las limitaciones del dataset y del modelo tras
el EDA (p. ej. variables anonimizadas vía PCA limitan la interpretabilidad de
negocio directa, ventana temporal acotada del dataset original)._

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 02-banking-fraud-detection/data/

# 2. Explorar el EDA
jupyter notebook 02-banking-fraud-detection/notebooks/

# 3. Correr el dashboard (una vez desarrollado)
streamlit run 02-banking-fraud-detection/dashboard/app.py
```
