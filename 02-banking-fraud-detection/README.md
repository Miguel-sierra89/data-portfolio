# 02 — Banca: Detección de Fraude en Transacciones

> Este es el dataset clásico de fraude (el de ULB), así que no esperaba
> sorpresas de calidad de datos. Igual apareció una: 1.081 filas duplicadas
> exactas, 19 de ellas fraude — nada grave, pero suficiente para arruinar
> la evaluación si se deduplica después de separar train/test en vez de
> antes. Es el tipo de detalle que se pasa por alto fácil si uno da por
> sentado que un dataset "conocido" ya viene limpio.

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
- **Tamaño:** 284.807 transacciones crudas (283.726 tras eliminar 1.081
  duplicados exactos), 473 fraudes (~0.167% — desbalance extremo), 30
  variables (28 componentes PCA anonimizados + `Time` + `Amount`), cubriendo
  ~48 horas continuas de transacciones.

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/` como `creditcard.csv`.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio (ver notebooks en
`notebooks/`, en orden):

1. **Business Understanding:** el costo de un falso negativo (fraude no
   detectado) se definió como el **monto real de la transacción** (`Amount`),
   no un valor inventado — el dataset lo permite. Solo el costo de investigar
   una alerta (`ADMIN_COST`) quedó como supuesto ilustrativo, ajustable en el
   dashboard.
2. **Data Understanding** (`01_eda.ipynb`): sin nulos, pero **1.081 filas
   duplicadas exactas** (19 de fraude). Se confirmó que la señal de fraude en
   los componentes PCA (`V17`, `V14`, `V12`) se concentra justo en los
   valores extremos, y se detectó un patrón horario (más fraude de
   madrugada).
3. **Data Preparation** (`02_data_preparation.ipynb`): deduplicación **antes**
   de separar train/test (para no filtrar duplicados entre ambos conjuntos),
   dos features nuevas (`hour_of_day`, `log_amount`), **sin** recorte de
   outliers en `V1`–`V28` (destruiría la señal), y split cronológico 80/20
   por `Time`.
4. **Modeling** (`03_modeling.ipynb`):
   - *Enfoque analista:* distribución de clases, patrón horario de fraude,
     distribución de montos por clase (`01_eda.ipynb`, replicado en el
     dashboard).
   - *Enfoque científico:* Logistic Regression + class weights, XGBoost +
     class weights, XGBoost + SMOTE, e Isolation Forest (no supervisado),
     comparados por **PR-AUC** — no accuracy, inútil con este desbalance.
     Interpretabilidad de los componentes PCA vía SHAP sobre el modelo
     ganador.
5. **Evaluation** (`04_evaluation.ipynb`): curva Precision-Recall y umbral de
   decisión optimizado contra una matriz de costo real (pérdida = monto de
   la transacción; alerta = costo fijo de investigación), no accuracy ni el
   umbral por defecto de 0.5.
6. **Deployment/Storytelling** (`dashboard/app.py`): dashboard en Streamlit
   con 4 vistas (resumen ejecutivo, análisis descriptivo, modelo predictivo
   con SHAP, simulador de costo interactivo con `ADMIN_COST` ajustable).

## Resultados clave (cuantificados)

- **XGBoost + SMOTE gana la comparación de modelos por PR-AUC = 0.791**,
  apenas por encima de XGBoost + class weights (0.783) y de Logistic
  Regression (0.767) — todos muy por encima del baseline trivial
  (PR-AUC ≈ prevalencia = 0.0013).
- **Isolation Forest (no supervisado) queda muy por detrás** (PR-AUC = 0.037,
  ~28x el baseline pero lejos de los modelos supervisados) — esperable, ya
  que nunca ve las etiquetas de fraude durante el entrenamiento. Su valor no
  está en ganar esta comparación, sino en poder detectar patrones de fraude
  *nuevos*, no vistos en el historial etiquetado.
- **En el umbral de costo óptimo (0.96, con `ADMIN_COST=10` ilustrativo), el
  costo total se reduce 60.6%** frente a no alertar nunca (de $7.727,67 a
  $3.041,01 en el set de test) — y **"alertar siempre" resultó ser la peor
  estrategia posible** ($567.460, muchísimo peor que no hacer nada), el
  argumento cuantitativo más claro a favor de usar el modelo.
- **Hallazgo contraintuitivo:** el umbral óptimo (0.96) es más alto que el
  0.5 por defecto, no más bajo — como cada alerta (acierto o error) cuesta
  lo mismo en revisión, subir el umbral filtra falsas alarmas (92→64
  transacciones marcadas) sin perder ni un fraude adicional (17 no
  detectados en ambos casos).
- **SHAP** identifica a `V4`, `V14` y `V12` como los componentes PCA de
  mayor peso en el modelo ganador, consistente con las correlaciones más
  fuertes ya detectadas en el EDA.

## Limitaciones

- **Componentes PCA anonimizados:** `V1`–`V28` no tienen significado de
  negocio recuperable. SHAP puede decir *cuánto* pesa cada componente, pero
  nunca *qué* variable original (comercio, ubicación, tipo de tarjeta, etc.)
  dispara el riesgo — a diferencia de project 01, acá no se puede verificar
  con certeza total la ausencia de leakage en esas variables.
- **`ADMIN_COST` es ilustrativo:** el costo del fraude en sí (`Amount`) es
  real, pero el costo de investigar una alerta es un supuesto — debe
  reemplazarse por la cifra real del equipo de operaciones antes de fijar
  un umbral en producción (la sección de sensibilidad de
  `04_evaluation.ipynb` y el dashboard muestran cómo cambia la recomendación
  según ese valor).
- **Ventana temporal corta:** el dataset cubre solo ~48 horas continuas — el
  patrón horario de fraude detectado en el EDA es una hipótesis razonable,
  no un patrón confirmado con múltiples ciclos día/noche.
- **El modelo ganador está lejos de ser perfecto** (PR-AUC 0.79, no 0.99
  como en project 01): en el umbral óptimo sigue dejando pasar 17 de 74
  fraudes del período de test. El ahorro de costo mostrado asume que el
  modelo mantiene ese desempeño fuera del período evaluado, lo cual debería
  revalidarse con más historia antes de producción.

## Dashboard en vivo

**[miguel-banking-frauddetection.streamlit.app](https://miguel-banking-frauddetection.streamlit.app/)**
— sin instalación, corre directo en el navegador (puede tardar unos segundos
en despertar si nadie lo visitó recientemente).

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 02-banking-fraud-detection/data/creditcard.csv

# 2. Correr los notebooks en orden (cada uno alimenta al siguiente)
jupyter notebook 02-banking-fraud-detection/notebooks/
#   01_eda.ipynb -> 02_data_preparation.ipynb -> 03_modeling.ipynb -> 04_evaluation.ipynb

# 3. Correr el dashboard
streamlit run 02-banking-fraud-detection/dashboard/app.py
```
