# 03 — Entretenimiento: Estrategia de Contenido

> Este fue el proyecto donde más tuve que resistir la tentación de
> "completar" el dataset con algo que no estaba ahí. El catálogo de Netflix
> no trae ninguna métrica de audiencia — ni vistas, ni rating de usuarios,
> nada. Podría haber inventado un proxy de "éxito" y seguir de largo, pero
> me pareció más honesto redefinir el problema a algo que el dato sí puede
> sostener: si una serie fue renovada o no. Menos vistoso, más real.

## Contexto de negocio

Las plataformas de streaming compiten por retención y engagement mediante la
composición de su catálogo. Adquirir o producir contenido es costoso, por lo
que decidir qué tipo de contenido priorizar (género, país de origen, formato,
rating) tiene impacto directo en el retorno de esa inversión. Identificar
brechas del catálogo actual frente a tendencias de consumo permite priorizar
mejor las decisiones de adquisición/producción.

## Pregunta / problema a resolver

¿Qué tipo de contenido debería adquirir o producir la plataforma para
maximizar el engagement, e identificar en qué combinaciones de
género/país/rating existe una brecha de catálogo frente a la demanda o
tendencia observada?

## Dataset

- **Nombre:** Netflix Movies and TV Shows
- **Fuente:** Kaggle
- **Link:** https://www.kaggle.com/datasets/shivamb/netflix-shows
- **Tamaño:** 8.807 títulos (6.131 películas, 2.676 series), con metadata de
  género, país, año, rating, elenco y descripción — sin ninguna métrica de
  audiencia (vistas, ratings de usuarios, retención).

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/` como `netflix_titles.csv`.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio (ver notebooks en
`notebooks/`, en orden):

1. **Business Understanding:** "maximizar engagement" no se puede medir
   directamente con este dataset (no trae datos de audiencia) — se acotó el
   alcance a lo que la metadata sí permite: brechas de catálogo (enfoque
   analista) y un recomendador content-based + un modelo sobre una definición
   honesta de "éxito" (enfoque científico), documentando explícitamente qué
   queda fuera de alcance.
2. **Data Understanding** (`01_eda.ipynb`): sin duplicados, pero con nulos
   altos y esperables en `director` (29.9%) y `cast`/`country` (9.4% cada
   uno). Se detectó un **bug de datos real**: 3 títulos con el valor de
   duración filtrado en la columna `rating` por un corrimiento de columna en
   el CSV original. `country` y `listed_in` son campos multivaluados (748
   combinaciones de país → 127 países individuales al separar).
3. **Data Preparation** (`02_data_preparation.ipynb`): corrección del bug,
   nulos imputados como `"Unknown"` (no se elimina ninguna fila), separación
   de `duration` en `duration_min` (películas) y `n_seasons` (series), y
   construcción de `content_soup` (texto combinado) para el recomendador.
4. **Modeling** (`03_modeling.ipynb`):
   - *Enfoque analista:* brechas de catálogo por país, género, rating y año
     de incorporación (`01_eda.ipynb`, replicado en el dashboard).
   - *Enfoque científico:* recomendador content-based (TF-IDF + similitud de
     coseno sobre `content_soup`) sobre todo el catálogo, y un modelo de
     clasificación (Logistic Regression, XGBoost) sobre `renewed` — la
     redefinición honesta de "éxito": si una serie fue renovada para una
     segunda temporada, en vez de un proxy inventado para todo el catálogo.
5. **Evaluation:** el recomendador se evaluó cualitativamente (no hay forma
   cuantitativa de validarlo sin datos de audiencia); el modelo de renovación
   se evaluó con ROC-AUC/PR-AUC (no accuracy). No se construyó un notebook de
   evaluación de costo — a diferencia de los proyectos 01 y 02, acá no existe
   una matriz de costo de negocio clara para ninguna de las dos piezas.
6. **Deployment/Storytelling** (`dashboard/app.py`): dashboard en Streamlit
   con 4 vistas (resumen ejecutivo, brechas de catálogo, recomendador
   interactivo, modelo de renovación).

## Resultados clave (cuantificados)

- **Concentración fuerte de catálogo:** Estados Unidos aparece en 41.9% de
  los títulos (3.689 de 8.807), India muy por detrás (1.046) — la brecha de
  catálogo más clara del dataset.
- **El recomendador content-based funciona cualitativamente** (agrupa por
  género/tono/reparto compartido), pero **no se puede validar
  cuantitativamente** contra engagement real — limitación estructural del
  dataset, no del método.
- **El modelo de renovación tiene ROC-AUC ≈ 0.72 / PR-AUC ≈ 0.55** sobre
  2.676 series (33% de tasa de renovación) — muy por debajo del 0.99 de
  project 01 o el 0.79 de project 02, consistente con que la metadata de
  catálogo es un proxy débil de una decisión que depende de datos de
  audiencia que este dataset no tiene.
- **País y género pesan más que descripción o director** en la importancia
  de features del modelo de renovación — ninguno de los dos aparece en el
  top 10.
- **Decisión metodológica documentada:** el split de train/test para el
  modelo de renovación es aleatorio estratificado, no cronológico —
  justificado porque el sesgo de madurez del target (títulos recientes
  tuvieron menos tiempo de renovarse) contamina todo el dataset, no solo el
  futuro.

## Limitaciones

- **Sin métrica de audiencia:** es la limitación central del proyecto — no
  hay vistas, ratings de usuarios ni retención. Todo lo "predictivo" de este
  proyecto es necesariamente más débil que en project 01/02, y se documentó
  así en cada paso en vez de inflar artificialmente el alcance.
- **`renewed` tiene sesgo de madurez:** una serie de 2021 tuvo mucho menos
  tiempo para ser renovada que una de 2015, independientemente de su
  calidad — ningún split evita esto, es una limitación de la definición del
  target.
- **Las películas quedan fuera del modelo predictivo** — no existe una señal
  equivalente a la renovación para ellas en este dataset; sí participan
  plenamente del recomendador.
- **El recomendador es puramente de contenido** (no collaborative
  filtering): agrupa por similitud textual/de género, no por patrones de
  co-consumo real entre usuarios, que este dataset no puede observar.
- **Cobertura temporal parcial:** el dataset llega hasta septiembre de 2021
  — cualquier análisis de tendencia de incorporación de catálogo debe leerse
  con esa fecha de corte en mente.

## Dashboard en vivo

**[miguel-entertainment-content-strategy.streamlit.app](https://miguel-entertainment-content-strategy.streamlit.app/)**
— sin instalación, corre directo en el navegador (puede tardar unos segundos
en despertar si nadie lo visitó recientemente).

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 03-entertainment-content-strategy/data/netflix_titles.csv

# 2. Correr los notebooks en orden (cada uno alimenta al siguiente)
jupyter notebook 03-entertainment-content-strategy/notebooks/
#   01_eda.ipynb -> 02_data_preparation.ipynb -> 03_modeling.ipynb

# 3. Correr el dashboard
streamlit run 03-entertainment-content-strategy/dashboard/app.py
```
