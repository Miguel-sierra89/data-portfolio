# 04 — Jira: Estimación de Esfuerzo en Equipos Ágiles

## Contexto de negocio

En equipos ágiles, la estimación de esfuerzo (story points) suele hacerse de
forma manual y subjetiva (planning poker), y es una fuente recurrente de
subestimación sistemática: los equipos consistentemente estiman por debajo
del esfuerzo real, lo que rompe la planificación de sprints, genera
sobrecarga de trabajo y afecta la previsibilidad de entregas frente a
negocio. Automatizar una estimación inicial basada en el texto del issue
puede servir como punto de referencia objetivo que reduzca ese sesgo.

## Pregunta / problema a resolver

¿Cómo automatizamos la estimación de story points a partir del texto de los
issues, para reducir la subestimación sistemática que rompe la planificación
de sprints en equipos ágiles?

## Dataset

- **Nombre:** Story Point Estimation Dataset (16 proyectos open-source)
- **Fuente:** el Public Jira Dataset original de Zenodo pesa **13.8 TB** y
  requiere login — inviable para un proyecto de portafolio (ver
  `01_eda.ipynb` para el detalle). Se usa en su lugar el dataset académico
  que originó esa línea de investigación (Choetkiertikul et al., *"A Deep
  Learning Model for Estimating Story Points"*, IEEE TSE 2018), público y
  sin login: https://github.com/morakotch/datasets
  (`storypoint/IEEE TSE2018/dataset/`).
- **Tamaño:** 23.313 issues reales de 16 proyectos (Apache Mesos, Spring XD,
  Moodle, Jira Software, Titanium, etc.), ~17 MB en 16 CSV — cada uno con
  `issuekey`, `title`, `description` y `storypoint`.

> Los archivos no se versionan en este repo. Se descargan directamente
> (público, sin login) a `data/` — ver la sección "Cómo correr este
> proyecto" para el comando exacto.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio (ver notebooks en
`notebooks/`, en orden):

1. **Business Understanding:** estimación de story points como problema de
   regresión sobre el texto del issue — una referencia objetiva que
   complementa, no reemplaza, el juicio del equipo.
2. **Data Understanding** (`01_eda.ipynb`): sin duplicados; nulos en
   `description` desparejos por proyecto (hasta 31%). **Hallazgo central:**
   las 16 escalas de story points no son comparables entre sí — de 5
   valores posibles (`usergrid`, Fibonacci estricto) a 79 (`datamanagement`,
   casi continua). Target con sesgo extremo (candidato a `log1p`). Sin
   leakage textual detectado.
3. **Data Preparation** (`02_data_preparation.ipynb`): texto combinado
   (`title` + `description`) vectorizado con `sentence-transformers`
   (`all-MiniLM-L6-v2`, 384 dim), target `log1p(storypoint)`, y split
   **cronológico por proyecto** — corregido tras verificar que el orden de
   fila del CSV no coincide perfectamente con el orden de creación del
   issue (hasta 13.8% de filas fuera de orden en `titanium`).
4. **Modeling** (`03_modeling.ipynb`):
   - *Enfoque científico:* comparación directa entre un modelo **pooled**
     (un solo regresor con `project` como feature) y **16 modelos por
     proyecto** independientes — la respuesta al hallazgo de heterogeneidad
     de escalas del EDA.
   - *Enfoque analista:* el alcance original incluía cycle time/lead
     time/cuellos de botella, **no posible con el dataset sustituto** (sin
     timestamps de flujo) — el análisis descriptivo se limita a la
     distribución de story points y características del texto por
     proyecto.
5. **Evaluation:** MAE/RMSE en la **escala original de story points**
   (revirtiendo `log1p`), no en escala log — interpretable en términos de
   negocio. Se reporta también el *bias* (predicción − real), no solo el
   error absoluto, para verificar si el modelo hereda el sesgo de
   subestimación que el proyecto busca corregir.
6. **Deployment/Storytelling** (`dashboard/app.py`): dashboard en Streamlit
   con 4 vistas, incluyendo un estimador de story points en vivo (se
   escribe un issue, se obtiene una predicción al instante).

## Resultados clave (cuantificados)

- **El modelo pooled (Ridge sobre embeddings) tiene MAE ≈ 2.97 story
  points** en test, con **bias = −0.38** — mejor que el baseline (MAE 3.26),
  pero lejos de una estimación precisa.
- **Ni pooled ni por-proyecto gana de forma universal:** pooled gana en 9 de
  16 proyectos, por-proyecto en 7. Pooled gana en proyectos grandes con
  escalas idiosincráticas (`datamanagement`, `moodle`); por-proyecto gana en
  proyectos chicos con escalas Fibonacci limpias (`duracloud`, `bamboo`,
  `mesos`, `usergrid`) — conecta directamente con el hallazgo de
  heterogeneidad de escalas del EDA.
- **El modelo hereda el sesgo que el proyecto busca corregir:** el bias
  negativo significa que, en promedio, el modelo también subestima —
  automatizar la estimación no elimina el problema de negocio por sí solo
  si no se corrige explícitamente ese sesgo.
- **XGBoost no superó a Ridge** sobre embeddings — con una representación ya
  densa y de bajo ruido, un modelo lineal regularizado capta casi toda la
  señal disponible.

## Limitaciones

- **Dataset sustituto, no el planeado originalmente:** el Zenodo original
  (13.8 TB) hubiera permitido el análisis de flujo (cycle time, lead time,
  cuellos de botella) que este proyecto no pudo hacer — documentado como
  decisión explícita, no oculto.
- **Heterogeneidad de escalas entre proyectos**, ya discutida arriba — un
  MAE agregado de ~3 puntos esconde variación real: de 0.67 en `duracloud` a
  8.06 en `moodle`.
- **Orden cronológico imperfecto en el CSV original** — se corrigió
  ordenando explícitamente por número de issue key, pero es un recordatorio
  de que no hay que confiar ciegamente en el orden de un archivo de origen
  desconocido.
- **Proyectos open-source, no necesariamente representativos de equipos
  corporativos** — la cultura de estimación (y su desbalance) puede diferir
  bastante entre un proyecto open-source y un equipo de producto interno.
- **MAE ≈ 3 puntos no es una estimación precisa** — el modelo es un punto de
  partida para asistir la conversación de planning poker, no un reemplazo.

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar los 16 CSV (público, sin login) a 04-jira-effort-estimation/data/
cd 04-jira-effort-estimation/data
for f in appceleratorstudio aptanastudio bamboo clover datamanagement duracloud \
         jirasoftware mesos moodle mule mulestudio springxd talenddataquality \
         talendesb titanium usergrid; do
  curl -sL -o "${f}.csv" \
    "https://raw.githubusercontent.com/morakotch/datasets/master/storypoint/IEEE%20TSE2018/dataset/${f}.csv"
done
cd ../..

# 2. Correr los notebooks en orden (cada uno alimenta al siguiente)
jupyter notebook 04-jira-effort-estimation/notebooks/
#   01_eda.ipynb -> 02_data_preparation.ipynb -> 03_modeling.ipynb

# 3. Correr el dashboard
streamlit run 04-jira-effort-estimation/dashboard/app.py
```
