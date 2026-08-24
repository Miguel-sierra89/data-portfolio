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

- **Nombre:** Public Jira Dataset
- **Fuente:** Zenodo (dataset real, proyectos open-source de Apache, Spring y
  JBoss)
- **Link:** https://zenodo.org/records/5901804
- **Tamaño:** por confirmar tras la descarga (dataset multi-proyecto con
  historial de issues de varios repositorios open-source)

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar los archivos en `data/`.

## Metodología

1. **Business Understanding:** enmarcar la estimación de story points como un
   problema de regresión (o clasificación ordinal) sobre el texto del issue,
   con el objetivo de servir como referencia objetiva que complemente —no
   reemplace— el juicio del equipo.
2. **Data Understanding:** EDA de nulos y consistencia entre proyectos
   (distintos equipos pueden usar la escala de story points de forma
   distinta), cardinalidad de tipos de issue y componentes, outliers en
   story points y en cycle time/lead time, y verificación de que no se
   filtren campos que solo existen después de resuelto el issue (data
   leakage, ej. tiempo de resolución real).
3. **Data Preparation:** limpieza y normalización del texto de issues
   (título + descripción), generación de embeddings de texto, y manejo de la
   heterogeneidad entre proyectos (posible normalización o modelado por
   proyecto).
4. **Modeling:**
   - *Enfoque analista:* análisis de cycle time, lead time, burndown y
     cuellos de botella del flujo de trabajo, para entender dónde se generan
     los retrasos más allá de la estimación inicial.
   - *Enfoque científico:* modelo de NLP (embeddings de sentence-transformers
     + regresión) que prediga story points a partir del texto del issue.
5. **Evaluation:** métricas de error de regresión (MAE, RMSE) interpretadas
   en términos de negocio (¿cuántos sprints se habrían planificado mejor?),
   comparando el error del modelo contra el error histórico de
   subestimación del equipo — no solo una métrica de ajuste aislada.
6. **Deployment/Storytelling:** dashboard en Streamlit con métricas de flujo
   ágil (cycle time, burndown) y el estimador de story points + este README
   como resumen ejecutivo.

## Resultados clave (cuantificados)

_Pendiente — se completará una vez ejecutado el análisis y el modelado._

## Limitaciones

_Pendiente — se documentarán las limitaciones del dataset y del modelo tras
el EDA (p. ej. heterogeneidad de criterios de estimación entre proyectos
open-source, posible no representatividad frente a equipos corporativos)._

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Zenodo indicado arriba
#    y colocarlo en 04-jira-effort-estimation/data/

# 2. Explorar el EDA
jupyter notebook 04-jira-effort-estimation/notebooks/

# 3. Correr el dashboard (una vez desarrollado)
streamlit run 04-jira-effort-estimation/dashboard/app.py
```
