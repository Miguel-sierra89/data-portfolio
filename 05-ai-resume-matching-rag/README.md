# 05 — IA Aplicada: Sistema de Matching de Candidatos con RAG

## Contexto de negocio

Los procesos de reclutamiento tradicionales filtran CVs por coincidencia
literal de palabras clave, lo que descarta candidatos relevantes que
describen su experiencia con terminología distinta a la de la vacante (ej.
"desarrollo de modelos predictivos" vs. "machine learning"). Un matching
semántico —que entienda el significado, no solo el texto exacto— permite
recuperar candidatos verdaderamente relevantes, y hacerlo con explicabilidad
es clave para que un reclutador confíe en el ranking y pueda justificarlo
ante el negocio.

## Pregunta / problema a resolver

¿Cómo automatizamos el matching semántico (no solo por keyword) entre
vacantes y CVs, entregando además una explicación clara de por qué cada
candidato fue rankeado en esa posición?

## Dataset

- **Nombre:** Resume Dataset
- **Fuente:** Kaggle
- **Link:** https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset
- **Tamaño:** ~2.400 CVs reales, categorizados por rol/industria

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar los archivos en `data/`.

## Metodología

1. **Business Understanding:** enmarcar el matching como un problema de
   recuperación semántica (retrieval) sobre un espacio vectorial de
   embeddings, seguido de una capa de explicabilidad sobre el resultado, en
   lugar de un simple filtro de keywords.
2. **Data Understanding:** EDA de la distribución de categorías/roles de los
   CVs, calidad y longitud del texto, nulos, y detección de posibles sesgos
   de representación (categorías sobre o sub-representadas) que puedan
   afectar el matching.
3. **Data Preparation:** limpieza de texto (normalización, remoción de
   ruido de formato propio de CVs extraídos de PDF/Word), y extracción
   estructurada de skills como paso previo a la vectorización.
4. **Modeling:**
   - Pipeline de embeddings con `sentence-transformers` para vacantes y
     CVs.
   - Búsqueda vectorial de similitud con FAISS para recuperar los candidatos
     más relevantes a una vacante dada.
   - Capa de LLM que extraiga skills estructurados desde el CV y justifique
     en lenguaje natural por qué un candidato fue rankeado en cada posición
     (explicabilidad del ranking, no solo un score).
   - Exposición como servicio mínimo vía FastAPI (endpoint de matching +
     endpoint de explicación).
5. **Evaluation:** evaluación cualitativa de relevancia del top-k
   recuperado (¿los candidatos rankeados arriba son razonables para la
   vacante?), y verificación de que las justificaciones generadas por el LLM
   sean consistentes con los skills realmente extraídos del CV (evitar
   alucinaciones en la explicación).
6. **Deployment/Storytelling:** servicio FastAPI mínimo como demo funcional
   + este README como resumen ejecutivo del enfoque.

## Resultados clave (cuantificados)

_Pendiente — se completará una vez ejecutado el análisis y el desarrollo del
pipeline._

## Limitaciones

_Pendiente — se documentarán las limitaciones del dataset y del sistema tras
el desarrollo (p. ej. dependencia de la calidad del texto extraído de los
CVs originales, y el hecho de que la explicación generada por LLM requiere
validación humana antes de usarse en decisiones reales de contratación)._

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 05-ai-resume-matching-rag/data/

# 2. Explorar el EDA y el desarrollo del pipeline
jupyter notebook 05-ai-resume-matching-rag/notebooks/

# 3. Levantar el servicio FastAPI (una vez desarrollado)
uvicorn app.main:app --reload --app-dir 05-ai-resume-matching-rag/app
```
