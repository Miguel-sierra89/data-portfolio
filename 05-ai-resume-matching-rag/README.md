# 05 — IA Aplicada: Sistema de Matching de Candidatos con RAG

> Este fue el proyecto con la sorpresa más agradable: pensé que iba a bajar
> ~2.400 CVs sueltos y tener que armar los pares candidato-vacante yo mismo.
> El CSV real ya venía con esos pares armados y un `matched_score`
> calculado — mejor punto de partida del que había planeado. También decidí
> a propósito **no** llamar a ningún LLM real acá: no tenía sentido dejar un
> notebook público de portafolio dependiendo de una API key de pago para
> poder correrse.

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

- **Nombre:** Resume Dataset (`resume_data.csv`)
- **Fuente:** Kaggle
- **Link:** https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset
- **Tamaño real:** el dataset descargado resultó ser distinto — y mejor para
  este proyecto — que lo planeado originalmente. No son CVs sueltos
  categorizados: son **9.544 pares candidato–vacante ya emparejados**, que
  se reducen a **344 CVs únicos** y **28 vacantes únicas** tras deduplicar,
  con un `matched_score` precalculado por el creador del dataset (usado como
  referencia externa de validación, no como ground truth humano).

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/` como `resume_data.csv`.

## Metodología

Siguiendo el framework CRISP-DM adaptado del portafolio (ver notebooks en
`notebooks/`, en orden):

1. **Business Understanding:** matching como recuperación semántica
   (embeddings + FAISS) sobre CVs y vacantes reales, con una capa de
   explicabilidad — deliberadamente **sin** depender de una API de LLM en el
   pipeline reproducible (ver punto 4).
2. **Data Understanding** (`01_eda.ipynb`): el dataset real trae campos
   estructurados de ambos lados (CV y vacante) más `matched_score` — no CVs
   sueltos. Se corrigieron dos bugs de datos: un BOM incrustado en el nombre
   de una columna, y dos columnas literalmente llamadas `responsibilities`
   que pandas desambiguó en silencio. Se verificó que `matched_score` no es
   un score trivial de overlap de keywords (correlación ~0.13).
3. **Data Preparation** (`02_data_preparation.ipynb`): **hallazgo nuevo** —
   casi todos los campos del CV venían serializados como strings de listas
   de Python (a veces anidadas, a veces con `None`), parseados de forma
   segura con `ast.literal_eval`. Deduplicación a 344 candidatos y 28
   vacantes, con perfil de texto combinado por cada uno.
4. **Modeling** (`03_modeling.ipynb`): embeddings (`sentence-transformers`,
   `all-MiniLM-L6-v2`) + índice FAISS sobre los 344 candidatos únicos.
   **Explicabilidad sin LLM por diseño:** en vez de llamar a una API de pago
   dentro de un notebook público de portafolio, se implementó una
   justificación determinista por overlap de skills, con el camino de
   mejora (LLM real) documentado para el servicio FastAPI.
5. **Evaluation:** validación cuantitativa del ranking semántico contra
   `matched_score` (correlación de Spearman), no solo evaluación cualitativa
   — reportada con honestidad como señal moderada, no un ajuste perfecto.
6. **Deployment** (`app/main.py`): servicio FastAPI mínimo con matching por
   `job_id` existente o por **texto libre de vacante** (el caso de uso real:
   una vacante nueva, no vista en el dataset).

## Resultados clave (cuantificados)

- **El pipeline de embeddings + FAISS funciona end-to-end** sobre los 344
  candidatos únicos, recuperando candidatos semánticamente afines a una
  vacante — incluyendo vacantes en texto libre no presentes en el dataset
  original, vía `POST /match/text`.
- **Validación contra `matched_score`:** correlación de Spearman ≈ **0.30**
  (pooled sobre ~9.500 pares, p ≈ 10⁻²⁰⁰ — altamente significativa), con un
  rango de 0.10 a 0.48 según la vacante. Señal real y no trivial, pero
  moderada: el embedding semántico y la referencia externa miden algo
  similar, no son la misma fórmula.
- **La explicabilidad por overlap de skills es limitada, y eso quedó
  documentado con datos reales:** en la demo, 4 de los 5 candidatos top por
  similitud semántica no comparten ningún skill textual con la vacante —
  el embedding capta afinidad conceptual que el overlap literal no siempre
  puede mostrar como evidencia.
- **Dos bugs de datos reales encontrados y corregidos** (no solo
  documentados): un BOM incrustado en un nombre de columna, y campos del CV
  serializados como listas de Python que habrían metido corchetes y la
  palabra "None" en el texto del embedding si no se parseaban primero.
- **Un bug real de la API FastAPI, encontrado probando contra un servidor
  vivo:** Starlette rechaza `NaN` en respuestas JSON (`allow_nan=False`),
  así que cualquier campo nulo (ej. `age_requirement`, ~43% nulo) tumbaba el
  endpoint con un 500 — corregido convirtiendo `NaN` a `None` antes de
  responder.

## Limitaciones

- **`matched_score` es un score de un tercero, no una etiqueta humana
  verificada** — útil como referencia de validación, pero no debe leerse
  como una medición objetiva perfecta de "el mejor candidato real".
- **Explicabilidad determinista, no generativa:** el overlap de skills es
  transparente y sin costo, pero no siempre encuentra evidencia textual
  aunque la similitud semántica sea alta — una capa de LLM real (fuera de
  alcance acá, por la razón ya explicada) explicaría mejor esos casos.
- **Recuperación de candidatos, no un pipeline de extracción de CVs desde
  PDF/Word** — el dataset ya trae los campos estructurados; un sistema en
  producción necesitaría ese paso previo (parsing de CVs reales), que este
  proyecto no cubre.
- **344 candidatos es un pool chico** para un sistema de producción — el
  índice FAISS escala bien a volúmenes mucho mayores, pero la validación
  cuantitativa (Spearman) se hizo sobre esa muestra.
- **El servicio FastAPI es una demo funcional, no un producto con
  autenticación, rate limiting ni persistencia** — pensado para mostrar el
  pipeline de matching, no como base directa de un sistema en producción.

## Dashboard en vivo

**[ai-resume-matching-rag.streamlit.app](https://ai-resume-matching-rag.streamlit.app/)**
— demo del motor de matching (mismo embedder, mismo índice FAISS y misma
explicabilidad que el servicio FastAPI de abajo), sin instalación. El
servicio FastAPI en sí no está desplegado (Docker en Hugging Face Spaces
pasó a requerir plan pago) — se corre local siguiendo los pasos de abajo.

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 05-ai-resume-matching-rag/data/resume_data.csv

# 2. Correr los notebooks en orden (cada uno alimenta al siguiente)
jupyter notebook 05-ai-resume-matching-rag/notebooks/
#   01_eda.ipynb -> 02_data_preparation.ipynb -> 03_modeling.ipynb

# 3. Levantar el servicio FastAPI (desde 05-ai-resume-matching-rag/)
cd 05-ai-resume-matching-rag
uvicorn app.main:app --reload
# Docs interactivas: http://127.0.0.1:8000/docs
```
