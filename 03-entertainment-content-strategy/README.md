# 03 — Entretenimiento: Estrategia de Contenido

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
- **Tamaño:** ~8.800 títulos (películas y series), con metadata de género,
  país, año, rating, elenco y descripción

> El archivo no se versiona en este repo. Descargar manualmente desde el link
> anterior y colocar el CSV en `data/`.

## Metodología

1. **Business Understanding:** traducir "maximizar engagement" en proxies
   medibles a partir de la metadata disponible (concentración de
   género/país/rating, año de incorporación, evolución del mix de contenido
   en el tiempo).
2. **Data Understanding:** EDA de nulos (especialmente en `director`,
   `cast`, `country`), cardinalidad de género y país (campos multivaluados),
   outliers en duración, y consistencia de fechas de incorporación al
   catálogo.
3. **Data Preparation:** normalización de campos multivaluados (género,
   país, elenco) a formato analizable, extracción de features de texto desde
   `description`, y construcción de variables temporales (año/década de
   producción vs. año de incorporación al catálogo).
4. **Modeling:**
   - *Enfoque analista:* EDA de brechas de contenido por género, país y
     rating, comparando la composición del catálogo contra tendencias
     observables en los propios datos (concentración vs. diversidad,
     evolución temporal del mix).
   - *Enfoque científico:* sistema de recomendación content-based basado en
     similitud de coseno sobre embeddings de texto (descripción + género +
     elenco), y un modelo que prediga la probabilidad de éxito de un título
     según sus atributos.
5. **Evaluation:** para el recomendador, evaluación cualitativa de
   similitud y cobertura de catálogo; para el modelo de éxito, métricas de
   clasificación (precision/recall/AUC) sobre el proxy de éxito definido,
   evitando sobreinterpretar accuracy dado el desbalance esperado entre
   contenido "exitoso" y el resto.
6. **Deployment/Storytelling:** dashboard en Streamlit para explorar brechas
   de catálogo y probar el recomendador + este README como resumen
   ejecutivo.

## Resultados clave (cuantificados)

_Pendiente — se completará una vez ejecutado el análisis y el modelado._

## Limitaciones

_Pendiente — se documentarán las limitaciones del dataset y del modelo tras
el EDA (p. ej. el dataset no incluye métricas reales de audiencia/engagement,
por lo que "éxito" debe aproximarse con un proxy construido a partir de la
metadata disponible)._

## Cómo correr este proyecto

```bash
# Desde la raíz del repo
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Descargar el dataset desde el link de Kaggle indicado arriba
#    y colocarlo en 03-entertainment-content-strategy/data/

# 2. Explorar el EDA
jupyter notebook 03-entertainment-content-strategy/notebooks/

# 3. Correr el dashboard (una vez desarrollado)
streamlit run 03-entertainment-content-strategy/dashboard/app.py
```
