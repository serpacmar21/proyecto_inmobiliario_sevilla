# Tasador Inteligente de Viviendas - Sevilla 

Este proyecto es una solución integral de ciencia de datos para la valoración automatizada de inmuebles en la provincia de Sevilla. Combina ingeniería de datos, análisis estadístico avanzado (R + Python), Big Data y Deep Learning para ofrecer predicciones precisas basadas en variables estructurales y socioeconómicas.

## Objetivo del Proyecto
El objetivo es proporcionar una herramienta interactiva que permita a usuarios y profesionales del sector inmobiliario estimar el precio de venta de una vivienda. El modelo no solo considera los metros cuadrados, sino que integra datos externos de **renta bruta media por municipio (INE)** y **distancia al centro de la capital**, permitiendo capturar la realidad del mercado sevillano.

## Problema que resuelve
La tasación manual de viviendas es un proceso lento y a menudo subjetivo. Esta aplicación resuelve:
1.  **La falta de transparencia y explicabilidad (XAI):** El sistema no es una "caja negra"; permite auditar el peso matemático de cada variable en las predicciones gracias a técnicas avanzadas como la Importancia por Permutación.
2.  **La fragmentación de datos:** Unifica fuentes masivas de precios, distancias geográficas y niveles de renta en un solo pipeline automatizado.
3.  **La escalabilidad:** Gracias al uso del procesamiento paralelo con Dask, el sistema está preparado para procesar grandes volúmenes de datos.

## Estructura del Proyecto
- `data/`: 
    - `raw/`: Datasets originales (precios, distancias y rentas).
    - `processed/`: Dataset final limpio y cruzado listo para inferencia.
- `models/`: Modelos congelados (Random Forest en `.pkl` y Red Neuronal en `.pth`), además de los *Label Encoders* y *Scalers* necesarios para evitar la fuga de datos (Data Leakage) en producción.
- `notebooks/`: 
    - `01_exploracion_y_limpieza.ipynb`: Ingeniería de datos y análisis estadístico interoperable (R/Python).
    - `02_entrenamiento_modelo.ipynb`: Entrenamiento de ML (Dask) y DL (PyTorch) con análisis de explicabilidad.
    - `grafico_renta_precio.png`: Análisis de correlación generado mediante R.
- `src/`:
    - `tasador_pipeline.py`: Orquestación del flujo de datos con Dagster.
- `app.py`: Interfaz de usuario multipágina (Tasador Interactivo + Análisis de Mercado con XAI) desarrollada en Streamlit utilizando Programación Orientada a Objetos (POO). Aplica internamente un factor de corrección para adaptar los precios aprendidos históricamente (2021) al mercado actual (2026).
- `pyproject.toml`: Archivo de configuración moderno para la gestión de dependencias y entornos con `uv`.
- `requirements.txt`: Dependencias para el entorno de producción (Web) mantenidas como respaldo de compatibilidad.

## Instalación y Reproducibilidad
Este proyecto utiliza `uv` y el estándar `pyproject.toml` para la gestión eficiente, rápida y determinista de entornos y dependencias.

### 1. Crear el entorno virtual
```bash
uv venv
```

### 2. Activar el entorno
- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

### 3. Instalar dependencias

Para cumplir con los requisitos del proyecto, utiliza el comando de sincronización:

Para instalar las dependencias base y ejecutar la web (Producción):

`uv sync`

Para instalar también las herramientas de desarrollo (Dask, Dagster y Jupyter Notebooks):

`uv sync --all-extras`

*Nota:* Si lo prefieres, también puedes usar el archivo tradicional para el entorno de producción:

## Ejecución de la Aplicación
Para lanzar la interfaz interactiva de tasación, ejecuta:
```bash
streamlit run app.py
```

## Orquestación
Si deseas ver el pipeline de datos en funcionamiento con **Dagster**:
```bash
dagster dev -f src/tasador_pipeline.py
```

## Tecnologías Destacadas

- **Big Data:** Entrenamiento en paralelo con un clúster local de Dask.
- **Deep Learning:** Redes neuronales convolucionales (1D CNN) con PyTorch y técnica de Fine-Tuning.
- **Explicabilidad (XAI):** Métrica nativa en Scikit-Learn (Random Forest) e Importancia por Permutación programada a mano para PyTorch.
- **Interoperabilidad:** Análisis de correlación generado mediante ejecución externa de Rscript (ggplot2) desde un entorno Python.
- **Orquestación:** Gestión de activos y scheduling mediante Dagster.
- **Despliegue Orientado a Objetos (POO):** Interfaz desarrollada en Streamlit bajo el paradigma de clases y objetos.
