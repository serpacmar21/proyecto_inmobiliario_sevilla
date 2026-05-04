# Tasador Inteligente de Viviendas - Sevilla 

Este proyecto es una solución integral de ciencia de datos para la valoración automatizada de inmuebles en la provincia de Sevilla. Combina ingeniería de datos, análisis estadístico avanzado (R + Python), Big Data y Deep Learning para ofrecer predicciones precisas basadas en variables estructurales y socioeconómicas.

## Objetivo del Proyecto
El objetivo es proporcionar una herramienta interactiva que permita a usuarios y profesionales del sector inmobiliario estimar el precio de venta de una vivienda. El modelo no solo considera los metros cuadrados, sino que integra datos externos de **renta neta media por persona (INE)** y **distancia al centro de la capital**, permitiendo capturar la realidad del mercado sevillano.

## Problema que resuelve
La tasación manual de viviendas es un proceso lento y a menudo subjetivo. Esta aplicación resuelve:
1.  **La falta de transparencia:** Al basarse en datos estadísticos reales.
2.  **La fragmentación de datos:** Unifica fuentes de precios, datos geográficos y niveles de renta en un solo pipeline.
3.  **La escalabilidad:** Gracias al uso de Dask, el sistema está preparado para procesar grandes volúmenes de datos.

## Estructura del Proyecto
- `data/`: Datasets originales y procesados.
- `notebooks/`: 
    - `01_exploracion_y_limpieza.ipynb`: Ingeniería de datos y análisis R/Python.
    - `02_entrenamiento_modelo.ipynb`: Entrenamiento de ML (Dask) y DL (PyTorch).
- `models/`: Modelos entrenados y encoders exportados (.pkl).
- `app.py`: Aplicación interactiva de usuario (Streamlit).
- `tasador_pipeline.py`: Orquestación del flujo de datos con Dagster.
- `requirements.txt`: Dependencias del sistema.

## Instalación y Reproducibilidad
Este proyecto utiliza `uv` para la gestión eficiente de entornos y dependencias. 

**Nota arquitectónica:** Para evitar el colapso de memoria en el despliegue web de la aplicación, las dependencias se han dividido siguiendo buenas prácticas de MLOps:
* `requirements.txt`: Contiene las librerías ligeras necesarias exclusivamente para desplegar la app en producción.
* `requirements_dev.txt`: Contiene el entorno completo (incluyendo Dask, PyTorch y Dagster) necesario para la experimentación, orquestación y evaluación del proyecto.

### 1. Crear el entorno virtual
Ejecuta los siguientes comandos (sustituyendo `3.10` por tu versión de Python, en nuestro caso hemos usado Python 3.10):
```bash
uv venv --python 3.10
```

### 2. Activar el entorno

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

### 3. Instalar dependencias
```bash
uv pip install -r requirements.txt
```

## Ejecución de la Aplicación
Para lanzar la interfaz interactiva de tasación, ejecuta:
```bash
streamlit run app.py
```

## Orquestación
Si deseas ver el pipeline de datos en funcionamiento con **Dagster**:
```bash
dagster dev -f tasador_pipeline.py
```

## Tecnologías Destacadas
- **Big Data:** Entrenamiento en paralelo con un clúster local de **Dask**.
- **Deep Learning:** Redes neuronales convolucionales (1D CNN) con **PyTorch**.
- **Interoperabilidad:** Análisis de correlación generado mediante ejecución externa de **Rscript (ggplot2)**.
- **Orquestación:** Gestión de activos de datos mediante **Dagster**.
