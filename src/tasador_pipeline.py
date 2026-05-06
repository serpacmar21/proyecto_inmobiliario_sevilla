from dagster import (
    asset, 
    Definitions, 
    get_dagster_logger, 
    define_asset_job, 
    ScheduleDefinition
)
import os
import time

# LOGGING ESTRUCTURADO
logger = get_dagster_logger()


# FASE 1: PREPARACIÓN DE DATOS

@asset(group_name="pipeline_inmobiliario", description="Fase 1: Limpieza paralela y cruce con INE")
def datos_limpios() -> str:
    """Simula la ejecución del ETL y limpieza de datos con Dask."""
    logger.info("Iniciando pipeline de ingesta de datos...")
    time.sleep(1) # Simulamos tiempo de cómputo
    
    ruta_csv = os.path.join("data", "processed", "viviendas_sevilla_limpio.csv")
    
    if os.path.exists(ruta_csv):
        logger.info(f"Datos limpios y estructurados localizados en: {ruta_csv}")
    else:
        logger.warning(f"Aviso: No se encontró {ruta_csv}. Usando ruta simulada para el DAG.")
        ruta_csv = "data/processed/viviendas_sevilla_limpio.csv"
        
    return ruta_csv


# FASE 2: ENTRENAMIENTO DE IA (Depende de Fase 1)

@asset(group_name="pipeline_inmobiliario", description="Fase 2: Entrenamiento Random Forest (Big Data) y CNN (Deep Learning)")
def modelo_entrenado(datos_limpios: str) -> dict:
    """Recibe los datos limpios y entrena los modelos de IA."""
    logger.info(f"Consumiendo datos de: {datos_limpios}")
    logger.info("Activando clúster Dask local para entrenamiento distribuido...")
    time.sleep(2)
    
    logger.info("Entrenando Red Neuronal (CNN 1D) con PyTorch...")
    time.sleep(1)
    
    # Ahora verificamos las dos arquitecturas clave
    ruta_rf = os.path.join("models", "modelo_casas_sevilla.pkl")
    ruta_dl = os.path.join("models", "modelo_pytorch.pth")
    
    modelos_listos = {
        "Random Forest": ruta_rf if os.path.exists(ruta_rf) else "Simulado_RF.pkl",
        "Deep Learning CNN": ruta_dl if os.path.exists(ruta_dl) else "Simulado_DL.pth"
    }
    
    for modelo, ruta in modelos_listos.items():
        if "Simulado" not in ruta:
            logger.info(f"{modelo} entrenado y serializado en: {ruta}")
        else:
            logger.warning(f"Aviso: No se encontró archivo real para {modelo}. Usando ruta simulada.")
            
    # Devolvemos un diccionario en lugar de un string único
    return modelos_listos


# FASE 3: DESPLIEGUE (Depende de Fase 2)

@asset(group_name="pipeline_inmobiliario", description="Fase 3: Despliegue de la App en Streamlit")
def app_desplegada(modelo_entrenado: dict) -> bool:
    """Verifica que los modelos están listos y simula el levantamiento de la App."""
    logger.info("Cargando los siguientes modelos en la memoria de producción:")
    for nombre, ruta in modelo_entrenado.items():
        logger.info(f" -> Cargando {nombre} desde: {ruta}")
    
    ruta_app = "app.py"
    if os.path.exists(ruta_app):
        logger.info(f"App lista. Ejecuta 'streamlit run {ruta_app}' en la terminal para interactuar.")
        return True
    else:
        logger.error(f"Error crítico: Falta el archivo {ruta_app}")
        return False


# AUTOMATIZACIÓN Y SCHEDULING 

# Definimos un "Trabajo" que agrupa todos nuestros assets
actualizacion_semanal_job = define_asset_job(
    name="actualizacion_semanal_tasador",
    selection="*" # Selecciona todos los assets definidos
)

# Definimos el "Horario" (Ej: Se ejecuta automáticamente cada Lunes a las 02:00 AM)
schedule_lunes_madrugada = ScheduleDefinition(
    job=actualizacion_semanal_job,
    cron_schedule="0 2 * * 1", # Formato CRON: Minuto 0, Hora 2, Todos los meses, Día 1 (Lunes)
)

# Empaquetamos todo para que Dagster lo reconozca
defs = Definitions(
    assets=[datos_limpios, modelo_entrenado, app_desplegada],
    schedules=[schedule_lunes_madrugada],
)