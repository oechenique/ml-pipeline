from datetime import datetime, timedelta
import os
import sys
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir /app al PYTHONPATH si no está ya
if '/app' not in sys.path:
    sys.path.append('/app')

# Argumentos por defecto
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Crear DAG maestro
dag = DAG(
    'bike_sharing_master_pipeline',
    default_args=default_args,
    description='Orquestación principal del pipeline de bike-sharing',
    schedule_interval='@daily',  # Ajustar según necesidades
    catchup=False,
    tags=['bike-sharing', 'master', 'pipeline'],
)

# Función para verificar prerequisitos
def check_environment():
    """Verifica que todo el entorno esté configurado correctamente."""
    logger.info("🔍 Verificando entorno y dependencias...")
    
    # Verificar Docker
    try:
        import docker
        client = docker.from_env()
        containers = client.containers.list()
        running_containers = [c.name for c in containers]
        logger.info(f"✅ Docker en ejecución. Contenedores: {running_containers}")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo conectar a Docker: {str(e)}")
    
    # Verificar conexión a PostgreSQL
    try:
        import psycopg2
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"✅ PostgreSQL conectado: {version[0]}")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"❌ Error conectando a PostgreSQL: {str(e)}")
        raise
    
    # Verificar dependencias Python
    try:
        import pandas as pd
        import geopandas as gpd
        import h3
        import shapely
        import folium
        logger.info("✅ Todas las dependencias Python están instaladas")
    except ImportError as e:
        logger.error(f"❌ Falta la dependencia: {str(e)}")
        raise
    
    # Verificar directorios de datos
    data_dirs = [
        "/app/data",
        "/app/data/raw",
        "/app/data/processed",
        "/app/data/processed/h3"
    ]
    for d in data_dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            logger.info(f"🔄 Creado directorio: {d}")
    
    logger.info("✅ Entorno verificado correctamente")
    return True

# Función para verificar data lakes/big data
def check_data_sources():
    """Verifica la disponibilidad de fuentes de datos."""
    logger.info("🔍 Verificando fuentes de datos...")
    
    # Verificar BigQuery
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        logger.info("✅ Conexión a BigQuery disponible")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo conectar a BigQuery: {str(e)}")
        logger.info("ℹ️ Se usará fallback a datos simulados")
    
    # Verificar disponibilidad de fuentes alternativas
    try:
        import requests
        kaggle_api_check = os.environ.get('KAGGLE_API_KEY') is not None
        if kaggle_api_check:
            logger.info("✅ Kaggle API disponible como fuente alternativa")
        else:
            logger.info("ℹ️ Kaggle API no configurada")
    except Exception:
        pass
    
    return True

# Crear flujo de tareas
with dag:
    # Tarea 1: Verificar entorno
    check_env_task = PythonOperator(
        task_id='check_environment',
        python_callable=check_environment,
    )
    
    # Tarea 2: Verificar fuentes de datos
    check_data_task = PythonOperator(
        task_id='check_data_sources',
        python_callable=check_data_sources,
    )
    
    # Grupo de tareas para ingesta de datos
    with TaskGroup("data_ingestion") as ingestion_group:
        # Trigger para DAG de ingesta de datos
        trigger_ingestion = TriggerDagRunOperator(
            task_id='trigger_ingestion',
            trigger_dag_id='bike_sharing_ingestion',
            wait_for_completion=True,
            poke_interval=60,  # Verificar cada minuto
            execution_date="{{ execution_date }}",
        )
        
        # Trigger para DAG de ingesta desde BigQuery (opcional)
        trigger_bigquery = TriggerDagRunOperator(
            task_id='trigger_bigquery_ingestion',
            trigger_dag_id='bigquery_ingestion',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Definir dependencias dentro del grupo
        trigger_ingestion >> trigger_bigquery
    
    # Grupo de tareas para almacenamiento
    with TaskGroup("data_storage") as storage_group:
        # Sensor para esperar a que termine la ingesta
        wait_for_ingestion = ExternalTaskSensor(
            task_id='wait_for_ingestion',
            external_dag_id='bike_sharing_ingestion',
            external_task_id=None,  # Esperar a que todo el DAG termine
            timeout=3600,  # 1 hora de timeout
            mode='reschedule',
            poke_interval=60,
            execution_date_fn=lambda dt: dt,
        )
        
        # Trigger para DAG de almacenamiento en PostgreSQL
        trigger_storage = TriggerDagRunOperator(
            task_id='trigger_postgres_storage',
            trigger_dag_id='bike_sharing_postgres_storage',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Definir dependencias dentro del grupo
        wait_for_ingestion >> trigger_storage
    
    # Grupo de tareas para feature engineering y H3
    with TaskGroup("feature_engineering") as feature_group:
        # Sensor para esperar a que termine el almacenamiento
        wait_for_storage = ExternalTaskSensor(
            task_id='wait_for_storage',
            external_dag_id='bike_sharing_postgres_storage',
            external_task_id=None,
            timeout=3600,
            mode='reschedule',
            poke_interval=60,
            execution_date_fn=lambda dt: dt,
        )
        
        # Trigger para DAG de indexación H3
        trigger_h3 = TriggerDagRunOperator(
            task_id='trigger_h3_indexing',
            trigger_dag_id='bike_sharing_h3_indexing',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Definir dependencias dentro del grupo
        wait_for_storage >> trigger_h3
    
    # Grupo de tareas para análisis y ML
    with TaskGroup("analytics_ml") as analytics_group:
        # Sensor para esperar a que termine el feature engineering
        wait_for_features = ExternalTaskSensor(
            task_id='wait_for_features',
            external_dag_id='bike_sharing_h3_indexing',
            external_task_id=None,
            timeout=3600,
            mode='reschedule',
            poke_interval=60,
            execution_date_fn=lambda dt: dt,
        )
        
        # Trigger para DAG de clustering
        trigger_clustering = TriggerDagRunOperator(
            task_id='trigger_clustering',
            trigger_dag_id='bike_sharing_clustering',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Trigger para DAG de predicción
        trigger_prediction = TriggerDagRunOperator(
            task_id='trigger_prediction',
            trigger_dag_id='bike_sharing_prediction',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Trigger para DAG de series temporales
        trigger_time_series = TriggerDagRunOperator(
            task_id='trigger_time_series',
            trigger_dag_id='bike_sharing_time_series',
            wait_for_completion=True,
            execution_date="{{ execution_date }}",
        )
        
        # Definir dependencias dentro del grupo
        wait_for_features >> [trigger_clustering, trigger_prediction, trigger_time_series]
    
    # Función para generar informe final
    def generate_summary():
        """Genera un informe final con estadísticas del pipeline."""
        import pandas as pd
        from datetime import datetime
        import os
        
        logger.info("📊 Generando informe final del pipeline...")
        
        # Conectar a PostgreSQL y extraer estadísticas
        try:
            import psycopg2
            import psycopg2.extras
            
            conn_params = {
                'host': 'db_service',
                'port': 5432,
                'database': 'geo_db',
                'user': 'geo_user',
                'password': 'NekoSakamoto448'
            }
            
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Estadísticas de estaciones
            cursor.execute("""
            SELECT 
                COUNT(*) as total_stations,
                COUNT(DISTINCT city) as total_cities,
                COUNT(h3_index) as stations_with_h3,
                AVG(capacity) as avg_capacity
            FROM bike_stations
            """)
            station_stats = cursor.fetchone()
            
            # Estadísticas de viajes
            cursor.execute("""
            SELECT 
                COUNT(*) as total_trips,
                AVG(duration_sec)/60 as avg_duration_min,
                COUNT(DISTINCT start_station_id) as unique_start_stations,
                COUNT(DISTINCT end_station_id) as unique_end_stations
            FROM bike_trips
            """)
            trip_stats = cursor.fetchone()
            
            # Crear DataFrame con estadísticas
            stats = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total Stations': station_stats['total_stations'],
                'Total Cities': station_stats['total_cities'],
                'Stations with H3': station_stats['stations_with_h3'],
                'Average Capacity': round(station_stats['avg_capacity'], 2),
                'Total Trips': trip_stats['total_trips'],
                'Average Trip Duration (min)': round(trip_stats['avg_duration_min'], 2),
                'Unique Start Stations': trip_stats['unique_start_stations'],
                'Unique End Stations': trip_stats['unique_end_stations']
            }
            
            # Cerrar conexión
            cursor.close()
            conn.close()
            
            # Crear DataFrame
            stats_df = pd.DataFrame([stats])
            
            # Guardar a CSV (append)
            output_path = '/app/data/reports/'
            os.makedirs(output_path, exist_ok=True)
            
            csv_path = os.path.join(output_path, 'pipeline_summary.csv')
            if os.path.exists(csv_path):
                # Append sin duplicar encabezados
                stats_df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                # Crear nuevo archivo
                stats_df.to_csv(csv_path, index=False)
            
            # Guardar informe de esta ejecución específica
            report_name = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            stats_df.to_csv(os.path.join(output_path, report_name), index=False)
            
            logger.info(f"✅ Informe guardado en {csv_path} y {report_name}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error generando informe: {str(e)}")
            return {"error": str(e)}
    
    # Tarea para generar informe final
    generate_report_task = PythonOperator(
        task_id='generate_summary_report',
        python_callable=generate_summary,
    )
    
    # Definir dependencias principales
    check_env_task >> check_data_task >> ingestion_group >> storage_group >> feature_group >> analytics_group >> generate_report_task