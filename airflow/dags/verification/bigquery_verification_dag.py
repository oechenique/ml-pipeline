import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import bigquery
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'verify_bigquery_connection',
    default_args=default_args,
    description='Verifica conexión a BigQuery con consulta detallada',
    schedule_interval=None,  # Solo manual
    catchup=False,
)

def test_bigquery_connection(**context):
    # Configurar logging
    logger = logging.getLogger(__name__)
    
    # Ruta al archivo de credenciales
    credentials_path = "/app/credentials/credentials.json"
    
    # Verificar que el archivo existe
    logger.info(f"Verificando credenciales en: {credentials_path}")
    if not os.path.exists(credentials_path):
        logger.error("❌ Archivo de credenciales NO encontrado")
        raise FileNotFoundError(f"No se encontró el archivo de credenciales en {credentials_path}")
    
    logger.info("✅ Archivo de credenciales encontrado")
    
    try:
        # Configurar cliente de BigQuery usando credenciales del archivo
        client = bigquery.Client.from_service_account_json(credentials_path)
        
        # Consulta de prueba con filtros y más columnas
        query = """
        SELECT 
            trip_id, 
            duration_sec, 
            start_date, 
            start_station_name, 
            start_station_id,
            end_station_name, 
            end_station_id,
            subscriber_type, 
            zip_code,
            member_birth_year,
            member_gender,
            bike_share_for_all_trip
        FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
        WHERE 
            duration_sec IS NOT NULL 
            AND start_station_name IS NOT NULL
            AND end_station_name IS NOT NULL
            AND subscriber_type IS NOT NULL
            AND member_birth_year IS NOT NULL
            AND member_gender IS NOT NULL
        LIMIT 10
        """
        
        logger.info("Ejecutando consulta a BigQuery...")
        
        # Ejecutar consulta
        query_job = client.query(query)
        
        # Recuperar resultados
        results = list(query_job)
        
        # Número de registros
        count = len(results)
        logger.info(f"✅ Conexión exitosa! Recuperados {count} registros.")
        
        # Mostrar registros
        for row in results:
            logger.info(f"Registro: {dict(row)}")
        
        return count
    
    except Exception as e:
        logger.error(f"❌ Error conectando a BigQuery: {str(e)}")
        raise e

# Crear tarea de Python
verify_bq_task = PythonOperator(
    task_id='verify_bigquery_connection_task',
    python_callable=test_bigquery_connection,
    provide_context=True,
    dag=dag,
)