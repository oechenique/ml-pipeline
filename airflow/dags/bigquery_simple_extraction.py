import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import bigquery

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bigquery_simple_extraction',
    default_args=default_args,
    description='Extrae datos de BigQuery usando el método simple de Colab',
    schedule_interval=None,  # Solo manual
    catchup=False,
)

def extract_citibike_data(**context):
    """
    Extrae datos de CitiBike de BigQuery usando el enfoque simple
    """
    print("🔄 Iniciando extracción de datos de CitiBike desde BigQuery...")
    
    # Verificar credenciales disponibles
    # Busca en varias ubicaciones posibles
    credential_paths = [
        "/app/credentials/credentials.json",
        "/app/credentials/bigquery_credentials.json",
        "/app/credentials/ee-gastonechenique-9870a2644d44.json"
    ]
    
    credentials_file = None
    for path in credential_paths:
        if os.path.exists(path):
            credentials_file = path
            print(f"✅ Credenciales encontradas en: {path}")
            break
    
    if not credentials_file:
        print("❌ No se encontraron credenciales en ninguna ubicación")
        raise FileNotFoundError("No se encontraron credenciales para BigQuery")
    
    # Configurar la variable de entorno (como en Colab)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    
    # Conectar a BigQuery
    try:
        client = bigquery.Client()
        print("✅ Cliente de BigQuery inicializado correctamente")
        
        # Consulta para CitiBike (similar a la de Colab)
        print("🔄 Ejecutando consulta para CitiBike...")
        query = """
        SELECT *
        FROM `bigquery-public-data.new_york_citibike.citibike_trips`
        WHERE start_station_id IS NOT NULL
        AND end_station_id IS NOT NULL
        LIMIT 100
        """
        
        # Ejecutar consulta
        query_job = client.query(query)
        
        # Convertir a DataFrame
        citibike_df = query_job.to_dataframe()
        print(f"✅ Datos de CitiBike recuperados: {len(citibike_df)} registros")
        
        # Consulta para San Francisco (para comparar)
        print("🔄 Ejecutando consulta para San Francisco...")
        sf_query = """
        SELECT *
        FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
        WHERE start_station_id IS NOT NULL
        AND end_station_id IS NOT NULL
        LIMIT 100
        """
        
        # Ejecutar consulta
        sf_job = client.query(sf_query)
        
        # Convertir a DataFrame
        sf_df = sf_job.to_dataframe()
        print(f"✅ Datos de San Francisco recuperados: {len(sf_df)} registros")
        
        # Guardar resultados para verificación
        output_dir = Path("/app/data/raw/bigquery_test")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Guardar CSVs para inspección
        citibike_path = output_dir / "citibike_sample.csv"
        sf_path = output_dir / "sf_sample.csv"
        
        citibike_df.to_csv(citibike_path, index=False)
        sf_df.to_csv(sf_path, index=False)
        
        print(f"✅ Datos guardados en {output_dir}")
        
        # Mostrar algunas columnas para verificación
        print("\n--- Muestra de datos CitiBike ---")
        print(citibike_df.columns.tolist())
        print(citibike_df[['starttime', 'start_station_id', 'end_station_id']].head(3))
        
        print("\n--- Muestra de datos San Francisco ---")
        print(sf_df.columns.tolist())
        print(sf_df[['start_date', 'start_station_id', 'end_station_id']].head(3))
        
        return {
            "citibike_count": len(citibike_df),
            "sf_count": len(sf_df),
            "output_dir": str(output_dir)
        }
        
    except Exception as e:
        print(f"❌ Error durante la extracción: {str(e)}")
        raise

# Tarea para extraer datos de BigQuery
extract_task = PythonOperator(
    task_id='extract_bigquery_data',
    python_callable=extract_citibike_data,
    provide_context=True,
    dag=dag,
)

# Solo una tarea
extract_task