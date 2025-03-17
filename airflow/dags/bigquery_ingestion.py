import os
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json
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
    'bigquery_ingestion',
    default_args=default_args,
    description='Ingesta de datos de BigQuery a formato Parquet',
    schedule_interval=None,  # Solo manual
    catchup=False,
)

def extract_bikeshare_data(output_dir="/app/data/processed/bike_sharing", batch_size=50000, **context):
    """
    Extrae datos de SF y NY desde BigQuery y los guarda en formato parquet por lotes
    """
    print("🚀 Iniciando extracción masiva de datos de bikesharing desde BigQuery...")
    
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
        
        # Diccionario para almacenar estadísticas
        extraction_stats = {
            "sf_total": 0,
            "nyc_total": 0,
            "partitions": []
        }
        
        # Crear directorio base
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # --- Extracción de datos de San Francisco ---
        print("🔄 Comenzando extracción de datos de San Francisco...")
        
        # Extraer datos de estaciones de SF
        station_query = """
        SELECT DISTINCT 
            station_id,
            name,
            'San Francisco' as city,
            lat as latitude,
            lon as longitude,
            capacity
        FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_station_info`
        """
        
        sf_stations = client.query(station_query).to_dataframe()
        print(f"✅ Extraídas {len(sf_stations)} estaciones de San Francisco")
        
        # Determinar el número total de registros en SF
        count_query = """
        SELECT COUNT(*) as total 
        FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
        WHERE 
            duration_sec IS NOT NULL 
            AND start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
        """
        
        total_sf = client.query(count_query).to_dataframe().iloc[0]['total']
        print(f"🔢 SF tiene un total de {total_sf} registros. Extrayendo por lotes de {batch_size}...")
        extraction_stats["sf_total"] = int(total_sf)
        
        # Extraer datos por lotes
        for offset in range(0, min(1000000, int(total_sf)), batch_size):  # Limitar a 1M por ahora
            # Construir consulta con OFFSET y LIMIT
            trips_query = f"""
            SELECT 
                trip_id,
                start_date as start_time,
                end_date as end_time,
                duration_sec,
                start_station_id,
                end_station_id,
                bike_number as bike_id,
                subscriber_type as user_type,
                member_birth_year as birth_year,
                member_gender as gender,
                'San Francisco' as city
            FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
            WHERE 
                duration_sec IS NOT NULL 
                AND start_station_id IS NOT NULL
                AND end_station_id IS NOT NULL
            ORDER BY start_date
            LIMIT {batch_size} OFFSET {offset}
            """
            
            print(f"🔄 Extrayendo lote SF #{offset//batch_size + 1}: registros {offset}-{offset+batch_size}")
            sf_batch = client.query(trips_query).to_dataframe()
            
            # Convertir fechas si es necesario
            if 'start_time' in sf_batch.columns and not pd.api.types.is_datetime64_dtype(sf_batch['start_time']):
                sf_batch['start_time'] = pd.to_datetime(sf_batch['start_time'])
            
            if 'end_time' in sf_batch.columns and not pd.api.types.is_datetime64_dtype(sf_batch['end_time']):
                sf_batch['end_time'] = pd.to_datetime(sf_batch['end_time'])
            
            # Determinar el mes y año para la partición
            first_date = sf_batch['start_time'].min()
            last_date = sf_batch['start_time'].max()
            
            # Formatear fechas para el nombre de la partición
            start_date_str = first_date.strftime('%Y-%m-%d')
            end_date_str = last_date.strftime('%Y-%m-%d')
            
            # Crear directorio de la partición
            partition_dir = output_path / f"partition_{start_date_str}_{end_date_str}"
            partition_dir.mkdir(exist_ok=True)
            
            # Guardar datos en formato parquet
            trips_path = partition_dir / "trips.parquet"
            sf_batch.to_parquet(trips_path, index=False)
            
            # Guardar también las estaciones
            stations_path = partition_dir / "stations.parquet"
            sf_stations.to_parquet(stations_path, index=False)
            
            # Crear archivo de estadísticas
            stats = {
                "partition": {
                    "start_date": start_date_str,
                    "end_date": end_date_str
                },
                "trips_count": len(sf_batch),
                "stations_count": len(sf_stations),
                "cities": ["San Francisco"],
                "source": "BigQuery"
            }
            
            # Guardar estadísticas
            with open(partition_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            print(f"✅ Guardado lote en {partition_dir}: {len(sf_batch)} viajes")
            
            # Actualizar estadísticas globales
            extraction_stats["partitions"].append({
                "path": str(partition_dir),
                "trips_count": len(sf_batch),
                "stations_count": len(sf_stations),
                "start_date": start_date_str,
                "end_date": end_date_str,
                "city": "San Francisco"
            })
            
            # Si el lote tiene menos registros de los solicitados, hemos terminado
            if len(sf_batch) < batch_size:
                break
        
        # --- Extracción de datos de NYC ---
        print("🔄 Comenzando extracción de datos de New York...")
        
        # Extraer datos de estaciones de NYC - Crear con start_stations desde los viajes
        nyc_stations_query = """
        SELECT DISTINCT
            start_station_id as station_id,
            start_station_name as name,
            'New York' as city,
            start_station_latitude as latitude,
            start_station_longitude as longitude,
            0 as capacity  -- No disponible en el esquema
        FROM `bigquery-public-data.new_york_citibike.citibike_trips`
        WHERE start_station_id IS NOT NULL
        AND start_station_name IS NOT NULL
        LIMIT 500  -- Limitar cantidad de estaciones
        """
        
        nyc_stations = client.query(nyc_stations_query).to_dataframe()
        print(f"✅ Extraídas {len(nyc_stations)} estaciones de New York")
        
        # Determinar el número total de registros en NYC
        nyc_count_query = """
        SELECT COUNT(*) as total 
        FROM `bigquery-public-data.new_york_citibike.citibike_trips`
        WHERE start_station_id IS NOT NULL
        AND end_station_id IS NOT NULL
        """
        
        total_nyc = client.query(nyc_count_query).to_dataframe().iloc[0]['total']
        print(f"🔢 NYC tiene un total de {total_nyc} registros. Extrayendo por lotes de {batch_size}...")
        extraction_stats["nyc_total"] = int(total_nyc)
        
        # Extraer datos por lotes
        for offset in range(0, min(1000000, int(total_nyc)), batch_size):  # Limitar a 1M por ahora
            # Construir consulta con OFFSET y LIMIT
            nyc_query = f"""
            SELECT 
                CAST(bikeid AS STRING) as trip_id,  -- Usando bikeid como trip_id
                starttime as start_time,
                stoptime as end_time,
                tripduration as duration_sec,
                start_station_id,
                end_station_id,
                CAST(bikeid AS STRING) as bike_id,
                usertype as user_type,
                birth_year,
                gender,
                'New York' as city
            FROM `bigquery-public-data.new_york_citibike.citibike_trips`
            WHERE 
                start_station_id IS NOT NULL
                AND end_station_id IS NOT NULL
            ORDER BY starttime
            LIMIT {batch_size} OFFSET {offset}
            """
            
            print(f"🔄 Extrayendo lote NYC #{offset//batch_size + 1}: registros {offset}-{offset+batch_size}")
            nyc_batch = client.query(nyc_query).to_dataframe()
            
            # Convertir fechas si es necesario
            if 'start_time' in nyc_batch.columns and not pd.api.types.is_datetime64_dtype(nyc_batch['start_time']):
                nyc_batch['start_time'] = pd.to_datetime(nyc_batch['start_time'])
            
            if 'end_time' in nyc_batch.columns and not pd.api.types.is_datetime64_dtype(nyc_batch['end_time']):
                nyc_batch['end_time'] = pd.to_datetime(nyc_batch['end_time'])
            
            # Determinar el mes y año para la partición
            first_date = nyc_batch['start_time'].min()
            last_date = nyc_batch['start_time'].max()
            
            # Formatear fechas para el nombre de la partición
            start_date_str = first_date.strftime('%Y-%m-%d')
            end_date_str = last_date.strftime('%Y-%m-%d')
            
            # Crear directorio de la partición
            partition_dir = output_path / f"partition_{start_date_str}_{end_date_str}"
            partition_dir.mkdir(exist_ok=True)
            
            # Guardar datos en formato parquet
            trips_path = partition_dir / "trips.parquet"
            nyc_batch.to_parquet(trips_path, index=False)
            
            # Guardar también las estaciones
            stations_path = partition_dir / "stations.parquet"
            nyc_stations.to_parquet(stations_path, index=False)
            
            # Crear archivo de estadísticas
            stats = {
                "partition": {
                    "start_date": start_date_str,
                    "end_date": end_date_str
                },
                "trips_count": len(nyc_batch),
                "stations_count": len(nyc_stations),
                "cities": ["New York"],
                "source": "BigQuery"
            }
            
            # Guardar estadísticas
            with open(partition_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            print(f"✅ Guardado lote en {partition_dir}: {len(nyc_batch)} viajes")
            
            # Actualizar estadísticas globales
            extraction_stats["partitions"].append({
                "path": str(partition_dir),
                "trips_count": len(nyc_batch),
                "stations_count": len(nyc_stations),
                "start_date": start_date_str,
                "end_date": end_date_str,
                "city": "New York"
            })
            
            # Si el lote tiene menos registros de los solicitados, hemos terminado
            if len(nyc_batch) < batch_size:
                break
        
        # Guardar estadísticas globales
        with open(output_path / "extraction_stats.json", "w") as f:
            json.dump(extraction_stats, f, indent=2)
        
        print(f"✅ Extracción completada. Total: {sum(p['trips_count'] for p in extraction_stats['partitions'])} viajes en {len(extraction_stats['partitions'])} particiones")
        
        return extraction_stats
        
    except Exception as e:
        print(f"❌ Error durante la extracción: {str(e)}")
        raise

# Tarea para extraer e ingestar datos de BigQuery
extract_task = PythonOperator(
    task_id='extract_bikeshare_data',
    python_callable=extract_bikeshare_data,
    provide_context=True,
    dag=dag,
)

# Solo una tarea por ahora
extract_task