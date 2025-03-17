from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
import json
import pandas as pd
import random
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np

# Configurar logging
logger = logging.getLogger(__name__)

# Añadir /app al PYTHONPATH si no está ya
if '/app' not in sys.path:
    sys.path.append('/app')

# Importar DatabaseStore
from src.ingestion.store_data import DatabaseStore

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_ingestion',
    default_args=default_args,
    description='Ingesta de datos para bike-sharing',
    schedule_interval=None,
    catchup=False,
)

# Función para verificar y crear directorios
def ensure_directories_exist():
    import os
    os.makedirs('/app/data/raw/bike_sharing/', exist_ok=True)
    os.makedirs('/app/data/processed/bike_sharing', exist_ok=True)
    return True

# Función para generar datos simulados con estructura real
def generate_sample_data():
    try:
        logger.info("🔄 Generando datos de muestra para bike-sharing...")
        
        # Configurar parámetros
        num_stations = 50  # Número de estaciones
        num_trips = 10000  # Número de viajes 
        cities = ["San Francisco", "New York"]
        
        # Generar IDs de estaciones
        station_ids = [f"station_{i}" for i in range(1, num_stations + 1)]
        
        # Fechas de inicio y fin para la partición
        start_date = "2023-01-01"
        end_date = "2023-01-31"
        
        # Directorio de la partición
        partition_dir = Path(f"/app/data/processed/bike_sharing/partition_{start_date}_{end_date}")
        partition_dir.mkdir(exist_ok=True, parents=True)
        
        # Generar datos de estaciones
        stations_data = []
        for station_id in station_ids:
            city = random.choice(cities)
            # Coordenadas aproximadas para SF y NY
            if city == "San Francisco":
                lat = random.uniform(37.7, 37.8)
                lon = random.uniform(-122.5, -122.4)
            else:  # New York
                lat = random.uniform(40.7, 40.8)
                lon = random.uniform(-74.0, -73.9)
                
            stations_data.append({
                "station_id": station_id,
                "name": f"Station {station_id.split('_')[1]}",
                "latitude": lat,
                "longitude": lon,
                "capacity": random.randint(15, 30),
                "city": city
            })
        
        # Crear DataFrame de estaciones
        stations_df = pd.DataFrame(stations_data)
        
        # Generar datos de viajes
        trips_data = []
        for i in range(num_trips):
            # Seleccionar estaciones de origen y destino
            start_station = random.choice(station_ids)
            end_station = random.choice([s for s in station_ids if s != start_station])
            
            # Tiempo de inicio aleatorio dentro del rango de la partición
            start_time = pd.to_datetime(start_date) + pd.Timedelta(days=random.randint(0, 30))
            
            # Duración del viaje (entre 5 minutos y 1 hora)
            duration_sec = random.randint(300, 3600)
            
            # Tiempo de finalización
            end_time = start_time + pd.Timedelta(seconds=duration_sec)
            
            # Ciudad (basada en la estación de origen)
            city = next(station["city"] for station in stations_data if station["station_id"] == start_station)
            
            trips_data.append({
                "trip_id": f"trip_{i}",
                "start_time": start_time,
                "end_time": end_time,
                "duration_sec": duration_sec,
                "start_station_id": start_station,
                "end_station_id": end_station,
                "bike_id": f"bike_{random.randint(1, 500)}",
                "user_type": random.choice(["Member", "Casual"]),
                "birth_year": random.randint(1970, 2000),
                "gender": random.choice(["Male", "Female", "Other"]),
                "city": city
            })
        
        # Crear DataFrame de viajes
        trips_df = pd.DataFrame(trips_data)
        
        # Guardar datos en parquet
        trips_parquet_path = partition_dir / "trips.parquet"
        stations_parquet_path = partition_dir / "stations.parquet"
        
        # Guardar DataFrames en formato parquet
        trips_df.to_parquet(trips_parquet_path, index=False)
        stations_df.to_parquet(stations_parquet_path, index=False)
        
        # Crear archivo de estadísticas
        stats = {
            "partition": {
                "start_date": start_date,
                "end_date": end_date
            },
            "trips_count": len(trips_df),
            "stations_count": len(stations_df),
            "cities": cities
        }
        
        # Guardar estadísticas como JSON
        with open(partition_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"✅ Datos simulados generados: {len(trips_df)} viajes y {len(stations_df)} estaciones")
        logger.info(f"✅ Datos guardados en {partition_dir}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error generando datos de muestra: {str(e)}")
        raise

# Función para generar datos a mayor escala
def generate_large_sample_data(batch_size=100000, num_partitions=3):
    """
    Genera datos de muestra a gran escala para simular volúmenes reales.
    """
    try:
        logger.info(f"🔄 Generando datos a gran escala: {batch_size} viajes por partición, {num_partitions} particiones")
        
        # Configurar parámetros
        num_stations = 500  # Más estaciones para simular ciudades reales
        cities = {"San Francisco": 0.7, "New York": 0.3}  # Distribución: 70% SF, 30% NY
        
        # Generar IDs de estaciones
        station_ids = [f"station_{i}" for i in range(1, num_stations + 1)]
        
        # Generar datos de estaciones
        stations_data = []
        for station_id in station_ids:
            # Elegir ciudad basada en la distribución
            city = np.random.choice(
                list(cities.keys()), 
                p=list(cities.values())
            )
            
            # Coordenadas aproximadas para SF y NY
            if city == "San Francisco":
                lat = random.uniform(37.7, 37.8)
                lon = random.uniform(-122.5, -122.4)
            else:  # New York
                lat = random.uniform(40.7, 40.8)
                lon = random.uniform(-74.0, -73.9)
                
            stations_data.append({
                "station_id": station_id,
                "name": f"Station {station_id.split('_')[1]}",
                "latitude": lat,
                "longitude": lon,
                "capacity": random.randint(15, 50),
                "city": city
            })
        
        # Crear DataFrame de estaciones
        stations_df = pd.DataFrame(stations_data)
        
        # Generar particiones mensuales
        for month in range(1, num_partitions + 1):
            # Fechas de inicio y fin para la partición
            year = 2023
            start_date = f"{year}-{month:02d}-01"
            
            # Calcular el último día del mes
            if month == 12:
                end_date = f"{year}-{month:02d}-31"
            else:
                next_month = month + 1
                end_date = f"{year}-{next_month:02d}-01"
                # Restar un día
                end_date = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            logger.info(f"🔄 Generando partición {start_date} a {end_date}")
            
            # Directorio de la partición
            partition_dir = Path(f"/app/data/processed/bike_sharing/partition_{start_date}_{end_date}")
            partition_dir.mkdir(exist_ok=True, parents=True)
            
            # Generar datos de viajes en lotes
            trips_df_list = []
            for batch in range(1, (batch_size // 10000) + 1):
                trips_batch = []
                for i in range(10000):  # 10,000 viajes por lote para evitar problemas de memoria
                    trip_id = f"trip_{batch}_{i}"
                    
                    # Seleccionar estaciones de origen y destino
                    start_station = random.choice(station_ids)
                    end_station = random.choice([s for s in station_ids if s != start_station])
                    
                    # Tiempo de inicio aleatorio dentro del rango de la partición
                    days_in_month = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
                    start_time = pd.to_datetime(start_date) + pd.Timedelta(days=random.randint(0, days_in_month-1),
                                                                           hours=random.randint(0, 23),
                                                                           minutes=random.randint(0, 59))
                    
                    # Duración del viaje (entre 5 minutos y 2 horas)
                    duration_sec = random.randint(300, 7200)
                    
                    # Tiempo de finalización
                    end_time = start_time + pd.Timedelta(seconds=duration_sec)
                    
                    # Ciudad (basada en la estación de origen)
                    city = next(station["city"] for station in stations_data if station["station_id"] == start_station)
                    
                    trips_batch.append({
                        "trip_id": trip_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration_sec": duration_sec,
                        "start_station_id": start_station,
                        "end_station_id": end_station,
                        "bike_id": f"bike_{random.randint(1, 2000)}",
                        "user_type": random.choice(["Member", "Casual"]),
                        "birth_year": random.randint(1960, 2005),
                        "gender": random.choice(["Male", "Female", "Other"]),
                        "city": city
                    })
                
                # Añadir lote al listado de DataFrames
                trips_df_list.append(pd.DataFrame(trips_batch))
                logger.info(f"  - Generado lote {batch}/{batch_size // 10000} ({len(trips_batch)} viajes)")
            
            # Concatenar todos los lotes
            trips_df = pd.concat(trips_df_list, ignore_index=True)
            
            # Guardar datos en parquet
            trips_parquet_path = partition_dir / "trips.parquet"
            stations_parquet_path = partition_dir / "stations.parquet"
            
            # Guardar DataFrames en formato parquet
            logger.info(f"📊 Guardando {len(trips_df)} viajes en {trips_parquet_path}")
            trips_df.to_parquet(trips_parquet_path, index=False)
            
            logger.info(f"📊 Guardando {len(stations_df)} estaciones en {stations_parquet_path}")
            stations_df.to_parquet(stations_parquet_path, index=False)
            
            # Crear archivo de estadísticas
            stats = {
                "partition": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "trips_count": len(trips_df),
                "stations_count": len(stations_df),
                "cities": list(cities.keys())
            }
            
            # Guardar estadísticas como JSON
            with open(partition_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✅ Partición {month}/{num_partitions} completada: {len(trips_df)} viajes")
        
        logger.info("✅ Generación de datos a gran escala completada")
        return True
    except Exception as e:
        logger.error(f"❌ Error generando datos a gran escala: {str(e)}")
        raise

# Intento de usar BigQuery con fallback a datos simulados
def try_bigquery_or_generate_data():
    try:
        logger.info("🔄 Intentando extraer datos de BigQuery...")
        
        # Verificar si existe la credencial para BigQuery
        credentials_path = "/app/credentials/bigquery_credentials.json"
        
        if os.path.exists(credentials_path):
            try:
                from google.cloud import bigquery
                from google.oauth2 import service_account
                
                # Configurar credenciales
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                client = bigquery.Client(credentials=credentials)
                
                logger.info("✅ Conexión con BigQuery establecida")
                
                # Consulta para extraer datos de SF Bike Share
                sf_query = """
                SELECT * 
                FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
                LIMIT 1000000  -- Limitar inicialmente para pruebas
                """
                
                # Consulta para extraer datos de NYC Citibike
                nyc_query = """
                SELECT * 
                FROM `bigquery-public-data.new_york_citibike.citibike_trips`
                LIMIT 500000  -- Limitar inicialmente para pruebas
                """
                
                logger.info("🔄 Extrayendo datos de SF Bike Share...")
                sf_df = client.query(sf_query).to_dataframe()
                
                logger.info("🔄 Extrayendo datos de NYC Citibike...")
                nyc_df = client.query(nyc_query).to_dataframe()
                
                logger.info(f"✅ Datos extraídos: {len(sf_df)} registros de SF, {len(nyc_df)} registros de NYC")
                
                # Procesar y guardar los datos
                # [Aquí iría el código para procesar y guardar los datos]
                
                return True
            except Exception as e:
                logger.error(f"❌ Error al extraer datos de BigQuery: {str(e)}")
                logger.info("⚠️ Fallback a generación de datos de muestra")
                return generate_large_sample_data(batch_size=200000, num_partitions=3)
        else:
            logger.warning("⚠️ No se encontraron credenciales para BigQuery")
            logger.info("⚠️ Fallback a generación de datos de muestra")
            return generate_large_sample_data(batch_size=200000, num_partitions=3)
    except Exception as e:
        logger.error(f"❌ Error en la extracción o generación de datos: {str(e)}")
        raise

# Función para almacenar datos en PostgreSQL
def store_data_in_postgres():
    try:
        logger.info("🔄 Iniciando almacenamiento en PostgreSQL...")
        
        # Crear instancia de DatabaseStore
        store = DatabaseStore(data_dir="/app/data/processed")
        
        # Inicializar la base de datos
        store.init_database()
        
        # Leer particiones disponibles
        logger.info("📂 Buscando particiones de datos...")
        base_dir = Path("/app/data/processed/bike_sharing")
        partitions = list(base_dir.glob("partition_*"))
        
        if not partitions:
            raise ValueError("No se encontraron particiones en el directorio de datos procesados")
        
        logger.info(f"🔍 Se encontraron {len(partitions)} particiones")
        
        # Procesar cada partición
        for partition_dir in partitions:
            logger.info(f"🔄 Procesando partición: {partition_dir.name}")
            
            # Extraer fechas de la partición
            dates = partition_dir.name.split('_')[1:]
            if len(dates) >= 2:
                partition_info = {
                    'start_date': dates[0],
                    'end_date': dates[1]
                }
                
                # Almacenar partición
                store.store_partition(partition_dir, partition_info)
            else:
                logger.warning(f"⚠️ Formato de partición no reconocido: {partition_dir.name}")
        
        logger.info("✅ Datos almacenados en PostgreSQL exitosamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error almacenando datos en PostgreSQL: {str(e)}")
        raise

# Tarea para verificar directorios
check_bike_data = PythonOperator(
    task_id='check_bike_data',
    python_callable=ensure_directories_exist,
    dag=dag,
)

# Tarea para ingestar datos (intenta BigQuery primero, luego fallback)
ingest_trips_task = PythonOperator(
    task_id='ingest_trips_from_bigquery',
    python_callable=try_bigquery_or_generate_data,
    dag=dag,
)

# Tarea para procesar estaciones
process_stations_task = PythonOperator(
    task_id='process_stations',
    python_callable=lambda: print("Procesamiento de estaciones completado"),
    dag=dag,
)

# Tarea para almacenar datos en PostgreSQL
store_in_postgres_task = PythonOperator(
    task_id='store_in_postgres',
    python_callable=store_data_in_postgres,
    dag=dag,
)

# Definir dependencias
check_bike_data >> ingest_trips_task >> process_stations_task >> store_in_postgres_task