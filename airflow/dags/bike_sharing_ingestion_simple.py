from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Añadir /app al PYTHONPATH si no está ya
if '/app' not in sys.path:
    sys.path.append('/app')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_postgres_storage',
    default_args=default_args,
    description='Almacenamiento de datos bike-sharing en PostgreSQL',
    schedule_interval=None,
    catchup=False,
)

# Función para verificar y crear directorios
def ensure_directories_exist():
    import os
    os.makedirs('/app/data/raw/bike_sharing/', exist_ok=True)
    os.makedirs('/app/data/processed/bike_sharing', exist_ok=True)
    return True

# Función para verificar los archivos parquet existentes
def check_parquet_files():
    import os
    import json
    from pathlib import Path
    
    data_dir = Path("/app/data/processed/bike_sharing")
    partitions = list(data_dir.glob("partition_*"))
    
    if not partitions:
        raise ValueError("No se encontraron particiones en el directorio de datos procesados")
    
    results = []
    for partition in partitions:
        # Verificar archivos
        trips_file = partition / "trips.parquet"
        stations_file = partition / "stations.parquet"
        stats_file = partition / "stats.json"
        
        if trips_file.exists() and stations_file.exists() and stats_file.exists():
            # Leer estadísticas
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            partition_info = {
                'partition_path': str(partition),
                'trips_count': stats.get('trips_count', 0),
                'stations_count': stats.get('stations_count', 0),
                'start_date': stats.get('partition', {}).get('start_date'),
                'end_date': stats.get('partition', {}).get('end_date')
            }
            results.append(partition_info)
    
    print(f"✅ Se encontraron {len(results)} particiones con datos:")
    for partition in results:
        print(f"  - {partition['partition_path']}: {partition['trips_count']} viajes, {partition['stations_count']} estaciones")
    
    return results

# Función para almacenar datos en PostgreSQL
def store_data_in_postgres():
    try:
        print("🔄 Iniciando almacenamiento en PostgreSQL...")
        
        # Importar la clase DatabaseStore
        from src.ingestion.store_data import DatabaseStore
        
        # Crear instancia de DatabaseStore
        store = DatabaseStore(data_dir="/app/data/processed")
        
        # Inicializar la base de datos
        store.init_database()
        
        # Leer particiones disponibles
        print("📂 Buscando particiones de datos...")
        base_dir = Path("/app/data/processed/bike_sharing")
        partitions = list(base_dir.glob("partition_*"))
        
        if not partitions:
            raise ValueError("No se encontraron particiones en el directorio de datos procesados")
        
        print(f"🔍 Se encontraron {len(partitions)} particiones")
        
        # Procesar cada partición
        for partition_dir in partitions:
            print(f"🔄 Procesando partición: {partition_dir.name}")
            
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
                print(f"⚠️ Formato de partición no reconocido: {partition_dir.name}")
        
        print("✅ Almacenamiento en PostgreSQL completado exitosamente")
        return True
    
    except Exception as e:
        print(f"❌ Error almacenando datos en PostgreSQL: {str(e)}")
        raise

# Tareas
check_dirs_task = PythonOperator(
    task_id='check_directories',
    python_callable=ensure_directories_exist,
    dag=dag,
)

check_files_task = PythonOperator(
    task_id='check_parquet_files',
    python_callable=check_parquet_files,
    dag=dag,
)

store_data_task = PythonOperator(
    task_id='store_in_postgres',
    python_callable=store_data_in_postgres,
    dag=dag,
)

# Dependencias
check_dirs_task >> check_files_task >> store_data_task