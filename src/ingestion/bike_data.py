import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, to_timestamp, unix_timestamp, year, month, 
    dayofmonth, hour, date_format, count, when, udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, IntegerType
)
import h3
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeDataProcessor:
    """Procesa datos de bike-sharing de múltiples ciudades."""
    
    # Schema unificado para viajes
    TRIP_SCHEMA = StructType([
        StructField("trip_id", StringType(), True),
        StructField("start_time", TimestampType(), True),
        StructField("end_time", TimestampType(), True),
        StructField("duration_sec", IntegerType(), True),
        StructField("start_station_id", StringType(), True),
        StructField("start_station_name", StringType(), True),
        StructField("start_lat", DoubleType(), True),
        StructField("start_lon", DoubleType(), True),
        StructField("end_station_id", StringType(), True),
        StructField("end_station_name", StringType(), True),
        StructField("end_lat", DoubleType(), True),
        StructField("end_lon", DoubleType(), True),
        StructField("bike_id", StringType(), True),
        StructField("user_type", StringType(), True),
        StructField("birth_year", IntegerType(), True),
        StructField("gender", StringType(), True),
        StructField("city", StringType(), True)
    ])

    def __init__(
        self,
        output_dir: str = "data/processed/bike_sharing",
        base_resolution: int = 9,
        partition_months: int = 12  # Número de meses de datos a procesar
    ):
        """
        Inicializa el procesador de datos de bike-sharing.
        
        Args:
            output_dir: Directorio para datos procesados
            base_resolution: Resolución H3 base
            partition_months: Número de meses a procesar
        """
        self.output_dir = Path(output_dir)
        self.base_resolution = base_resolution
        self.partition_months = partition_months
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar Spark
        self.spark = self._init_spark()
        
        # Cargar credenciales GCP
        load_dotenv()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark con configuración para BigQuery y procesamiento de datos grandes."""
        return SparkSession.builder \
            .appName("Bike-Sharing-Analysis") \
            .config("spark.jars.packages", 
                   "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.1") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.default.parallelism", "100") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
    
    def get_date_partitions(self) -> List[Dict[str, str]]:
        """
        Genera rangos de fechas para particionamiento.
        
        Returns:
            Lista de diccionarios con fechas inicio/fin
        """
        end_date = datetime.now()
        partitions = []
        
        for i in range(self.partition_months):
            start_date = end_date - timedelta(days=30)
            partitions.append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            })
            end_date = start_date
        
        return partitions
    
    def fetch_ny_data(self, partition: Dict[str, str]) -> DataFrame:
        """
        Obtiene datos de NYC Citibike para un rango de fechas.
        
        Args:
            partition: Diccionario con fechas inicio/fin
            
        Returns:
            DataFrame con datos de viajes
        """
        query = f"""
        SELECT 
            CAST(bikeid AS STRING) as trip_id,
            starttime as start_time,
            stoptime as end_time,
            tripduration as duration_sec,
            CAST(start_station_id AS STRING) as start_station_id,
            start_station_name,
            start_station_latitude as start_lat,
            start_station_longitude as start_lon,
            CAST(end_station_id AS STRING) as end_station_id,
            end_station_name,
            end_station_latitude as end_lat,
            end_station_longitude as end_lon,
            CAST(bikeid AS STRING) as bike_id,
            usertype as user_type,
            birth_year,
            gender,
            'nyc' as city
        FROM `bigquery-public-data.new_york_citibike.citibike_trips`
        WHERE DATE(starttime) BETWEEN '{partition['start_date']}' AND '{partition['end_date']}'
            AND start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
        """
        
        return self.spark.read.format('bigquery') \
            .option('query', query) \
            .load()
    
    def fetch_sf_data(self, partition: Dict[str, str]) -> DataFrame:
        """
        Obtiene datos de San Francisco bike share para un rango de fechas.
        
        Args:
            partition: Diccionario con fechas inicio/fin
            
        Returns:
            DataFrame con datos de viajes
        """
        query = f"""
        SELECT 
            trip_id,
            start_date as start_time,
            end_date as end_time,
            duration_sec,
            CAST(start_station_id AS STRING) as start_station_id,
            start_station_name,
            start_station_latitude as start_lat,
            start_station_longitude as start_lon,
            CAST(end_station_id AS STRING) as end_station_id,
            end_station_name,
            end_station_latitude as end_lat,
            end_station_longitude as end_lon,
            CAST(bike_number AS STRING) as bike_id,
            subscriber_type as user_type,
            member_birth_year as birth_year,
            member_gender as gender,
            'sf' as city
        FROM `bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`
        WHERE DATE(start_date) BETWEEN '{partition['start_date']}' AND '{partition['end_date']}'
            AND start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
        """
        
        return self.spark.read.format('bigquery') \
            .option('query', query) \
            .load()
    
    def process_station_data(self, trips_df: DataFrame) -> DataFrame:
        """
        Procesa datos de estaciones, incluyendo H3 y métricas de uso.
        
        Args:
            trips_df: DataFrame con viajes
            
        Returns:
            DataFrame con datos de estaciones procesados
        """
        # Extraer estaciones únicas
        start_stations = trips_df.select(
            col("start_station_id").alias("station_id"),
            col("start_station_name").alias("name"),
            col("start_lat").alias("latitude"),
            col("start_lon").alias("longitude"),
            col("city"),
            col("start_time")
        )
        
        end_stations = trips_df.select(
            col("end_station_id").alias("station_id"),
            col("end_station_name").alias("name"),
            col("end_lat").alias("latitude"),
            col("end_lon").alias("longitude"),
            col("city"),
            col("end_time").alias("start_time")
        )
        
        # Calcular métricas por estación
        station_metrics = start_stations.union(end_stations) \
            .groupBy("station_id", "name", "latitude", "longitude", "city") \
            .agg(
                count("*").alias("total_trips"),
                count(
                    when(
                        hour("start_time").between(7, 9) |
                        hour("start_time").between(16, 18), 
                        True
                    )
                ).alias("peak_hour_trips")
            )
        
        # Agregar índices H3
        def get_h3_index(lat, lon):
            try:
                return h3.geo_to_h3(lat, lon, self.base_resolution)
            except:
                return None
        
        get_h3_udf = udf(get_h3_index, StringType())
        
        return station_metrics.withColumn(
            "h3_index",
            get_h3_udf(col("latitude"), col("longitude"))
        )
    
    def save_partition(
        self,
        trips_df: DataFrame,
        stations_df: DataFrame,
        partition: Dict[str, str]
    ) -> None:
        """
        Guarda datos de una partición.
        
        Args:
            trips_df: DataFrame de viajes
            stations_df: DataFrame de estaciones
            partition: Información de la partición
        """
        # Crear directorio para la partición
        partition_dir = self.output_dir / f"partition_{partition['start_date']}_{partition['end_date']}"
        partition_dir.mkdir(exist_ok=True)
        
        # Guardar viajes en parquet
        trips_path = partition_dir / "trips.parquet"
        trips_df.write.parquet(str(trips_path), mode="overwrite")
        
        # Guardar estaciones en parquet
        stations_path = partition_dir / "stations.parquet"
        stations_df.write.parquet(str(stations_path), mode="overwrite")
        
        logger.info(f"💾 Datos guardados en {partition_dir}")
        
        # Guardar estadísticas
        stats = {
            'partition': partition,
            'trips_count': trips_df.count(),
            'stations_count': stations_df.count(),
            'processed_at': datetime.now().isoformat()
        }
        
        stats_path = partition_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def process(self) -> None:
        """Ejecuta el pipeline completo de procesamiento."""
        try:
            logger.info("🚀 Iniciando procesamiento de datos de bike-sharing")
            
            # Generar particiones
            partitions = self.get_date_partitions()
            
            # Procesar cada partición
            for partition in partitions:
                logger.info(f"🔄 Procesando partición: {partition['start_date']} - {partition['end_date']}")
                
                # Obtener datos
                ny_trips = self.fetch_ny_data(partition)
                sf_trips = self.fetch_sf_data(partition)
                
                # Unir datos
                all_trips = ny_trips.union(sf_trips)
                
                # Procesar estaciones
                stations = self.process_station_data(all_trips)
                
                # Guardar partición
                self.save_partition(all_trips, stations, partition)
                
                logger.info(f"✅ Partición completada: {partition['start_date']}")
            
            logger.info("✅ Procesamiento completado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error en el procesamiento: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar el procesamiento."""
    processor = BikeDataProcessor()
    processor.process()

if __name__ == "__main__":
    main()