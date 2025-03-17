"""
Process bike sharing trip data from multiple cities.
This module handles the processing of trip data from different bike sharing systems,
standardizing formats and calculating metrics.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, hour, dayofweek, month, year, 
    date_format, to_timestamp, lit, unix_timestamp,
    expr, datediff, abs, array, min, max, avg, count,
    percentile_approx, window, desc
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType, BooleanType
)
from datetime import datetime, timedelta
import json
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from dotenv import load_dotenv
import h3
from geopy.distance import geodesic

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class TripProcessor:
    """Procesa datos de viajes de sistemas de bike-sharing."""
    
    # Schema unificado para viajes
    TRIPS_SCHEMA = StructType([
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
        StructField("city", StringType(), True)
    ])
    
    def __init__(
        self,
        input_dir: str = "data/raw/trips",
        output_dir: str = "data/processed/trips",
        h3_resolution: int = 9,
        city_configs: Optional[Dict] = None
    ):
        """
        Inicializa el procesador de viajes.
        
        Args:
            input_dir: Directorio con datos crudos
            output_dir: Directorio para datos procesados
            h3_resolution: Resolución H3 para análisis espacial
            city_configs: Configuraciones por ciudad
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.h3_resolution = h3_resolution
        
        # Configuración por defecto de ciudades
        self.city_configs = city_configs or {
            "nyc": {
                "file_pattern": "citibike_trips_*.csv",
                "has_header": True,
                "date_format": "yyyy-MM-dd HH:mm:ss"
            },
            "sf": {
                "file_pattern": "sf_bikeshare_trips_*.csv",
                "has_header": True,
                "date_format": "yyyy-MM-dd HH:mm:ss"
            }
        }
        
        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar Spark
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa una sesión de Spark optimizada para procesamiento de viajes."""
        return SparkSession.builder \
            .appName("Bike-Trip-Processing") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .config("spark.sql.files.maxPartitionBytes", "128m") \
            .getOrCreate()
    
    def load_trips_from_bigquery(
        self,
        city: str,
        start_date: str,
        end_date: str,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Carga datos de viajes desde BigQuery.
        
        Args:
            city: Ciudad a procesar ('nyc' o 'sf')
            start_date: Fecha de inicio (YYYY-MM-DD)
            end_date: Fecha de fin (YYYY-MM-DD)
            limit: Límite opcional de registros
            
        Returns:
            DataFrame con datos de viajes
        """
        logger.info(f"🔄 Cargando datos de {city} desde BigQuery")
        
        # Seleccionar tabla y mapeo de campos según la ciudad
        if city == 'nyc':
            table = "`bigquery-public-data.new_york_citibike.citibike_trips`"
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
                'nyc' as city
            FROM {table}
            WHERE starttime BETWEEN '{start_date}' AND '{end_date}'
            AND start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
            """
        elif city == 'sf':
            table = "`bigquery-public-data.san_francisco_bikeshare.bikeshare_trips`"
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
                'sf' as city
            FROM {table}
            WHERE start_date BETWEEN '{start_date}' AND '{end_date}'
            AND start_station_id IS NOT NULL
            AND end_station_id IS NOT NULL
            """
        else:
            raise ValueError(f"Ciudad {city} no soportada")
        
        # Agregar límite si especificado
        if limit:
            query += f"\nLIMIT {limit}"
        
        # Ejecutar query
        df = self.spark.read.format('bigquery') \
            .option('query', query) \
            .load()
        
        count = df.count()
        logger.info(f"✅ Cargados {count:,} viajes de {city}")
        
        return df
    
    def load_trips_from_files(
        self,
        city: str
    ) -> pd.DataFrame:
        """
        Carga datos de viajes desde archivos.
        
        Args:
            city: Ciudad a procesar
            
        Returns:
            DataFrame con datos de viajes
        """
        logger.info(f"🔄 Cargando datos de {city} desde archivos")
        
        city_config = self.city_configs.get(city)
        if not city_config:
            raise ValueError(f"Configuración para ciudad {city} no encontrada")
        
        # Buscar archivos
        city_dir = self.input_dir / city
        files = list(city_dir.glob(city_config["file_pattern"]))
        
        if not files:
            raise FileNotFoundError(f"No se encontraron archivos para {city}")
        
        # Leer archivos con Spark
        df = self.spark.read.option(
            "header", city_config["has_header"]
        ).csv(str(city_dir / city_config["file_pattern"]))
        
        # Mapear a schema unificado
        if city == 'nyc':
            df = df.select(
                col("bikeid").cast("string").alias("trip_id"),
                to_timestamp(col("starttime"), city_config["date_format"]).alias("start_time"),
                to_timestamp(col("stoptime"), city_config["date_format"]).alias("end_time"),
                col("tripduration").cast("int").alias("duration_sec"),
                col("start_station_id").cast("string").alias("start_station_id"),
                col("start_station_name"),
                col("start_station_latitude").cast("double").alias("start_lat"),
                col("start_station_longitude").cast("double").alias("start_lon"),
                col("end_station_id").cast("string").alias("end_station_id"),
                col("end_station_name"),
                col("end_station_latitude").cast("double").alias("end_lat"),
                col("end_station_longitude").cast("double").alias("end_lon"),
                col("bikeid").cast("string").alias("bike_id"),
                col("usertype").alias("user_type"),
                lit(city).alias("city")
            )
        elif city == 'sf':
            df = df.select(
                col("trip_id").cast("string"),
                to_timestamp(col("start_date"), city_config["date_format"]).alias("start_time"),
                to_timestamp(col("end_date"), city_config["date_format"]).alias("end_time"),
                col("duration_sec").cast("int"),
                col("start_station_id").cast("string").alias("start_station_id"),
                col("start_station_name"),
                col("start_station_latitude").cast("double").alias("start_lat"),
                col("start_station_longitude").cast("double").alias("start_lon"),
                col("end_station_id").cast("string").alias("end_station_id"),
                col("end_station_name"),
                col("end_station_latitude").cast("double").alias("end_lat"),
                col("end_station_longitude").cast("double").alias("end_lon"),
                col("bike_number").cast("string").alias("bike_id"),
                col("subscriber_type").alias("user_type"),
                lit(city).alias("city")
            )
        
        count = df.count()
        logger.info(f"✅ Cargados {count:,} viajes de {city}")
        
        return df
    
    def clean_trips(self, df) -> pd.DataFrame:
        """
        Limpia y valida los datos de viajes.
        
        Args:
            df: DataFrame con datos de viajes
            
        Returns:
            DataFrame limpio
        """
        logger.info("🧹 Limpiando datos de viajes")
        
        # Filtrar registros con coordenadas válidas
        clean_df = df.filter(
            (col("start_lat").isNotNull()) &
            (col("start_lon").isNotNull()) &
            (col("end_lat").isNotNull()) &
            (col("end_lon").isNotNull()) &
            (col("start_lat") != 0) &
            (col("start_lon") != 0) &
            (col("end_lat") != 0) &
            (col("end_lon") != 0)
        )
        
        # Filtrar duraciones válidas (entre 60 segundos y 24 horas)
        clean_df = clean_df.filter(
            (col("duration_sec") >= 60) &
            (col("duration_sec") <= 86400)
        )
        
        # Filtrar fechas válidas
        clean_df = clean_df.filter(
            (col("start_time") <= col("end_time")) &
            (col("start_time") >= to_timestamp(lit("2010-01-01"))) &
            (col("end_time") <= to_timestamp(lit("2025-12-31")))
        )
        
        # Estandarizar user_type
        clean_df = clean_df.withColumn(
            "user_type",
            when(
                col("user_type").isin(
                    "Subscriber", "Member", "member", "subscriber"
                ),
                "member"
            ).otherwise(
                when(
                    col("user_type").isin(
                        "Customer", "Casual", "casual", "customer"
                    ),
                    "casual"
                ).otherwise("unknown")
            )
        )
        
        # Calcular estadísticas de limpieza
        initial_count = df.count()
        final_count = clean_df.count()
        removed = initial_count - final_count
        
        logger.info(f"✅ Limpieza completada: {removed:,} registros eliminados ({removed/initial_count:.1%})")
        
        return clean_df
    
    def enrich_trips(self, df) -> pd.DataFrame:
        """
        Enriquece los datos de viajes con métricas adicionales.
        
        Args:
            df: DataFrame con datos de viajes
            
        Returns:
            DataFrame enriquecido
        """
        logger.info("🔄 Enriqueciendo datos de viajes")
        
        # Calcular distancia en línea recta
        def calculate_distance(lat1, lon1, lat2, lon2):
            """Calcula distancia en km."""
            try:
                return geodesic((lat1, lon1), (lat2, lon2)).kilometers
            except:
                return None
        
        calculate_distance_udf = self.spark.udf.register(
            "calculate_distance",
            calculate_distance,
            DoubleType()
        )
        
        enriched_df = df.withColumn(
            "distance_km",
            calculate_distance_udf(
                col("start_lat"), col("start_lon"),
                col("end_lat"), col("end_lon")
            )
        )
        
        # Agregar indice H3 para inicio y fin
        def get_h3_index(lat, lon, resolution):
            try:
                return h3.geo_to_h3(lat, lon, resolution)
            except:
                return None
        
        get_h3_udf = self.spark.udf.register(
            "get_h3_index",
            lambda lat, lon: get_h3_index(lat, lon, self.h3_resolution),
            StringType()
        )
        
        enriched_df = enriched_df.withColumn(
            "start_h3",
            get_h3_udf(col("start_lat"), col("start_lon"))
        ).withColumn(
            "end_h3",
            get_h3_udf(col("end_lat"), col("end_lon"))
        )
        
        # Agregar variables temporales
        enriched_df = enriched_df.withColumn(
            "hour_of_day", hour("start_time")
        ).withColumn(
            "day_of_week", dayofweek("start_time")
        ).withColumn(
            "month", month("start_time")
        ).withColumn(
            "year", year("start_time")
        ).withColumn(
            "is_weekend",
            when(col("day_of_week").isin([1, 7]), 1).otherwise(0)
        ).withColumn(
            "is_rush_hour",
            when(
                (hour("start_time").between(7, 9)) |
                (hour("start_time").between(16, 18)),
                1
            ).otherwise(0)
        )
        
        # Calcular velocidad media (km/h)
        enriched_df = enriched_df.withColumn(
            "speed_kmh",
            when(
                col("duration_sec") > 0,
                col("distance_km") / (col("duration_sec") / 3600)
            ).otherwise(None)
        )
        
        # Filtrar velocidades no razonables (menos de 0.5 km/h o más de 50 km/h)
        enriched_df = enriched_df.filter(
            (col("speed_kmh").isNull()) |
            ((col("speed_kmh") >= 0.5) & (col("speed_kmh") <= 50))
        )
        
        # Calcular estadísticas de enriquecimiento
        stats = enriched_df.select(
            avg("duration_sec").alias("avg_duration"),
            avg("distance_km").alias("avg_distance"),
            avg("speed_kmh").alias("avg_speed")
        ).first()
        
        logger.info(
            f"✅ Enriquecimiento completado: "
            f"duración media={stats['avg_duration']:.1f}s, "
            f"distancia media={stats['avg_distance']:.2f}km, "
            f"velocidad media={stats['avg_speed']:.2f}km/h"
        )
        
        return enriched_df
    
    def calculate_station_metrics(self, trips_df) -> pd.DataFrame:
        """
        Calcula métricas por estación a partir de viajes.
        
        Args:
            trips_df: DataFrame con viajes
            
        Returns:
            DataFrame con métricas por estación
        """
        logger.info("🔄 Calculando métricas por estación")
        
        # Agrupar por estación de inicio
        start_metrics = trips_df.groupBy("start_station_id") \
            .agg(
                first("start_station_name").alias("station_name"),
                first("start_lat").alias("latitude"),
                first("start_lon").alias("longitude"),
                first("city").alias("city"),
                first("start_h3").alias("h3_index"),
                count("*").alias("departures"),
                avg("duration_sec").alias("avg_trip_duration"),
                percentile_approx("duration_sec", 0.5).alias("median_trip_duration")
            )
        
        # Agrupar por estación de fin
        end_metrics = trips_df.groupBy("end_station_id") \
            .agg(
                count("*").alias("arrivals")
            )
        
        # Unir y calcular balance
        station_metrics = start_metrics.join(
            end_metrics,
            start_metrics.start_station_id == end_metrics.end_station_id,
            "left"
        ).select(
            col("start_station_id").alias("station_id"),
            col("station_name"),
            col("latitude"),
            col("longitude"),
            col("city"),
            col("h3_index"),
            col("departures"),
            col("arrivals").fillna(0).cast("long"),
            col("avg_trip_duration"),
            col("median_trip_duration"),
            (col("arrivals") - col("departures")).alias("net_flow")
        )
        
        # Clasificar estaciones por desbalanceo
        station_metrics = station_metrics.withColumn(
            "rebalancing_need",
            when(
                col("net_flow") < -10, "high_outflow"
            ).when(
                col("net_flow") > 10, "high_inflow"
            ).otherwise("balanced")
        )
        
        # Calcular estadísticas
        stats = station_metrics.agg(
            count("*").alias("station_count"),
            avg("departures").alias("avg_departures"),
            avg("arrivals").alias("avg_arrivals"),
            count(when(col("rebalancing_need") == "high_outflow", 1)).alias("outflow_stations"),
            count(when(col("rebalancing_need") == "high_inflow", 1)).alias("inflow_stations")
        ).first()
        
        logger.info(
            f"✅ Métricas por estación calculadas: "
            f"{stats['station_count']} estaciones, "
            f"{stats['outflow_stations']} con alto outflow, "
            f"{stats['inflow_stations']} con alto inflow"
        )
        
        return station_metrics
    
    def save_processed_data(
        self,
        trips_df,
        stations_df,
        partition_col: str = "year"
    ) -> None:
        """
        Guarda datos procesados.
        
        Args:
            trips_df: DataFrame con viajes
            stations_df: DataFrame con estaciones
            partition_col: Columna para particionar datos de viajes
        """
        logger.info("💾 Guardando datos procesados")
        
        # Guardar viajes particionados
        trips_path = self.output_dir / "trips"
        trips_df.write.partitionBy(partition_col) \
            .parquet(str(trips_path), mode="overwrite")
        
        # Guardar estaciones
        stations_path = self.output_dir / "stations"
        stations_df.write.parquet(str(stations_path), mode="overwrite")
        
        # Guardar versión unificada para análisis
        # Para evitar problemas de memoria, tomamos una muestra de viajes
        trips_sample = trips_df.sample(fraction=0.1, seed=42)
        
        # Unificar datos
        all_data_path = self.output_dir / "all_data.parquet"
        trips_sample.write.parquet(str(all_data_path), mode="overwrite")
        
        # Guardar metadata
        trips_count = trips_df.count()
        stations_count = stations_df.count()
        
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "trips_count": trips_count,
            "stations_count": stations_count,
            "cities": trips_df.select("city").distinct().rdd.flatMap(lambda x: x).collect(),
            "date_range": {
                "start": trips_df.select(min("start_time")).first()[0].isoformat(),
                "end": trips_df.select(max("end_time")).first()[0].isoformat()
            },
            "partition_column": partition_col
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(
            f"✅ Datos guardados: "
            f"{trips_count:,} viajes, "
            f"{stations_count:,} estaciones"
        )
    
    def process(
        self,
        use_bigquery: bool = True,
        start_date: str = "2022-01-01",
        end_date: str = "2022-12-31",
        limit_per_city: Optional[int] = None
    ) -> None:
        """
        Ejecuta el pipeline completo de procesamiento.
        
        Args:
            use_bigquery: Si usar BigQuery o archivos locales
            start_date: Fecha de inicio para filtrado (YYYY-MM-DD)
            end_date: Fecha de fin para filtrado (YYYY-MM-DD)
            limit_per_city: Límite opcional de registros por ciudad
        """
        try:
            cities = list(self.city_configs.keys())
            all_trips = []
            
            # Procesar cada ciudad
            for city in cities:
                try:
                    # Cargar datos
                    if use_bigquery:
                        city_trips = self.load_trips_from_bigquery(
                            city, start_date, end_date, limit_per_city
                        )
                    else:
                        city_trips = self.load_trips_from_files(city)
                    
                    all_trips.append(city_trips)
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando {city}: {str(e)}")
            
            # Unir datos de todas las ciudades
            if not all_trips:
                raise ValueError("No se pudieron cargar datos de ninguna ciudad")
                
            trips_df = all_trips[0]
            for df in all_trips[1:]:
                trips_df = trips_df.union(df)
            
            # Limpiar datos
            trips_df = self.clean_trips(trips_df)
            
            # Enriquecer con métricas adicionales
            trips_df = self.enrich_trips(trips_df)
            
            # Calcular métricas por estación
            station_metrics = self.calculate_station_metrics(trips_df)
            
            # Guardar resultados
            self.save_processed_data(trips_df, station_metrics)
            
            logger.info("✅ Procesamiento de viajes completado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error en el procesamiento: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar el procesamiento."""
    processor = TripProcessor()
    processor.process(use_bigquery=True, limit_per_city=1000000)  # Limitar para pruebas

if __name__ == "__main__":
    main()