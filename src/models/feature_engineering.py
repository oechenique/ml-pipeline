import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import geopandas as gpd
import h3
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, expr, udf, array, lit, hour, dayofweek, 
    month, year, count, avg, stddev, sum,
    to_timestamp, date_format, when
)
from pyspark.sql.types import StringType, DoubleType, ArrayType
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeFeatureEngineer:
    """Genera features para el análisis de bike-sharing."""
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/ml_ready",
        h3_resolution: int = 9
    ):
        """
        Inicializa el generador de features.
        
        Args:
            input_dir: Directorio con datos procesados
            output_dir: Directorio para features procesados
            h3_resolution: Nivel de resolución H3
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.h3_resolution = h3_resolution
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar Spark
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark con configuración optimizada."""
        return SparkSession.builder \
            .appName("Bike-Feature-Engineering") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def load_trip_data(self) -> pd.DataFrame:
        """Carga datos de viajes."""
        logger.info("🔄 Cargando datos de viajes")
        
        trips_path = self.input_dir / "bike_sharing/trips.parquet"
        df = pd.read_parquet(trips_path)
        
        logger.info(f"✅ Cargados {len(df)} viajes")
        return df
    
    def load_station_data(self) -> pd.DataFrame:
        """Carga datos de estaciones."""
        logger.info("🔄 Cargando datos de estaciones")
        
        stations_path = self.input_dir / "bike_sharing/stations.parquet"
        df = pd.read_parquet(stations_path)
        
        logger.info(f"✅ Cargadas {len(df)} estaciones")
        return df
    
    def calculate_temporal_features(self, trips_df) -> pd.DataFrame:
        """
        Calcula features temporales.
        
        Args:
            trips_df: DataFrame con viajes
            
        Returns:
            DataFrame con features temporales
        """
        logger.info("🔄 Calculando features temporales")
        
        # Convertir a Spark DataFrame
        sdf = self.spark.createDataFrame(trips_df)
        
        # Extraer componentes temporales
        temporal_features = sdf \
            .withColumn("hour", hour("start_time")) \
            .withColumn("day_of_week", dayofweek("start_time")) \
            .withColumn("month", month("start_time")) \
            .withColumn("year", year("start_time")) \
            .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
            .withColumn("is_rush_hour", 
                when(
                    (hour("start_time").between(7, 9)) |
                    (hour("start_time").between(16, 18)),
                    1
                ).otherwise(0)
            )
        
        return temporal_features.toPandas()
    
    def calculate_station_features(self, trips_df, stations_df) -> pd.DataFrame:
        """
        Calcula features por estación.
        
        Args:
            trips_df: DataFrame con viajes
            stations_df: DataFrame con estaciones
            
        Returns:
            DataFrame con features por estación
        """
        logger.info("🔄 Calculando features por estación")
        
        # Convertir a Spark DataFrames
        trips_sdf = self.spark.createDataFrame(trips_df)
        stations_sdf = self.spark.createDataFrame(stations_df)
        
        # Calcular métricas por estación
        station_metrics = trips_sdf \
            .groupBy("start_station_id") \
            .agg(
                count("*").alias("total_trips"),
                avg("duration_sec").alias("avg_duration"),
                stddev("duration_sec").alias("std_duration"),
                sum("is_rush_hour").alias("rush_hour_trips"),
                avg("is_weekend").alias("weekend_ratio")
            )
        
        # Calcular balance de viajes
        outbound = trips_sdf \
            .groupBy("start_station_id") \
            .agg(count("*").alias("outbound_trips"))
            
        inbound = trips_sdf \
            .groupBy("end_station_id") \
            .agg(count("*").alias("inbound_trips"))
        
        # Unir métricas
        station_features = station_metrics \
            .join(outbound, "start_station_id") \
            .join(
                inbound,
                station_metrics.start_station_id == inbound.end_station_id,
                "left"
            ) \
            .withColumn(
                "trip_balance",
                col("outbound_trips") - col("inbound_trips")
            ) \
            .withColumn(
                "balance_ratio",
                col("inbound_trips") / col("outbound_trips")
            )
        
        return station_features.toPandas()
    
    def calculate_network_features(self, trips_df, stations_df) -> pd.DataFrame:
        """
        Calcula features de red y conectividad.
        
        Args:
            trips_df: DataFrame con viajes
            stations_df: DataFrame con estaciones
            
        Returns:
            DataFrame con features de red
        """
        logger.info("🔄 Calculando features de red")
        
        # Calcular matriz de conectividad
        trips_sdf = self.spark.createDataFrame(trips_df)
        
        connections = trips_sdf \
            .groupBy("start_station_id", "end_station_id") \
            .agg(count("*").alias("connection_strength"))
        
        # Calcular métricas de red por estación
        network_metrics = connections \
            .groupBy("start_station_id") \
            .agg(
                count("end_station_id").alias("unique_destinations"),
                sum("connection_strength").alias("total_connections"),
                avg("connection_strength").alias("avg_connection_strength")
            )
        
        return network_metrics.toPandas()
    
    def calculate_geographic_features(self, stations_df) -> pd.DataFrame:
        """
        Calcula features geográficos.
        
        Args:
            stations_df: DataFrame con estaciones
            
        Returns:
            DataFrame con features geográficos
        """
        logger.info("🔄 Calculando features geográficos")
        
        # Convertir a GeoDataFrame
        gdf = gpd.GeoDataFrame(
            stations_df,
            geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude)
        )
        
        # Calcular distancias a hitos importantes
        # (esto se expandiría con datos reales de POIs)
        gdf['dist_to_center'] = gdf.geometry.distance(gdf.geometry.centroid)
        
        # Agregar índice H3
        def get_h3_index(lat, lon):
            try:
                return h3.geo_to_h3(lat, lon, self.h3_resolution)
            except:
                return None
                
        gdf['h3_index'] = gdf.apply(
            lambda x: get_h3_index(x.latitude, x.longitude),
            axis=1
        )
        
        return gdf
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza features numéricos.
        
        Args:
            df: DataFrame con features
            
        Returns:
            DataFrame con features normalizados
        """
        logger.info("🔄 Normalizando features")
        
        # Identificar columnas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Excluir columnas que no deben normalizarse
        exclude_columns = ['station_id', 'h3_index', 'year']
        columns_to_normalize = [
            col for col in numeric_columns 
            if col not in exclude_columns
        ]
        
        # Normalizar usando Spark
        sdf = self.spark.createDataFrame(df)
        
        for column in columns_to_normalize:
            # Calcular estadísticas
            stats = sdf.select(
                avg(col(column)).alias('mean'),
                stddev(col(column)).alias('std')
            ).collect()[0]
            
            # Aplicar normalización
            sdf = sdf.withColumn(
                f'{column}_norm',
                (col(column) - stats['mean']) / stats['std']
            )
        
        return sdf.toPandas()
    
    def process(self) -> pd.DataFrame:
        """Ejecuta el pipeline completo de feature engineering."""
        try:
            # Cargar datos
            trips_df = self.load_trip_data()
            stations_df = self.load_station_data()
            
            # Calcular features
            temporal = self.calculate_temporal_features(trips_df)
            station = self.calculate_station_features(trips_df, stations_df)
            network = self.calculate_network_features(trips_df, stations_df)
            geographic = self.calculate_geographic_features(stations_df)
            
            # Combinar todos los features
            all_features = pd.merge(
                station,
                network,
                on="start_station_id",
                how="left"
            )
            
            all_features = pd.merge(
                all_features,
                geographic[['station_id', 'h3_index', 'dist_to_center']],
                left_on="start_station_id",
                right_on="station_id",
                how="left"
            )
            
            # Normalizar
            normalized_features = self.normalize_features(all_features)
            
            # Guardar resultados
            output_path = self.output_dir / f"bike_features_h3_{self.h3_resolution}"
            normalized_features.to_parquet(f"{output_path}.parquet")
            
            # Guardar metadata
            feature_metadata = {
                'n_features': len(normalized_features.columns),
                'n_samples': len(normalized_features),
                'feature_groups': {
                    'temporal': [col for col in normalized_features.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend'])],
                    'station': [col for col in normalized_features.columns if any(x in col for x in ['trips', 'duration'])],
                    'network': [col for col in normalized_features.columns if any(x in col for x in ['connection', 'destinations'])],
                    'geographic': [col for col in normalized_features.columns if any(x in col for x in ['dist_', 'h3_'])]
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(f"{output_path}_metadata.json", 'w') as f:
                import json
                json.dump(feature_metadata, f, indent=2)
            
            logger.info(f"💾 Features guardados en {output_path}")
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"❌ Error en feature engineering: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar feature engineering."""
    engineer = BikeFeatureEngineer()
    features = engineer.process()
    
    logger.info(f"✅ Generados {len(features.columns)} features para {len(features)} estaciones")

if __name__ == "__main__":
    main()