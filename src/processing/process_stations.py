"""
Process bike sharing station data from multiple cities.
This module handles the processing of station data from different bike sharing systems,
extracting metrics and supplementing with geographic data.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import geopandas as gpd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, array, struct, collect_list,
    count, sum, avg, min, max, first
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, ArrayType, MapType
)
import json
import requests
import time
from datetime import datetime
import h3
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from dotenv import load_dotenv
import backoff

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class StationProcessor:
    """Procesa datos de estaciones de sistemas de bike-sharing."""
    
    # Schema unificado para estaciones
    STATIONS_SCHEMA = StructType([
        StructField("station_id", StringType(), True),
        StructField("name", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("capacity", IntegerType(), True),
        StructField("installation_date", StringType(), True),
        StructField("city", StringType(), True),
        StructField("h3_index", StringType(), True),
        StructField("neighborhood", StringType(), True)
    ])
    
    def __init__(
        self,
        input_dir: str = "data/raw/stations",
        output_dir: str = "data/processed/stations",
        trips_dir: str = "data/processed/trips",
        census_dir: str = "data/processed/census",
        h3_resolution: int = 9,
        city_configs: Optional[Dict] = None
    ):
        """
        Inicializa el procesador de estaciones.
        
        Args:
            input_dir: Directorio con datos crudos
            output_dir: Directorio para datos procesados
            trips_dir: Directorio con datos procesados de viajes
            census_dir: Directorio con datos procesados del censo
            h3_resolution: Resolución H3 para análisis espacial
            city_configs: Configuraciones por ciudad
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.trips_dir = Path(trips_dir)
        self.census_dir = Path(census_dir)
        self.h3_resolution = h3_resolution
        
        # Configuración por defecto de ciudades
        self.city_configs = city_configs or {
            "nyc": {
                "file_pattern": "citibike_stations_*.json",
                "has_capacity": True,
                "api_url": "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
            },
            "sf": {
                "file_pattern": "sf_bikeshare_stations_*.json",
                "has_capacity": True,
                "api_url": "https://gbfs.baywheels.com/gbfs/en/station_information.json"
            }
        }
        
        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar Spark
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa una sesión de Spark."""
        return SparkSession.builder \
            .appName("Bike-Station-Processing") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
    
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3
    )
    def fetch_station_data(self, city: str) -> pd.DataFrame:
        """
        Obtiene datos actualizados de estaciones desde API.
        
        Args: