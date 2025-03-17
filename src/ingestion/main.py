import os
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sedona.register import SedonaRegistrator
from sedona.utils import KryoSerializer, SedonaKryoRegistrator
from dotenv import load_dotenv

from bike_data import BikeDataProcessor
from census_data import CensusDataProcessor
from process_osm import BikeInfraProcessor
from store_data import DatabaseStore

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeSharingPipeline:
    """Pipeline principal para análisis de bike-sharing."""
    
    # Configuración de ciudades
    CITIES = {
        'New York': (-74.0479, 40.6829, -73.9067, 40.7964),  # Manhattan + parte de Brooklyn
        'San Francisco': (-122.5158, 37.7079, -122.3558, 37.8324)  # SF proper
    }
    
    def __init__(
        self,
        base_dir: str = "data",
        h3_resolution: int = 9
    ):
        """
        Inicializa el pipeline.
        
        Args:
            base_dir: Directorio base para datos
            h3_resolution: Resolución H3 base
        """
        # Configurar directorios
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.final_dir = self.base_dir / "final"
        
        # Crear directorios
        for dir_path in [self.raw_dir, self.processed_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Parámetros
        self.h3_resolution = h3_resolution
        
        # Inicializar Spark con Sedona
        self.spark = self._init_spark()
        
        # Cargar variables de entorno
        load_dotenv()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark con todas las configuraciones necesarias."""
        return SparkSession.builder \
            .appName("Bike-Sharing-Pipeline") \
            .config("spark.serializer", KryoSerializer.getName) \
            .config("spark.kryo.registrator", SedonaKryoRegistrator.getName) \
            .config("spark.jars.packages",
                   "org.apache.sedona:sedona-python-adapter-3.4_2.12:1.4.1,"
                   "org.apache.sedona:sedona-viz-3.4_2.12:1.4.1,"
                   "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.27.1") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def process_bike_data(self) -> None:
        """Procesa datos de viajes de bike-sharing."""
        logger.info("🚲 Iniciando procesamiento de datos de bike-sharing")
        
        processor = BikeDataProcessor(
            output_dir=str(self.processed_dir / "bike_sharing"),
            base_resolution=self.h3_resolution
        )
        processor.process()
    
    def process_census_data(self) -> None:
        """Procesa datos geográficos del Census Bureau."""
        logger.info("🗺️ Iniciando procesamiento de datos del Census")
        
        processor = CensusDataProcessor(
            cache_dir=str(self.raw_dir / "census"),
            output_dir=str(self.processed_dir / "census")
        )
        processor.process()
    
    def process_infrastructure(self) -> None:
        """Procesa datos de infraestructura de OpenStreetMap."""
        logger.info("🚴 Iniciando procesamiento de infraestructura")
        
        processor = BikeInfraProcessor(
            cache_dir=str(self.raw_dir / "osm"),
            output_dir=str(self.processed_dir / "osm"),
            base_resolution=self.h3_resolution
        )
        processor.process(self.CITIES)
    
    def store_results(self) -> None:
        """Almacena todos los resultados en PostgreSQL."""
        logger.info("💾 Iniciando almacenamiento en base de datos")
        
        store = DatabaseStore(data_dir=str(self.processed_dir))
        store.process()
    
    def run(self) -> None:
        """Ejecuta el pipeline completo."""
        try:
            logger.info("🚀 Iniciando pipeline de bike-sharing")
            start_time = datetime.now()
            
            # 1. Procesar datos del Census (límites y geografía)
            self.process_census_data()
            
            # 2. Obtener datos de infraestructura
            self.process_infrastructure()
            
            # 3. Procesar datos de viajes
            self.process_bike_data()
            
            # 4. Almacenar todo en PostgreSQL
            self.store_results()
            
            # Guardar metadata del proceso
            duration = datetime.now() - start_time
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration.total_seconds(),
                'cities_processed': list(self.CITIES.keys()),
                'h3_resolution': self.h3_resolution,
                'spark_config': {
                    'driver_memory': '4g',
                    'executor_memory': '8g'
                }
            }
            
            metadata_path = self.final_dir / "pipeline_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Pipeline completado en {duration.total_seconds():.1f} segundos")
            
        except Exception as e:
            logger.error(f"❌ Error en el pipeline: {str(e)}")
            raise
        
        finally:
            # Limpiar
            if self.spark:
                self.spark.stop()

def main():
    """Función principal para ejecutar el pipeline."""
    try:
        # Parámetros configurables
        pipeline = BikeSharingPipeline(
            base_dir="data",
            h3_resolution=9
        )
        
        # Ejecutar pipeline
        pipeline.run()
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()