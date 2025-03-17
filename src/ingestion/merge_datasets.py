import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import geopandas as gpd
import h3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
from sedona.register import SedonaRegistrator
from sedona.utils import KryoSerializer, SedonaKryoRegistrator
import json
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetMerger:
    """Integra datasets procesados en una estructura unificada multi-resolución."""
    
    # Configuraciones H3
    H3_CONFIGS = {
        9: {
            'name': 'general',
            'area_km2': 0.1,
            'description': 'Análisis general y suburbano'
        },
        10: {
            'name': 'detailed',
            'area_km2': 0.03,
            'description': 'Análisis urbano detallado'
        }
    }
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/final",
        base_resolution: int = 9
    ):
        """
        Inicializa el merger de datasets.
        
        Args:
            input_dir: Directorio con datos procesados
            output_dir: Directorio para resultados finales
            base_resolution: Resolución H3 base
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.base_resolution = base_resolution
        
        # Crear directorios de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar Spark con Sedona
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark con Sedona para procesamiento espacial."""
        spark = SparkSession.builder \
            .appName("Dataset-Integration") \
            .config("spark.serializer", KryoSerializer.getName) \
            .config("spark.kryo.registrator", SedonaKryoRegistrator.getName) \
            .config("spark.jars.packages",
                   "org.apache.sedona:sedona-python-adapter-3.4_2.12:1.4.1,"
                   "org.apache.sedona:sedona-viz-3.4_2.12:1.4.1") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Registrar funciones Sedona
        SedonaRegistrator.registerAll(spark)
        
        return spark
    
    def load_administrative_boundaries(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Carga límites administrativos.
        
        Returns:
            Diccionario con GeoDataFrames por nivel
        """
        logger.info("🔄 Cargando límites administrativos")
        
        boundaries = {}
        
        # Cargar diferentes niveles
        for level in ['provincias', 'departamentos']:
            try:
                path = self.input_dir / "ign" / f"{level}.geojson"
                gdf = gpd.read_file(path)
                boundaries[level] = gdf
                logger.info(f"✅ {level.title()}: {len(gdf)} registros")
            except Exception as e:
                logger.error(f"❌ Error cargando {level}: {str(e)}")
        
        return boundaries
    
    def load_population_data(self) -> Dict[int, gpd.GeoDataFrame]:
        """
        Carga datos de población por resolución.
        
        Returns:
            Diccionario con GeoDataFrames por resolución
        """
        logger.info("🔄 Cargando datos de población")
        
        population = {}
        
        for resolution in self.H3_CONFIGS.keys():
            try:
                path = self.input_dir / "population" / f"resolution_{resolution}" / \
                       f"population_h3_{resolution}.geojson"
                gdf = gpd.read_file(path)
                population[resolution] = gdf
                logger.info(
                    f"✅ Resolución {resolution}: {len(gdf)} hexágonos, "
                    f"población total: {gdf['population'].sum():,.0f}"
                )
            except Exception as e:
                logger.error(f"❌ Error cargando población resolución {resolution}: {str(e)}")
        
        return population
    
    def load_commercial_data(self) -> Dict[int, gpd.GeoDataFrame]:
        """
        Carga datos comerciales por resolución.
        
        Returns:
            Diccionario con GeoDataFrames por resolución
        """
        logger.info("🔄 Cargando datos comerciales")
        
        commercial = {}
        
        for resolution in self.H3_CONFIGS.keys():
            try:
                path = self.input_dir / "osm" / f"resolution_{resolution}" / \
                       f"commercial_h3_{resolution}.geojson"
                gdf = gpd.read_file(path)
                commercial[resolution] = gdf
                logger.info(
                    f"✅ Resolución {resolution}: {len(gdf)} hexágonos, "
                    f"POIs totales: {gdf['poi_count'].sum():,.0f}"
                )
            except Exception as e:
                logger.error(f"❌ Error cargando comercios resolución {resolution}: {str(e)}")
        
        return commercial
    
    def merge_h3_data(
        self,
        population_gdf: gpd.GeoDataFrame,
        commercial_gdf: gpd.GeoDataFrame,
        resolution: int
    ) -> gpd.GeoDataFrame:
        """
        Combina datos de población y comerciales a nivel H3.
        
        Args:
            population_gdf: GeoDataFrame con datos de población
            commercial_gdf: GeoDataFrame con datos comerciales
            resolution: Resolución H3
            
        Returns:
            GeoDataFrame combinado
        """
        logger.info(f"🔄 Combinando datos resolución {resolution}")
        
        # Convertir a Spark DataFrames
        pop_df = self.spark.createDataFrame(
            population_gdf.drop(columns=['geometry'])
        )
        
        com_df = self.spark.createDataFrame(
            commercial_gdf.drop(columns=['geometry'])
        )
        
        # Realizar join por h3_index
        merged_df = pop_df.join(
            com_df,
            on='h3_index',
            how='outer'
        ).fillna({
            'poi_count': 0,
            'population': 0
        })
        
        # Calcular métricas
        final_df = merged_df \
            .withColumn(
                'commercial_density',
                expr('poi_count / CASE WHEN population > 0 THEN population ELSE 1 END * 1000')  # POIs por 1000 habitantes
            ).withColumn(
                'population_density',
                expr(f'population / {self.H3_CONFIGS[resolution]["area_km2"]}')  # Habitantes por km²
            )
        
        # Convertir a Pandas
        pdf = final_df.toPandas()
        
        # Agregar geometrías H3
        pdf['geometry'] = pdf['h3_index'].apply(self.h3_to_polygon)
        
        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(pdf, geometry='geometry', crs="EPSG:4326")
        
        logger.info(
            f"✅ Resolución {resolution}: {len(gdf)} hexágonos integrados"
        )
        return gdf
    
    def assign_administrative_areas(
        self,
        h3_data: gpd.GeoDataFrame,
        boundaries: Dict[str, gpd.GeoDataFrame]
    ) -> gpd.GeoDataFrame:
        """
        Asigna áreas administrativas a hexágonos H3.
        
        Args:
            h3_data: GeoDataFrame con datos H3
            boundaries: Diccionario con límites administrativos
            
        Returns:
            GeoDataFrame con información administrativa
        """
        logger.info("🔄 Asignando áreas administrativas")
        
        result = h3_data.copy()
        
        # Asignar provincia
        provinces = boundaries['provincias']
        result = gpd.sjoin(
            result,
            provinces[['nombre', 'geometry']],
            how="left",
            predicate="intersects"
        ).rename(columns={'nombre': 'provincia'})
        
        # Asignar departamento
        departments = boundaries['departamentos']
        result = gpd.sjoin(
            result,
            departments[['nombre', 'geometry']],
            how="left",
            predicate="intersects"
        ).rename(columns={'nombre': 'departamento'})
        
        # Limpiar columnas duplicadas
        result = result.loc[:,~result.columns.duplicated()]
        
        logger.info("✅ Áreas administrativas asignadas")
        return result
    
    def h3_to_polygon(self, h3_index: str) -> Polygon:
        """Convierte índice H3 a polígono."""
        boundary = h3.cell_to_boundary(h3_index)
        coords = [(lng, lat) for lat, lng in boundary]
        return Polygon(coords)
    
    def calculate_statistics(
        self,
        results: Dict[int, gpd.GeoDataFrame]
    ) -> Dict:
        """
        Calcula estadísticas del dataset integrado.
        
        Args:
            results: Diccionario con GeoDataFrames por resolución
            
        Returns:
            Diccionario con estadísticas
        """
        stats = {}
        
        for resolution, gdf in results.items():
            # Estadísticas generales
            general_stats = {
                'hexagon_count': len(gdf),
                'total_population': gdf['population'].sum(),
                'total_pois': gdf['poi_count'].sum(),
                'avg_commercial_density': gdf['commercial_density'].mean(),
                'avg_population_density': gdf['population_density'].mean(),
                'area_covered_km2': len(gdf) * self.H3_CONFIGS[resolution]['area_km2']
            }
            
            # Estadísticas por provincia
            province_stats = {}
            for province in gdf['provincia'].unique():
                if pd.isna(province):
                    continue
                    
                prov_data = gdf[gdf['provincia'] == province]
                province_stats[province] = {
                    'hexagon_count': len(prov_data),
                    'population': prov_data['population'].sum(),
                    'pois': prov_data['poi_count'].sum(),
                    'avg_commercial_density': prov_data['commercial_density'].mean()
                }
            
            stats[resolution] = {
                'general': general_stats,
                'by_province': province_stats
            }
        
        return stats
    
    def save_results(
        self,
        results: Dict[int, gpd.GeoDataFrame],
        stats: Dict
    ) -> None:
        """
        Guarda resultados y estadísticas.
        
        Args:
            results: Diccionario con GeoDataFrames por resolución
            stats: Estadísticas calculadas
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for resolution, gdf in results.items():
            # Crear directorio para la resolución
            res_dir = self.output_dir / f"resolution_{resolution}"
            res_dir.mkdir(exist_ok=True)
            
            # Guardar GeoJSON
            geojson_path = res_dir / f"integrated_data_{timestamp}.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            
            # Guardar CSV
            csv_path = res_dir / f"integrated_data_{timestamp}.csv"
            gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
        
        # Guardar estadísticas
        stats_path = self.output_dir / f"statistics_{timestamp}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados guardados en {self.output_dir}")
    
    def process(self) -> Dict[int, gpd.GeoDataFrame]:
        """
        Ejecuta el pipeline completo de integración.
        
        Returns:
            Diccionario con GeoDataFrames por resolución
        """
        try:
            # Cargar datos
            boundaries = self.load_administrative_boundaries()
            population = self.load_population_data()
            commercial = self.load_commercial_data()
            
            # Integrar datos por resolución
            results = {}
            for resolution in self.H3_CONFIGS.keys():
                if resolution in population and resolution in commercial:
                    # Combinar datos H3
                    h3_data = self.merge_h3_data(
                        population[resolution],
                        commercial[resolution],
                        resolution
                    )
                    
                    # Asignar áreas administrativas
                    final_data = self.assign_administrative_areas(
                        h3_data,
                        boundaries
                    )
                    
                    results[resolution] = final_data
            
            # Calcular estadísticas
            stats = self.calculate_statistics(results)
            
            # Guardar resultados
            self.save_results(results, stats)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en la integración: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar la integración."""
    try:
        merger = DatasetMerger()
        results = merger.process()
        
        logger.info("✅ Integración completada exitosamente")
        
        # Mostrar resumen
        for resolution, gdf in results.items():
            logger.info(
                f"\nResolución {resolution}:"
                f"\n- Hexágonos: {len(gdf):,}"
                f"\n- Población total: {gdf['population'].sum():,.0f}"
                f"\n- POIs totales: {gdf['poi_count'].sum():,.0f}"
                f"\n- Densidad comercial promedio: {gdf['commercial_density'].mean():.2f}"
                f"\n- Provincias cubiertas: {gdf['provincia'].nunique()}"
                f"\n- Departamentos cubiertos: {gdf['departamento'].nunique()}"
                f"\n- Área total cubierta: {len(gdf) * merger.H3_CONFIGS[resolution]['area_km2']:,.1f} km²"
            )
        
        # Guardar un archivo de metadata del proceso
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "resolutions_processed": list(results.keys()),
            "data_sources": {
                "boundaries": "IGN Argentina WFS",
                "population": "WorldPop 2020",
                "commercial": "OpenStreetMap"
            },
            "process_info": {
                "base_resolution": merger.base_resolution,
                "h3_configs": merger.H3_CONFIGS,
                "crs": "EPSG:4326"
            }
        }
        
        with open(merger.output_dir / "process_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()