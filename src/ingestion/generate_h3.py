import os
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Set
import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import folium
from folium import GeoJson, Marker
import json
from tqdm import tqdm
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class H3GridGenerator:
    """Genera grids H3 para análisis espacial con soporte multi-resolución."""
    
    # Definiciones de resolución
    RESOLUTION_CONFIGS = {
        9: {
            'name': 'general',
            'area_km2': 0.1,
            'use_case': 'Análisis urbano general, patrones de barrio',
            'min_population': 0,  # Sin límite
            'max_density': float('inf')  # Sin límite
        },
        10: {
            'name': 'detailed',
            'area_km2': 0.03,
            'use_case': 'Análisis micro-urbano, alta densidad',
            'min_population': 5000,  # Ejemplo: áreas más pobladas
            'max_density': 15000  # Ejemplo: densidad por km²
        }
    }
    
    def __init__(
        self,
        base_resolution: int = 9,
        output_dir: str = "data/processed/h3",
        auto_resolution: bool = True
    ):
        """
        Inicializa el generador de grids H3.
        
        Args:
            base_resolution: Resolución base H3 (default: 9)
            output_dir: Directorio para resultados
            auto_resolution: Si ajustar automáticamente la resolución
        """
        self.base_resolution = base_resolution
        self.output_dir = Path(output_dir)
        self.auto_resolution = auto_resolution
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validar resolución base
        if base_resolution not in self.RESOLUTION_CONFIGS:
            raise ValueError(f"Resolución {base_resolution} no soportada")
    
    def determine_resolution(
        self,
        population: float,
        density: float
    ) -> int:
        """
        Determina la resolución óptima basada en población y densidad.
        
        Args:
            population: Población del área
            density: Densidad poblacional
            
        Returns:
            Resolución H3 recomendada
        """
        if not self.auto_resolution:
            return self.base_resolution
            
        # Usar resolución 10 para áreas densas/pobladas
        if (population >= self.RESOLUTION_CONFIGS[10]['min_population'] or
            density >= self.RESOLUTION_CONFIGS[10]['max_density']):
            return 10
            
        return self.base_resolution
    
    def process_polygons_multi_resolution(
        self,
        gdf: gpd.GeoDataFrame,
        population_col: Optional[str] = None,
        area_col: Optional[str] = None
    ) -> Dict[int, gpd.GeoDataFrame]:
        """
        Procesa polígonos con resolución adaptativa.
        
        Args:
            gdf: GeoDataFrame con polígonos
            population_col: Nombre de columna de población
            area_col: Nombre de columna de área
            
        Returns:
            Diccionario con GeoDataFrames por resolución
        """
        logger.info(f"🔄 Procesando {len(gdf)} polígonos con resolución adaptativa")
        
        # Inicializar resultados por resolución
        results = {res: [] for res in self.RESOLUTION_CONFIGS.keys()}
        
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
            # Calcular densidad si hay datos
            population = row[population_col] if population_col else 0
            area = row.geometry.area if not area_col else row[area_col]
            density = population / area if area > 0 else 0
            
            # Determinar resolución
            resolution = self.determine_resolution(population, density)
            
            # Obtener hexágonos para el polígono
            hexagons = self.polygon_to_h3(row.geometry, resolution)
            
            # Crear registros para cada hexágono
            for h3_index in hexagons:
                record = {
                    'h3_index': h3_index,
                    'h3_resolution': resolution,
                    'geometry': self.h3_to_polygon(h3_index),
                    'population': population,
                    'density': density
                }
                
                # Agregar nombre del área si está especificado
                if area_col and area_col in row:
                    record['area_name'] = row[area_col]
                
                results[resolution].append(record)
        
        # Convertir resultados a GeoDataFrames
        return {
            res: gpd.GeoDataFrame(data, geometry='geometry', crs=gdf.crs)
            for res, data in results.items()
            if data  # Solo incluir resoluciones con datos
        }
    
    def create_multi_resolution_map(
        self,
        results: Dict[int, gpd.GeoDataFrame],
        center: Optional[List[float]] = None,
        filename: str = "h3_multi_resolution_map.html"
    ) -> folium.Map:
        """
        Crea mapa interactivo con hexágonos de múltiples resoluciones.
        
        Args:
            results: Diccionario con GeoDataFrames por resolución
            center: Centro opcional del mapa [lat, lon]
            filename: Nombre del archivo HTML
            
        Returns:
            Mapa Folium
        """
        # Determinar centro del mapa
        if center is None:
            # Usar el centro del primer GeoDataFrame disponible
            first_gdf = next(iter(results.values()))
            center = [
                first_gdf.geometry.centroid.y.mean(),
                first_gdf.geometry.centroid.x.mean()
            ]
        
        # Crear mapa base
        m = folium.Map(location=center, zoom_start=11)
        
        # Colores por resolución
        colors = {9: 'blue', 10: 'red'}
        
        # Agregar hexágonos por resolución
        for resolution, gdf in results.items():
            # Crear grupo de capa para la resolución
            feature_group = folium.FeatureGroup(
                name=f"Resolución {resolution} - {self.RESOLUTION_CONFIGS[resolution]['name']}"
            )
            
            for _, row in gdf.iterrows():
                # Crear polígono para cada hexágono
                folium.Polygon(
                    locations=[(lat, lng) for lng, lat in row.geometry.exterior.coords],
                    color=colors[resolution],
                    fill=True,
                    fill_opacity=0.4,
                    weight=1,
                    tooltip=f"H3 Index: {row['h3_index']}<br>"
                           f"Resolución: {resolution}<br>"
                           f"Área aprox: {self.RESOLUTION_CONFIGS[resolution]['area_km2']} km²"
                ).add_to(feature_group)
            
            feature_group.add_to(m)
        
        # Agregar control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa
        output_path = self.output_dir / filename
        m.save(str(output_path))
        logger.info(f"💾 Mapa multi-resolución guardado en {output_path}")
        
        return m
    
    def save_multi_resolution_results(
        self,
        results: Dict[int, gpd.GeoDataFrame],
        name: str
    ) -> None:
        """
        Guarda resultados de múltiples resoluciones.
        
        Args:
            results: Diccionario con GeoDataFrames por resolución
            name: Nombre base para los archivos
        """
        for resolution, gdf in results.items():
            # Crear subdirectorio para la resolución
            res_dir = self.output_dir / f"resolution_{resolution}"
            res_dir.mkdir(exist_ok=True)
            
            # Guardar GeoJSON
            geojson_path = res_dir / f"{name}_h3_{resolution}.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            
            # Guardar CSV
            csv_path = res_dir / f"{name}_h3_{resolution}.csv"
            gdf[['h3_index', 'h3_resolution', 'population', 'density']].to_csv(
                csv_path,
                index=False
            )
            
            logger.info(f"💾 Resultados resolución {resolution} guardados en {res_dir}")
    
    # Los métodos point_to_h3, polygon_to_h3 y h3_to_polygon se mantienen igual,
    # solo agregamos el parámetro de resolución donde sea necesario

def main():
    """Función principal para pruebas."""
    try:
        # Cargar datos de prueba
        provinces = gpd.read_file("data/processed/ign/provincias.geojson")
        
        # Inicializar generador con resolución adaptativa
        generator = H3GridGenerator(auto_resolution=True)
        
        # Procesar provincias con múltiples resoluciones
        results = generator.process_polygons_multi_resolution(
            provinces,
            population_col='population',  # Ajustar según tus datos
            area_col='nombre'
        )
        
        # Crear mapa multi-resolución
        generator.create_multi_resolution_map(results)
        
        # Guardar resultados
        generator.save_multi_resolution_results(results, "provincias")
        
        # Mostrar estadísticas
        for resolution, gdf in results.items():
            logger.info(
                f"Resolución {resolution}: {len(gdf)} hexágonos generados "
                f"({self.RESOLUTION_CONFIGS[resolution]['name']})"
            )
        
        logger.info("✅ Proceso multi-resolución completado exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error en el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()