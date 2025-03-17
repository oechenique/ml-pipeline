import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
import requests
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon, LineString
from tqdm import tqdm
import backoff

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeInfraProcessor:
    """Procesa datos de infraestructura ciclista y POIs relevantes de OpenStreetMap."""
    
    # Categorías de datos a extraer
    OSM_QUERIES = {
        'bike_infra': {
            'ways': [
                # Ciclovías y carriles bici
                'highway=cycleway',
                'cycleway=*',
                'bicycle=designated',
                'cycleway:left=*',
                'cycleway:right=*'
            ],
            'nodes': [
                # Puntos relacionados con bicicletas
                'amenity=bicycle_parking',
                'amenity=bicycle_rental',
                'amenity=bicycle_repair_station'
            ]
        },
        'transit': {
            'nodes': [
                # Estaciones de transporte público
                'railway=station',
                'railway=subway_entrance',
                'amenity=bus_station',
                'public_transport=station'
            ]
        },
        'attractions': {
            'nodes': [
                # Puntos de interés que generan demanda
                'tourism=*',
                'leisure=park',
                'amenity=university',
                'amenity=college',
                'amenity=theatre',
                'amenity=arts_centre'
            ],
            'areas': [
                # Áreas comerciales y recreativas
                'landuse=retail',
                'landuse=commercial',
                'leisure=park'
            ]
        }
    }
    
    def __init__(
        self,
        cache_dir: str = "data/raw/osm",
        output_dir: str = "data/processed/osm",
        base_resolution: int = 9
    ):
        """
        Inicializa el procesador de datos OSM.
        
        Args:
            cache_dir: Directorio para datos crudos
            output_dir: Directorio para datos procesados
            base_resolution: Resolución H3 base
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.base_resolution = base_resolution
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Endpoint de Overpass
        self.overpass_url = "https://overpass-api.de/api/interpreter"
    
    def build_query(
        self,
        bbox: Tuple[float, float, float, float],
        category: str
    ) -> str:
        """
        Construye query Overpass para una categoría.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            category: Categoría de datos ('bike_infra', 'transit', 'attractions')
            
        Returns:
            Query Overpass
        """
        if category not in self.OSM_QUERIES:
            raise ValueError(f"Categoría {category} no válida")
            
        query_parts = []
        config = self.OSM_QUERIES[category]
        
        # Procesar ways si existen
        if 'ways' in config:
            for way_filter in config['ways']:
                query_parts.append(f"""
                way[{way_filter}]
                    ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                """)
        
        # Procesar nodes
        if 'nodes' in config:
            for node_filter in config['nodes']:
                query_parts.append(f"""
                node[{node_filter}]
                    ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                """)
        
        # Procesar areas
        if 'areas' in config:
            for area_filter in config['areas']:
                query_parts.append(f"""
                way[{area_filter}]
                    ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                relation[{area_filter}]
                    ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
                """)
        
        # Construir query completa
        query = f"""
        [out:json][timeout:300];
        (
            {' '.join(query_parts)}
        );
        out body geom;
        """
        
        return query
    
    @backoff.on_exception(
        backoff.expo,
        requests.exceptions.RequestException,
        max_tries=3
    )
    def download_data(
        self,
        bbox: Tuple[float, float, float, float],
        category: str,
        cache_file: Optional[Path] = None
    ) -> Dict:
        """
        Descarga datos de OSM usando Overpass API.
        
        Args:
            bbox: Bounding box
            category: Categoría de datos
            cache_file: Archivo para cachear resultados
            
        Returns:
            Datos OSM en formato JSON
        """
        if cache_file and cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        query = self.build_query(bbox, category)
        
        response = requests.post(
            self.overpass_url,
            data=query,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=300
        )
        response.raise_for_status()
        
        data = response.json()
        
        if cache_file:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        
        # Esperar para respetar rate limits
        time.sleep(1)
        
        return data
    
    def process_bike_infrastructure(
        self,
        data: Dict,
        city: str
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Procesa infraestructura ciclista.
        
        Args:
            data: Datos OSM
            city: Nombre de la ciudad
            
        Returns:
            Tuple de (rutas, puntos) como GeoDataFrames
        """
        routes = []
        points = []
        
        for element in data['elements']:
            try:
                # Procesar ways (rutas)
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(p['lon'], p['lat']) for p in element['geometry']]
                    if len(coords) >= 2:
                        route = {
                            'osm_id': element['id'],
                            'geometry': LineString(coords),
                            'type': element.get('tags', {}).get('highway', 'unknown'),
                            'name': element.get('tags', {}).get('name'),
                            'city': city
                        }
                        routes.append(route)
                
                # Procesar nodes (puntos)
                elif element['type'] == 'node':
                    point = {
                        'osm_id': element['id'],
                        'geometry': Point(element['lon'], element['lat']),
                        'type': element.get('tags', {}).get('amenity'),
                        'name': element.get('tags', {}).get('name'),
                        'city': city
                    }
                    points.append(point)
            
            except Exception as e:
                logger.warning(f"Error procesando elemento {element.get('id')}: {str(e)}")
        
        # Crear GeoDataFrames
        routes_gdf = gpd.GeoDataFrame(routes, crs="EPSG:4326")
        points_gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
        
        return routes_gdf, points_gdf
    
    def process_transit_points(
        self,
        data: Dict,
        city: str
    ) -> gpd.GeoDataFrame:
        """
        Procesa puntos de transporte público.
        
        Args:
            data: Datos OSM
            city: Nombre de la ciudad
            
        Returns:
            GeoDataFrame con puntos de transporte
        """
        points = []
        
        for element in data['elements']:
            try:
                if element['type'] == 'node':
                    point = {
                        'osm_id': element['id'],
                        'geometry': Point(element['lon'], element['lat']),
                        'type': element.get('tags', {}).get('railway') or 
                               element.get('tags', {}).get('amenity'),
                        'name': element.get('tags', {}).get('name'),
                        'network': element.get('tags', {}).get('network'),
                        'operator': element.get('tags', {}).get('operator'),
                        'city': city
                    }
                    points.append(point)
            
            except Exception as e:
                logger.warning(f"Error procesando elemento {element.get('id')}: {str(e)}")
        
        return gpd.GeoDataFrame(points, crs="EPSG:4326")
    
    def process_attractions(
        self,
        data: Dict,
        city: str
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Procesa puntos y áreas de interés.
        
        Args:
            data: Datos OSM
            city: Nombre de la ciudad
            
        Returns:
            Tuple de (áreas, puntos) como GeoDataFrames
        """
        areas = []
        points = []
        
        for element in data['elements']:
            try:
                tags = element.get('tags', {})
                
                if element['type'] == 'way' and 'geometry' in element:
                    # Procesar áreas
                    coords = [(p['lon'], p['lat']) for p in element['geometry']]
                    if len(coords) >= 3 and coords[0] == coords[-1]:
                        area = {
                            'osm_id': element['id'],
                            'geometry': Polygon(coords),
                            'type': tags.get('landuse') or tags.get('leisure'),
                            'name': tags.get('name'),
                            'city': city
                        }
                        areas.append(area)
                
                elif element['type'] == 'node':
                    # Procesar puntos
                    point = {
                        'osm_id': element['id'],
                        'geometry': Point(element['lon'], element['lat']),
                        'type': tags.get('tourism') or tags.get('leisure') or 
                               tags.get('amenity'),
                        'name': tags.get('name'),
                        'city': city
                    }
                    points.append(point)
            
            except Exception as e:
                logger.warning(f"Error procesando elemento {element.get('id')}: {str(e)}")
        
        areas_gdf = gpd.GeoDataFrame(areas, crs="EPSG:4326")
        points_gdf = gpd.GeoDataFrame(points, crs="EPSG:4326")
        
        return areas_gdf, points_gdf
    
    def process_city(
        self,
        city_name: str,
        bbox: Tuple[float, float, float, float]
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Procesa todos los datos para una ciudad.
        
        Args:
            city_name: Nombre de la ciudad
            bbox: Bounding box
            
        Returns:
            Diccionario con GeoDataFrames por categoría
        """
        results = {}
        
        for category in self.OSM_QUERIES.keys():
            try:
                # Preparar archivo cache
                cache_file = self.cache_dir / f"{city_name}_{category}_osm.json"
                
                # Descargar datos
                data = self.download_data(bbox, category, cache_file)
                
                # Procesar según categoría
                if category == 'bike_infra':
                    routes, points = self.process_bike_infrastructure(data, city_name)
                    results['bike_routes'] = routes
                    results['bike_points'] = points
                
                elif category == 'transit':
                    results['transit_points'] = self.process_transit_points(data, city_name)
                
                elif category == 'attractions':
                    areas, points = self.process_attractions(data, city_name)
                    results['attraction_areas'] = areas
                    results['attraction_points'] = points
                
                logger.info(f"✅ {city_name} - {category}: procesado exitosamente")
                
            except Exception as e:
                logger.error(f"❌ Error procesando {category} en {city_name}: {str(e)}")
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Dict[str, gpd.GeoDataFrame]]
    ) -> None:
        """
        Guarda resultados procesados.
        
        Args:
            results: Diccionario de resultados por ciudad
        """
        # Crear directorio para cada ciudad
        for city, city_data in results.items():
            city_dir = self.output_dir / city.lower().replace(' ', '_')
            city_dir.mkdir(exist_ok=True)
            
            for name, gdf in city_data.items():
                if len(gdf) > 0:
                    # Guardar GeoJSON
                    output_path = city_dir / f"{name}.geojson"
                    gdf.to_file(output_path, driver='GeoJSON')
                    
                    # Guardar CSV con datos no geométricos
                    csv_path = city_dir / f"{name}.csv"
                    gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
                    
                    logger.info(f"💾 {city} - {name}: {len(gdf)} elementos guardados")
    
    def process(self, cities: Dict[str, Tuple[float, float, float, float]]) -> None:
        """
        Ejecuta el pipeline completo de procesamiento.
        
        Args:
            cities: Diccionario de ciudades y sus bounding boxes
        """
        try:
            logger.info("🚀 Iniciando procesamiento de datos OSM")
            
            results = {}
            for city_name, bbox in cities.items():
                logger.info(f"🔄 Procesando {city_name}")
                results[city_name] = self.process_city(city_name, bbox)
            
            # Guardar resultados
            self.save_results(results)
            
            logger.info("✅ Procesamiento completado exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error en el procesamiento: {str(e)}")
            raise

def main():
    """Función principal para ejecutar el procesamiento."""
    # Definir bounding boxes para cada ciudad
    cities = {
        'New York': (-74.0479, 40.6829, -73.9067, 40.7964),  # Manhattan + parte de Brooklyn
        'San Francisco': (-122.5158, 37.7079, -122.3558, 37.8324)  # SF proper
    }
    
    processor = BikeInfraProcessor()
    processor.process(cities)

if __name__ == "__main__":
    main()