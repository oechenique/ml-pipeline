"""
Process OpenStreetMap POI data.
This module handles the extraction and processing of commercial Points of Interest
(POIs) from OpenStreetMap, categorizing them and aggregating to H3 hexagons.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import json
import time
from datetime import datetime
import requests
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon, box
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit
from pyspark.sql.types import StringType, ArrayType
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuraciones H3
H3_CONFIGS = {
    9: {
        'name': 'general',
        'area_km2': 0.1,
        'min_pois': 0  # Sin límite mínimo
    },
    10: {
        'name': 'detailed',
        'area_km2': 0.03,
        'min_pois': 10  # Mínimo POIs para resolución detallada
    }
}

# Categorización de POIs comerciales
COMMERCIAL_CATEGORIES = {
    'retail': {
        'shop': '*',  # Todas las tiendas
        'tags': ['shop']
    },
    'food_drink': {
        'amenity': [
            'restaurant', 'cafe', 'bar', 'fast_food',
            'food_court', 'ice_cream', 'pub'
        ],
        'tags': ['amenity', 'cuisine']
    },
    'services': {
        'amenity': [
            'bank', 'pharmacy', 'clinic', 'dentist',
            'doctors', 'veterinary'
        ],
        'shop': [
            'hairdresser', 'beauty', 'optician',
            'laundry', 'dry_cleaning'
        ],
        'tags': ['amenity', 'shop', 'service']
    },
    'entertainment': {
        'leisure': [
            'fitness_centre', 'sports_centre', 'cinema',
            'theatre', 'dance'
        ],
        'amenity': ['nightclub', 'casino'],
        'tags': ['leisure', 'amenity']
    },
    'education': {
        'amenity': [
            'school', 'university', 'college',
            'language_school', 'music_school'
        ],
        'tags': ['amenity', 'school']
    },
    'office': {
        'office': [
            'company', 'insurance', 'estate_agent',
            'lawyer', 'accountant', 'coworking'
        ],
        'tags': ['office']
    }
}

def process_osm_data(
    output_dir: str = "data/processed/osm",
    cache_dir: str = "data/raw/osm",
    force_download: bool = False,
    boundaries_path: Optional[str] = "data/processed/ign/provincias.geojson"
) -> Dict[int, gpd.GeoDataFrame]:
    """
    Procesa datos de POIs comerciales de OSM.
    
    Args:
        output_dir: Directorio para datos procesados
        cache_dir: Directorio para caché de datos crudos
        force_download: Si forzar descarga ignorando caché
        boundaries_path: Path a archivo de límites administrativos
        
    Returns:
        Diccionario con GeoDataFrames por resolución
    """
    try:
        # Crear directorios
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Cargar límites administrativos
        boundaries = gpd.read_file(boundaries_path)
        logger.info(f"📊 Procesando {len(boundaries)} áreas administrativas")
        
        # Inicializar Spark
        spark = init_spark()
        
        try:
            # Procesar cada área administrativa
            all_pois = []
            for idx, row in boundaries.iterrows():
                try:
                    # Obtener bbox
                    minx, miny, maxx, maxy = row.geometry.bounds
                    
                    # Procesar área
                    area_pois = process_area(
                        row['nombre'],
                        (minx, miny, maxx, maxy),
                        cache_dir,
                        force_download
                    )
                    
                    all_pois.append(area_pois)
                    
                    logger.info(f"✅ {row['nombre']}: {len(area_pois)} POIs")
                    
                except Exception as e:
                    logger.error(f"❌ Error en {row['nombre']}: {str(e)}")
            
            # Combinar todos los POIs
            combined_df = pd.concat(all_pois, ignore_index=True)
            
            # Procesar para cada resolución
            results = {}
            for resolution in H3_CONFIGS.keys():
                logger.info(f"🔄 Procesando resolución H3 {resolution}")
                
                # Agregar a H3
                h3_gdf = points_to_h3(combined_df, resolution, spark)
                
                # Filtrar por densidad mínima
                min_pois = H3_CONFIGS[resolution]['min_pois']
                if min_pois > 0:
                    h3_gdf = h3_gdf[h3_gdf['poi_count'] >= min_pois]
                
                # Guardar resultados
                save_resolution_data(h3_gdf, resolution, output_dir)
                
                results[resolution] = h3_gdf
                
                logger.info(
                    f"✅ Resolución {resolution}: {len(h3_gdf)} hexágonos, "
                    f"POIs totales: {h3_gdf['poi_count'].sum():,.0f}"
                )
            
            # Guardar metadata
            save_metadata(results, output_dir)
            
            return results
            
        finally:
            spark.stop()
            
    except Exception as e:
        logger.error(f"❌ Error procesando datos OSM: {str(e)}")
        raise

def init_spark() -> SparkSession:
    """Inicializa sesión de Spark."""
    return SparkSession.builder \
        .appName("OSM-Processing") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

def process_area(
    area_name: str,
    bbox: Tuple[float, float, float, float],
    cache_dir: str,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Procesa POIs en un área específica.
    
    Args:
        area_name: Nombre del área
        bbox: Bounding box (minx, miny, maxx, maxy)
        cache_dir: Directorio para caché
        force_download: Si forzar descarga
        
    Returns:
        DataFrame con POIs procesados
    """
    all_pois = []
    
    # Procesar cada categoría comercial
    for category in COMMERCIAL_CATEGORIES.keys():
        try:
            # Preparar caché
            cache_path = Path(cache_dir) / f"{area_name}_{category}_osm.json"
            
            if not force_download and cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
            else:
                # Construir query
                query = build_overpass_query(bbox, category)
                
                # Descargar datos
                response = requests.post(
                    os.getenv("OSM_OVERPASS_URL", "https://overpass-api.de/api/interpreter"),
                    data=query,
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=300
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Guardar en caché
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
                
                # Esperar entre requests
                time.sleep(1)
            
            # Procesar POIs
            for element in data['elements']:
                poi = process_poi(element, category)
                if poi:
                    all_pois.append(poi)
            
            logger.info(f"✅ {category}: {len(data['elements'])} elementos")
            
        except Exception as e:
            logger.error(f"❌ Error en {category}: {str(e)}")
    
    return pd.DataFrame(all_pois)

def build_overpass_query(
    bbox: Tuple[float, float, float, float],
    category: str
) -> str:
    """
    Construye query Overpass para una categoría.
    
    Args:
        bbox: Bounding box
        category: Categoría comercial
        
    Returns:
        Query Overpass
    """
    category_config = COMMERCIAL_CATEGORIES[category]
    filters = []
    
    for key, values in category_config.items():
        if key != 'tags':
            if values == '*':
                filters.append(f'["{key}"]')
            else:
                for value in values:
                    filters.append(f'["{key}"="{value}"]')
    
    filter_str = '\n    '.join(filters)
    
    query = f"""
    [out:json][timeout:300];
    (
      node{filter_str}
        ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
      way{filter_str}
        ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
      relation{filter_str}
        ({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out center;
    """
    
    return query

def process_poi(element: Dict, category: str) -> Optional[Dict]:
    """
    Procesa un POI individual.
    
    Args:
        element: Elemento OSM
        category: Categoría comercial
        
    Returns:
        Diccionario procesado o None
    """
    try:
        # Extraer coordenadas
        if 'center' in element:
            lat = element['center']['lat']
            lon = element['center']['lon']
        else:
            lat = element.get('lat')
            lon = element.get('lon')
        
        if not (lat and lon):
            return None
        
        # Extraer tags
        tags = element.get('tags', {})
        
        return {
            'osm_id': element['id'],
            'category': category,
            'name': tags.get('name'),
            'latitude': lat,
            'longitude': lon,
            'tags': json.dumps(tags)
        }
        
    except Exception as e:
        logger.warning(f"Error procesando POI {element.get('id')}: {str(e)}")
        return None

def points_to_h3(
    df: pd.DataFrame,
    resolution: int,
    spark: SparkSession
) -> gpd.GeoDataFrame:
    """
    Convierte puntos a hexágonos H3.
    
    Args:
        df: DataFrame con puntos
        resolution: Resolución H3
        spark: Sesión de Spark
        
    Returns:
        GeoDataFrame con hexágonos
    """
    # Convertir a Spark DataFrame
    sdf = spark.createDataFrame(df)
    
    # UDF para conversión a H3
    @udf(returnType=StringType())
    def to_h3_index(lat, lon):
        return h3.geo_to_h3(lat, lon, resolution)
    
    # Agregar índice H3
    sdf = sdf.withColumn(
        'h3_index',
        to_h3_index('latitude', 'longitude')
    )
    
    # Agregar por H3
    h3_df = sdf.groupBy('h3_index').agg({
        'osm_id': 'count',
        'category': 'collect_set',
        'latitude': 'mean',
        'longitude': 'mean'
    })
    
    # Convertir a Pandas
    pdf = h3_df.toPandas()
    
    # Crear geometrías
    pdf['geometry'] = pdf['h3_index'].apply(lambda h: Polygon(
        h3.h3_to_geo_boundary(h, geo_json=True)
    ))
    
    # Crear GeoDataFrame
    gdf = gpd.GeoDataFrame(
        pdf,
        geometry='geometry',
        crs="EPSG:4326"
    ).rename(columns={
        'count(osm_id)': 'poi_count',
        'collect_set(category)': 'categories',
        'avg(latitude)': 'latitude',
        'avg(longitude)': 'longitude'
    })
    
    # Calcular densidad
    gdf['area_km2'] = H3_CONFIGS[resolution]['area_km2']
    gdf['poi_density'] = gdf['poi_count'] / gdf['area_km2']
    
    return gdf

def save_resolution_data(
    gdf: gpd.GeoDataFrame,
    resolution: int,
    output_dir: str
):
    """
    Guarda datos de una resolución específica.
    
    Args:
        gdf: GeoDataFrame con datos
        resolution: Resolución H3
        output_dir: Directorio de salida
    """
    # Crear directorio para la resolución
    res_dir = Path(output_dir) / f"resolution_{resolution}"
    res_dir.mkdir(exist_ok=True)
    
    # Guardar GeoJSON
    geojson_path = res_dir / f"commercial_h3_{resolution}.geojson"
    gdf.to_file(geojson_path, driver='GeoJSON')
    
    # Guardar CSV
    csv_path = res_dir / f"commercial_h3_{resolution}.csv"
    gdf[[
        'h3_index', 'poi_count', 'categories',
        'poi_density', 'latitude', 'longitude'
    ]].to_csv(csv_path, index=False)

def save_metadata(
    results: Dict[int, gpd.GeoDataFrame],
    output_dir: str
):
    """
    Guarda metadata del procesamiento.
    
    Args:
        results: Diccionario con resultados
        output_dir: Directorio de salida
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'h3_configs': H3_CONFIGS,
        'categories': COMMERCIAL_CATEGORIES,
        'resolutions': {}
    }
    
    for resolution, gdf in results.items():
        # Estadísticas por categoría
        category_stats = {}
        for category in COMMERCIAL_CATEGORIES.keys():
            cat_pois = gdf[gdf['categories'].apply(lambda x: category in x)]
            category_stats[category] = {
                'hexagon_count': len(cat_pois),
                'poi_count': cat_pois['poi_count'].sum(),
                'avg_pois_per_hex': float(cat_pois['poi_count'].mean())
            }
        
        metadata['resolutions'][resolution] = {
            'hexagon_count': len(gdf),
            'total_pois': int(gdf['poi_count'].sum()),
            'mean_density': float(gdf['poi_density'].mean()),
            'area_covered_km2': float(len(gdf) * H3_CONFIGS[resolution]['area_km2']),
            'categories': category_stats
        }
    
    metadata_path = Path(output_dir) / 'commercial_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"📝 Metadata guardada en {metadata_path}")

def main():
    """Función principal para ejecutar el procesamiento."""
    try:
        logger.info("🚀 Iniciando procesamiento de datos OSM")
        
        results = process_osm_data()
        
        logger.info("✅ Procesamiento OSM completado exitosamente")
        
        # Mostrar resumen
        for resolution, gdf in results.items():
            category_counts = {}
            for category in COMMERCIAL_CATEGORIES.keys():
                cat_pois = gdf[gdf['categories'].apply(lambda x: category in x)]
                category_counts[category] = cat_pois['poi_count'].sum()
            
            logger.info(
                f"\nResolución {resolution}:"
                f"\n- Hexágonos: {len(gdf):,}"
                f"\n- POIs totales: {gdf['poi_count'].sum():,.0f}"
                f"\n- Densidad media: {gdf['poi_density'].mean():.1f} POIs/km²"
                f"\n- Área cubierta: {len(gdf) * H3_CONFIGS[resolution]['area_km2']:,.1f} km²"
                f"\n\nPOIs por categoría:"
            )
            
            for category, count in category_counts.items():
                logger.info(f"- {category}: {count:,}")
            
    except Exception as e:
        logger.error(f"❌ Error en proceso principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()