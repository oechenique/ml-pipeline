import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
import zipfile
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CensusDataProcessor:
    """Procesa datos geográficos del US Census Bureau."""
    
    # URLs de los shapefiles
    CENSUS_FILES = {
        'counties': {
            'url': 'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip',
            'filename': 'cb_2023_us_county_500k.shp',
            'description': 'US Counties - 1:500k'
        },
        'cbsa': {
            'url': 'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_cbsa_500k.zip',
            'filename': 'cb_2023_us_cbsa_500k.shp',
            'description': 'Core Based Statistical Areas - 1:500k'
        },
        'places': {
            'url': 'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_place_500k.zip',
            'filename': 'cb_2023_us_place_500k.shp',
            'description': 'Places - 1:500k'
        }
    }
    
    # Estados relevantes para el análisis
    TARGET_STATES = {
        'NY': 'New York',
        'CA': 'California'
    }
    
    def __init__(
        self,
        cache_dir: str = "data/raw/census",
        output_dir: str = "data/processed/census",
        force_download: bool = False
    ):
        """
        Inicializa el procesador de datos del Census.
        
        Args:
            cache_dir: Directorio para datos crudos
            output_dir: Directorio para datos procesados
            force_download: Si forzar descarga aunque exista cache
        """
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.force_download = force_download
        
        # Crear directorios
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """
        Descarga archivo con barra de progreso.
        
        Args:
            url: URL del archivo
            filename: Nombre del archivo
            
        Returns:
            Path al archivo descargado
        """
        cache_path = self.cache_dir / filename
        
        if cache_path.exists() and not self.force_download:
            logger.info(f"🔄 Usando archivo en caché: {cache_path}")
            return cache_path
        
        logger.info(f"📥 Descargando {filename}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(cache_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc="Downloading"
            ) as pbar:
                for data in response.iter_content(8192):
                    f.write(data)
                    pbar.update(len(data))
        
        logger.info(f"✅ Archivo descargado: {cache_path}")
        return cache_path
    
    def extract_zip(self, zip_path: Path, extract_dir: Path) -> None:
        """
        Extrae archivo ZIP.
        
        Args:
            zip_path: Path al archivo ZIP
            extract_dir: Directorio de extracción
        """
        logger.info(f"📦 Extrayendo {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def load_shapefile(
        self,
        name: str,
        filter_states: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Carga y procesa shapefile.
        
        Args:
            name: Nombre del dataset ('counties', 'cbsa', o 'places')
            filter_states: Si filtrar solo estados objetivo
            
        Returns:
            GeoDataFrame procesado
        """
        if name not in self.CENSUS_FILES:
            raise ValueError(f"Dataset {name} no válido")
        
        config = self.CENSUS_FILES[name]
        
        # Descargar y extraer si es necesario
        zip_path = self.download_file(config['url'], f"{name}.zip")
        extract_dir = self.cache_dir / name
        
        if not extract_dir.exists() or self.force_download:
            self.extract_zip(zip_path, extract_dir)
        
        # Cargar shapefile
        shp_path = extract_dir / config['filename']
        gdf = gpd.read_file(shp_path)
        
        # Filtrar estados si es necesario
        if filter_states:
            if name == 'counties':
                gdf = gdf[gdf['STATEFP'].isin(self.TARGET_STATES.keys())]
            elif name == 'places':
                gdf = gdf[gdf['STATEFP'].isin(self.TARGET_STATES.keys())]
            elif name == 'cbsa':
                # Para CBSAs, filtrar si intersectan con estados objetivo
                states = self.load_shapefile('states', filter_states=True)
                gdf = gpd.overlay(gdf, states, how='intersection')
        
        logger.info(f"✅ {name.title()}: {len(gdf)} elementos cargados")
        return gdf
    
    def process_counties(self) -> gpd.GeoDataFrame:
        """
        Procesa datos de condados.
        
        Returns:
            GeoDataFrame con condados procesados
        """
        counties = self.load_shapefile('counties')
        
        # Simplificar columnas
        counties = counties[[
            'GEOID', 'NAME', 'STATEFP', 'NAMELSAD',
            'ALAND', 'AWATER', 'geometry'
        ]].rename(columns={
            'GEOID': 'county_id',
            'NAME': 'county_name',
            'STATEFP': 'state_id',
            'NAMELSAD': 'county_full_name',
            'ALAND': 'land_area',
            'AWATER': 'water_area'
        })
        
        return counties
    
    def process_cbsa(self) -> gpd.GeoDataFrame:
        """
        Procesa datos de áreas metropolitanas.
        
        Returns:
            GeoDataFrame con CBSAs procesados
        """
        cbsa = self.load_shapefile('cbsa')
        
        # Simplificar columnas
        cbsa = cbsa[[
            'GEOID', 'NAME', 'NAMELSAD', 'LSAD',
            'ALAND', 'AWATER', 'geometry'
        ]].rename(columns={
            'GEOID': 'cbsa_id',
            'NAME': 'cbsa_name',
            'NAMELSAD': 'cbsa_full_name',
            'LSAD': 'cbsa_type',
            'ALAND': 'land_area',
            'AWATER': 'water_area'
        })
        
        return cbsa
    
    def process_places(self) -> gpd.GeoDataFrame:
        """
        Procesa datos de lugares.
        
        Returns:
            GeoDataFrame con lugares procesados
        """
        places = self.load_shapefile('places')
        
        # Simplificar columnas
        places = places[[
            'GEOID', 'NAME', 'STATEFP', 'PLACEFP',
            'NAMELSAD', 'LSAD', 'geometry'
        ]].rename(columns={
            'GEOID': 'place_id',
            'NAME': 'place_name',
            'STATEFP': 'state_id',
            'PLACEFP': 'place_fips',
            'NAMELSAD': 'place_full_name',
            'LSAD': 'place_type'
        })
        
        return places
    
    def save_results(self, data: Dict[str, gpd.GeoDataFrame]) -> None:
        """
        Guarda resultados procesados.
        
        Args:
            data: Diccionario con GeoDataFrames por tipo
        """
        for name, gdf in data.items():
            # Guardar GeoJSON
            output_path = self.output_dir / f"{name}.geojson"
            gdf.to_file(output_path, driver='GeoJSON')
            
            # Guardar CSV con datos no geométricos
            csv_path = self.output_dir / f"{name}.csv"
            gdf.drop(columns=['geometry']).to_csv(csv_path, index=False)
            
            logger.info(f"💾 {name.title()} guardado en {output_path}")
    
    def process(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Ejecuta el pipeline completo de procesamiento.
        
        Returns:
            Diccionario con GeoDataFrames procesados
        """
        try:
            logger.info("🚀 Iniciando procesamiento de datos del Census")
            
            # Procesar cada tipo de dato
            results = {
                'counties': self.process_counties(),
                'cbsa': self.process_cbsa(),
                'places': self.process_places()
            }
            
            # Guardar resultados
            self.save_results(results)
            
            logger.info("✅ Procesamiento completado exitosamente")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en el procesamiento: {str(e)}")
            raise

def main():
    """Función principal para ejecutar el procesamiento."""
    processor = CensusDataProcessor()
    processor.process()

if __name__ == "__main__":
    main()