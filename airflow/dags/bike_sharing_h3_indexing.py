from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import json
import h3
from shapely.geometry import Polygon, Point
from pathlib import Path
import psycopg2
import psycopg2.extras
import geopandas as gpd
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir /app al PYTHONPATH si no está ya
if '/app' not in sys.path:
    sys.path.append('/app')

# Importar clases propias
from generate_h3 import H3GridGenerator
from store_data import DatabaseStore

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_h3_indexing',
    default_args=default_args,
    description='Generación de índices H3 para estaciones de bike-sharing',
    schedule_interval=None,  # Solo manual
    catchup=False,
)

def check_dependencies():
    """
    Verifica que todas las dependencias necesarias estén instaladas.
    """
    try:
        import h3
        import shapely
        import psycopg2
        print("✅ Todas las dependencias están instaladas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error: Falta la dependencia {str(e)}")
        print("Instala las dependencias requeridas con:")
        print("pip install h3 shapely psycopg2-binary")
        raise

def get_stations_from_postgres():
    """
    Extrae datos de estaciones desde PostgreSQL para procesarlos con H3.
    """
    print("🔄 Extrayendo datos de estaciones desde PostgreSQL...")
    
    # Configuración de conexión a PostgreSQL
    conn_params = {
        'host': 'db_service',  # Nombre del servicio Docker
        'port': 5432,
        'database': 'geo_db',
        'user': 'geo_user',
        'password': 'NekoSakamoto448'
    }
    
    try:
        # Conectar a PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Consulta para obtener estaciones con coordenadas
        query = """
        SELECT 
            station_id, 
            name, 
            city,
            latitude, 
            longitude, 
            capacity 
        FROM bike_stations
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convertir a DataFrame
        stations_df = pd.DataFrame(rows, columns=['station_id', 'name', 'city', 'latitude', 'longitude', 'capacity'])
        
        print(f"✅ Extraídas {len(stations_df)} estaciones con coordenadas válidas")
        
        # Guardar una copia temporal para debug (opcional)
        debug_dir = Path("/app/data/processed/h3")
        debug_dir.mkdir(exist_ok=True, parents=True)
        stations_df.to_csv(debug_dir / "stations_for_h3.csv", index=False)
        
        cursor.close()
        conn.close()
        
        return stations_df
    
    except Exception as e:
        print(f"❌ Error extrayendo estaciones desde PostgreSQL: {str(e)}")
        raise

def generate_h3_indices(df, auto_resolution=True, base_resolution=9):
    """
    Genera índices H3 para cada estación en el DataFrame utilizando H3GridGenerator.
    
    Args:
        df: DataFrame con columnas 'latitude' y 'longitude'
        auto_resolution: Si se debe ajustar la resolución automáticamente
        base_resolution: Resolución base para H3 (9 o 10)
        
    Returns:
        DataFrame con columnas adicionales para H3
    """
    logger.info(f"🔄 Generando índices H3 (resolución base: {base_resolution}, auto: {auto_resolution})...")
    
    # Verificar que el DataFrame tenga las columnas necesarias
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("El DataFrame debe contener columnas 'latitude' y 'longitude'")
    
    # Crear directorio de salida
    output_dir = Path("/app/data/processed/h3")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Inicializar el generador H3
    h3_generator = H3GridGenerator(
        base_resolution=base_resolution,
        output_dir=str(output_dir),
        auto_resolution=auto_resolution
    )
    
    # Convertir DataFrame a GeoDataFrame con puntos
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    
    # Crear una copia del DataFrame original
    df_h3 = df.copy()
    
    # Función para procesar cada estación
    for idx, row in gdf.iterrows():
        point = row.geometry
        # Determinar resolución basada en configuración
        resolution = h3_generator.determine_resolution(
            population=row.get('total_trips', 0),  # Usar trips como proxy de población
            density=row.get('peak_hour_trips', 0)   # Usar peak_hour_trips como proxy de densidad
        )
        
        # Obtener índice H3
        try:
            h3_index = h3.geo_to_h3(point.y, point.x, resolution)
            df_h3.at[idx, 'h3_index'] = h3_index
            df_h3.at[idx, 'h3_resolution'] = resolution
            
            # Obtener polígono H3
            boundary = h3.h3_to_geo_boundary(h3_index)
            coords = [(lng, lat) for lat, lng in boundary]
            h3_polygon = Polygon(coords)
            
            # Almacenar como WKT para facilitar serialización
            df_h3.at[idx, 'h3_polygon_wkt'] = h3_polygon.wkt
        except Exception as e:
            logger.error(f"❌ Error procesando H3 para estación {row.get('station_id')}: {str(e)}")
    
    # Contar índices válidos
    valid_count = df_h3['h3_index'].notna().sum()
    logger.info(f"✅ Generados {valid_count} índices H3 válidos de {len(df_h3)} estaciones")
    
    # Guardar una copia para verificación
    df_h3.to_csv(output_dir / "stations_with_h3.csv", index=False)
    
    # Si hay resoluciones múltiples, mostrar estadísticas
    if auto_resolution:
        res_counts = df_h3['h3_resolution'].value_counts()
        for res, count in res_counts.items():
            logger.info(f"  - Resolución {res}: {count} estaciones ({count/len(df_h3)*100:.1f}%)")
    
    return df_h3

def store_h3_in_postgres(df_h3, create_new_table=False):
    """
    Almacena los índices H3 en PostgreSQL utilizando DatabaseStore,
    ya sea actualizando la tabla existente o creando una nueva.
    
    Args:
        df_h3: DataFrame con índices H3 y polígonos (como WKT)
        create_new_table: Si es True, crea una nueva tabla; si es False, actualiza la existente
    """
    logger.info("🔄 Almacenando índices H3 en PostgreSQL...")
    
    try:
        # Crear instancia de DatabaseStore para aprovechar sus optimizaciones
        db_store = DatabaseStore()
        
        # Asegurarse de que las tablas existen
        db_store.init_database()
        
        if create_new_table:
            # Código existente para crear nueva tabla...
            # (se mantiene igual)
            logger.info("🔄 Creando nueva tabla bike_stations_h3...")
            # ... (código existente)
        else:
            # Actualizar tabla existente
            logger.info("🔄 Actualizando tabla bike_stations con índices H3...")
            
            # Preparar datos para actualización
            stations_for_update = df_h3.copy()
            
            # Primero, usar store_stations para actualizar campos básicos y el h3_index
            db_store.store_stations(stations_for_update)
            
            # Después, actualizar/crear una columna específica para los polígonos H3
            conn_params = {
                'host': 'db_service',
                'port': 5432,
                'database': 'geo_db',
                'user': 'geo_user',
                'password': 'NekoSakamoto448'
            }
            
            try:
                conn = psycopg2.connect(**conn_params)
                cursor = conn.cursor()
                
                # Primero, verificar si la columna h3_polygon existe, si no, crearla
                cursor.execute("""
                ALTER TABLE bike_stations 
                ADD COLUMN IF NOT EXISTS h3_polygon GEOMETRY(POLYGON, 4326);
                """)
                conn.commit()
                
                # Optimizar PostgreSQL para actualizaciones masivas
                db_store.optimize_postgres_for_bulk_load(cursor)
                
                # Actualizar cada estación con el polígono H3 en la columna h3_polygon
                updated = 0
                for idx, row in stations_for_update.iterrows():
                    if 'h3_polygon_wkt' in row and row['h3_polygon_wkt'] and 'station_id' in row:
                        # Actualizar la columna h3_polygon con el polígono H3
                        cursor.execute("""
                        UPDATE bike_stations
                        SET h3_polygon = ST_SetSRID(ST_GeomFromText(%s), 4326)
                        WHERE station_id = %s
                        """, (row['h3_polygon_wkt'], row['station_id']))
                        updated += 1
                
                conn.commit()
                logger.info(f"✅ Actualizadas {updated} geometrías con hexágonos H3")
                
                # Restaurar configuración de PostgreSQL
                db_store.restore_postgres_settings(cursor)
                cursor.close()
                conn.close()
                
            except Exception as e:
                logger.error(f"❌ Error actualizando geometrías H3: {str(e)}")
                raise
        
        logger.info("✅ Índices H3 almacenados correctamente en PostgreSQL")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error almacenando índices H3 en PostgreSQL: {str(e)}")
        raise

def generate_h3_visualization():
    """
    Genera una visualización HTML de las estaciones con hexágonos H3 utilizando H3GridGenerator.
    Crea visualizaciones separadas por resolución.
    """
    logger.info("🔄 Generando visualización de hexágonos H3...")
    
    try:
        import folium
        
        # Configuración de conexión a PostgreSQL
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'geo_password'
        }
        
        # Conectar a PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Consulta para obtener estaciones con índices H3
        query = """
        SELECT 
            station_id, 
            name, 
            city,
            latitude, 
            longitude, 
            h3_index,
            h3_resolution,
            ST_AsText(h3_polygon) as polygon_wkt,
            COALESCE(total_trips, 0) as total_trips,
            COALESCE(peak_hour_trips, 0) as peak_hour_trips
        FROM bike_stations
        WHERE h3_index IS NOT NULL
        LIMIT 2000
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Cerrar recursos de PostgreSQL
        cursor.close()
        conn.close()
        
        if not rows:
            logger.warning("⚠️ No hay estaciones con índices H3 para visualizar")
            return False
            
        # Crear DataFrame con los resultados
        stations_df = pd.DataFrame(rows, columns=[
            'station_id', 'name', 'city', 'latitude', 'longitude', 
            'h3_index', 'h3_resolution', 'polygon_wkt', 'total_trips', 'peak_hour_trips'
        ])
        
        # Convertir WKT a geometría para GeoDataFrame
        from shapely import wkt
        stations_df['geometry'] = stations_df['polygon_wkt'].apply(
            lambda x: wkt.loads(x) if x and isinstance(x, str) else None
        )
        
        # Convertir a GeoDataFrame
        stations_gdf = gpd.GeoDataFrame(stations_df, geometry='geometry', crs="EPSG:4326")
        
        # Directorio para visualizaciones
        viz_dir = Path("/app/data/processed/h3")
        viz_dir.mkdir(exist_ok=True, parents=True)
        
        # Dividir por resolución para visualización
        resolutions = stations_gdf['h3_resolution'].unique()
        
        # Crear un diccionario de GeoDataFrames por resolución
        resolution_gdfs = {
            res: stations_gdf[stations_gdf['h3_resolution'] == res]
            for res in resolutions
        }
        
        # Inicializar generador de H3
        h3_generator = H3GridGenerator(
            base_resolution=9,
            output_dir=str(viz_dir),
            auto_resolution=True
        )
        
        # Crear mapa multi-resolución
        map_path = h3_generator.create_multi_resolution_map(
            resolution_gdfs,
            center=[stations_gdf['latitude'].mean(), stations_gdf['longitude'].mean()],
            filename="h3_stations_map.html"
        )
        
        # También guardar mapas individuales por ciudad
        cities = stations_gdf['city'].unique()
        
        for city in cities:
            # Filtrar por ciudad
            city_gdf = stations_gdf[stations_gdf['city'] == city]
            
            # Dividir por resolución
            city_res_gdfs = {
                res: city_gdf[city_gdf['h3_resolution'] == res]
                for res in city_gdf['h3_resolution'].unique()
                if len(city_gdf[city_gdf['h3_resolution'] == res]) > 0
            }
            
            if city_res_gdfs:
                # Crear mapa multi-resolución para la ciudad
                city_filename = f"h3_stations_{city.lower().replace(' ', '_')}.html"
                h3_generator.create_multi_resolution_map(
                    city_res_gdfs,
                    center=[city_gdf['latitude'].mean(), city_gdf['longitude'].mean()],
                    filename=city_filename
                )
                logger.info(f"✅ Mapa para {city} guardado como {city_filename}")
        
        # Guardar resultados como GeoJSON
        try:
            for res, gdf in resolution_gdfs.items():
                if len(gdf) > 0:
                    # Crear subdirectorio para resolución
                    res_dir = viz_dir / f"resolution_{res}"
                    res_dir.mkdir(exist_ok=True)
                    
                    # Guardar GeoJSON
                    geojson_path = res_dir / f"bike_stations_h3_{res}.geojson"
                    gdf.to_file(geojson_path, driver='GeoJSON')
                    logger.info(f"✅ GeoJSON para resolución {res} guardado en {geojson_path}")
        except Exception as e:
            logger.warning(f"⚠️ Error al guardar GeoJSON: {str(e)}")
        
        logger.info(f"✅ Visualizaciones H3 guardadas en {viz_dir}")
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️ Dependencia no instalada: {str(e)}. Saltando visualización.")
        return False
    except Exception as e:
        logger.error(f"❌ Error generando visualización: {str(e)}")
        return False

def pipeline_h3_indexing(auto_resolution=True, create_new_table=False):
    """
    Pipeline completo para la generación y almacenamiento de índices H3
    utilizando H3GridGenerator.
    
    Args:
        auto_resolution: Si se debe usar resolución adaptativa (9/10)
        create_new_table: Si se debe crear una tabla nueva para H3
    """
    try:
        logger.info("🚀 Iniciando pipeline de indexación H3...")
        
        # 1. Extraer estaciones de PostgreSQL
        stations_df = get_stations_from_postgres()
        
        # 2. Generar índices H3 (con posibilidad de multi-resolución)
        stations_h3 = generate_h3_indices(
            stations_df, 
            auto_resolution=auto_resolution,
            base_resolution=9
        )
        
        # 3. Almacenar en PostgreSQL
        store_h3_in_postgres(stations_h3, create_new_table=create_new_table)
        
        # 4. Generar visualización
        generate_h3_visualization()
        
        logger.info("✅ Pipeline de indexación H3 completado exitosamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error en el pipeline de H3: {str(e)}")
        raise

# Crear argumentos para nuestros operadores
h3_indexing_args = {
    'auto_resolution': True,      # Usar resolución adaptativa (9 o 10 según densidad)
    'create_new_table': False     # Actualizar tabla existente en lugar de crear nueva
}

# Tareas
check_deps_task = PythonOperator(
    task_id='check_dependencies',
    python_callable=check_dependencies,
    dag=dag,
)

get_stations_task = PythonOperator(
    task_id='get_stations_from_postgres',
    python_callable=get_stations_from_postgres,
    dag=dag,
)

generate_h3_task = PythonOperator(
    task_id='generate_h3_indices',
    python_callable=lambda: generate_h3_indices(
        get_stations_from_postgres(), 
        auto_resolution=h3_indexing_args['auto_resolution'],
        base_resolution=9
    ),
    dag=dag,
)

store_h3_task = PythonOperator(
    task_id='store_h3_in_postgres',
    python_callable=lambda: store_h3_in_postgres(
        generate_h3_indices(get_stations_from_postgres(), 
                           auto_resolution=h3_indexing_args['auto_resolution'],
                           base_resolution=9),
        create_new_table=h3_indexing_args['create_new_table']
    ),
    dag=dag,
)

visualization_task = PythonOperator(
    task_id='generate_visualization',
    python_callable=generate_h3_visualization,
    dag=dag,
)

pipeline_task = PythonOperator(
    task_id='h3_indexing_pipeline',
    python_callable=lambda: pipeline_h3_indexing(
        auto_resolution=h3_indexing_args['auto_resolution'],
        create_new_table=h3_indexing_args['create_new_table']
    ),
    dag=dag,
)

# Definir dependencias - Opción 1: Pipeline completo (por defecto)
check_deps_task >> pipeline_task

# Definir dependencias - Opción 2: Tareas individuales (descomenta para usar)
# check_deps_task >> get_stations_task >> generate_h3_task >> store_h3_task >> visualization_task