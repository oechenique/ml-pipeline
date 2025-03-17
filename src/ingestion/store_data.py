import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import psycopg2
from psycopg2.extras import execute_values, execute_batch
from sqlalchemy import create_engine
import pandas as pd
import geopandas as gpd
from datetime import datetime
from dotenv import load_dotenv
import calendar
import io
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseStore:
    """Almacena datos procesados en PostgreSQL/PostGIS con optimizaciones para grandes volúmenes."""
    
    # Definiciones de tablas
    TABLE_SCHEMAS = {
        'trips': """
            CREATE TABLE IF NOT EXISTS bike_trips (
                trip_id VARCHAR,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_sec INTEGER,
                start_station_id VARCHAR,
                end_station_id VARCHAR,
                bike_id VARCHAR,
                user_type VARCHAR,
                birth_year INTEGER,
                gender VARCHAR,
                city VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (trip_id, start_time)
            )
            PARTITION BY RANGE (start_time);
        """,
        'stations': """
            CREATE TABLE IF NOT EXISTS bike_stations (
                station_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                city VARCHAR,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                h3_index VARCHAR,
                total_trips INTEGER,
                peak_hour_trips INTEGER,
                geom GEOMETRY(Point, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        'bike_infrastructure': """
            CREATE TABLE IF NOT EXISTS bike_infrastructure (
                id SERIAL PRIMARY KEY,
                osm_id VARCHAR,
                type VARCHAR,
                name VARCHAR,
                city VARCHAR,
                geom GEOMETRY(Geometry, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """,
        'transit_connections': """
            CREATE TABLE IF NOT EXISTS transit_connections (
                id SERIAL PRIMARY KEY,
                osm_id VARCHAR,
                type VARCHAR,
                name VARCHAR,
                city VARCHAR,
                operator VARCHAR,
                geom GEOMETRY(Point, 4326),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
    }
    
    def __init__(self, data_dir: str = "data/processed", batch_size: int = 100000):
        """
        Inicializa el almacenamiento en base de datos.
        
        Args:
            data_dir: Directorio con datos procesados
            batch_size: Tamaño de lote para operaciones masivas
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Cargar variables de entorno
        load_dotenv()
        
        # Configuración de la base de datos
        self.db_params = {
            'dbname': os.getenv('POSTGRES_DB', 'bike_sharing'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
        
        # Conexión SQLAlchemy para pandas
        self.engine = create_engine(
            f"postgresql://{self.db_params['user']}:{self.db_params['password']}"
            f"@{self.db_params['host']}:{self.db_params['port']}"
            f"/{self.db_params['dbname']}"
        )
        
        # Configurar índices
        self.indices = {
            'bike_trips': [
                "CREATE INDEX IF NOT EXISTS idx_trips_start_station ON bike_trips(start_station_id);",
                "CREATE INDEX IF NOT EXISTS idx_trips_end_station ON bike_trips(end_station_id);",
                "CREATE INDEX IF NOT EXISTS idx_trips_start_time ON bike_trips(start_time);",
            ],
            'bike_stations': [
                "CREATE INDEX IF NOT EXISTS idx_stations_geom ON bike_stations USING GIST(geom);",
                "CREATE INDEX IF NOT EXISTS idx_stations_h3 ON bike_stations(h3_index);",
            ],
            'bike_infrastructure': [
                "CREATE INDEX IF NOT EXISTS idx_infrastructure_geom ON bike_infrastructure USING GIST(geom);",
                "CREATE INDEX IF NOT EXISTS idx_infrastructure_type ON bike_infrastructure(type);",
            ],
            'transit_connections': [
                "CREATE INDEX IF NOT EXISTS idx_transit_geom ON transit_connections USING GIST(geom);",
                "CREATE INDEX IF NOT EXISTS idx_transit_type ON transit_connections(type);",
            ]
        }
    
    def init_database(self) -> None:
        """Inicializa esquema de base de datos con optimizaciones."""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Habilitar PostGIS
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    
                    # Crear tablas
                    for table_name, schema in self.TABLE_SCHEMAS.items():
                        cur.execute(schema)
                        logger.info(f"✅ Tabla {table_name} creada/verificada")
                    
                    # Crear índices (solo después de crear tablas)
                    for table, index_list in self.indices.items():
                        for index_sql in index_list:
                            cur.execute(index_sql)
                        logger.info(f"✅ Índices para {table} creados/verificados")
                    
                    conn.commit()
                    
            logger.info("✅ Base de datos inicializada correctamente")
                    
        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos: {str(e)}")
            raise
    
    def create_partition(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """
        Crea partición para rango de fechas con optimizaciones.
        
        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            
        Returns:
            str: Nombre de la partición creada
        """
        # Asegurarnos de crear particiones por mes completo
        year = start_date.year
        month = start_date.month
        
        # Determinar el primer día del mes
        first_day = datetime(year, month, 1)
        
        # Determinar el primer día del siguiente mes
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        
        partition_name = f"trips_{year}{month:02d}"
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Verificar si la partición ya existe
                    cur.execute(f"""
                        SELECT 1 FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = '{partition_name}';
                    """)
                    
                    if cur.fetchone() is None:
                        # Crear partición si no existe
                        cur.execute(f"""
                            CREATE TABLE IF NOT EXISTS {partition_name}
                            PARTITION OF bike_trips
                            FOR VALUES FROM ('{first_day}') TO ('{next_month}');
                        """)
                        conn.commit()
                        logger.info(f"✅ Partición {partition_name} creada para rango {first_day} a {next_month}")
                    else:
                        logger.info(f"✅ Partición {partition_name} ya existe")
            
            # Para cubrir todo el rango de datos, también verificamos si necesitamos crear más particiones
            current_date = first_day
            while current_date < end_date:
                if current_date.month == 12:
                    next_date = datetime(current_date.year + 1, 1, 1)
                else:
                    next_date = datetime(current_date.year, current_date.month + 1, 1)
                
                if current_date > first_day:  # Evitar duplicar la primera partición
                    partition_name = f"trips_{current_date.year}{current_date.month:02d}"
                    with psycopg2.connect(**self.db_params) as conn:
                        with conn.cursor() as cur:
                            # Verificar si la partición ya existe
                            cur.execute(f"""
                                SELECT 1 FROM pg_class c
                                JOIN pg_namespace n ON n.oid = c.relnamespace
                                WHERE c.relname = '{partition_name}';
                            """)
                            
                            if cur.fetchone() is None:
                                # Crear partición si no existe
                                cur.execute(f"""
                                    CREATE TABLE IF NOT EXISTS {partition_name}
                                    PARTITION OF bike_trips
                                    FOR VALUES FROM ('{current_date}') TO ('{next_date}');
                                """)
                                conn.commit()
                                logger.info(f"✅ Partición adicional {partition_name} creada para rango {current_date} a {next_date}")
                            else:
                                logger.info(f"✅ Partición {partition_name} ya existe")
                
                current_date = next_date
            
            return partition_name
            
        except Exception as e:
            logger.error(f"❌ Error creando partición: {str(e)}")
            raise
    
    def optimize_postgres_for_bulk_load(self, cur):
        """Configura PostgreSQL para optimizar la carga masiva de datos."""
        # Usar solo parámetros que se pueden cambiar a nivel de sesión
        
        # Aumentar memoria para mantenimiento (parámetro de sesión)
        cur.execute("SET maintenance_work_mem = '1GB';")
        
        # Desactivar sincronización para mejorar rendimiento (parámetro de sesión)
        cur.execute("SET synchronous_commit = off;")
        
        # Usar work_mem para operaciones de ordenamiento (parámetro de sesión)
        cur.execute("SET work_mem = '256MB';")
        
        logger.info("✅ PostgreSQL optimizado para carga masiva (configuración básica)")

    def restore_postgres_settings(self, cur):
        """Restaura la configuración por defecto de PostgreSQL."""
        # Restaurar solo los parámetros que modificamos
        cur.execute("SET maintenance_work_mem = DEFAULT;")
        cur.execute("SET synchronous_commit = on;")
        cur.execute("SET work_mem = DEFAULT;")
        
        logger.info("✅ Configuración de PostgreSQL restaurada")
    
    # Reemplaza solo esta función en tu archivo store_data.py:

    def store_trips(
        self,
        trips_df: pd.DataFrame,
        partition_info: Dict[str, str],
        truncate_first: bool = False  # Parámetro opcional para limpiar la partición
    ) -> None:
        """
        Almacena datos de viajes optimizado para cargas masivas.
        
        Args:
            trips_df: DataFrame con viajes
            partition_info: Información de la partición
            truncate_first: Si es True, trunca la partición antes de insertar
        """
        try:
            # Crear partición si no existe
            start_date = datetime.strptime(partition_info['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(partition_info['end_date'], '%Y-%m-%d')
            partition_name = self.create_partition(start_date, end_date)
            
            # Limpiar la partición si se solicita
            if truncate_first and partition_name:
                try:
                    with psycopg2.connect(**self.db_params) as conn:
                        with conn.cursor() as cur:
                            logger.info(f"🧹 Limpiando datos existentes en la partición {partition_name}...")
                            cur.execute(f"TRUNCATE TABLE {partition_name};")
                            conn.commit()
                            logger.info(f"✅ Partición {partition_name} limpiada correctamente")
                except Exception as e:
                    logger.warning(f"⚠️ Error al limpiar la partición {partition_name}: {str(e)}")
            
            # Filtrar solo las columnas que existen en la tabla bike_trips
            table_columns = [
                'trip_id', 'start_time', 'end_time', 'duration_sec', 
                'start_station_id', 'end_station_id', 'bike_id', 
                'user_type', 'birth_year', 'gender', 'city'
            ]
            
            # Filtrar el DataFrame para incluir solo las columnas que coinciden con la tabla
            valid_columns = [col for col in table_columns if col in trips_df.columns]
            filtered_trips_df = trips_df[valid_columns].copy()
            
            # Convertir tipos de datos para compatibilidad con PostgreSQL
            if 'birth_year' in filtered_trips_df.columns:
                # Convertir birth_year a entero, manejando los NaN
                filtered_trips_df['birth_year'] = pd.to_numeric(filtered_trips_df['birth_year'], errors='coerce')
                # Sustituir NaN por valores nulos
                filtered_trips_df['birth_year'] = filtered_trips_df['birth_year'].fillna(pd.NA)
                
                # Convertir flotantes a enteros donde sea posible (valores no nulos)
                mask = ~filtered_trips_df['birth_year'].isna()
                if mask.any():
                    filtered_trips_df.loc[mask, 'birth_year'] = filtered_trips_df.loc[mask, 'birth_year'].astype(int)
            
            # Asegurar que las fechas estén en formato correcto para PostgreSQL
            if 'start_time' in filtered_trips_df.columns:
                filtered_trips_df['start_time'] = pd.to_datetime(filtered_trips_df['start_time'])
            
            if 'end_time' in filtered_trips_df.columns:
                filtered_trips_df['end_time'] = pd.to_datetime(filtered_trips_df['end_time'])
            
            # Convertir duration_sec a entero si es necesario
            if 'duration_sec' in filtered_trips_df.columns and filtered_trips_df['duration_sec'].dtype != 'int64':
                filtered_trips_df['duration_sec'] = pd.to_numeric(filtered_trips_df['duration_sec'], errors='coerce')
                mask = ~filtered_trips_df['duration_sec'].isna()
                if mask.any():
                    filtered_trips_df.loc[mask, 'duration_sec'] = filtered_trips_df.loc[mask, 'duration_sec'].astype(int)
            
            # Obtener número total de registros
            total_rows = len(filtered_trips_df)
            logger.info(f"🔄 Procesando {total_rows} viajes para almacenamiento")
            
            # Conexión a PostgreSQL
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Optimizar PostgreSQL para carga masiva
                    self.optimize_postgres_for_bulk_load(cur)
                    
                    # Procesar en lotes para evitar problemas de memoria
                    num_batches = (total_rows + self.batch_size - 1) // self.batch_size
                    logger.info(f"📊 Dividiendo en {num_batches} lotes de hasta {self.batch_size} registros")
                    
                    # Procesar en lotes
                    for i in range(0, total_rows, self.batch_size):
                        end_idx = min(i + self.batch_size, total_rows)
                        batch_df = filtered_trips_df.iloc[i:end_idx]
                        
                        logger.info(f"🔄 Procesando lote {i//self.batch_size + 1}/{num_batches} ({len(batch_df)} registros)")
                        
                        # Usar SQL COPY con una tabla temporal
                        try:
                            # IMPORTANTE: Crear tabla temporal dentro del bucle para cada lote
                            cur.execute("""
                            CREATE TEMP TABLE temp_trips (
                                trip_id VARCHAR,
                                start_time TIMESTAMP,
                                end_time TIMESTAMP,
                                duration_sec INTEGER,
                                start_station_id VARCHAR,
                                end_station_id VARCHAR,
                                bike_id VARCHAR,
                                user_type VARCHAR,
                                birth_year INTEGER,
                                gender VARCHAR,
                                city VARCHAR
                            ) ON COMMIT DROP;
                            """)
                            
                            # Preparar datos para CSV
                            csv_data = io.StringIO()
                            
                            # Usar to_csv con NULL String personalizado para valores nulos
                            batch_df.to_csv(csv_data, index=False, header=False, sep='\t', na_rep=r'\N')
                            csv_data.seek(0)
                            
                            # Usar COPY con NULL explícito
                            cur.copy_expert(
                                """
                                COPY temp_trips FROM STDIN WITH (
                                    FORMAT CSV,
                                    DELIMITER E'\\t',
                                    NULL '\\N'
                                )
                                """, 
                                csv_data
                            )
                            
                            # UPSERT desde tabla temporal a tabla principal
                            cur.execute("""
                            INSERT INTO bike_trips(
                                trip_id, start_time, end_time, duration_sec, 
                                start_station_id, end_station_id, bike_id, user_type, 
                                birth_year, gender, city
                            )
                            SELECT 
                                trip_id, start_time, end_time, duration_sec, 
                                start_station_id, end_station_id, bike_id, user_type, 
                                birth_year, gender, city
                            FROM temp_trips
                            ON CONFLICT (trip_id, start_time) 
                            DO NOTHING;
                            """)
                            
                            # Commit cada lote - la tabla temporal se eliminará debido a ON COMMIT DROP
                            conn.commit()
                            logger.info(f"✅ Lote {i//self.batch_size + 1}/{num_batches} completado")
                            
                        except Exception as e:
                            logger.error(f"❌ Error en lote {i//self.batch_size + 1}/{num_batches}: {str(e)}")
                            # Intentar continuar con el siguiente lote
                            conn.rollback()
                    
                    # Restaurar configuración de PostgreSQL
                    self.restore_postgres_settings(cur)
                    
                    logger.info(f"✅ {total_rows} viajes procesados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error almacenando viajes: {str(e)}")
            raise
    
    def store_stations(self, stations_df):
        """
        Almacena datos de estaciones de forma segura, procesando una a una.
        
        Args:
            stations_df: DataFrame con estaciones
        """
        try:
            # Verificar si ya es un GeoDataFrame o necesitamos convertirlo
            if not isinstance(stations_df, gpd.GeoDataFrame) and 'geometry' not in stations_df.columns:
                if 'longitude' in stations_df.columns and 'latitude' in stations_df.columns:
                    # Convertir a GeoDataFrame
                    stations_gdf = gpd.GeoDataFrame(
                        stations_df,
                        geometry=gpd.points_from_xy(
                            stations_df.longitude,
                            stations_df.latitude
                        ),
                        crs="EPSG:4326"
                    )
                else:
                    raise ValueError("El DataFrame de estaciones debe tener columnas 'longitude' y 'latitude'")
            else:
                stations_gdf = stations_df
            
            # Eliminar duplicados para evitar el error de PostgreSQL
            stations_gdf = stations_gdf.drop_duplicates(subset=['station_id'], keep='first')
            logger.info(f"🔄 Procesando {len(stations_gdf)} estaciones únicas para almacenamiento")
            
            # Asegurar que tenemos las columnas necesarias
            required_cols = ['station_id', 'name', 'city', 'latitude', 'longitude']
            for col in required_cols:
                if col not in stations_gdf.columns:
                    if col == 'city' and 'city' not in stations_gdf.columns:
                        # Si falta ciudad, usar valor predeterminado
                        stations_gdf['city'] = 'Unknown'
                    else:
                        raise ValueError(f"Columna requerida '{col}' no encontrada en el DataFrame de estaciones")
            
            # Añadir columnas opcionales si no existen
            if 'h3_index' not in stations_gdf.columns:
                stations_gdf['h3_index'] = ''
            
            if 'total_trips' not in stations_gdf.columns:
                stations_gdf['total_trips'] = 0
            
            if 'peak_hour_trips' not in stations_gdf.columns:
                stations_gdf['peak_hour_trips'] = 0
            
            # Convertir a formato PostGIS
            stations_gdf['geom'] = stations_gdf.geometry.apply(lambda x: f"SRID=4326;{x.wkt}")
            
            # Procesar cada estación por separado para evitar el error
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Usar execute_batch para procesar en pequeños lotes
                    insert_sql = """
                    INSERT INTO bike_stations(
                        station_id, name, city, latitude, longitude,
                        h3_index, total_trips, peak_hour_trips, geom
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::geometry(Point, 4326))
                    ON CONFLICT (station_id) 
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        city = EXCLUDED.city,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        h3_index = EXCLUDED.h3_index,
                        total_trips = EXCLUDED.total_trips,
                        peak_hour_trips = EXCLUDED.peak_hour_trips,
                        geom = EXCLUDED.geom,
                        updated_at = CURRENT_TIMESTAMP;
                    """
                    
                    # Crear valores para la inserción
                    values = [
                        (
                            row.station_id,
                            row.name,
                            row.city,
                            row.latitude,
                            row.longitude,
                            row.h3_index,
                            row.total_trips,
                            row.peak_hour_trips,
                            row.geom
                        )
                        for _, row in stations_gdf.iterrows()
                    ]
                    
                    # Usar execute_batch con un tamaño de lote pequeño
                    execute_batch(cur, insert_sql, values, page_size=10)
                    
                    # Commit cambios
                    conn.commit()
                    
                    logger.info(f"✅ {len(stations_gdf)} estaciones almacenadas exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error almacenando estaciones: {str(e)}")
            raise
    
    def store_infrastructure(
        self,
        infra_gdf: gpd.GeoDataFrame,
        type_name: str
    ) -> None:
        """
        Almacena datos de infraestructura optimizado.
        
        Args:
            infra_gdf: GeoDataFrame con infraestructura
            type_name: Tipo de infraestructura
        """
        try:
            # Verificar que sea un GeoDataFrame válido
            if not isinstance(infra_gdf, gpd.GeoDataFrame):
                raise ValueError("Se requiere un GeoDataFrame para almacenar infraestructura")
            
            if len(infra_gdf) == 0:
                logger.warning(f"⚠️ GeoDataFrame de infraestructura '{type_name}' está vacío, omitiendo")
                return
            
            # Convertir a formato PostGIS
            infra_gdf['geom'] = infra_gdf.geometry.apply(lambda x: f"SRID=4326;{x.wkt}")
            
            # Determinar tabla destino
            table_name = 'bike_infrastructure' if type_name == 'bike' else 'transit_connections'
            
            logger.info(f"🔄 Procesando {len(infra_gdf)} elementos de {type_name} para almacenamiento")
            
            # Optimizar inserción usando COPY
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Optimizar PostgreSQL para carga masiva
                    self.optimize_postgres_for_bulk_load(cur)
                    
                    # Crear tabla temporal con estructura adecuada
                    if table_name == 'bike_infrastructure':
                        cur.execute("""
                        CREATE TEMP TABLE temp_infrastructure (
                            osm_id VARCHAR,
                            type VARCHAR,
                            name VARCHAR,
                            city VARCHAR,
                            geom TEXT
                        ) ON COMMIT DROP;
                        """)
                        
                        # Preparar columnas
                        if 'osm_id' not in infra_gdf.columns:
                            infra_gdf['osm_id'] = ''
                        
                        if 'type' not in infra_gdf.columns:
                            infra_gdf['type'] = type_name
                        
                        if 'name' not in infra_gdf.columns:
                            infra_gdf['name'] = ''
                        
                        if 'city' not in infra_gdf.columns:
                            infra_gdf['city'] = 'Unknown'
                        
                        # Seleccionar columnas para exportar
                        export_cols = infra_gdf[['osm_id', 'type', 'name', 'city', 'geom']]
                        
                    else:  # transit_connections
                        cur.execute("""
                        CREATE TEMP TABLE temp_infrastructure (
                            osm_id VARCHAR,
                            type VARCHAR,
                            name VARCHAR,
                            city VARCHAR,
                            operator VARCHAR,
                            geom TEXT
                        ) ON COMMIT DROP;
                        """)
                        
                        # Preparar columnas
                        if 'osm_id' not in infra_gdf.columns:
                            infra_gdf['osm_id'] = ''
                        
                        if 'type' not in infra_gdf.columns:
                            infra_gdf['type'] = type_name
                        
                        if 'name' not in infra_gdf.columns:
                            infra_gdf['name'] = ''
                        
                        if 'city' not in infra_gdf.columns:
                            infra_gdf['city'] = 'Unknown'
                        
                        if 'operator' not in infra_gdf.columns:
                            infra_gdf['operator'] = ''
                        
                        # Seleccionar columnas para exportar
                        export_cols = infra_gdf[['osm_id', 'type', 'name', 'city', 'operator', 'geom']]
                    
                    # Crear buffer CSV en memoria
                    buffer = io.StringIO()
                    export_cols.to_csv(buffer, index=False, header=False, sep='\t')
                    buffer.seek(0)
                    
                    # Usar COPY para cargar en tabla temporal
                    cur.copy_from(
                        buffer,
                        'temp_infrastructure',
                        sep='\t',
                        null=''
                    )
                    
                    # Insertar desde tabla temporal a tabla principal
                    if table_name == 'bike_infrastructure':
                        cur.execute("""
                        INSERT INTO bike_infrastructure(osm_id, type, name, city, geom)
                        SELECT osm_id, type, name, city, geom::geometry(Geometry, 4326)
                        FROM temp_infrastructure;
                        """)
                    else:
                        cur.execute("""
                        INSERT INTO transit_connections(osm_id, type, name, city, operator, geom)
                        SELECT osm_id, type, name, city, operator, geom::geometry(Point, 4326)
                        FROM temp_infrastructure;
                        """)
                    
                    # Commit cambios
                    conn.commit()
                    
                    # Restaurar configuración de PostgreSQL
                    self.restore_postgres_settings(cur)
                    
                    logger.info(f"✅ {len(infra_gdf)} elementos de {type_name} almacenados exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error almacenando infraestructura: {str(e)}")
            raise
    
    def store_partition(
        self,
        partition_dir: Path,
        partition_info: Dict[str, str],
        truncate_first: bool = False  # Nuevo parámetro para limpiar particiones
    ) -> None:
        """
        Almacena todos los datos de una partición de manera optimizada.
        
        Args:
            partition_dir: Directorio de la partición
            partition_info: Información de la partición
            truncate_first: Si es True, limpia las particiones antes de insertar
        """
        try:
            logger.info(f"🔄 Procesando partición: {partition_dir}")
            
            # Verificar archivos
            trips_file = partition_dir / "trips.parquet"
            stations_file = partition_dir / "stations.parquet"
            
            if not trips_file.exists():
                raise FileNotFoundError(f"Archivo de viajes no encontrado: {trips_file}")
            
            if not stations_file.exists():
                raise FileNotFoundError(f"Archivo de estaciones no encontrado: {stations_file}")
            
            # Leer datos - optimizado para grandes archivos
            logger.info(f"📊 Leyendo estaciones desde {stations_file}")
            stations_df = pd.read_parquet(stations_file)
            
            # Almacenar estaciones (primero para mantener integridad referencial)
            self.store_stations(stations_df)
            
            # Procesar viajes en fragmentos para archivos muy grandes
            logger.info(f"📊 Leyendo viajes desde {trips_file}")
            
            # Detectar si el archivo es demasiado grande (más de 1GB)
            file_size = trips_file.stat().st_size
            
            if file_size > 1e9:  # Más de 1GB
                logger.info(f"⚠️ Archivo de viajes grande detectado ({file_size / 1e9:.2f} GB), procesando en fragmentos")
                
                # Leer y procesar en fragmentos
                chunk_size = self.batch_size  # Usar el mismo tamaño de lote
                trip_chunks = pd.read_parquet(trips_file, chunksize=chunk_size)
                
                chunk_num = 1
                total_rows = 0
                
                for chunk_df in trip_chunks:
                    logger.info(f"🔄 Procesando fragmento {chunk_num} con {len(chunk_df)} viajes")
                    # Solo pasar truncate_first=True para el primer fragmento
                    self.store_trips(chunk_df, partition_info, truncate_first=(chunk_num == 1 and truncate_first))
                    total_rows += len(chunk_df)
                    chunk_num += 1
                
                logger.info(f"✅ Procesados {total_rows} viajes en {chunk_num-1} fragmentos")
            else:
                # Archivo pequeño, procesar todo junto
                trips_df = pd.read_parquet(trips_file)
                logger.info(f"📊 Leyendo {len(trips_df)} viajes desde {trips_file}")
                
                # Almacenar viajes
                self.store_trips(trips_df, partition_info, truncate_first=truncate_first)
            
            logger.info(f"✅ Partición {partition_info['start_date']} almacenada correctamente")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error procesando partición {partition_dir}: {str(e)}")
            raise
    
    def process(self, truncate_first: bool = False) -> None:
        """
        Ejecuta el pipeline completo de almacenamiento con optimizaciones.
        
        Args:
            truncate_first: Si es True, limpia las particiones antes de insertar
        """
        try:
            logger.info("🚀 Iniciando almacenamiento en base de datos")
            start_time = time.time()
            
            # Inicializar base de datos
            self.init_database()
            
            # Procesar cada partición en el directorio de datos
            bike_data_dir = self.data_dir / "bike_sharing"
            
            if not bike_data_dir.exists():
                logger.warning(f"⚠️ Directorio {bike_data_dir} no encontrado")
                # Intentar con ruta alternativa
                bike_data_dir = Path(self.data_dir)
                
                if not bike_data_dir.exists():
                    raise FileNotFoundError(f"Directorio no encontrado: {bike_data_dir}")
            
            partitions = list(bike_data_dir.glob("partition_*"))
            
            if not partitions:
                raise ValueError(f"No se encontraron particiones en {bike_data_dir}")
            
            logger.info(f"🔍 Se encontraron {len(partitions)} particiones")
            
            # Procesar cada partición
            for partition_dir in partitions:
                # Extraer fechas de la partición
                dates = partition_dir.name.split('_')[1:]
                if len(dates) >= 2:
                    partition_info = {
                        'start_date': dates[0],
                        'end_date': dates[1]
                    }
                    
                    # Almacenar partición
                    self.store_partition(partition_dir, partition_info, truncate_first=truncate_first)
                else:
                    logger.warning(f"⚠️ Formato de partición no reconocido: {partition_dir.name}")
            
            # Procesar infraestructura OSM si existe
            osm_dir = self.data_dir / "osm"
            if osm_dir.exists():
                for city_dir in osm_dir.glob("*"):
                    if city_dir.is_dir():
                        city = city_dir.name
                        
                        # Infraestructura ciclista
                        bike_routes_file = city_dir / "bike_routes.geojson"
                        bike_points_file = city_dir / "bike_points.geojson"
                        
                        if bike_routes_file.exists() and bike_points_file.exists():
                            bike_routes = gpd.read_file(bike_routes_file)
                            bike_points = gpd.read_file(bike_points_file)
                            self.store_infrastructure(
                                pd.concat([bike_routes, bike_points]),
                                'bike'
                            )
                        
                        # Conexiones de transporte
                        transit_file = city_dir / "transit_points.geojson"
                        if transit_file.exists():
                            transit = gpd.read_file(transit_file)
                            self.store_infrastructure(transit, 'transit')
            
            end_time = time.time()
            logger.info(f"✅ Almacenamiento completado exitosamente en {end_time - start_time:.2f} segundos")
            
        except Exception as e:
            logger.error(f"❌ Error en el proceso: {str(e)}")
            raise

def main():
    """Función principal para ejecutar el almacenamiento."""
    # Por defecto sin truncar, pasar truncate_first=True para limpiar datos existentes
    store = DatabaseStore()
    store.process(truncate_first=False)

if __name__ == "__main__":
    main()