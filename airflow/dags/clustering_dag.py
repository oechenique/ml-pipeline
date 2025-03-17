from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
import pandas as pd
import numpy as np
import h3
import psycopg2
import psycopg2.extras
import json
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_clustering',
    default_args=default_args,
    description='Análisis de clusters de bike-sharing',
    schedule_interval=None,  # Triggered by master DAG
    catchup=False,
)

def run_clustering_direct():
    """
    Ejecuta clustering directamente con datos de PostgreSQL sin dependencias externas.
    """
    try:
        print("🚀 Iniciando clustering directo desde PostgreSQL")
        
        # 1. Extraer datos de estaciones con H3 de PostgreSQL
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Consulta para obtener estaciones con H3
        query = """
        SELECT 
            station_id, 
            name, 
            city,
            latitude, 
            longitude, 
            h3_index,
            ST_AsText(CASE WHEN h3_polygon IS NOT NULL THEN h3_polygon ELSE geom END) as geometry_wkt,
            capacity
        FROM bike_stations
        WHERE h3_index IS NOT NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            raise ValueError("No hay estaciones con índices H3 en la base de datos")
            
        # Crear DataFrame con nombres de columnas explícitos
        column_names = [
            'station_id', 'name', 'city', 'latitude', 'longitude', 
            'h3_index', 'geometry_wkt', 'capacity'
        ]
        
        df = pd.DataFrame(rows, columns=column_names)
        print(f"✅ Extraídas {len(df)} estaciones con índices H3")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        
        # 2. Generar features simulados
        # Añadir variables aleatorias para clustering
        np.random.seed(42)  # Para reproducibilidad
        
        # Generar métricas básicas simuladas para cada estación
        df['total_trips_sim'] = np.random.poisson(500, size=len(df))  # Número simulado de viajes
        df['avg_duration'] = np.random.uniform(300, 1800, size=len(df))  # Entre 5 y 30 minutos
        df['rush_hour_trips'] = df['total_trips_sim'] * np.random.uniform(0.3, 0.7, size=len(df))
        df['weekend_ratio'] = np.random.uniform(0.2, 0.8, size=len(df))
        
        # Features de balance
        df['trip_balance'] = np.random.uniform(-50, 50, size=len(df))
        df['balance_ratio'] = np.random.uniform(0.5, 1.5, size=len(df))
        
        # Features temporales
        hours = np.arange(24)
        for h in hours:
            hourly_factor = 1.0
            if 7 <= h <= 9:  # Hora pico mañana
                hourly_factor = 1.5
            elif 16 <= h <= 18:  # Hora pico tarde
                hourly_factor = 1.8
            elif 0 <= h <= 5:  # Noche
                hourly_factor = 0.2
                
            df[f'hour_{h}'] = np.random.poisson(
                df['total_trips_sim'] * hourly_factor * 0.04, 
                size=len(df)
            )
        
        # Normalizar features
        features_to_normalize = [
            'total_trips_sim', 'rush_hour_trips', 'avg_duration', 
            'weekend_ratio', 'balance_ratio', 'trip_balance'
        ]
        
        for col in features_to_normalize:
            mean = df[col].mean()
            std = df[col].std() or 1.0  # Evitar división por cero
            df[f'{col}_norm'] = (df[col] - mean) / std
        
        # 3. Realizar clustering usando K-means básico
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Seleccionar features para clustering
        cluster_features = [f'{col}_norm' for col in features_to_normalize]
        
        # Asegurarse de que no hay NaN
        X = df[cluster_features].fillna(0).values
        
        # Determinar número óptimo de clusters (método del codo)
        distortions = []
        K_range = range(2, 8)  # Reducido para mayor velocidad
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)
        
        # Identificar "codo" en la curva
        deltas = np.diff(distortions)
        optimal_k = K_range[np.argmax(np.abs(np.diff(deltas))) + 1]
        
        print(f"✅ Número óptimo de clusters: {optimal_k}")
        
        # Entrenar modelo con k óptimo
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X)
        
        # 4. Generar análisis de clusters
        cluster_analysis = {}
        
        for cluster in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster]
            
            # Estadísticas básicas
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'avg_trips': float(cluster_data['total_trips_sim'].mean()),
                'avg_duration': float(cluster_data['avg_duration'].mean()),
                'city_distribution': cluster_data['city'].value_counts().to_dict(),
                'feature_patterns': {
                    feature: float(cluster_data[feature].mean())
                    for feature in features_to_normalize
                }
            }
            
            cluster_analysis[f'cluster_{cluster}'] = stats
        
        # 5. Guardar resultados
        output_dir = Path("/app/data/clusters")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar DataFrame con clusters
        clustered_df = df[['station_id', 'name', 'city', 'latitude', 'longitude', 'h3_index', 'cluster']]
        clustered_df.to_csv(output_dir / "station_clusters.csv", index=False)
        
        # Almacenar resultados de clustering en PostgreSQL para integración con otros DAGs
        print("💾 Almacenando resultados de clustering en PostgreSQL...")
            
        try:
            # Reconectar a PostgreSQL
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()
            
            # Crear tabla si no existe
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS station_clusters (
                id SERIAL PRIMARY KEY,
                station_id VARCHAR UNIQUE,
                name VARCHAR,
                city VARCHAR,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                h3_index VARCHAR,
                cluster INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Para una carga más eficiente, crear tabla temporal
            cursor.execute("""
            CREATE TEMP TABLE temp_clusters (
                station_id VARCHAR,
                name VARCHAR,
                city VARCHAR,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                h3_index VARCHAR,
                cluster INTEGER
            ) ON COMMIT DROP;
            """)
            
            # Preparar datos para carga
            from io import StringIO
            buffer = StringIO()
            clustered_df.to_csv(buffer, index=False, header=False, sep='\t')
            buffer.seek(0)
            
            # Cargar datos en tabla temporal
            cursor.copy_from(buffer, 'temp_clusters', sep='\t', null='')
            
            # Upsert a la tabla principal (MERGE en PostgreSQL >= 15, pero usamos ON CONFLICT para compatibilidad)
            cursor.execute("""
            INSERT INTO station_clusters 
            (station_id, name, city, latitude, longitude, h3_index, cluster)
            SELECT 
                station_id, name, city, latitude, longitude, h3_index, cluster
            FROM temp_clusters
            ON CONFLICT (station_id) 
            DO UPDATE SET
                name = EXCLUDED.name,
                city = EXCLUDED.city,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                h3_index = EXCLUDED.h3_index,
                cluster = EXCLUDED.cluster,
                created_at = CURRENT_TIMESTAMP;
            """)
            
            # Guardar estadísticas de clustering para fácil acceso
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS clustering_stats (
                id SERIAL PRIMARY KEY,
                n_clusters INTEGER,
                feature_importance JSONB,
                cluster_analysis JSONB,
                timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # Convertir análisis a formato JSON para PostgreSQL
            import json
            cluster_stats_json = json.dumps(cluster_analysis)
            feature_importance_json = json.dumps(
                dict(zip(
                    cluster_features, 
                    [float(x) for x in np.std(kmeans.cluster_centers_, axis=0)]
                ))
            )
            
            # Almacenar estadísticas actualizadas
            cursor.execute("""
            INSERT INTO clustering_stats 
            (n_clusters, feature_importance, cluster_analysis, timestamp)
            VALUES (%s, %s, %s, %s);
            """, (
                optimal_k,
                feature_importance_json,
                cluster_stats_json,
                datetime.now()
            ))
            
            # Commit cambios
            conn.commit()
            
            # Verificar resultados
            cursor.execute("SELECT COUNT(*) FROM station_clusters;")
            count = cursor.fetchone()[0]
            print(f"✅ Almacenados {count} registros de clusters en PostgreSQL")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error almacenando clusters en PostgreSQL: {str(e)}")
        
        # Continuar con la ejecución aunque falle PostgreSQL (los archivos se guardarán        
        # Guardar resultados y análisis
        results = {
            'n_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'feature_importance': dict(zip(
                cluster_features, 
                np.std(kmeans.cluster_centers_, axis=0)
            )),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / 'clustering_results.json', 'w') as f:
            # Convertir cualquier valor numpy a tipos nativos de Python
            json_results = json.loads(
                json.dumps(results, default=lambda x: float(x) if isinstance(x, np.number) else x)
            )
            json.dump(json_results, f, indent=2)
        
        # 6. Generar visualización básica con HTML
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Análisis de Clusters de Estaciones</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .cluster-card {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; }}
                .cluster-0 {{ background-color: #e6f7ff; }}
                .cluster-1 {{ background-color: #fff7e6; }}
                .cluster-2 {{ background-color: #f6ffed; }}
                .cluster-3 {{ background-color: #fff1f0; }}
                .cluster-4 {{ background-color: #f9f0ff; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Análisis de Clusters de Estaciones</h1>
            <p>Número de clusters: {optimal_k}</p>
            <h2>Resumen de Clusters</h2>
        """
        
        for cluster_id, stats in cluster_analysis.items():
            html_output += f"""
            <div class="cluster-card cluster-{cluster_id.split('_')[1]}">
                <h3>{cluster_id.replace('_', ' ').title()}</h3>
                <p>Estaciones: {stats['size']} ({stats['percentage']:.1f}%)</p>
                <p>Viajes promedio: {stats['avg_trips']:.1f}</p>
                <p>Duración promedio: {stats['avg_duration']:.1f} segundos</p>
                
                <h4>Distribución por ciudad:</h4>
                <ul>
            """
            
            for city, count in stats['city_distribution'].items():
                html_output += f"<li>{city}: {count} estaciones</li>"
                
            html_output += """
                </ul>
                
                <h4>Patrones de features:</h4>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Valor promedio</th>
                    </tr>
            """
            
            for feature, value in stats['feature_patterns'].items():
                html_output += f"""
                    <tr>
                        <td>{feature.replace('_', ' ').title()}</td>
                        <td>{value:.2f}</td>
                    </tr>
                """
                
            html_output += """
                </table>
            </div>
            """
            
        html_output += """
        </body>
        </html>
        """
        
        with open(output_dir / 'clustering_visualization.html', 'w') as f:
            f.write(html_output)
        
        print(f"✅ Resultados guardados en {output_dir}")
        
        # Imprimir resumen
        print("\n📊 Resumen de clusters:")
        for cluster_id, stats in cluster_analysis.items():
            print(f"\n{cluster_id}:")
            print(f"- Tamaño: {stats['size']} estaciones ({stats['percentage']:.1f}%)")
            print(f"- Viajes promedio: {stats['avg_trips']:.1f}")
            print(f"- Duración promedio: {stats['avg_duration']:.1f} segundos")
        
        return results
        
    except Exception as e:
        print(f"❌ Error en clustering directo: {str(e)}")
        raise

# Tarea que ejecuta el clustering directamente
run_clustering_task = PythonOperator(
    task_id='run_clustering',
    python_callable=run_clustering_direct,
    dag=dag,
)