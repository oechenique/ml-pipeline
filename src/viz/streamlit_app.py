import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap, MarkerCluster
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configuración de la página - Usando emojis Unicode correctos
# IMPORTANTE: esta llamada debe estar al principio y solo ocurrir una vez
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .chart-container {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 10px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #f0f9ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .stDataFrame {
        padding: 5px;
    }
    .stTable {
        padding: 5px;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    /* Ocultar el botón de fullscreen del mapa */
    .folium-map .leaflet-control-zoom {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Funciones de conexión a base de datos y carga de datos
def get_db_connection():
    """Establece conexión con PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host="db_service",
            port=5432,
            database="geo_db",
            user="geo_user",
            password="NekoSakamoto448"
        )
        return conn
    except Exception as e:
        st.error(f"No se pudo conectar a la base de datos: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_station_data():
    """Carga datos de estaciones desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo o crear muestra si no hay conexión
            # Intentar cargar desde archivos generados si existen
            backup_path = Path("/app/data/clusters/station_clusters.csv")
            if backup_path.exists():
                return pd.read_csv(backup_path)
            return create_sample_stations()
        
        query = """
        SELECT 
            s.station_id, 
            s.name, 
            s.city,
            s.latitude, 
            s.longitude, 
            s.capacity,
            s.h3_index,
            COALESCE(c.cluster, -1) as cluster,
            ST_AsGeoJSON(s.geom) as geometry,
            ST_AsGeoJSON(s.h3_polygon) as h3_geometry
        FROM bike_stations s
        LEFT JOIN station_clusters c ON s.station_id = c.station_id
        WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convertir a GeoDataFrame si geometry existe
        if 'geometry' in df.columns and df['geometry'].notna().any():
            df['geometry'] = df['geometry'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else None)
            geometry = [g for g in df['geometry'] if g]
            if geometry:
                gdf = gpd.GeoDataFrame.from_features(
                    [{"type": "Feature", "geometry": g, "properties": {}} for g in geometry],
                    crs="EPSG:4326"
                )
                # Copiar columnas del DataFrame original
                for col in df.columns:
                    if col != 'geometry':
                        gdf[col] = df[col].values
                return gdf
        return df
        
    except Exception as e:
        st.error(f"Error al cargar datos de estaciones: {str(e)}")
        return create_sample_stations()

@st.cache_data(ttl=3600)
def load_cluster_data():
    """Carga análisis de clusters desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Intentar cargar desde archivos generados si existen
            backup_path = Path("/app/data/clusters/clustering_results.json")
            if backup_path.exists():
                with open(backup_path) as f:
                    return json.load(f).get("cluster_analysis", {})
            return create_sample_clusters()
        
        query = """
        SELECT 
            cluster_analysis
        FROM clustering_stats
        ORDER BY created_at DESC
        LIMIT 1;
        """
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row and row[0]:
            return row[0]
        else:
            # Intentar cargar desde archivos generados si existen
            backup_path = Path("/app/data/clusters/clustering_results.json")
            if backup_path.exists():
                with open(backup_path) as f:
                    return json.load(f).get("cluster_analysis", {})
            return create_sample_clusters()
            
    except Exception as e:
        st.error(f"Error al cargar datos de clustering: {str(e)}")
        return create_sample_clusters()

@st.cache_data(ttl=3600)
def load_prediction_data():
    """Carga predicciones de demanda desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Intentar cargar desde archivos generados si existen
            backup_path = Path("/app/data/predictions/demand_predictions.csv")
            if backup_path.exists():
                return pd.read_csv(backup_path)
            return create_sample_predictions()
        
        query = """
        SELECT 
            station_id,
            prediction_date,
            prediction_hour,
            predicted_demand,
            is_weekend,
            cluster
        FROM bike_demand_predictions
        WHERE prediction_date >= CURRENT_DATE - INTERVAL '1 day'
        ORDER BY prediction_date, prediction_hour
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) == 0:
            # Intentar cargar desde archivos generados si existen
            backup_path = Path("/app/data/predictions/demand_predictions.csv")
            if backup_path.exists():
                return pd.read_csv(backup_path)
            return create_sample_predictions()
            
        return df
        
    except Exception as e:
        st.error(f"Error al cargar predicciones: {str(e)}")
        return create_sample_predictions()

# Funciones para generar datos de muestra solo cuando sea necesario
def create_sample_stations(n=50):
    """Crea datos de estaciones de muestra."""
    # Coordenadas base para San Francisco y Nueva York
    sf_coords = (37.7749, -122.4194)
    ny_coords = (40.7128, -74.0060)
    
    data = []
    for i in range(1, n+1):
        city = "San Francisco" if i % 3 != 0 else "New York"
        base_coords = sf_coords if city == "San Francisco" else ny_coords
        
        # Generar coordenadas con ruido para distribuir puntos
        lat = base_coords[0] + (np.random.random() - 0.5) * 0.05
        lon = base_coords[1] + (np.random.random() - 0.5) * 0.05
        
        # Asignar cluster para visualización
        cluster = i % 3
        
        data.append({
            "station_id": f"station_{i}",
            "name": f"Station {i}",
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "capacity": np.random.randint(15, 30),
            "h3_index": f"8a2a100d27fffff{i:02d}",  # ID de H3 ficticio
            "cluster": cluster
        })
    
    return pd.DataFrame(data)

def create_sample_clusters():
    """Crea datos de clusters de muestra."""
    return {
        "cluster_0": {
            "size": 20,
            "percentage": 40.0,
            "avg_trips": 120.5,
            "avg_duration": 850.2,
            "city_distribution": {"San Francisco": 15, "New York": 5},
            "feature_patterns": {
                "total_trips_norm": 1.2,
                "rush_hour_trips_norm": 1.5,
                "avg_duration_norm": 0.8,
                "weekend_ratio_norm": 0.9,
                "balance_ratio_norm": 1.1,
                "trip_balance_norm": 0.7
            }
        },
        "cluster_1": {
            "size": 15,
            "percentage": 30.0,
            "avg_trips": 85.3,
            "avg_duration": 720.1,
            "city_distribution": {"San Francisco": 5, "New York": 10},
            "feature_patterns": {
                "total_trips_norm": 0.7,
                "rush_hour_trips_norm": 0.5,
                "avg_duration_norm": 1.2,
                "weekend_ratio_norm": 1.5,
                "balance_ratio_norm": 0.8,
                "trip_balance_norm": 1.3
            }
        },
        "cluster_2": {
            "size": 15,
            "percentage": 30.0,
            "avg_trips": 150.8,
            "avg_duration": 950.5,
            "city_distribution": {"San Francisco": 8, "New York": 7},
            "feature_patterns": {
                "total_trips_norm": 1.8,
                "rush_hour_trips_norm": 1.9,
                "avg_duration_norm": 0.5,
                "weekend_ratio_norm": 0.4,
                "balance_ratio_norm": 1.5,
                "trip_balance_norm": 0.4
            }
        }
    }

def create_sample_predictions(n_stations=50):
    """Crea datos de predicciones de muestra."""
    today = datetime.now().date()
    data = []
    
    # Usar los datos de demanda por hora del HTML proporcionado
    hourly_pattern = [
        1.16, 1.04, 0.95, 0.62, 0.66, 0.70, 1.06, 1.09, 
        1.29, 1.72, 2.13, 2.57, 2.94, 3.20, 3.46, 3.40, 
        3.63, 3.81, 3.51, 3.07, 2.49, 2.01, 1.76, 1.43
    ]
    
    for station_id in range(1, n_stations+1):
        # Factor de escala para variar la demanda entre estaciones
        scale_factor = np.random.uniform(0.5, 3.0)
        
        # Factor de cluster para agregar consistencia
        cluster = station_id % 3
        cluster_factor = [1.0, 1.2, 0.8][cluster]
        
        for hour in range(24):
            # Calcular demanda base con patrón horario
            base_demand = hourly_pattern[hour] * scale_factor * cluster_factor
            
            # Añadir ruido
            predicted_demand = max(0.5, base_demand * np.random.normal(1, 0.1))
            
            # Determinar si es fin de semana
            is_weekend = 1 if today.weekday() >= 5 else 0
            
            data.append({
                "station_id": f"station_{station_id}",
                "prediction_date": today.strftime("%Y-%m-%d"),
                "prediction_hour": hour,
                "predicted_demand": round(predicted_demand, 2),
                "is_weekend": is_weekend,
                "cluster": cluster
            })
    
    return pd.DataFrame(data)

# Funciones de visualización
def create_station_map(stations_df, prediction_data=None):
    """Crea mapa interactivo de estaciones con clusters y hexágonos H3."""
    if stations_df is None or len(stations_df) == 0:
        st.warning("No hay datos de estaciones disponibles para el mapa.")
        return None
    
    # Crear mapa centrado en los datos
    center = [stations_df['latitude'].mean(), stations_df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron", 
                  scrollWheelZoom=True, zoom_control=False)  # Desactivar controles de zoom
    
    # Colores para los clusters
    colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#FF5722', '#607D8B']
    
    # Crear grupos de clusters
    cluster_groups = {}
    
    # Añadir hexágonos H3 si están disponibles
    if 'h3_geometry' in stations_df.columns and stations_df['h3_geometry'].notna().any():
        h3_group = folium.FeatureGroup(name="H3 Hexagons")
        
        for idx, row in stations_df.iterrows():
            if row['h3_geometry']:
                try:
                    h3_geom = json.loads(row['h3_geometry']) if isinstance(row['h3_geometry'], str) else row['h3_geometry']
                    
                    # Determinar cluster y color
                    cluster_id = int(row['cluster']) if 'cluster' in row and row['cluster'] is not None else -1
                    if cluster_id < 0:
                        color = '#9E9E9E'  # Gris para estaciones sin asignar
                    else:
                        color = colors[cluster_id % len(colors)]
                    
                    # Añadir hexágono
                    folium.GeoJson(
                        h3_geom,
                        style_function=lambda x, color=color: {
                            'fillColor': color,
                            'color': 'white',
                            'weight': 1,
                            'fillOpacity': 0.3
                        }
                    ).add_to(h3_group)
                except Exception as e:
                    pass
        
        h3_group.add_to(m)
    
    # Añadir marcadores de estaciones
    for idx, row in stations_df.iterrows():
        # Determinar cluster y color
        cluster_id = int(row['cluster']) if 'cluster' in row and pd.notna(row['cluster']) else -1
        if cluster_id < 0:
            color = '#9E9E9E'  # Gris para estaciones sin asignar
        else:
            color = colors[cluster_id % len(colors)]
        
        # Crear o acceder al grupo del cluster
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = folium.FeatureGroup(name=f"Cluster {cluster_id if cluster_id >= 0 else 'Unassigned'}")
            m.add_child(cluster_groups[cluster_id])
        
        # Calcular demanda si los datos de predicción están disponibles
        demand_info = ""
        radius = 5
        
        if prediction_data is not None:
            # Si el prediction_data tiene station_id como cadena, convertir para comparación
            if isinstance(row['station_id'], str) and isinstance(prediction_data['station_id'].iloc[0], str):
                station_predictions = prediction_data[prediction_data['station_id'] == row['station_id']]
            else:
                # Intentar conversión si los tipos no coinciden
                try:
                    station_predictions = prediction_data[prediction_data['station_id'].astype(str) == str(row['station_id'])]
                except:
                    station_predictions = pd.DataFrame()
            
            if len(station_predictions) > 0:
                avg_demand = station_predictions['predicted_demand'].mean()
                max_demand = station_predictions['predicted_demand'].max()
                
                demand_info = f"""
                <br><b>Predicted Demand:</b>
                <br>Average: {avg_demand:.1f} trips
                <br>Maximum: {max_demand:.1f} trips
                """
                
                # Ajustar radio basado en la demanda
                radius = 5 + min(15, avg_demand / 2)
        
        # Crear popup con información
        name = row['name'] if 'name' in row else f"Station {row['station_id']}"
        city = row['city'] if 'city' in row else "Unknown"
        capacity = row['capacity'] if 'capacity' in row else "N/A"
        
        popup_text = f"""
        <div style="font-family: Arial; max-width: 200px;">
            <h3 style="color: {color};">{name}</h3>
            <b>ID:</b> {row['station_id']}<br>
            <b>City:</b> {city}<br>
            <b>Capacity:</b> {capacity}<br>
            <b>Cluster:</b> {cluster_id if cluster_id >= 0 else 'Unassigned'}<br>
            {demand_info}
        </div>
        """
        
        # Añadir marcador
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(cluster_groups[cluster_id])
    
    # Añadir capa de heatmap si hay datos de predicción disponibles
    if prediction_data is not None:
        # Agregar predicciones por estación
        pred_by_station = prediction_data.groupby('station_id')['predicted_demand'].mean().reset_index()
        
        # Unir con datos de estaciones
        if isinstance(pred_by_station['station_id'].iloc[0], str) and isinstance(stations_df['station_id'].iloc[0], str):
            heat_data = pd.merge(stations_df, pred_by_station, on='station_id', how='inner')
        else:
            # Intentar conversión si los tipos no coinciden
            try:
                pred_by_station['station_id'] = pred_by_station['station_id'].astype(str)
                heat_data = pd.merge(stations_df, pred_by_station, left_on='station_id', right_on='station_id', how='inner')
            except:
                heat_data = pd.DataFrame()
        
        if len(heat_data) > 0:
            # Preparar datos para heatmap
            heat_points = [[row['latitude'], row['longitude'], row['predicted_demand']] 
                        for idx, row in heat_data.iterrows()]
            
            # Añadir heatmap
            heat_group = folium.FeatureGroup(name="Demand Heatmap")
            HeatMap(
                heat_points,
                radius=15, 
                gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1: 'red'},
                min_opacity=0.5,
                blur=10
            ).add_to(heat_group)
            
            heat_group.add_to(m)
    
    # NO añadir control de capas para mantener el mapa limpio
    # folium.LayerControl(collapsed=False).add_to(m)
    
    return m

def create_hourly_demand_chart(prediction_data):
    """Crea gráfico de demanda por hora."""
    if prediction_data is None or len(prediction_data) == 0:
        return None
    
    # Agregar por hora
    hourly_demand = prediction_data.groupby('prediction_hour')['predicted_demand'].mean().reset_index()
    
    # Asegurar orden correcto
    hourly_demand = hourly_demand.sort_values('prediction_hour')
    
    # Crear gráfico
    fig = px.line(
        hourly_demand,
        x='prediction_hour',
        y='predicted_demand',
        labels={
            'prediction_hour': 'Hour of day', 
            'predicted_demand': 'Average demand'
        },
        title='Demand by hour of day',
        markers=True
    )
    
    # Mejorar diseño
    fig.update_traces(
        line=dict(color='#1E88E5', width=3),
        marker=dict(size=8, color='#1E88E5')
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=3,
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h}:00" for h in range(0, 24, 3)]
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_cluster_distribution_chart(cluster_data):
    """Crea gráfico de distribución de estaciones por cluster."""
    if not cluster_data:
        return None
    
    # Extraer datos
    cluster_sizes = []
    
    for cluster_id, data in cluster_data.items():
        cluster_num = cluster_id.split('_')[1] if '_' in cluster_id else cluster_id
        cluster_sizes.append({
            'Cluster': f"Cluster {cluster_num}",
            'Stations': data.get('size', 0),
            'Percentage': data.get('percentage', 0)
        })
    
    # Convertir a DataFrame
    df = pd.DataFrame(cluster_sizes)
    
    # Crear gráfico
    fig = px.bar(
        df,
        x='Cluster',
        y='Stations',
        text='Percentage',
        color='Stations',
        color_continuous_scale='Viridis',
        labels={
            'Stations': 'Number of stations'
        },
        title='Distribution by cluster'
    )
    
    # Mejorar formato
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_weekday_heatmap(prediction_data):
    """Crea heatmap de demanda por día de la semana y hora."""
    if prediction_data is None or len(prediction_data) == 0:
        return None
    
    # Si no hay columna 'day_of_week', intentar añadirla
    if 'day_of_week' not in prediction_data.columns and 'prediction_date' in prediction_data.columns:
        try:
            prediction_data['day_of_week'] = pd.to_datetime(prediction_data['prediction_date']).dt.dayofweek
        except:
            # Si falla, usar simplemente 'is_weekend'
            if 'is_weekend' in prediction_data.columns:
                prediction_data['day_of_week'] = prediction_data['is_weekend'].apply(lambda x: 5 if x == 1 else 1)  # 5=Sábado, 1=Martes
    
    # Si aún no hay 'day_of_week', retornar None
    if 'day_of_week' not in prediction_data.columns:
        return None
    
    # Crear tabla pivote
    heatmap_data = prediction_data.pivot_table(
        index='day_of_week',
        columns='prediction_hour',
        values='predicted_demand',
        aggfunc='mean'
    ).fillna(0)
    
    # Nombres de los días
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Crear heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x='Hour of day', y='Day of week', color='Demand'),
        x=[f"{h}" for h in range(24)],
        y=[day_names[d] if d < len(day_names) else f"Day {d}" for d in heatmap_data.index],
        color_continuous_scale='Viridis',
        title='Weekly demand pattern'
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h}:00" for h in range(0, 24, 3)]
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_stations_table(prediction_data, stations_data, top_n=5, show_highest=True):
    """Crea tabla de estaciones con mayor/menor demanda predicha."""
    if prediction_data is None or len(prediction_data) == 0:
        return None
    
    # Agregar por estación
    station_demand = prediction_data.groupby('station_id').agg({
        'predicted_demand': ['mean', 'max'],
        'cluster': 'first'
    }).reset_index()
    
    # Aplanar columnas multi-índice
    station_demand.columns = ['station_id', 'avg_demand', 'max_demand', 'cluster']
    
    # Ordenar por demanda promedio
    if show_highest:
        station_demand = station_demand.sort_values('avg_demand', ascending=False).head(top_n)
    else:
        station_demand = station_demand.sort_values('avg_demand').head(top_n)
    
    # Fusionar con datos de estaciones si es posible
    if stations_data is not None and 'name' in stations_data.columns:
        # Convertir tipos para asegurar compatibilidad
        stations_data['station_id'] = stations_data['station_id'].astype(str)
        station_demand['station_id'] = station_demand['station_id'].astype(str)
        
        try:
            station_demand = pd.merge(
                station_demand,
                stations_data[['station_id', 'name']],
                on='station_id',
                how='left'
            )
        except:
            station_demand['name'] = station_demand['station_id']
    else:
        station_demand['name'] = station_demand['station_id']
    
    # Renombrar columnas para mejor visualización
    station_demand = station_demand.rename(columns={
        'station_id': 'ID',
        'name': 'Name',
        'avg_demand': 'Average Demand',
        'max_demand': 'Maximum Demand',
        'cluster': 'Cluster'
    })
    
    # Formatear valores numéricos
    station_demand['Average Demand'] = station_demand['Average Demand'].round(1)
    station_demand['Maximum Demand'] = station_demand['Maximum Demand'].round(1)
    
    # Reordenar columnas para mejor visualización
    cols = ['ID', 'Name', 'Cluster', 'Average Demand', 'Maximum Demand']
    station_demand = station_demand[[c for c in cols if c in station_demand.columns]]
    
    return station_demand

# Aplicación principal
# Aplicación principal
def main():
    # Cargar datos
    with st.spinner("Loading system data..."):
        stations_data = load_station_data()
        cluster_data = load_cluster_data()
        prediction_data = load_prediction_data()
    
    # Título principal
    st.markdown('<div class="main-header">🚲 Bike-Sharing Dashboard</div>', unsafe_allow_html=True)
    
    # Estructura principal: 3 columnas (métrica, mapa, métrica)
    left_col, map_col, right_col = st.columns([1, 2, 1])
    
    # Columna izquierda: Métricas principales y distribución por cluster
    with left_col:
        # Número de estaciones
        stations_count = len(stations_data) if stations_data is not None else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stations_count}</div>
                <div class="metric-label">Stations</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Número de clusters
        clusters_count = len(cluster_data) if cluster_data else 0
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{clusters_count}</div>
                <div class="metric-label">Clusters</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Distribución por cluster
        st.markdown("<h3>Distribution by cluster</h3>", unsafe_allow_html=True)
        cluster_chart = create_cluster_distribution_chart(cluster_data)
        if cluster_chart:
            st.plotly_chart(cluster_chart, use_container_width=True, config={'displayModeBar': False})
        
        # Tabla de estaciones con mayor demanda
        st.markdown("<h3>Stations with highest demand</h3>", unsafe_allow_html=True)
        top_stations = create_stations_table(prediction_data, stations_data, top_n=5, show_highest=True)
        if top_stations is not None:
            st.dataframe(top_stations, hide_index=True, height=180)
        
    # Columna central: Mapa interactivo
    with map_col:
        st.markdown("<h3>Stations and predictions map</h3>", unsafe_allow_html=True)
        
        # Creamos una fila con dos columnas para los filtros
        filter_cols = st.columns(2)
        with filter_cols[0]:
            # Filtro de cluster si hay datos de clustering
            if stations_data is not None and 'cluster' in stations_data.columns:
                unique_clusters = sorted(stations_data['cluster'].unique())
                cluster_options = ['All'] + [str(c) for c in unique_clusters if pd.notna(c)]
                selected_cluster = st.selectbox("Cluster:", cluster_options)
            else:
                selected_cluster = "All"
                
        with filter_cols[1]:
            # Filtro de ciudad si hay múltiples ciudades
            if stations_data is not None and 'city' in stations_data.columns:
                cities = ['All'] + sorted(stations_data['city'].unique())
                selected_city = st.selectbox("City:", cities)
            else:
                selected_city = "All"
        
        # Aplicar filtros a los datos
        filtered_stations = stations_data
        
        if selected_city != "All" and 'city' in stations_data.columns:
            filtered_stations = filtered_stations[filtered_stations['city'] == selected_city]
        
        if selected_cluster != "All" and 'cluster' in stations_data.columns:
            try:
                cluster_val = int(selected_cluster)
                filtered_stations = filtered_stations[filtered_stations['cluster'] == cluster_val]
            except:
                pass
        
        # Mostrar mapa con datos filtrados
        map_container = st.container()
        with map_container:
            if filtered_stations is not None and len(filtered_stations) > 0:
                m = create_station_map(filtered_stations, prediction_data)
                if m:
                    folium_static(m, width=600, height=450)
            else:
                st.warning("No stations available with selected filters.")
    
    # Columna derecha: Gráficos de predicción y demanda
    with right_col:
        # Demanda total predicha para hoy
        if prediction_data is not None and len(prediction_data) > 0:
            today = datetime.now().strftime('%Y-%m-%d')
            todays_pred = prediction_data[prediction_data['prediction_date'] == today]
            total_demand = todays_pred['predicted_demand'].sum() if len(todays_pred) > 0 else 0
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{int(total_demand):,}</div>
                    <div class="metric-label">Total demand today</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de demanda por hora
        st.markdown("<h3>Demand by hour</h3>", unsafe_allow_html=True)
        hourly_chart = create_hourly_demand_chart(prediction_data)
        if hourly_chart:
            st.plotly_chart(hourly_chart, use_container_width=True, config={'displayModeBar': False})
        
        # Heatmap por día de la semana
        st.markdown("<h3>Weekly pattern</h3>", unsafe_allow_html=True)
        heatmap = create_weekday_heatmap(prediction_data)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True, config={'displayModeBar': False})
        
        # Tabla de estaciones con menor demanda
        st.markdown("<h3>Stations with lowest demand</h3>", unsafe_allow_html=True)
        bottom_stations = create_stations_table(prediction_data, stations_data, top_n=5, show_highest=False)
        if bottom_stations is not None:
            st.dataframe(bottom_stations, hide_index=True, height=180)
    
    # Fila inferior: Resumen del modelo
    st.markdown("<hr>", unsafe_allow_html=True)
    model_cols = st.columns(4)
    
    with model_cols[0]:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">84,057</div>
                <div class="metric-label">Historical records</div>
            </div>
        """, unsafe_allow_html=True)
    
    with model_cols[1]:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">3</div>
                <div class="metric-label">Clusters identified</div>
            </div>
        """, unsafe_allow_html=True)
    
    with model_cols[2]:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">0.60</div>
                <div class="metric-label">Model R²</div>
            </div>
        """, unsafe_allow_html=True)
    
    with model_cols[3]:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">2.29</div>
                <div class="metric-label">RMSE</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Nota de pie
    st.markdown("""
        <div style="text-align: center; margin-top: 20px; font-size: 0.8rem; color: #666;">
            Dashboard created with Streamlit | Data processed with PostgreSQL and H3 | Last update: {0}
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()