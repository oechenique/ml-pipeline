import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
from pathlib import Path

# Configuración de página
st.set_page_config(
    page_title="Bike Sharing ML Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #f0f9ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 10px;
    }
    .footer {
        margin-top: 2rem;
        text-align: center;
        color: #888;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Funciones de conexión a la base de datos
def get_db_connection():
    """Establece conexión con la base de datos PostgreSQL."""
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
        st.error(f"Error de conexión a la base de datos: {str(e)}")
        return None

# Funciones para cargar datos
@st.cache_data(ttl=3600)
def load_station_data():
    """Carga datos de estaciones desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo si no hay conexión
            return pd.read_csv("data/examples/stations_sample.csv")
        
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
            ST_AsGeoJSON(s.geom) as geometry
        FROM bike_stations s
        LEFT JOIN station_clusters c ON s.station_id = c.station_id
        WHERE s.latitude IS NOT NULL AND s.longitude IS NOT NULL
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convertir a GeoDataFrame
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: json.loads(x) if x else None)
            gdf = gpd.GeoDataFrame.from_features(
                [{"type": "Feature", "geometry": g, "properties": {}} for g in df['geometry']],
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
        # Cargar datos de respaldo
        return pd.read_csv("data/examples/stations_sample.csv")

@st.cache_data(ttl=3600)
def load_trip_data(limit=10000):
    """Carga datos de viajes desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo si no hay conexión
            return pd.read_csv("data/examples/trips_sample.csv")
        
        query = f"""
        SELECT 
            trip_id, 
            start_time, 
            end_time,
            duration_sec,
            start_station_id,
            end_station_id,
            user_type,
            city
        FROM bike_trips
        ORDER BY start_time DESC
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de viajes: {str(e)}")
        # Cargar datos de respaldo
        return pd.read_csv("data/examples/trips_sample.csv")

@st.cache_data(ttl=3600)
def load_cluster_data():
    """Carga datos de clustering desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo si no hay conexión
            return None
        
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
        
        if row and row[0]:
            # JSONB viene como diccionario en Python
            cluster_data = row[0]
        else:
            cluster_data = {}
            
        cursor.close()
        conn.close()
        return cluster_data
    except Exception as e:
        st.error(f"Error al cargar datos de clustering: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_prediction_data():
    """Carga datos de predicciones desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo si no hay conexión
            return None
        
        query = """
        SELECT 
            station_id,
            prediction_date,
            prediction_hour,
            predicted_demand,
            is_weekend,
            cluster
        FROM bike_demand_predictions
        WHERE prediction_date >= CURRENT_DATE
        ORDER BY prediction_date, prediction_hour
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error al cargar datos de predicciones: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_time_series_data():
    """Carga datos de series temporales desde PostgreSQL."""
    try:
        conn = get_db_connection()
        if conn is None:
            # Cargar datos de respaldo si no hay conexión
            return None
        
        # Pronósticos de series temporales
        forecast_query = """
        SELECT 
            forecast_date,
            forecast_value,
            lower_bound,
            upper_bound,
            model_type
        FROM bike_demand_forecasts
        ORDER BY forecast_date, model_type
        """
        
        forecasts = pd.read_sql(forecast_query, conn)
        
        # Datos históricos agregados por día para comparación
        historic_query = """
        SELECT 
            date_trunc('day', start_time) as day,
            COUNT(*) as trip_count
        FROM bike_trips
        GROUP BY date_trunc('day', start_time)
        ORDER BY date_trunc('day', start_time)
        """
        
        historic = pd.read_sql(historic_query, conn)
        conn.close()
        
        return {
            'forecasts': forecasts,
            'historic': historic
        }
    except Exception as e:
        st.error(f"Error al cargar datos de series temporales: {str(e)}")
        return None

# Función para cargar datos de archivos locales (respaldo)
def load_local_data(file_path):
    """Carga datos de archivos locales como respaldo."""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.geojson'):
            return gpd.read_file(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            st.warning(f"Formato de archivo no soportado: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error al cargar archivo local {file_path}: {str(e)}")
        return None
        
# Funciones para visualizaciones
def create_station_map(stations_gdf):
    """Crea un mapa interactivo de estaciones con clusters."""
    
    if stations_gdf is None or len(stations_gdf) == 0:
        st.warning("No hay datos de estaciones disponibles para el mapa.")
        return
    
    # Crear mapa centrado en los datos
    center = [stations_gdf['latitude'].mean(), stations_gdf['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    
    # Definir colores para los clusters
    colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#FF5722', '#607D8B']
    
    # Crear grupos para cada cluster
    cluster_groups = {}
    
    for idx, row in stations_gdf.iterrows():
        # Determinar cluster y color
        cluster_id = int(row['cluster']) if 'cluster' in row and row['cluster'] is not None else -1
        if cluster_id < 0:
            color = '#9E9E9E'  # Gris para estaciones sin cluster
        else:
            color = colors[cluster_id % len(colors)]
        
        # Crear o acceder al grupo del cluster
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = folium.FeatureGroup(name=f"Cluster {cluster_id if cluster_id >= 0 else 'No asignado'}")
            m.add_child(cluster_groups[cluster_id])
        
        # Crear popup con información
        popup_text = f"""
        <b>{row['name']}</b><br>
        ID: {row['station_id']}<br>
        Ciudad: {row['city']}<br>
        Capacidad: {row['capacity']}<br>
        Cluster: {cluster_id if cluster_id >= 0 else 'No asignado'}<br>
        """
        
        # Añadir marcador
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(cluster_groups[cluster_id])
    
    # Añadir control de capas
    folium.LayerControl().add_to(m)
    
    # Mostrar mapa
    return m

def create_heatmap(stations_df, predictions_df=None):
    """Crea un mapa de calor de actividad o predicciones."""
    
    if stations_df is None or len(stations_df) == 0:
        st.warning("No hay datos disponibles para el mapa de calor.")
        return
    
    # Crear mapa base
    center = [stations_df['latitude'].mean(), stations_df['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter")
    
    # Si hay predicciones, usarlas para el mapa de calor
    if predictions_df is not None and len(predictions_df) > 0:
        # Agregar predicciones por estación
        pred_by_station = predictions_df.groupby('station_id')['predicted_demand'].mean().reset_index()
        
        # Unir con datos de estaciones
        heat_data = pd.merge(
            stations_df,
            pred_by_station,
            on='station_id',
            how='inner'
        )
        
        # Preparar datos para el mapa de calor
        heat_points = [[row['latitude'], row['longitude'], row['predicted_demand']] 
                     for idx, row in heat_data.iterrows()]
        
        # Añadir mapa de calor
        HeatMap(
            heat_points,
            radius=15, 
            gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1: 'red'},
            min_opacity=0.5,
            blur=10
        ).add_to(m)
    else:
        # Si no hay predicciones, solo mostrar densidad de estaciones
        heat_points = [[row['latitude'], row['longitude'], 1] 
                     for idx, row in stations_df.iterrows()]
        
        # Añadir mapa de calor
        HeatMap(
            heat_points,
            radius=15, 
            gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'yellow', 1: 'red'},
            min_opacity=0.5,
            blur=10
        ).add_to(m)
    
    return m

def create_cluster_analysis_chart(cluster_data):
    """Crea visualizaciones para análisis de clusters."""
    
    if not cluster_data:
        st.warning("No hay datos de clustering disponibles.")
        return
    
    # Extraer datos de clusters
    cluster_sizes = []
    feature_patterns = []
    
    for cluster_id, data in cluster_data.items():
        cluster_sizes.append({
            'cluster': cluster_id,
            'size': data.get('size', 0),
            'percentage': data.get('percentage', 0)
        })
        
        # Extraer patterns para radar chart
        if 'feature_patterns' in data:
            patterns = data['feature_patterns']
            patterns['cluster'] = cluster_id
            feature_patterns.append(patterns)
    
    # Convertir a DataFrames
    df_sizes = pd.DataFrame(cluster_sizes)
    df_patterns = pd.DataFrame(feature_patterns)
    
    # Crear gráfico de barras para tamaños de cluster
    fig_sizes = px.bar(
        df_sizes,
        x='cluster',
        y='size',
        text='percentage',
        labels={'cluster': 'Cluster', 'size': 'Número de estaciones', 'percentage': 'Porcentaje'},
        title='Distribución de estaciones por cluster',
        color='size',
        color_continuous_scale='Viridis'
    )
    
    fig_sizes.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    # Crear gráfico de radar para patrones de feature
    if len(df_patterns) > 0:
        # Seleccionar features para radar (excluyendo 'cluster')
        feature_cols = [col for col in df_patterns.columns if col != 'cluster']
        
        # Crear figura con subplots para cada cluster
        fig_patterns = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'polar'}]]
        )
        
        # Colores para cada cluster
        colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#FF5722', '#607D8B']
        
        # Añadir datos de cada cluster
        for i, row in df_patterns.iterrows():
            cluster_id = row['cluster']
            color_idx = int(cluster_id.split('_')[1]) % len(colors)
            
            fig_patterns.add_trace(
                go.Scatterpolar(
                    r=[row[col] for col in feature_cols],
                    theta=feature_cols,
                    fill='toself',
                    name=cluster_id,
                    line_color=colors[color_idx],
                    opacity=0.7
                )
            )
        
        fig_patterns.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df_patterns[feature_cols].max().max() * 1.1]
                )
            ),
            title='Patrones de característica por cluster',
            showlegend=True
        )
        
        return fig_sizes, fig_patterns
    
    return fig_sizes, None

def create_demand_prediction_charts(predictions_df):
    """Crea visualizaciones para predicciones de demanda."""
    
    if predictions_df is None or len(predictions_df) == 0:
        st.warning("No hay datos de predicciones disponibles.")
        return None, None
    
    # Agregación por hora
    hourly_demand = predictions_df.groupby('prediction_hour')['predicted_demand'].mean().reset_index()
    
    # Agregación por día y hora (para heatmap)
    # Añadir día de la semana
    predictions_df['day_of_week'] = pd.to_datetime(predictions_df['prediction_date']).dt.dayofweek
    
    # Crear pivot table
    heatmap_data = predictions_df.pivot_table(
        index='day_of_week',
        columns='prediction_hour',
        values='predicted_demand',
        aggfunc='mean'
    ).fillna(0)
    
    # Gráfico de línea para demanda por hora
    fig_hourly = px.line(
        hourly_demand,
        x='prediction_hour',
        y='predicted_demand',
        labels={'prediction_hour': 'Hora del día', 'predicted_demand': 'Demanda predicha promedio'},
        title='Predicción de demanda por hora del día',
        markers=True
    )
    
    fig_hourly.update_traces(line=dict(color='#1E88E5', width=3))
    fig_hourly.update_layout(
        xaxis=dict(tickmode='linear', dtick=2),
        yaxis=dict(title='Demanda predicha (viajes)')
    )
    
    # Heatmap para demanda por día y hora
    day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x='Hora del día', y='Día de la semana', color='Demanda predicha'),
        x=[str(h) for h in range(24)],
        y=day_names,
        color_continuous_scale='Viridis',
        title='Patrón semanal de demanda predicha'
    )
    
    fig_heatmap.update_layout(
        xaxis=dict(tickmode='linear', dtick=2)
    )
    
    return fig_hourly, fig_heatmap

def create_time_series_charts(ts_data):
    """Crea visualizaciones para series temporales."""
    
    if ts_data is None:
        st.warning("No hay datos de series temporales disponibles.")
        return None, None
    
    forecasts = ts_data.get('forecasts')
    historic = ts_data.get('historic')
    
    if forecasts is None or len(forecasts) == 0:
        st.warning("No hay pronósticos disponibles.")
        return None, None
    
    # Filtrar y preparar datos
    prophet_data = forecasts[forecasts['model_type'] == 'prophet']
    xgboost_data = forecasts[forecasts['model_type'] == 'xgboost']
    
    # Gráfico de pronósticos
    fig_forecast = go.Figure()
    
    # Añadir datos históricos si están disponibles
    if historic is not None and len(historic) > 0:
        historic['day'] = pd.to_datetime(historic['day'])
        fig_forecast.add_trace(
            go.Scatter(
                x=historic['day'],
                y=historic['trip_count'],
                mode='lines',
                name='Histórico',
                line=dict(color='#888888', width=2)
            )
        )
    
    # Añadir pronóstico de Prophet
    if len(prophet_data) > 0:
        prophet_data['forecast_date'] = pd.to_datetime(prophet_data['forecast_date'])
        fig_forecast.add_trace(
            go.Scatter(
                x=prophet_data['forecast_date'],
                y=prophet_data['forecast_value'],
                mode='lines',
                name='Prophet',
                line=dict(color='#1E88E5', width=3)
            )
        )
        
        # Añadir intervalo de confianza
        fig_forecast.add_trace(
            go.Scatter(
                x=pd.concat([prophet_data['forecast_date'], prophet_data['forecast_date'].iloc[::-1]]),
                y=pd.concat([prophet_data['upper_bound'], prophet_data['lower_bound'].iloc[::-1]]),
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Prophet IC'
            )
        )
    
    # Añadir pronóstico de XGBoost
    if len(xgboost_data) > 0:
        xgboost_data['forecast_date'] = pd.to_datetime(xgboost_data['forecast_date'])
        fig_forecast.add_trace(
            go.Scatter(
                x=xgboost_data['forecast_date'],
                y=xgboost_data['forecast_value'],
                mode='lines',
                name='XGBoost',
                line=dict(color='#4CAF50', width=3, dash='dash')
            )
        )
    
    fig_forecast.update_layout(
        title='Pronóstico de demanda de viajes',
        xaxis_title='Fecha',
        yaxis_title='Demanda de viajes',
        legend_title='Modelo',
        hovermode='x unified'
    )
    
    # Gráfico de comparación de modelos
    # Calcular errores promedio (si hay datos suficientes)
    model_metrics = None
    if 'model_metrics' in ts_data:
        model_metrics = ts_data['model_metrics']
    else:
        # Intentar calcular métricas básicas
        if historic is not None and len(historic) > 0 and len(prophet_data) > 0 and len(xgboost_data) > 0:
            # Buscar fechas que se solapen para comparación
            comparison_dates = []
            for date in prophet_data['forecast_date']:
                if date in historic['day'].values and date in xgboost_data['forecast_date'].values:
                    comparison_dates.append(date)
            
            if len(comparison_dates) > 0:
                metrics = {
                    'Prophet': {
                        'RMSE': 0,
                        'MAE': 0,
                        'MAPE': 0
                    },
                    'XGBoost': {
                        'RMSE': 0,
                        'MAE': 0,
                        'MAPE': 0
                    }
                }
                model_metrics = pd.DataFrame(metrics)
    
    # Si hay métricas disponibles, crear gráfico de comparación
    if model_metrics is not None:
        fig_comparison = px.bar(
            model_metrics,
            barmode='group',
            title='Comparación de modelos de pronóstico',
            labels={'value': 'Valor', 'variable': 'Métrica'}
        )
    else:
        # Simplemente comparar valores pronosticados por cada modelo
        comparison_df = pd.DataFrame({
            'Fecha': prophet_data['forecast_date'],
            'Prophet': prophet_data['forecast_value'],
            'XGBoost': xgboost_data['forecast_value'].values if len(xgboost_data) == len(prophet_data) else np.nan
        })
        
        comparison_df = comparison_df.melt(
            id_vars=['Fecha'],
            value_vars=['Prophet', 'XGBoost'],
            var_name='Modelo',
            value_name='Pronóstico'
        )
        
        fig_comparison = px.box(
            comparison_df,
            x='Modelo',
            y='Pronóstico',
            color='Modelo',
            title='Distribución de pronósticos por modelo',
            color_discrete_map={'Prophet': '#1E88E5', 'XGBoost': '#4CAF50'}
        )
    
    return fig_forecast, fig_comparison
    
# Aplicación principal
def main():
    # Título principal
    st.markdown('<div class="main-header">🚲 Dashboard de Análisis de Bike-Sharing</div>', unsafe_allow_html=True)
    
    # Barra lateral para navegación y filtros
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Ir a:",
        ["Resumen general", "Mapa de estaciones", "Análisis de clusters", 
         "Predicciones de demanda", "Series temporales"]
    )
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        stations_data = load_station_data()
        trip_data = load_trip_data()
        cluster_data = load_cluster_data()
        prediction_data = load_prediction_data()
        time_series_data = load_time_series_data()
    
    # Información básica
    trip_count = len(trip_data) if trip_data is not None else 0
    station_count = len(stations_data) if stations_data is not None else 0
    
    # Fecha de última actualización
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.sidebar.write(f"Última actualización: {last_update}")
    
    # Panel de Resumen General
    if page == "Resumen general":
        st.markdown('<div class="sub-header">📊 Resumen General del Sistema</div>', unsafe_allow_html=True)
        
        # Métricas clave en 4 columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{station_count}</div>
                    <div class="metric-label">Estaciones</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{trip_count:,}</div>
                    <div class="metric-label">Viajes analizados</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Calcular clusters si hay datos disponibles
            cluster_count = len(cluster_data) if cluster_data else 0
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{cluster_count}</div>
                    <div class="metric-label">Clusters identificados</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calcular ciudades únicas
            if stations_data is not None and 'city' in stations_data.columns:
                city_count = stations_data['city'].nunique()
            else:
                city_count = 0
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{city_count}</div>
                    <div class="metric-label">Ciudades analizadas</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Mapa general
        st.markdown('<div class="sub-header">🌐 Distribución de Estaciones</div>', unsafe_allow_html=True)
        
        if stations_data is not None and len(stations_data) > 0:
            map_col1, map_col2 = st.columns([3, 1])
            
            with map_col1:
                m = create_station_map(stations_data)
                if m:
                    folium_static(m, width=700, height=500)
            
            with map_col2:
                st.markdown('<div class="highlight">El mapa muestra las estaciones agrupadas por clusters. Los colores indican los diferentes grupos identificados por el algoritmo de clustering.</div>', unsafe_allow_html=True)
                
                # Mostrar información de viajes recientes
                if trip_data is not None and len(trip_data) > 0:
                    st.markdown("### Viajes recientes")
                    # Mostrar los 5 viajes más recientes
                    recent_trips = trip_data.head(5)
                    
                    if 'start_time' in recent_trips.columns:
                        recent_trips['start_time'] = pd.to_datetime(recent_trips['start_time']).dt.strftime('%Y-%m-%d %H:%M')
                    
                    if 'duration_sec' in recent_trips.columns:
                        recent_trips['duration_min'] = (recent_trips['duration_sec'] / 60).round(1)
                    
                    # Mostrar viajes recientes en una tabla simple
                    st.dataframe(recent_trips[['start_time', 'start_station_id', 'end_station_id', 'duration_min']], height=200)
        
        # Resumen de predicciones
        st.markdown('<div class="sub-header">📈 Resumen de Predicciones</div>', unsafe_allow_html=True)
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            # Mostrar predicción de demanda
            if prediction_data is not None and len(prediction_data) > 0:
                # Calcular demanda total predicha para hoy
                today = datetime.now().strftime('%Y-%m-%d')
                todays_pred = prediction_data[prediction_data['prediction_date'] == today]
                total_demand = todays_pred['predicted_demand'].sum()
                
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{int(total_demand):,}</div>
                        <div class="metric-label">Demanda total predicha para hoy</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No hay datos de predicción disponibles.")
        
        with pred_col2:
            # Mostrar información de series temporales
            if time_series_data and 'forecasts' in time_series_data:
                forecasts = time_series_data['forecasts']
                # Calcular valor promedio pronosticado para el próximo mes
                if len(forecasts) > 0:
                    today = datetime.now()
                    next_month = today + timedelta(days=30)
                    
                    future_forecasts = forecasts[
                        (forecasts['forecast_date'] >= today.strftime('%Y-%m-%d')) & 
                        (forecasts['forecast_date'] <= next_month.strftime('%Y-%m-%d'))
                    ]
                    
                    avg_forecast = future_forecasts['forecast_value'].mean()
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{int(avg_forecast):,}</div>
                            <div class="metric-label">Demanda diaria promedio (próx. 30 días)</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No hay pronósticos disponibles.")
            else:
                st.info("No hay datos de series temporales disponibles.")
        
        # Mostrar distribución de viajes por cluster
        if cluster_data:
            st.markdown('<div class="sub-header">🔍 Distribución por Cluster</div>', unsafe_allow_html=True)
            
            fig_sizes, _ = create_cluster_analysis_chart(cluster_data)
            if fig_sizes:
                st.plotly_chart(fig_sizes, use_container_width=True)
        
        # Pie de página
        st.markdown("""
        <div class="footer">
            Dashboard creado con Streamlit | Datos procesados con PostgreSQL y H3
        </div>
        """, unsafe_allow_html=True)
        
# Panel de Mapa de Estaciones
    elif page == "Mapa de estaciones":
        st.markdown('<div class="sub-header">🗺️ Exploración Geoespacial de Estaciones</div>', unsafe_allow_html=True)
        
        # Filtros para el mapa
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Filtro de ciudad si hay múltiples ciudades
            if stations_data is not None and 'city' in stations_data.columns:
                cities = ['Todas'] + sorted(stations_data['city'].unique().tolist())
                selected_city = st.selectbox("Filtrar por ciudad:", cities)
            else:
                selected_city = "Todas"
        
        with filter_col2:
            # Filtro de cluster si hay datos de clustering
            if stations_data is not None and 'cluster' in stations_data.columns:
                clusters = ['Todos'] + sorted([str(c) for c in stations_data['cluster'].unique().tolist()])
                selected_cluster = st.selectbox("Filtrar por cluster:", clusters)
            else:
                selected_cluster = "Todos"
        
        # Aplicar filtros a los datos
        filtered_stations = stations_data
        
        if selected_city != "Todas" and 'city' in stations_data.columns:
            filtered_stations = filtered_stations[filtered_stations['city'] == selected_city]
        
        if selected_cluster != "Todos" and 'cluster' in stations_data.columns:
            try:
                cluster_val = int(selected_cluster)
                filtered_stations = filtered_stations[filtered_stations['cluster'] == cluster_val]
            except:
                pass
        
        # Mostrar mapa con datos filtrados
        if filtered_stations is not None and len(filtered_stations) > 0:
            st.write(f"Mostrando {len(filtered_stations)} estaciones")
            m = create_station_map(filtered_stations)
            if m:
                folium_static(m, width=1000, height=600)
        else:
            st.warning("No hay estaciones disponibles con los filtros seleccionados.")
        
        # Mostrar también un mapa de calor con predicciones si están disponibles
        if prediction_data is not None and len(prediction_data) > 0:
            st.markdown('<div class="sub-header">🔥 Mapa de Calor de Demanda Predicha</div>', unsafe_allow_html=True)
            m_heat = create_heatmap(filtered_stations, prediction_data)
            if m_heat:
                folium_static(m_heat, width=1000, height=600)
                st.info("El mapa de calor muestra la intensidad de demanda predicha para cada estación. Las áreas rojas indican mayor demanda.")
    
    # Panel de Análisis de Clusters
    elif page == "Análisis de clusters":
        st.markdown('<div class="sub-header">🧩 Análisis de Clusters de Estaciones</div>', unsafe_allow_html=True)
        
        if cluster_data:
            # Mostrar número de clusters y distribución
            cluster_count = len(cluster_data)
            st.write(f"Se han identificado {cluster_count} clusters distintos de estaciones.")
            
            # Crear visualizaciones
            fig_sizes, fig_patterns = create_cluster_analysis_chart(cluster_data)
            
            if fig_sizes:
                st.plotly_chart(fig_sizes, use_container_width=True)
            
            if fig_patterns:
                st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Mostrar detalles de cada cluster
            st.markdown('<div class="sub-header">📊 Características de los Clusters</div>', unsafe_allow_html=True)
            
            for cluster_id, data in cluster_data.items():
                with st.expander(f"Cluster {cluster_id}"):
                    # Información general
                    st.write(f"**Tamaño**: {data.get('size', 0)} estaciones ({data.get('percentage', 0):.1f}%)")
                    
                    # Distribución por ciudad si está disponible
                    if 'city_distribution' in data:
                        st.write("**Distribución por ciudad**:")
                        city_df = pd.DataFrame([
                            {"Ciudad": city, "Estaciones": count}
                            for city, count in data['city_distribution'].items()
                        ])
                        st.dataframe(city_df, hide_index=True)
                    
                    # Patrones de características
                    if 'feature_patterns' in data:
                        st.write("**Patrones de características**:")
                        patterns_df = pd.DataFrame([{
                            "Característica": key.replace('_', ' ').title(),
                            "Valor promedio": value
                        } for key, value in data['feature_patterns'].items()])
                        st.dataframe(patterns_df, hide_index=True)
                    
                    # Lista de estaciones en este cluster
                    if stations_data is not None and 'cluster' in stations_data.columns:
                        cluster_stations = stations_data[stations_data['cluster'] == int(cluster_id.split('_')[1])]
                        if len(cluster_stations) > 0:
                            st.write(f"**Estaciones en este cluster**: {len(cluster_stations)}")
                            st.dataframe(
                                cluster_stations[['station_id', 'name', 'city', 'capacity']].head(10),
                                hide_index=True
                            )
                            if len(cluster_stations) > 10:
                                st.write(f"... y {len(cluster_stations) - 10} más.")
            
            # Mapa de estaciones por cluster
            st.markdown('<div class="sub-header">🗺️ Mapa de Clusters</div>', unsafe_allow_html=True)
            
            # Filtrar por un cluster específico para el mapa
            cluster_options = ['Todos'] + [str(c_id.split('_')[1]) for c_id in cluster_data.keys()]
            map_cluster = st.selectbox("Seleccionar cluster para visualizar:", cluster_options)
            
            if stations_data is not None:
                if map_cluster != 'Todos':
                    cluster_num = int(map_cluster)
                    map_stations = stations_data[stations_data['cluster'] == cluster_num]
                else:
                    map_stations = stations_data
                
                if len(map_stations) > 0:
                    m = create_station_map(map_stations)
                    if m:
                        folium_static(m, width=1000, height=600)
        else:
            st.warning("No hay datos de clustering disponibles. El análisis de clustering debe ejecutarse primero.")
    
    # Panel de Predicciones de Demanda
    elif page == "Predicciones de demanda":
        st.markdown('<div class="sub-header">📈 Predicciones de Demanda</div>', unsafe_allow_html=True)
        
        if prediction_data is not None and len(prediction_data) > 0:
            # Información general
            today = datetime.now().strftime('%Y-%m-%d')
            todays_pred = prediction_data[prediction_data['prediction_date'] == today]
            
            if len(todays_pred) > 0:
                total_demand = int(todays_pred['predicted_demand'].sum())
                avg_demand = float(todays_pred['predicted_demand'].mean())
                max_demand = float(todays_pred['predicted_demand'].max())
                
                # Métricas de predicción
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_demand:,}</div>
                            <div class="metric-label">Demanda total para hoy</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{avg_demand:.1f}</div>
                            <div class="metric-label">Demanda promedio por estación</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{max_demand:.1f}</div>
                            <div class="metric-label">Demanda máxima por estación</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Gráficos de predicción
            hourly_chart, heatmap_chart = create_demand_prediction_charts(prediction_data)
            
            if hourly_chart:
                st.plotly_chart(hourly_chart, use_container_width=True)
            
            if heatmap_chart:
                st.plotly_chart(heatmap_chart, use_container_width=True)
            
            # Mapa de calor de demanda
            st.markdown('<div class="sub-header">🔥 Mapa de Calor de Demanda</div>', unsafe_allow_html=True)
            
            if stations_data is not None:
                m_heat = create_heatmap(stations_data, prediction_data)
                if m_heat:
                    folium_static(m_heat, width=1000, height=600)
            
            # Tabla de predicciones por estación
            st.markdown('<div class="sub-header">📋 Predicciones por Estación</div>', unsafe_allow_html=True)
            
            # Agrupar predicciones por estación
            station_pred = prediction_data.groupby('station_id')['predicted_demand'].agg(['mean', 'sum']).reset_index()
            station_pred.columns = ['Estación', 'Demanda media', 'Demanda total']
            
            # Redondear valores
            station_pred['Demanda media'] = station_pred['Demanda media'].round(1)
            station_pred['Demanda total'] = station_pred['Demanda total'].round(0).astype(int)
            
            # Ordenar por demanda total
            station_pred = station_pred.sort_values('Demanda total', ascending=False)
            
            # Mostrar tabla
            st.dataframe(station_pred, hide_index=True)
        else:
            st.warning("No hay datos de predicción disponibles. El modelo de predicción debe ejecutarse primero.")
    
    # Panel de Series Temporales
    elif page == "Series temporales":
        st.markdown('<div class="sub-header">📊 Análisis de Series Temporales</div>', unsafe_allow_html=True)
        
        if time_series_data:
            forecasts = time_series_data.get('forecasts')
            historic = time_series_data.get('historic')
            
            if forecasts is not None and len(forecasts) > 0:
                # Información general
                today = datetime.now().strftime('%Y-%m-%d')
                forecast_days = len(forecasts['forecast_date'].unique())
                
                st.write(f"Se han generado pronósticos para {forecast_days} días futuros.")
                
                # Gráficos de series temporales
                forecast_chart, comparison_chart = create_time_series_charts(time_series_data)
                
                if forecast_chart:
                    st.plotly_chart(forecast_chart, use_container_width=True)
                
                if comparison_chart:
                    st.plotly_chart(comparison_chart, use_container_width=True)
                
                # Mostrar detalles de pronósticos
                st.markdown('<div class="sub-header">📅 Pronósticos Diarios</div>', unsafe_allow_html=True)
                
                # Preparar datos para mostrar
                show_forecasts = forecasts.copy()
                show_forecasts['forecast_date'] = pd.to_datetime(show_forecasts['forecast_date'])
                show_forecasts['forecast_date'] = show_forecasts['forecast_date'].dt.strftime('%Y-%m-%d')
                
                # Agrupar por fecha y modelo
                daily_forecasts = show_forecasts.groupby(['forecast_date', 'model_type'])['forecast_value'].mean().reset_index()
                daily_forecasts.columns = ['Fecha', 'Modelo', 'Predicción']
                
                # Redondear valores
                daily_forecasts['Predicción'] = daily_forecasts['Predicción'].round(0).astype(int)
                
                # Pivot table para mostrar los modelos en columnas
                daily_pivot = daily_forecasts.pivot(index='Fecha', columns='Modelo', values='Predicción').reset_index()
                
                # Mostrar tabla
                st.dataframe(daily_pivot, hide_index=True)
                
                # Componentes del modelo (si están disponibles)
                st.markdown('<div class="sub-header">🔄 Detalle de Componentes Temporales</div>', unsafe_allow_html=True)
                
                # Intentar cargar visualizaciones de componentes desde archivos locales
                time_series_path = Path("/app/data/time_series")
                component_path = time_series_path / "prophet_components.png"
                
                if component_path.exists():
                    st.image(str(component_path), caption="Componentes del modelo Prophet", use_column_width=True)
                else:
                    st.info("No hay imágenes de componentes disponibles.")
            else:
                st.warning("No hay pronósticos de series temporales disponibles.")
        else:
            st.warning("No hay datos de series temporales disponibles. El análisis de series temporales debe ejecutarse primero.")
    
    # Pie de página
    st.markdown("""
    <div class="footer">
        Dashboard creado con Streamlit | Datos procesados con PostgreSQL y H3
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()