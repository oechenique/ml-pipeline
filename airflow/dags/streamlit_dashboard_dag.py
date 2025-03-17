from datetime import datetime, timedelta
import os
import sys
import subprocess
import signal
import atexit
import time
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'launch_streamlit_dashboard',
    default_args=default_args,
    description='Launch Streamlit dashboard with bike-sharing pipeline results',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=['viz', 'dashboard', 'streamlit'],
)

# Function to set up Streamlit files
def setup_streamlit_files(**kwargs):
    """
    Creates necessary directories and saves the Streamlit file to the correct location.
    """
    print("?? Preparing files for Streamlit dashboard...")
    
    # Create directory if it doesn't exist
    viz_dir = Path('/app/src/viz')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for examples/fallbacks
    examples_dir = Path('/app/data/examples')
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy streamlit_app.py content to the correct location
    dashboard_path = viz_dir / 'streamlit_app.py'
    
    # Check if we need to create the dashboard file
    if not dashboard_path.exists():
        # Import the dashboard content from our artifact
        # In a real-world scenario, this would be the content of your Streamlit app
        dashboard_content = """
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
import psycopg2
import psycopg2.extras
from datetime import datetime, timedelta
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Bike Sharing ML Dashboard",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS style
st.markdown(\"\"\"
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
\"\"\", unsafe_allow_html=True)

# Database connection functions
def get_db_connection():
    \"\"\"Establishes connection with PostgreSQL database.\"\"\"
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
        st.error(f"Database connection error: {str(e)}")
        return None

# Data loading functions
@st.cache_data(ttl=3600)
def load_station_data():
    \"\"\"Loads station data from PostgreSQL including H3 and cluster information.\"\"\"
    try:
        conn = get_db_connection()
        if conn is None:
            # Load backup data if connection fails
            return pd.read_csv("/app/data/examples/stations_sample.csv")
        
        query = \"\"\"
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
        \"\"\"
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert to GeoDataFrame if geometry column exists
        if 'geometry' in df.columns:
            df['geometry'] = df['geometry'].apply(lambda x: json.loads(x) if x else None)
            gdf = gpd.GeoDataFrame.from_features(
                [{"type": "Feature", "geometry": g, "properties": {}} for g in df['geometry'] if g],
                crs="EPSG:4326"
            )
            # Copy columns from original DataFrame
            for col in df.columns:
                if col != 'geometry':
                    gdf[col] = df[col].values
            return gdf
        return df
    except Exception as e:
        st.error(f"Error loading station data: {str(e)}")
        # Load backup data
        return pd.read_csv("/app/data/examples/stations_sample.csv")

@st.cache_data(ttl=3600)
def load_trip_data(limit=10000):
    \"\"\"Loads trip data from PostgreSQL.\"\"\"
    try:
        conn = get_db_connection()
        if conn is None:
            # Load backup data if connection fails
            return pd.read_csv("/app/data/examples/trips_sample.csv")
        
        query = f\"\"\"
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
        \"\"\"
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading trip data: {str(e)}")
        # Load backup data
        return pd.read_csv("/app/data/examples/trips_sample.csv")

@st.cache_data(ttl=3600)
def load_cluster_data():
    \"\"\"Loads clustering analysis from PostgreSQL.\"\"\"
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        query = \"\"\"
        SELECT 
            cluster_analysis
        FROM clustering_stats
        ORDER BY created_at DESC
        LIMIT 1;
        \"\"\"
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row and row[0]:
            # JSONB comes as a dictionary in Python
            cluster_data = row[0]
        else:
            cluster_data = {}
            
        cursor.close()
        conn.close()
        return cluster_data
    except Exception as e:
        st.error(f"Error loading cluster data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_prediction_data():
    \"\"\"Loads demand predictions from PostgreSQL.\"\"\"
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        query = \"\"\"
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
        \"\"\"
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading prediction data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_time_series_data():
    \"\"\"Loads time series forecasts from PostgreSQL.\"\"\"
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        
        # Time series forecasts
        forecast_query = \"\"\"
        SELECT 
            forecast_date,
            forecast_value,
            lower_bound,
            upper_bound,
            model_type
        FROM bike_demand_forecasts
        ORDER BY forecast_date, model_type
        \"\"\"
        
        forecasts = pd.read_sql(forecast_query, conn)
        
        # Historical aggregated data for comparison
        historic_query = \"\"\"
        SELECT 
            date_trunc('day', start_time) as day,
            COUNT(*) as trip_count
        FROM bike_trips
        GROUP BY date_trunc('day', start_time)
        ORDER BY date_trunc('day', start_time)
        \"\"\"
        
        historic = pd.read_sql(historic_query, conn)
        conn.close()
        
        return {
            'forecasts': forecasts,
            'historic': historic
        }
    except Exception as e:
        st.error(f"Error loading time series data: {str(e)}")
        return None

# Visualization functions
def create_station_map(stations_gdf, prediction_data=None):
    \"\"\"Creates an interactive map of stations with clusters and H3 hexagons.\"\"\"
    
    if stations_gdf is None or len(stations_gdf) == 0:
        st.warning("No station data available for the map.")
        return
    
    # Create map centered on the data
    center = [stations_gdf['latitude'].mean(), stations_gdf['longitude'].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    
    # Define colors for the clusters
    colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#FF5722', '#607D8B']
    
    # Create cluster groups
    cluster_groups = {}
    
    # Add H3 hexagons if available
    if 'h3_geometry' in stations_gdf.columns:
        h3_group = folium.FeatureGroup(name="H3 Hexagons")
        
        for idx, row in stations_gdf.iterrows():
            if row['h3_geometry']:
                try:
                    h3_geom = json.loads(row['h3_geometry'])
                    
                    # Determine cluster and color
                    cluster_id = int(row['cluster']) if 'cluster' in row and row['cluster'] is not None else -1
                    if cluster_id < 0:
                        color = '#9E9E9E'  # Gray for unassigned stations
                    else:
                        color = colors[cluster_id % len(colors)]
                    
                    # Add hexagon
                    folium.GeoJson(
                        h3_geom,
                        style_function=lambda x, color=color: {
                            'fillColor': color,
                            'color': 'white',
                            'weight': 1,
                            'fillOpacity': 0.3
                        }
                    ).add_to(h3_group)
                except:
                    pass
        
        h3_group.add_to(m)
    
    # Add station markers with demand prediction if available
    for idx, row in stations_gdf.iterrows():
        # Determine cluster and color
        cluster_id = int(row['cluster']) if 'cluster' in row and row['cluster'] is not None else -1
        if cluster_id < 0:
            color = '#9E9E9E'  # Gray for unassigned stations
        else:
            color = colors[cluster_id % len(colors)]
        
        # Create or access the cluster group
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = folium.FeatureGroup(name=f"Cluster {cluster_id if cluster_id >= 0 else 'Unassigned'}")
            m.add_child(cluster_groups[cluster_id])
        
        # Calculate demand if prediction data is available
        demand_info = ""
        if prediction_data is not None:
            station_predictions = prediction_data[prediction_data['station_id'] == row['station_id']]
            if len(station_predictions) > 0:
                avg_demand = station_predictions['predicted_demand'].mean()
                max_demand = station_predictions['predicted_demand'].max()
                demand_info = f\"\"\"
                <br>Avg. Predicted Demand: {avg_demand:.1f}
                <br>Max. Predicted Demand: {max_demand:.1f}
                \"\"\"
                
                # Adjust radius based on demand
                radius = 5 + min(10, avg_demand / 2)
            else:
                radius = 5
        else:
            radius = 5
        
        # Create popup with information
        popup_text = f\"\"\"
        <b>{row['name']}</b><br>
        ID: {row['station_id']}<br>
        City: {row['city']}<br>
        Capacity: {row['capacity']}<br>
        Cluster: {cluster_id if cluster_id >= 0 else 'Unassigned'}<br>
        {demand_info}
        \"\"\"
        
        # Add marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(cluster_groups[cluster_id])
    
    # Add heat map layer if prediction data is available
    if prediction_data is not None:
        # Aggregate predictions by station
        pred_by_station = prediction_data.groupby('station_id')['predicted_demand'].mean().reset_index()
        
        # Merge with station data
        heat_data = pd.merge(
            stations_gdf,
            pred_by_station,
            on='station_id',
            how='inner'
        )
        
        if len(heat_data) > 0:
            # Prepare data for heat map
            heat_points = [[row['latitude'], row['longitude'], row['predicted_demand']] 
                        for idx, row in heat_data.iterrows()]
            
            # Add heat map
            heat_group = folium.FeatureGroup(name="Demand Heatmap")
            HeatMap(
                heat_points,
                radius=15, 
                gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'orange', 1: 'red'},
                min_opacity=0.5,
                blur=10
            ).add_to(heat_group)
            
            heat_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_cluster_analysis_chart(cluster_data):
    \"\"\"Creates visualizations for cluster analysis.\"\"\"
    
    if not cluster_data:
        st.warning("No clustering data available.")
        return
    
    # Extract cluster data
    cluster_sizes = []
    feature_patterns = []
    
    for cluster_id, data in cluster_data.items():
        cluster_sizes.append({
            'cluster': cluster_id,
            'size': data.get('size', 0),
            'percentage': data.get('percentage', 0)
        })
        
        # Extract patterns for radar chart
        if 'feature_patterns' in data:
            patterns = data['feature_patterns']
            patterns['cluster'] = cluster_id
            feature_patterns.append(patterns)
    
    # Convert to DataFrames
    df_sizes = pd.DataFrame(cluster_sizes)
    df_patterns = pd.DataFrame(feature_patterns)
    
    # Create bar chart for cluster sizes
    fig_sizes = px.bar(
        df_sizes,
        x='cluster',
        y='size',
        text='percentage',
        labels={'cluster': 'Cluster', 'size': 'Number of stations', 'percentage': 'Percentage'},
        title='Distribution of Stations by Cluster',
        color='size',
        color_continuous_scale='Viridis'
    )
    
    fig_sizes.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    
    # Create radar chart for feature patterns
    if len(df_patterns) > 0:
        # Select features for radar (excluding 'cluster')
        feature_cols = [col for col in df_patterns.columns if col != 'cluster']
        
        # Create figure with subplots for each cluster
        fig_patterns = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'polar'}]]
        )
        
        # Colors for each cluster
        colors = ['#1E88E5', '#FFC107', '#4CAF50', '#E91E63', '#9C27B0', '#FF5722', '#607D8B']
        
        # Add data for each cluster
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
            title='Feature Patterns by Cluster',
            showlegend=True
        )
        
        return fig_sizes, fig_patterns
    
    return fig_sizes, None

def create_demand_prediction_charts(predictions_df):
    \"\"\"Creates visualizations for demand predictions.\"\"\"
    
    if predictions_df is None or len(predictions_df) == 0:
        st.warning("No prediction data available.")
        return None, None
    
    # Aggregation by hour
    hourly_demand = predictions_df.groupby('prediction_hour')['predicted_demand'].mean().reset_index()
    
    # Aggregation by day and hour (for heatmap)
    # Add day of week
    predictions_df['day_of_week'] = pd.to_datetime(predictions_df['prediction_date']).dt.dayofweek
    
    # Create pivot table
    heatmap_data = predictions_df.pivot_table(
        index='day_of_week',
        columns='prediction_hour',
        values='predicted_demand',
        aggfunc='mean'
    ).fillna(0)
    
    # Line chart for hourly demand
    fig_hourly = px.line(
        hourly_demand,
        x='prediction_hour',
        y='predicted_demand',
        labels={'prediction_hour': 'Hour of Day', 'predicted_demand': 'Average Predicted Demand'},
        title='Predicted Demand by Hour of Day',
        markers=True
    )
    
    fig_hourly.update_traces(line=dict(color='#1E88E5', width=3))
    fig_hourly.update_layout(
        xaxis=dict(tickmode='linear', dtick=2),
        yaxis=dict(title='Predicted Demand (trips)')
    )
    
    # Heatmap for demand by day and hour
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x='Hour of Day', y='Day of Week', color='Predicted Demand'),
        x=[str(h) for h in range(24)],
        y=day_names,
        color_continuous_scale='Viridis',
        title='Weekly Demand Pattern'
    )
    
    fig_heatmap.update_layout(
        xaxis=dict(tickmode='linear', dtick=2)
    )
    
    return fig_hourly, fig_heatmap

def create_time_series_charts(ts_data):
    \"\"\"Creates visualizations for time series forecasts.\"\"\"
    
    if ts_data is None:
        st.warning("No time series data available.")
        return None, None
    
    forecasts = ts_data.get('forecasts')
    historic = ts_data.get('historic')
    
    if forecasts is None or len(forecasts) == 0:
        st.warning("No forecasts available.")
        return None, None
    
    # Filter and prepare data
    prophet_data = forecasts[forecasts['model_type'] == 'prophet']
    xgboost_data = forecasts[forecasts['model_type'] == 'xgboost']
    
    # Create forecast chart
    fig_forecast = go.Figure()
    
    # Add historical data if available
    if historic is not None and len(historic) > 0:
        historic['day'] = pd.to_datetime(historic['day'])
        fig_forecast.add_trace(
            go.Scatter(
                x=historic['day'],
                y=historic['trip_count'],
                mode='lines',
                name='Historical',
                line=dict(color='#888888', width=2)
            )
        )
    
    # Add Prophet forecast
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
        
        # Add confidence interval
        fig_forecast.add_trace(
            go.Scatter(
                x=pd.concat([prophet_data['forecast_date'], prophet_data['forecast_date'].iloc[::-1]]),
                y=pd.concat([prophet_data['upper_bound'], prophet_data['lower_bound'].iloc[::-1]]),
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Prophet CI'
            )
        )
    
    # Add XGBoost forecast
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
        title='Trip Demand Forecast',
        xaxis_title='Date',
        yaxis_title='Trip Demand',
        legend_title='Model',
        hovermode='x unified'
    )
    
    # Model comparison chart
    # Calculate metrics if available
    model_metrics = None
    if 'model_metrics' in ts_data:
        model_metrics = ts_data['model_metrics']
    else:
        # Try to calculate basic metrics
        if historic is not None and len(historic) > 0 and len(prophet_data) > 0 and len(xgboost_data) > 0:
            # Look for overlapping dates for comparison
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
    
    # If metrics available, create comparison chart
    if model_metrics is not None:
        fig_comparison = px.bar(
            model_metrics,
            barmode='group',
            title='Forecast Model Comparison',
            labels={'value': 'Value', 'variable': 'Metric'}
        )
    else:
        # Simply compare forecasted values by each model
        comparison_df = pd.DataFrame({
            'Date': prophet_data['forecast_date'],
            'Prophet': prophet_data['forecast_value'],
            'XGBoost': xgboost_data['forecast_value'].values if len(xgboost_data) == len(prophet_data) else np.nan
        })
        
        comparison_df = comparison_df.melt(
            id_vars=['Date'],
            value_vars=['Prophet', 'XGBoost'],
            var_name='Model',
            value_name='Forecast'
        )
        
        fig_comparison = px.box(
            comparison_df,
            x='Model',
            y='Forecast',
            color='Model',
            title='Distribution of Forecasts by Model',
            color_discrete_map={'Prophet': '#1E88E5', 'XGBoost': '#4CAF50'}
        )
    
    return fig_forecast, fig_comparison

# Main application
def main():
    # Main title
    st.markdown('<div class="main-header">?? Bike-Sharing Analysis Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for navigation and filters
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Station Map", "Cluster Analysis", 
         "Demand Predictions", "Time Series"]
    )
    
    # Load data
    with st.spinner("Loading data..."):
        stations_data = load_station_data()
        trip_data = load_trip_data()
        cluster_data = load_cluster_data()
        prediction_data = load_prediction_data()
        time_series_data = load_time_series_data()
    
    # Basic information
    trip_count = len(trip_data) if trip_data is not None else 0
    station_count = len(stations_data) if stations_data is not None else 0
    
    # Last update timestamp
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.sidebar.write(f"Last updated: {last_update}")
    
    # Overview Panel
    if page == "Overview":
        st.markdown('<div class="sub-header">?? System Overview</div>', unsafe_allow_html=True)
        
        # Key metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f\"\"\"
                <div class="metric-card">
                    <div class="metric-value">{station_count}</div>
                    <div class="metric-label">Stations</div>
                </div>
            \"\"\", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f\"\"\"
                <div class="metric-card">
                    <div class="metric-value">{trip_count:,}</div>
                    <div class="metric-label">Trips analyzed</div>
                </div>
            \"\"\", unsafe_allow_html=True)
        
        with col3:
            # Calculate clusters if data available
            cluster_count = len(cluster_data) if cluster_data else 0
            metric_html = f'''
                <div class="metric-card">
                    <div class="metric-value">{cluster_count}</div>
                    <div class="metric-label">Clusters identified</div>
                </div>
            '''
            st.markdown(metric_html, unsafe_allow_html=True)
        
        with col4:
            # Calculate unique cities
            if stations_data is not None and 'city' in stations_data.columns:
                city_count = stations_data['city'].nunique()
            else:
                city_count = 0
            
            metric_html = f'''
                <div class="metric-card">
                    <div class="metric-value">{city_count}</div>
                    <div class="metric-label">Cities analyzed</div>
                </div>
            '''
            st.markdown(metric_html, unsafe_allow_html=True)
        
        # Show the rest of the Overview tab...
    
    # Station Map Panel
    elif page == "Station Map":
        st.markdown('<div class="sub-header">??? Geospatial Station Exploration</div>', unsafe_allow_html=True)
        
        # Show the rest of the Station Map tab...
    
    # Cluster Analysis Panel
    elif page == "Cluster Analysis":
        st.markdown('<div class="sub-header">?? Station Cluster Analysis</div>', unsafe_allow_html=True)
        
        # Show the rest of the Cluster Analysis tab...
    
    # Demand Predictions Panel
    elif page == "Demand Predictions":
        st.markdown('<div class="sub-header">?? Demand Predictions</div>', unsafe_allow_html=True)
        
        # Show the rest of the Demand Predictions tab...
    
    # Time Series Panel
    elif page == "Time Series":
        st.markdown('<div class="sub-header">?? Time Series Analysis</div>', unsafe_allow_html=True)
        
        # Show the rest of the Time Series tab...
    
    # Footer
    st.markdown(\"\"\"
    <div class="footer">
        Dashboard created with Streamlit | Data processed with PostgreSQL and H3
    </div>
    \"\"\", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
"""
        
        # Write the code to a file
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)
    
    # Create examples of backup data
    stations_sample = """station_id,name,city,latitude,longitude,capacity,h3_index,cluster
station_1,Downtown Station,San Francisco,37.7749,-122.4194,20,,0
station_2,Market St Station,San Francisco,37.7846,-122.4089,15,,0
station_3,Mission District,San Francisco,37.7599,-122.4148,18,,1
station_4,Golden Gate Park,San Francisco,37.7694,-122.4862,25,,2
station_5,Fisherman's Wharf,San Francisco,37.8080,-122.4177,30,,1
"""
    
    trips_sample = """trip_id,start_time,end_time,duration_sec,start_station_id,end_station_id,user_type,city
trip_001,2023-01-01 08:30:00,2023-01-01 08:45:00,900,station_1,station_3,Member,San Francisco
trip_002,2023-01-01 09:15:00,2023-01-01 09:35:00,1200,station_2,station_1,Casual,San Francisco
trip_003,2023-01-01 10:00:00,2023-01-01 10:30:00,1800,station_3,station_5,Member,San Francisco
trip_004,2023-01-01 12:30:00,2023-01-01 13:00:00,1800,station_5,station_4,Casual,San Francisco
trip_005,2023-01-01 15:45:00,2023-01-01 16:15:00,1800,station_4,station_2,Member,San Francisco
"""
    
    with open(examples_dir / 'stations_sample.csv', 'w') as f:
        f.write(stations_sample)
    
    with open(examples_dir / 'trips_sample.csv', 'w') as f:
        f.write(trips_sample)
    
    print(f"? Dashboard saved to {dashboard_path}")
    print(f"? Sample data saved to {examples_dir}")
    
    return {
        'status': 'success',
        'streamlit_path': str(dashboard_path),
        'examples_dir': str(examples_dir)
    }

# Function to launch the Streamlit service
def launch_streamlit_dashboard(**kwargs):
    """
    Launches the Streamlit dashboard as a web service.
    """
    ti = kwargs.get('ti')
    setup_result = ti.xcom_pull(task_ids='setup_streamlit_files')
    
    if not setup_result or setup_result.get('status') != 'success':
        print("? Error: File preparation did not complete successfully")
        return False
    
    streamlit_path = setup_result.get('streamlit_path')
    
    print(f"?? Launching Streamlit dashboard from {streamlit_path}")
    
    # Command to run Streamlit
    cmd = [
        "streamlit", "run", streamlit_path,
        "--server.port=8501",
        "--server.address=0.0.0.0"
    ]
    
    try:
        # Start Streamlit as an independent process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Register the PID to terminate the process later
        pid = process.pid
        print(f"? Dashboard started with PID {pid}")
        
        # Function to terminate the process on exit
        def terminate_process():
            if process.poll() is None:  # Process still running
                print(f"?? Terminating Streamlit process with PID {pid}")
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        
        # Register the function to terminate the process when Airflow terminates
        atexit.register(terminate_process)
        
        # Wait for Streamlit to start
        time.sleep(5)
        
        # Check if the process is still running
        if process.poll() is None:
            print("? Dashboard running successfully")
            print("?? Access the dashboard at: http://localhost:8501")
            
            # Message for airflow logs
            print("\n" + "*"*80)
            print("*  STREAMLIT DASHBOARD RUNNING AT: http://localhost:8501")
            print("*  (If you're using Docker, make sure to map port 8501)")
            print("*"*80 + "\n")
            
            return {
                'status': 'success',
                'pid': pid,
                'url': 'http://localhost:8501'
            }
        else:
            # Capture any startup errors
            stdout, stderr = process.communicate()
            print(f"? Error starting Streamlit: {stderr}")
            return False
    except Exception as e:
        print(f"? Error launching Streamlit: {str(e)}")
        return False

# Tasks
setup_task = PythonOperator(
    task_id='setup_streamlit_files',
    python_callable=setup_streamlit_files,
    dag=dag,
)

launch_task = PythonOperator(
    task_id='launch_dashboard',
    python_callable=launch_streamlit_dashboard,
    provide_context=True,
    dag=dag,
)

# Dependencies
setup_task >> launch_task