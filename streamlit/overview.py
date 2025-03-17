import streamlit as st
import pandas as pd
import geopandas as gpd
from keplergl import KeplerGl
from components.maps import create_kepler_map
from components.charts import (
    plot_density_distribution,
    plot_business_types,
    plot_metrics_evolution
)
from components.filters import apply_data_filters
from typing import Dict
import json

def load_data() -> gpd.GeoDataFrame:
    """Carga datos procesados."""
    try:
        # Cargar datos integrados
        gdf = gpd.read_file("data/final/integrated_data_h3_9.geojson")
        
        # Cargar resultados de clustering
        with open("data/clusters/clustering_results.json") as f:
            cluster_results = json.load(f)
        
        # Cargar predicciones
        predictions = pd.read_parquet("data/predictions/commercial_predictions.parquet")
        
        # Combinar datos
        gdf = gdf.merge(
            predictions[['h3_index', 'prediction']],
            on='h3_index',
            how='left'
        )
        
        return gdf
        
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None

def show_metrics(data: gpd.GeoDataFrame):
    """Muestra métricas principales."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Hexágonos",
            f"{len(data):,}",
            help="Número total de áreas H3 analizadas"
        )
    
    with col2:
        st.metric(
            "Población Total",
            f"{data['population'].sum():,.0f}",
            help="Población total en áreas analizadas"
        )
    
    with col3:
        st.metric(
            "Total Comercios",
            f"{data['poi_count'].sum():,.0f}",
            help="Número total de comercios registrados"
        )
    
    with col4:
        st.metric(
            "Densidad Comercial Promedio",
            f"{data['commercial_density'].mean():.2f}",
            help="Comercios por cada 1000 habitantes"
        )

def show_overview(data: gpd.GeoDataFrame, filters: Dict):
    """Muestra vista general de datos."""
    
    st.title("🏪 Análisis Comercial Argentina")
    st.markdown("""
        Análisis de oportunidades comerciales basado en datos geoespaciales,
        densidad poblacional y patrones comerciales existentes.
    """)
    
    # Aplicar filtros
    filtered_data = apply_data_filters(data, filters, 'geometry')
    
    # Mostrar métricas
    show_metrics(filtered_data)
    
    # Mapa Kepler.gl
    st.markdown("### 🗺️ Distribución Espacial")
    
    map_config = {
        "version": "v1",
        "config": {
            "visState": {
                "layers": [
                    {
                        "id": "h3_cells",
                        "type": "geojson",
                        "config": {
                            "dataId": "h3_data",
                            "label": "Hexágonos H3",
                            "color": [18, 147, 154],
                            "highlightColor": [252, 242, 26, 255],
                            "columns": {
                                "geojson": "geometry"
                            },
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.8,
                                "strokeOpacity": 1,
                                "thickness": 0.5,
                                "strokeColor": [255, 255, 255],
                                "colorRange": {
                                    "name": "Global Warming",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]
                                },
                                "strokeColorRange": {
                                    "name": "Global Warming",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#5A1846", "#900C3F", "#C70039", "#E3611C", "#F1920E", "#FFC300"]