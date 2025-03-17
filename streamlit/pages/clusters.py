import streamlit as st
import pandas as pd
import geopandas as gpd
import json
from typing import Dict
from components.maps import create_cluster_map
from components.charts import plot_cluster_characteristics
from components.filters import apply_data_filters

def load_cluster_data() -> Dict:
    """Carga datos de clusters y resultados."""
    try:
        # Cargar datos integrados con clusters
        gdf = gpd.read_file("data/final/integrated_data_h3_9.geojson")
        
        # Cargar resultados de clustering
        with open("data/clusters/clustering_results.json") as f:
            results = json.load(f)
            
        return {
            'data': gdf,
            'results': results
        }
    except Exception as e:
        st.error(f"Error cargando datos de clusters: {str(e)}")
        return None

def show_cluster_metrics(results: Dict):
    """Muestra métricas de clustering."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Número de Clusters",
            results['n_clusters'],
            help="Número óptimo de clusters identificados"
        )
    
    with col2:
        st.metric(
            "Silhouette Score",
            f"{results['silhouette_score']:.3f}",
            help="Medida de calidad del clustering (mayor es mejor)"
        )
    
    with col3:
        st.metric(
            "Clusters Balanceados",
            "Sí" if results.get('balanced_clusters', True) else "No",
            help="Indica si los clusters tienen tamaños similares"
        )

def show_cluster_analysis(results: Dict):
    """Muestra análisis detallado de clusters."""
    st.markdown("### 📊 Análisis de Clusters")
    
    # Selector de cluster
    cluster = st.selectbox(
        "Seleccionar Cluster",
        options=list(results['cluster_analysis'].keys()),
        format_func=lambda x: f"Cluster {x.split('_')[1]}"
    )
    
    # Mostrar características del cluster
    cluster_info = results['cluster_analysis'][cluster]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Estadísticas")
        st.write(f"- Tamaño: {cluster_info['size']:,} hexágonos")
        st.write(f"- Porcentaje: {cluster_info['percentage']:.1f}%")
        st.write(f"- Población promedio: {cluster_info['avg_population']:,.0f}")
        st.write(f"- Comercios promedio: {cluster_info['avg_poi_count']:.1f}")
    
    with col2:
        st.markdown("#### Tipos de Negocio Principales")
        for business_type, count in list(cluster_info['most_common_business_types'].items())[:5]:
            st.write(f"- {business_type}: {count}")

def show(filters: Dict):
    """Muestra página de análisis de clusters."""
    st.title("🎯 Análisis de Clusters")
    st.markdown("""
        Análisis de patrones comerciales identificados mediante técnicas de clustering.
        Los clusters agrupan áreas con características similares en términos de densidad
        comercial, población y tipos de negocios.
    """)
    
    # Cargar datos
    data = load_cluster_data()
    if not data:
        return
    
    # Aplicar filtros
    filtered_data = apply_data_filters(data['data'], filters, 'geometry')
    
    # Mostrar métricas
    show_cluster_metrics(data['results'])
    
    # Mapa de clusters
    st.markdown("### 🗺️ Distribución Espacial de Clusters")
    kepler_map = create_cluster_map(filtered_data)
    st.components.v1.html(kepler_map.to_html(), height=600)
    
    # Análisis de clusters
    show_cluster_analysis(data['results'])
    
    # Visualización de características
    st.markdown("### 📈 Características de Clusters")
    
    # Seleccionar features para visualizar
    feature_options = [
        col for col in filtered_data.columns
        if col.endswith('_norm') and not col.startswith('h3_')
    ]
    
    selected_features = st.multiselect(
        "Seleccionar Características",
        options=feature_options,
        default=feature_options[:5],
        format_func=lambda x: x.replace('_norm', '').replace('_', ' ').title()
    )
    
    if selected_features:
        fig = plot_cluster_characteristics(filtered_data, selected_features)
        st.plotly_chart(fig, use_container_width=True)
    
    # Descargar datos
    st.markdown("### 💾 Descargar Datos")
    
    if st.button("Preparar Datos para Descarga"):
        # Preparar datos
        download_data = filtered_data[[
            'h3_index', 'cluster', 'population', 'poi_count',
            'commercial_density', 'business_types'
        ]].copy()
        
        # Convertir a CSV
        csv = download_data.to_csv(index=False)
        
        # Botón de descarga
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="clusters_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show({})  # Para pruebas locales