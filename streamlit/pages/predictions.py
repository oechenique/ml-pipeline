import streamlit as st
import pandas as pd
import geopandas as gpd
import json
from typing import Dict
from components.maps import create_prediction_map
from components.charts import plot_prediction_comparison, plot_density_distribution
from components.filters import apply_data_filters

def load_prediction_data() -> Dict:
    """Carga datos y resultados de predicciones."""
    try:
        # Cargar datos con predicciones
        gdf = gpd.read_file("data/final/integrated_data_h3_9.geojson")
        
        # Cargar resultados del modelo
        with open("data/predictions/prediction_results.json") as f:
            results = json.load(f)
            
        # Cargar predicciones
        predictions = pd.read_parquet("data/predictions/commercial_predictions.parquet")
        
        # Combinar datos
        gdf = gdf.merge(
            predictions[['h3_index', 'prediction']],
            on='h3_index',
            how='left'
        )
        
        return {
            'data': gdf,
            'results': results
        }
    except Exception as e:
        st.error(f"Error cargando datos de predicciones: {str(e)}")
        return None

def show_model_metrics(results: Dict):
    """Muestra métricas del modelo predictivo."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "R² Score",
            f"{results['model_metrics']['r2']:.3f}",
            help="Coeficiente de determinación (mayor es mejor)"
        )
    
    with col2:
        st.metric(
            "RMSE",
            f"{results['model_metrics']['rmse']:.3f}",
            help="Error cuadrático medio"
        )
    
    with col3:
        st.metric(
            "MAE",
            f"{results['model_metrics']['mae']:.3f}",
            help="Error absoluto medio"
        )

def show_feature_importance(results: Dict):
    """Muestra importancia de features."""
    st.markdown("### 🎯 Importancia de Features")
    
    # Ordenar features por importancia
    importance = pd.DataFrame(
        results['feature_importance'].items(),
        columns=['Feature', 'Importance']
    ).sort_values('Importance', ascending=True)
    
    # Mostrar gráfico de barras horizontales
    fig = go.Figure(go.Bar(
        x=importance['Importance'],
        y=importance['Feature'],
        orientation='h'
    ))
    
    fig.update_layout(
        title="Importancia de Features",
        xaxis_title="Importancia",
        yaxis_title="Feature"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show(filters: Dict):
    """Muestra página de análisis de predicciones."""
    st.title("🎯 Predicciones de Potencial Comercial")
    st.markdown("""
        Análisis de predicciones del modelo de Machine Learning para identificar
        áreas con alto potencial comercial. El modelo considera múltiples factores
        como densidad poblacional, infraestructura existente y patrones comerciales.
    """)
    
    # Cargar datos
    data = load_prediction_data()
    if not data:
        return
    
    # Aplicar filtros
    filtered_data = apply_data_filters(data['data'], filters, 'geometry')
    
    # Mostrar métricas del modelo
    show_model_metrics(data['results'])
    
    # Mapa de predicciones
    st.markdown("### 🗺️ Mapa de Potencial Comercial")
    kepler_map = create_prediction_map(filtered_data)
    st.components.v1.html(kepler_map.to_html(), height=600)
    
    # Análisis de predicciones
    st.markdown("### 📊 Análisis de Predicciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de scores
        fig = plot_density_distribution(
            filtered_data,
            'prediction',
            'Distribución de Scores'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Predicción vs Real
        fig = plot_prediction_comparison(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Importancia de features
    show_feature_importance(data['results'])
    
    # Top áreas con potencial
    st.markdown("### 🌟 Áreas con Mayor Potencial")
    
    top_n = st.slider(
        "Número de áreas a mostrar",
        min_value=5,
        max_value=50,
        value=10
    )
    
    top_areas = filtered_data.nlargest(top_n, 'prediction')[[
        'h3_index', 'prediction', 'population', 'poi_count',
        'commercial_density', 'provincia', 'departamento'
    ]]
    
    st.dataframe(top_areas)
    
    # Descargar predicciones
    st.markdown("### 💾 Descargar Predicciones")
    
    if st.button("Preparar Predicciones para Descarga"):
        # Preparar datos
        download_data = filtered_data[[
            'h3_index', 'prediction', 'population', 'poi_count',
            'commercial_density', 'provincia', 'departamento'
        ]].copy()
        
        # Convertir a CSV
        csv = download_data.to_csv(index=False)
        
        # Botón de descarga
        st.download_button(
            label="Descargar CSV",
            data=csv,
            file_name="predictions_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    show({})  # Para pruebas locales