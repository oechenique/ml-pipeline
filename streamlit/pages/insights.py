import streamlit as st
import pandas as pd
import geopandas as gpd
import json
from typing import Dict, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from components.charts import plot_business_types, plot_metrics_evolution
from components.filters import apply_data_filters

def load_insight_data() -> Dict:
    """Carga datos para análisis de insights."""
    try:
        # Cargar datos integrados
        gdf = gpd.read_file("data/final/integrated_data_h3_9.geojson")
        
        # Cargar resultados de clustering
        with open("data/clusters/clustering_results.json") as f:
            cluster_results = json.load(f)
            
        # Cargar resultados de predicciones
        with open("data/predictions/prediction_results.json") as f:
            prediction_results = json.load(f)
            
        return {
            'data': gdf,
            'cluster_results': cluster_results,
            'prediction_results': prediction_results
        }
    except Exception as e:
        st.error(f"Error cargando datos para insights: {str(e)}")
        return None

def generate_market_insights(data: gpd.GeoDataFrame) -> List[Dict]:
    """Genera insights de mercado basados en los datos."""
    insights = []
    
    # Insight 1: Concentración de mercado
    total_comercios = data['poi_count'].sum()
    top_areas = data.nlargest(10, 'commercial_density')
    concentration = (top_areas['poi_count'].sum() / total_comercios) * 100
    
    insights.append({
        'title': 'Concentración de Mercado',
        'description': f'El {concentration:.1f}% de los comercios están concentrados en el top 10 de áreas más densas.',
        'type': 'market_concentration',
        'metrics': {
            'concentration': concentration,
            'total_comercios': total_comercios
        }
    })
    
    # Insight 2: Oportunidades por tipo de negocio
    business_types = pd.Series(
        [bt for bts in data['business_types'] for bt in eval(bts)]
    ).value_counts()
    
    opportunities = []
    for bt, count in business_types.items():
        density = count / len(data)
        if density < 0.1:  # Umbral arbitrario para identificar oportunidades
            opportunities.append((bt, density))
    
    insights.append({
        'title': 'Oportunidades por Tipo de Negocio',
        'description': f'Se identificaron {len(opportunities)} tipos de negocio con baja penetración en el mercado.',
        'type': 'business_opportunities',
        'metrics': {
            'opportunities': opportunities
        }
    })
    
    # Insight 3: Áreas desatendidas
    underserved = data[
        (data['population'] > data['population'].mean()) &
        (data['commercial_density'] < data['commercial_density'].mean())
    ]
    
    insights.append({
        'title': 'Áreas Desatendidas',
        'description': f'Se identificaron {len(underserved)} áreas con alta población pero baja densidad comercial.',
        'type': 'underserved_areas',
        'metrics': {
            'underserved_count': len(underserved),
            'potential_customers': underserved['population'].sum()
        }
    })
    
    return insights

def show_insight_card(insight: Dict):
    """Muestra un insight en formato de tarjeta."""
    st.markdown(f"### {insight['title']}")
    st.markdown(insight['description'])
    
    # Mostrar métricas específicas según el tipo de insight
    if insight['type'] == 'market_concentration':
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Concentración",
                f"{insight['metrics']['concentration']:.1f}%"
            )
        with col2:
            st.metric(
                "Total Comercios",
                f"{insight['metrics']['total_comercios']:,}"
            )
            
    elif insight['type'] == 'business_opportunities':
        st.markdown("#### Top Oportunidades:")
        for bt, density in insight['metrics']['opportunities'][:5]:
            st.write(f"- {bt}: {density:.3f} comercios por área")
            
    elif insight['type'] == 'underserved_areas':
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Áreas Desatendidas",
                f"{insight['metrics']['underserved_count']:,}"
            )
        with col2:
            st.metric(
                "Clientes Potenciales",
                f"{insight['metrics']['potential_customers']:,.0f}"
            )

def show_recommendations(data: gpd.GeoDataFrame):
    """Muestra recomendaciones basadas en el análisis."""
    st.markdown("## 📋 Recomendaciones")
    
    # Identificar mejores ubicaciones
    top_locations = data.nlargest(5, 'prediction')
    
    st.markdown("### 🎯 Mejores Ubicaciones para Nuevos Negocios")
    
    for idx, row in top_locations.iterrows():
        with st.expander(f"📍 {row['provincia']} - {row['departamento']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Score de Predicción", f"{row['prediction']:.3f}")
                st.metric("Población", f"{row['population']:,.0f}")
                
            with col2:
                st.metric("Densidad Comercial", f"{row['commercial_density']:.3f}")
                st.metric("Comercios Actuales", f"{row['poi_count']}")
            
            st.markdown("#### Tipos de Negocio Recomendados:")
            existing_types = eval(row['business_types'])
            all_types = pd.Series(
                [bt for bts in data['business_types'] for bt in eval(bts)]
            ).value_counts()
            
            # Identificar tipos de negocio faltantes
            missing_types = [
                bt for bt in all_types.index[:10]
                if bt not in existing_types
            ]
            
            for bt in missing_types[:3]:
                st.write(f"- {bt}")

def show(filters: Dict):
    """Muestra página de insights comerciales."""
    st.title("💡 Insights Comerciales")
    st.markdown("""
        Análisis detallado del mercado comercial y recomendaciones basadas en
        patrones identificados mediante técnicas de Machine Learning.
    """)
    
    # Cargar datos
    data = load_insight_data()
    if not data:
        return
    
    # Aplicar filtros
    filtered_data = apply_data_filters(data['data'], filters, 'geometry')
    
    # Generar y mostrar insights
    insights = generate_market_insights(filtered_data)
    
    # Mostrar insights en tabs
    tabs = st.tabs([insight['title'] for insight in insights])
    for tab, insight in zip(tabs, insights):
        with tab:
            show_insight_card(insight)
    
    # Visualizaciones complementarias
    st.markdown("## 📊 Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de tipos de negocio
        fig = plot_business_types(filtered_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Evolución de métricas
        fig = plot_metrics_evolution(
            filtered_data,
            ['commercial_density', 'population']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar recomendaciones
    show_recommendations(filtered_data)
    
    # Exportar insights
    st.markdown("### 💾 Exportar Análisis")
    
    if st.button("Generar Reporte"):
        # Preparar datos para reporte
        report_data = {
            'insights': insights,
            'top_locations': filtered_data.nlargest(10, 'prediction')[[
                'h3_index', 'provincia', 'departamento',
                'prediction', 'population', 'commercial_density'
            ]].to_dict('records'),
            'metrics': {
                'total_areas': len(filtered_data),
                'total_population': filtered_data['population'].sum(),
                'total_comercios': filtered_data['poi_count'].sum(),
                'avg_density': filtered_data['commercial_density'].mean()
            }
        }
        
        # Convertir a JSON
        report_json = json.dumps(report_data, indent=2)
        
        # Botón de descarga
        st.download_button(
            label="Descargar Reporte (JSON)",
            data=report_json,
            file_name="insights_report.json",
            mime="application/json"
        )

if __name__ == "__main__":
    show({})  # Para pruebas locales