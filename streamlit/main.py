import streamlit as st
from pages import overview, clusters, predictions, insights
from components.filters import initialize_filters
import config

# Configuración de la página
st.set_page_config(
    page_title="Análisis Comercial Argentina",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .reportview-container .main .block-container {
            max-width: 95%;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    """Función principal del dashboard."""
    
    # Sidebar
    st.sidebar.title("🏪 Análisis Comercial")
    
    # Selector de página
    page = st.sidebar.selectbox(
        "Seleccionar Vista",
        ["Vista General", "Clusters", "Predicciones", "Insights"],
        key="page_selection"
    )
    
    # Inicializar filtros en sidebar
    filters = initialize_filters()
    
    # Renderizar página seleccionada
    if page == "Vista General":
        overview.show(filters)
    elif page == "Clusters":
        clusters.show(filters)
    elif page == "Predicciones":
        predictions.show(filters)
    else:
        insights.show(filters)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Desarrollado con ❤️ usando "
        "[Streamlit](https://streamlit.io) | "
        "[GitHub](https://github.com/tuuser/geospatial-ml-engineering)"
    )

if __name__ == "__main__":
    main()