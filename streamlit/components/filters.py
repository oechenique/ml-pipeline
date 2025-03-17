import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import geopandas as gpd
from pathlib import Path

def load_boundary_data() -> Dict[str, List[str]]:
    """Carga datos de límites administrativos."""
    try:
        provinces = pd.read_parquet("data/processed/boundaries/provinces.parquet")
        departments = pd.read_parquet("data/processed/boundaries/departments.parquet")
        
        # Crear diccionario de departamentos por provincia
        dept_by_province = {}
        for province in provinces['nombre'].unique():
            dept_by_province[province] = departments[
                departments['provincia'] == province
            ]['nombre'].tolist()
            
        return {
            'provinces': provinces['nombre'].tolist(),
            'departments_by_province': dept_by_province
        }
    except Exception as e:
        st.error(f"Error cargando datos administrativos: {str(e)}")
        return {'provinces': [], 'departments_by_province': {}}

def initialize_filters() -> Dict:
    """Inicializa filtros interactivos en sidebar."""
    
    st.sidebar.markdown("## 🎯 Filtros")
    
    # Cargar datos de límites
    boundary_data = load_boundary_data()
    
    # Filtro de provincia
    province = st.sidebar.selectbox(
        "Provincia",
        ["Todas"] + boundary_data['provinces']
    )
    
    # Filtro de departamento (dependiente de provincia)
    departments = (
        ["Todos"] + boundary_data['departments_by_province'].get(province, [])
        if province != "Todas"
        else ["Todos"]
    )
    department = st.sidebar.selectbox("Departamento", departments)
    
    # Filtro de tipo de negocio
    business_types = [
        "Todos",
        "Retail",
        "Gastronomía",
        "Servicios",
        "Oficinas"
    ]
    business_type = st.sidebar.multiselect(
        "Tipo de Negocio",
        business_types,
        default=["Todos"]
    )
    
    # Filtros de densidad
    st.sidebar.markdown("### 📊 Rangos")
    
    # Slider de densidad poblacional
    pop_density_range = st.sidebar.slider(
        "Densidad Poblacional",
        min_value=0,
        max_value=1000,
        value=(0, 1000),
        step=50
    )
    
    # Slider de densidad comercial
    com_density_range = st.sidebar.slider(
        "Densidad Comercial",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.1
    )
    
    # Botón de aplicar filtros
    apply_filters = st.sidebar.button("Aplicar Filtros")
    
    # Reset filtros
    if st.sidebar.button("Reset Filtros"):
        st.experimental_rerun()
    
    return {
        'province': province,
        'department': department,
        'business_type': business_type,
        'pop_density_range': pop_density_range,
        'com_density_range': com_density_range,
        'apply_filters': apply_filters
    }

def apply_data_filters(
    df: pd.DataFrame,
    filters: Dict,
    geometry_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aplica filtros al DataFrame.
    
    Args:
        df: DataFrame a filtrar
        filters: Diccionario de filtros
        geometry_col: Nombre de columna geometría para GeoDataFrame
        
    Returns:
        DataFrame filtrado
    """
    if not filters['apply_filters']:
        return df
        
    # Crear copia para no modificar original
    filtered_df = df.copy()
    
    # Filtros geográficos
    if filters['province'] != "Todas":
        filtered_df = filtered_df[
            filtered_df['provincia'] == filters['province']
        ]
    
    if filters['department'] != "Todos":
        filtered_df = filtered_df[
            filtered_df['departamento'] == filters['department']
        ]
    
    # Filtros de tipo de negocio
    if "Todos" not in filters['business_type']:
        business_mask = filtered_df['business_types'].apply(
            lambda x: any(bt in x for bt in filters['business_type'])
        )
        filtered_df = filtered_df[business_mask]
    
    # Filtros de densidad
    filtered_df = filtered_df[
        (filtered_df['population_density'] >= filters['pop_density_range'][0]) &
        (filtered_df['population_density'] <= filters['pop_density_range'][1]) &
        (filtered_df['commercial_density'] >= filters['com_density_range'][0]) &
        (filtered_df['commercial_density'] <= filters['com_density_range'][1])
    ]
    
    # Convertir a GeoDataFrame si es necesario
    if geometry_col and len(filtered_df) > 0:
        filtered_df = gpd.GeoDataFrame(
            filtered_df,
            geometry=geometry_col,
            crs="EPSG:4326"
        )
    
    return filtered_df