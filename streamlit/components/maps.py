import streamlit as st
from keplergl import KeplerGl
import geopandas as gpd
from typing import Dict, Optional

def create_kepler_map(
    data: gpd.GeoDataFrame,
    height: int = 600,
    config: Optional[Dict] = None
) -> KeplerGl:
    """
    Crea un mapa interactivo con Kepler.gl.
    
    Args:
        data: GeoDataFrame con datos a visualizar
        height: Altura del mapa en píxeles
        config: Configuración opcional del mapa
        
    Returns:
        Objeto KeplerGl
    """
    # Configuración por defecto
    default_config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "h3_layer",
                        "type": "geojson",
                        "config": {
                            "dataId": "data",
                            "label": "Hexágonos H3",
                            "color": [18, 147, 154],
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
                                },
                                "radius": 100
                            }
                        },
                        "visualChannels": {
                            "colorField": {
                                "name": "commercial_density",
                                "type": "real"
                            },
                            "colorScale": "quantile"
                        }
                    }
                ]
            }
        }
    }
    
    # Usar configuración proporcionada o default
    map_config = config if config else default_config
    
    # Crear mapa
    kepler_map = KeplerGl(
        height=height,
        config=map_config
    )
    
    # Agregar datos
    kepler_map.add_data(
        data=data,
        name='data'
    )
    
    return kepler_map

def create_cluster_map(
    data: gpd.GeoDataFrame,
    height: int = 600
) -> KeplerGl:
    """
    Crea un mapa específico para visualizar clusters.
    
    Args:
        data: GeoDataFrame con datos de clusters
        height: Altura del mapa en píxeles
        
    Returns:
        Objeto KeplerGl
    """
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "cluster_layer",
                        "type": "geojson",
                        "config": {
                            "dataId": "data",
                            "label": "Clusters",
                            "color": [18, 147, 154],
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
                                    "name": "Paired",
                                    "type": "qualitative",
                                    "category": "Uber",
                                    "colors": ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"]
                                }
                            }
                        },
                        "visualChannels": {
                            "colorField": {
                                "name": "cluster",
                                "type": "integer"
                            },
                            "colorScale": "ordinal"
                        }
                    }
                ]
            }
        }
    }
    
    return create_kepler_map(data, height, config)

def create_prediction_map(
    data: gpd.GeoDataFrame,
    height: int = 600
) -> KeplerGl:
    """
    Crea un mapa específico para visualizar predicciones.
    
    Args:
        data: GeoDataFrame con predicciones
        height: Altura del mapa en píxeles
        
    Returns:
        Objeto KeplerGl
    """
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "prediction_layer",
                        "type": "geojson",
                        "config": {
                            "dataId": "data",
                            "label": "Predicciones",
                            "color": [18, 147, 154],
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
                                    "name": "Viridis",
                                    "type": "sequential",
                                    "category": "Uber",
                                    "colors": ["#440154", "#404387", "#29788E", "#22A784", "#79D151", "#FDE725"]
                                }
                            }
                        },
                        "visualChannels": {
                            "colorField": {
                                "name": "prediction",
                                "type": "real"
                            },
                            "colorScale": "quantile"
                        }
                    }
                ]
            }
        }
    }
    
    return create_kepler_map(data, height, config)