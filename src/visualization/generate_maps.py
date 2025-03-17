import geopandas as gpd
import keplergl
import os

# Paths de datos y salida
DATA_PATH = "data/final/integrated_data_h3_9.geojson"
OUTPUT_HTML = "outputs/kepler_map.html"

def load_data():
    """Carga los datos procesados en un GeoDataFrame."""
    print("📥 Cargando datos de GeoJSON...")
    try:
        gdf = gpd.read_file(DATA_PATH)
        print("✅ Datos cargados correctamente")
        return gdf
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        return None

def generate_kepler_map(gdf):
    """Genera un mapa interactivo con Kepler.gl."""
    print("🗺️ Generando mapa en Kepler.gl...")
    
    # Crear mapa con Kepler
    map_ = keplergl.KeplerGl(height=600)
    map_.add_data(data=gdf, name="GeoSpatial Data")

    # Crear directorio de salida si no existe
    os.makedirs("outputs", exist_ok=True)
    
    # Guardar mapa en HTML
    map_.save_to_html(file_name=OUTPUT_HTML)
    print(f"✅ Mapa guardado en {OUTPUT_HTML}")

def main():
    """Ejecuta la generación del mapa."""
    gdf = load_data()
    if gdf is not None:
        generate_kepler_map(gdf)
        print("🚀 Mapa generado exitosamente!")

if __name__ == "__main__":
    main()