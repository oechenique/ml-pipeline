import geopandas as gpd
import pandas as pd
import os

# Definir paths de entrada y salida
DATA_DIR = "data/processed"
OUTPUT_DIR = "data/final"

POPULATION_FILE = os.path.join(DATA_DIR, "population_h3_9.geojson")
OSM_FILE = os.path.join(DATA_DIR, "osm_pois_h3_9.geojson")
IGN_FILE = os.path.join(DATA_DIR, "admin_boundaries_h3_9.geojson")
OUTPUT_FILE_GEOJSON = os.path.join(OUTPUT_DIR, "integrated_data_h3_9.geojson")
OUTPUT_FILE_PARQUET = os.path.join(OUTPUT_DIR, "integrated_data_h3_9.parquet")

def load_data():
    """Carga los datasets de población, OSM y límites administrativos."""
    print("📥 Cargando datos...")
    try:
        gdf_population = gpd.read_file(POPULATION_FILE)
        gdf_osm = gpd.read_file(OSM_FILE)
        gdf_ign = gpd.read_file(IGN_FILE)
        print("✅ Datos cargados correctamente")
        return gdf_population, gdf_osm, gdf_ign
    except Exception as e:
        print(f"❌ Error cargando los datos: {str(e)}")
        return None, None, None

def merge_datasets(gdf_population, gdf_osm, gdf_ign):
    """Fusiona los datasets en un GeoDataFrame único basado en H3."""
    print("🔗 Combinando datasets...")

    # Asegurar que los datasets contienen la columna H3
    if "h3_index" not in gdf_population.columns or \
       "h3_index" not in gdf_osm.columns or \
       "h3_index" not in gdf_ign.columns:
        raise ValueError("Uno o más datasets no contienen la columna 'h3_index'")

    # Unir datasets por H3
    gdf = gdf_population.merge(gdf_osm, on="h3_index", how="left", suffixes=("_pop", "_osm"))
    gdf = gdf.merge(gdf_ign, on="h3_index", how="left", suffixes=("", "_ign"))

    # Rellenar NaN en columnas numéricas con 0
    numeric_cols = gdf.select_dtypes(include=["number"]).columns
    gdf[numeric_cols] = gdf[numeric_cols].fillna(0)

    print("✅ Datasets combinados correctamente")
    return gdf

def save_data(gdf):
    """Guarda el GeoDataFrame resultante en formatos GeoJSON y Parquet."""
    print("💾 Guardando datos integrados...")

    # Crear directorio de salida si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Guardar en GeoJSON y Parquet
    gdf.to_file(OUTPUT_FILE_GEOJSON, driver="GeoJSON")
    gdf.to_parquet(OUTPUT_FILE_PARQUET)

    print(f"✅ Datos guardados en {OUTPUT_FILE_GEOJSON} y {OUTPUT_FILE_PARQUET}")

def main():
    """Ejecuta el pipeline de merge de datasets."""
    gdf_population, gdf_osm, gdf_ign = load_data()
    if gdf_population is None or gdf_osm is None or gdf_ign is None:
        print("❌ No se pudo cargar algún dataset, abortando...")
        return
    
    gdf_final = merge_datasets(gdf_population, gdf_osm, gdf_ign)
    save_data(gdf_final)
    print("🚀 Merge de datasets completado con éxito!")

if __name__ == "__main__":
    main()