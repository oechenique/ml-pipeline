import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import mlflow
import mlflow.spark
from sklearn.metrics import silhouette_score
import json
from datetime import datetime
import plotly.express as px
import geopandas as gpd
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeStationClustering:
    """Análisis de patrones de uso mediante clustering de estaciones."""
    
    def __init__(
        self,
        input_dir: str = "data/ml_ready",
        output_dir: str = "data/clusters",
        h3_resolution: int = 9,
        experiment_name: str = "bike_station_clustering"
    ):
        """
        Inicializa el análisis de clustering.
        
        Args:
            input_dir: Directorio con features procesados
            output_dir: Directorio para resultados
            h3_resolution: Nivel de resolución H3
            experiment_name: Nombre del experimento MLflow
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.h3_resolution = h3_resolution
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar MLflow
        mlflow.set_experiment(experiment_name)
        
        # Inicializar Spark
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark para clustering."""
        return SparkSession.builder \
            .appName("Bike-Station-Clustering") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .getOrCreate()
    
    def load_features(self) -> pd.DataFrame:
        """Carga features procesados."""
        logger.info("🔄 Cargando features")
        
        features_path = self.input_dir / f"bike_features_h3_{self.h3_resolution}.parquet"
        df = pd.read_parquet(features_path)
        
        logger.info(f"✅ Cargados {len(df)} registros con {len(df.columns)} features")
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_groups: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepara features para clustering.
        
        Args:
            df: DataFrame con features
            feature_groups: Lista opcional de grupos de features a usar
            
        Returns:
            DataFrame preparado y lista de features usados
        """
        logger.info("🔄 Preparando features para clustering")
        
        # Cargar metadata de features
        metadata_path = self.input_dir / f"bike_features_h3_{self.h3_resolution}_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Seleccionar features por grupo
        if feature_groups:
            selected_features = []
            for group in feature_groups:
                selected_features.extend(metadata['feature_groups'][group])
        else:
            # Por defecto, usar features de uso y temporales
            selected_features = metadata['feature_groups']['station'] + \
                              metadata['feature_groups']['temporal']
        
        # Usar solo features normalizados
        selected_features = [f for f in selected_features if f.endswith('_norm')]
        
        # Remover valores faltantes
        clean_features = df[selected_features].dropna()
        
        logger.info(f"✅ Preparados {len(selected_features)} features para clustering")
        return clean_features, selected_features
    
    def find_optimal_k(
        self,
        features_df: pd.DataFrame,
        max_k: int = 10
    ) -> Tuple[int, List[float]]:
        """
        Encuentra número óptimo de clusters usando método del codo.
        
        Args:
            features_df: DataFrame con features
            max_k: Máximo número de clusters a probar
            
        Returns:
            Número óptimo de clusters y lista de costos
        """
        logger.info("🔄 Buscando número óptimo de clusters")
        
        # Convertir a Spark DataFrame
        spark_df = self.spark.createDataFrame(features_df)
        
        # Preparar features
        assembler = VectorAssembler(
            inputCols=features_df.columns,
            outputCol="features"
        )
        vector_df = assembler.transform(spark_df)
        
        # Calcular costos para diferentes k
        costs = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(k=k, seed=42)
            model = kmeans.fit(vector_df)
            cost = model.computeCost(vector_df)
            costs.append(cost)
            logger.info(f"K={k}, Cost={cost:.2f}")
        
        # Encontrar punto de codo
        diffs = np.diff(costs)
        diffs_2 = np.diff(diffs)
        optimal_k = np.argmax(np.abs(diffs_2)) + 2
        
        logger.info(f"✅ Número óptimo de clusters: {optimal_k}")
        return optimal_k, costs
    
    def train_clustering(
        self,
        features_df: pd.DataFrame,
        n_clusters: int,
        method: str = 'kmeans'
    ) -> Tuple[object, pd.DataFrame]:
        """
        Entrena modelo de clustering.
        
        Args:
            features_df: DataFrame con features
            n_clusters: Número de clusters
            method: Método de clustering ('kmeans' o 'bisecting')
            
        Returns:
            Modelo entrenado y predicciones
        """
        logger.info(f"🔄 Entrenando modelo de clustering ({method})")
        
        # Convertir a Spark DataFrame
        spark_df = self.spark.createDataFrame(features_df)
        
        # Preparar features
        assembler = VectorAssembler(
            inputCols=features_df.columns,
            outputCol="features"
        )
        vector_df = assembler.transform(spark_df)
        
        # Seleccionar modelo
        if method == 'kmeans':
            model = KMeans(k=n_clusters, seed=42)
        elif method == 'bisecting':
            model = BisectingKMeans(k=n_clusters, seed=42)
        else:
            raise ValueError(f"Método {method} no soportado")
        
        # Entrenar modelo
        with mlflow.start_run(run_name=f"{method}_clustering"):
            # Log parámetros
            mlflow.log_params({
                'method': method,
                'n_clusters': n_clusters,
                'n_features': len(features_df.columns)
            })
            
            # Entrenar
            trained_model = model.fit(vector_df)
            
            # Obtener predicciones
            predictions = trained_model.transform(vector_df)
            
            # Evaluar
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            
            # Log métricas
            mlflow.log_metric('silhouette_score', silhouette)
            
            # Log modelo
            mlflow.spark.log_model(trained_model, "model")
        
        logger.info(f"✅ Modelo entrenado (silhouette={silhouette:.3f})")
        return trained_model, predictions.toPandas()
    
    def analyze_clusters(
        self,
        original_df: pd.DataFrame,
        predictions: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict:
        """
        Analiza características de los clusters.
        
        Args:
            original_df: DataFrame original
            predictions: DataFrame con predicciones
            feature_names: Nombres de features usados
            
        Returns:
            Diccionario con análisis de clusters
        """
        logger.info("🔄 Analizando clusters")
        
        # Combinar datos originales con predicciones
        analysis_df = original_df.copy()
        analysis_df['cluster'] = predictions['prediction']
        
        # Analizar cada cluster
        cluster_analysis = {}
        
        for cluster in sorted(analysis_df['cluster'].unique()):
            cluster_data = analysis_df[analysis_df['cluster'] == cluster]
            
            # Calcular estadísticas
            stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(analysis_df) * 100,
                'avg_trips': cluster_data['total_trips'].mean(),
                'avg_duration': cluster_data['avg_duration'].mean(),
                'peak_usage': {
                    'rush_hour_ratio': cluster_data['rush_hour_trips'].mean() / cluster_data['total_trips'].mean(),
                    'weekend_ratio': cluster_data['weekend_ratio'].mean()
                },
                'balance': {
                    'inbound_outbound_ratio': cluster_data['balance_ratio'].mean(),
                    'net_flow': cluster_data['trip_balance'].mean()
                },
                'feature_patterns': {
                    feature: cluster_data[feature].mean()
                    for feature in feature_names
                }
            }
            
            cluster_analysis[f'cluster_{cluster}'] = stats
        
        logger.info("✅ Análisis de clusters completado")
        return cluster_analysis
    
    def visualize_clusters(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame
    ) -> None:
        """
        Genera visualizaciones de clusters.
        
        Args:
            df: DataFrame original
            predictions: DataFrame con predicciones
        """
        logger.info("🔄 Generando visualizaciones")
        
        # Combinar datos
        viz_df = df.copy()
        viz_df['cluster'] = predictions['prediction']
        
        # Crear directorio para visualizaciones
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Mapa de clusters
        if all(col in viz_df.columns for col in ['latitude', 'longitude']):
            gdf = gpd.GeoDataFrame(
                viz_df,
                geometry=gpd.points_from_xy(viz_df.longitude, viz_df.latitude)
            )
            
            fig_map = px.scatter_mapbox(
                gdf,
                lat='latitude',
                lon='longitude',
                color='cluster',
                size='total_trips',
                hover_data=['station_id', 'total_trips', 'avg_duration'],
                title='Clusters de Estaciones'
            )
            fig_map.write_html(viz_dir / 'cluster_map.html')
        
        # 2. Características por cluster
        fig_features = px.box(
            viz_df.melt(
                id_vars=['cluster'],
                value_vars=[c for c in viz_df.columns if c.endswith('_norm')]
            ),
            x='cluster',
            y='value',
            facet_col='variable',
            title='Distribución de Features por Cluster'
        )
        fig_features.write_html(viz_dir / 'cluster_features.html')
        
        # 3. Uso temporal por cluster
        if 'hour' in viz_df.columns:
            hourly_usage = viz_df.groupby(['cluster', 'hour'])['total_trips'].mean().reset_index()
            fig_temporal = px.line(
                hourly_usage,
                x='hour',
                y='total_trips',
                color='cluster',
                title='Patrones de Uso por Hora y Cluster'
            )
            fig_temporal.write_html(viz_dir / 'cluster_temporal.html')
        
        logger.info(f"✅ Visualizaciones guardadas en {viz_dir}")
    
    def process(self) -> Dict:
        """Ejecuta pipeline completo de clustering."""
        try:
            # Cargar y preparar datos
            df = self.load_features()
            features_df, feature_names = self.prepare_features(df)
            
            # Encontrar número óptimo de clusters
            n_clusters, costs = self.find_optimal_k(features_df)
            
            # Entrenar modelo
            model, predictions = self.train_clustering(
                features_df,
                n_clusters,
                method='kmeans'
            )
            
            # Analizar clusters 
            cluster_analysis = self.analyze_clusters(
                df,
                predictions,
                feature_names
            )
            
            # Generar visualizaciones
            self.visualize_clusters(df, predictions)
            
            # Guardar resultados
            results = {
                'n_clusters': n_clusters,
                'silhouette_score': float(predictions['prediction'].value_counts().mean()),
                'cluster_analysis': cluster_analysis,
                'temporal_patterns': {
                    str(cluster): {
                        'peak_hours': df[df['cluster'] == cluster]['hour'].mode().tolist(),
                        'weekend_ratio': float(df[df['cluster'] == cluster]['weekend_ratio'].mean())
                    }
                    for cluster in range(n_clusters)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'clustering_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("✅ Análisis de clustering completado")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en clustering: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar análisis de clustering."""
    clustering = BikeStationClustering()
    results = clustering.process()
    
    logger.info("\n📊 Resumen de clusters:")
    for cluster, stats in results['cluster_analysis'].items():
        logger.info(f"\n{cluster}:")
        logger.info(f"- Tamaño: {stats['size']} estaciones ({stats['percentage']:.1f}%)")
        logger.info(f"- Viajes promedio: {stats['avg_trips']:.1f}")
        logger.info(f"- Duración promedio: {stats['avg_duration']:.1f} segundos")
        logger.info(f"- Ratio hora pico: {stats['peak_usage']['rush_hour_ratio']:.2f}")
        logger.info(f"- Balance entrada/salida: {stats['balance']['inbound_outbound_ratio']:.2f}")

if __name__ == "__main__":
    main()