import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark
from sklearn.model_selection import train_test_split
import pulp
import geopandas as gpd
import json
from datetime import datetime
import plotly.express as px
import shap
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeSharingPredictor:
    """Predicción y optimización para sistema de bike-sharing."""
    
    def __init__(
        self,
        input_dir: str = "data/ml_ready",
        output_dir: str = "data/predictions",
        h3_resolution: int = 9,
        experiment_name: str = "bike_sharing_predictions"
    ):
        """
        Inicializa el predictor de bike-sharing.
        
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
        """Inicializa Spark para predicciones."""
        return SparkSession.builder \
            .appName("Bike-Sharing-Predictions") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .getOrCreate()
    
    def create_demand_prediction_pipeline(
        self,
        feature_cols: List[str],
        model_type: str = 'rf'
    ) -> Pipeline:
        """
        Crea pipeline de predicción de demanda.
        
        Args:
            feature_cols: Lista de features a usar
            model_type: Tipo de modelo ('rf' o 'gbt')
            
        Returns:
            Pipeline de Spark ML
        """
        # Feature assembly
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        # Seleccionar modelo
        if model_type == 'rf':
            model = RandomForestRegressor(
                featuresCol="features",
                labelCol="trip_count",
                numTrees=100,
                maxDepth=10
            )
        else:
            model = GBTRegressor(
                featuresCol="features",
                labelCol="trip_count",
                maxIter=100
            )
        
        return Pipeline(stages=[assembler, model])
    
    def optimize_rebalancing(
        self,
        current_inventory: pd.DataFrame,
        demand_forecast: pd.DataFrame,
        n_vehicles: int = 10
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Optimiza operaciones de rebalanceo usando demanda pronosticada.
        
        Args:
            current_inventory: Inventario actual por estación
            demand_forecast: Demanda pronosticada por estación
            n_vehicles: Número de vehículos disponibles
            
        Returns:
            Plan de rebalanceo por vehículo
        """
        logger.info("🔄 Optimizando plan de rebalanceo")
        
        # Crear problema de optimización
        model = pulp.LpProblem("BikeRebalancing", pulp.LpMinimize)
        
        # Obtener lista de estaciones
        stations = current_inventory['station_id'].unique()
        
        # Variables de decisión
        pickup = pulp.LpVariable.dicts(
            "pickup",
            ((v, s) for v in range(n_vehicles) for s in stations),
            lowBound=0,
            cat='Integer'
        )
        
        dropoff = pulp.LpVariable.dicts(
            "dropoff",
            ((v, s) for v in range(n_vehicles) for s in stations),
            lowBound=0,
            cat='Integer'
        )
        
        # Función objetivo: minimizar distancia total recorrida
        # TODO: Implementar cálculo de distancias entre estaciones
        model += pulp.lpSum([
            pickup[v,s] + dropoff[v,s]
            for v in range(n_vehicles)
            for s in stations
        ])
        
        # Restricciones
        for s in stations:
            # Balance de inventario
            current = current_inventory[current_inventory['station_id'] == s]['bikes'].values[0]
            target = demand_forecast[demand_forecast['station_id'] == s]['predicted_demand'].values[0]
            
            model += (
                current +
                pulp.lpSum([dropoff[v,s] for v in range(n_vehicles)]) -
                pulp.lpSum([pickup[v,s] for v in range(n_vehicles)]) ==
                target
            )
        
        # Resolver
        model.solve()
        
        # Extraer solución
        solution = {
            f'vehicle_{v}': [
                (s, pickup[v,s].value())
                for s in stations
                if pickup[v,s].value() > 0
            ]
            for v in range(n_vehicles)
        }
        
        logger.info("✅ Plan de rebalanceo optimizado")
        return solution
    
    def recommend_station_locations(
        self,
        existing_stations: gpd.GeoDataFrame,
        demand_heatmap: gpd.GeoDataFrame,
        constraints: gpd.GeoDataFrame,
        n_recommendations: int = 10
    ) -> gpd.GeoDataFrame:
        """
        Recomienda ubicaciones para nuevas estaciones.
        
        Args:
            existing_stations: Estaciones existentes
            demand_heatmap: Heatmap de demanda por H3
            constraints: Restricciones geográficas
            n_recommendations: Número de recomendaciones
            
        Returns:
            GeoDataFrame con ubicaciones recomendadas
        """
        logger.info("🔄 Generando recomendaciones de ubicación")
        
        # Crear grid de ubicaciones candidatas
        candidates = self._create_candidate_grid(demand_heatmap)
        
        # Filtrar por restricciones
        valid_candidates = gpd.sjoin(
            candidates,
            constraints,
            how='inner',
            predicate='within'
        )
        
        # Calcular scores
        scores = self._score_locations(
            valid_candidates,
            existing_stations,
            demand_heatmap
        )
        
        # Seleccionar mejores ubicaciones
        recommendations = scores.nlargest(n_recommendations, 'score')
        
        logger.info(f"✅ Generadas {len(recommendations)} recomendaciones")
        return recommendations
    
    def _create_candidate_grid(
        self,
        demand_heatmap: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Crea grid de ubicaciones candidatas."""
        # Obtener bounds del área
        bounds = demand_heatmap.total_bounds
        
        # Crear grid regular
        x_range = np.arange(bounds[0], bounds[2], 0.01)
        y_range = np.arange(bounds[1], bounds[3], 0.01)
        
        grid_points = []
        for x in x_range:
            for y in y_range:
                grid_points.append({
                    'geometry': Point(x, y),
                    'h3_index': h3.geo_to_h3(y, x, self.h3_resolution)
                })
        
        return gpd.GeoDataFrame(grid_points)
    
    def _score_locations(
        self,
        candidates: gpd.GeoDataFrame,
        existing_stations: gpd.GeoDataFrame,
        demand_heatmap: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Calcula scores para ubicaciones candidatas."""
        # Calcular distancia a estaciones existentes
        candidates['min_distance'] = candidates.geometry.apply(
            lambda x: existing_stations.distance(x).min()
        )
        
        # Obtener demanda del heatmap
        candidates = candidates.merge(
            demand_heatmap[['h3_index', 'predicted_demand']],
            on='h3_index',
            how='left'
        )
        
        # Calcular score combinado
        candidates['score'] = (
            candidates['predicted_demand'] * 
            candidates['min_distance'].clip(0.1, None)  # Evitar divisiones por cero
        )
        
        return candidates
    
    def process(self) -> Dict:
        """Ejecuta pipeline completo de predicción y optimización."""
        try:
            logger.info("🚀 Iniciando pipeline de predicción")
            
            # 1. Cargar datos
            features = pd.read_parquet(
                self.input_dir / f"bike_features_h3_{self.h3_resolution}.parquet"
            )
            
            # 2. Entrenar modelo de demanda
            pipeline = self.create_demand_prediction_pipeline(
                [c for c in features.columns if c.endswith('_norm')]
            )
            
            with mlflow.start_run():
                # Entrenar
                model = pipeline.fit(features)
                
                # Generar predicciones
                predictions = model.transform(features)
                
                # Evaluar
                evaluator = RegressionEvaluator(
                    labelCol="trip_count",
                    predictionCol="prediction",
                    metricName="rmse"
                )
                rmse = evaluator.evaluate(predictions)
                
                # Log métricas
                mlflow.log_metric("rmse", rmse)
            
            # 3. Optimizar rebalanceo
            rebalancing_plan = self.optimize_rebalancing(
                current_inventory=features[['station_id', 'current_bikes']],
                demand_forecast=predictions[['station_id', 'prediction']]
            )
            
            # 4. Recomendar nuevas ubicaciones
            recommendations = self.recommend_station_locations(
                existing_stations=gpd.GeoDataFrame(
                    features,
                    geometry=gpd.points_from_xy(
                        features.longitude,
                        features.latitude
                    )
                ),
                demand_heatmap=gpd.GeoDataFrame(
                    predictions,
                    geometry='geometry'
                ),
                constraints=gpd.read_file(
                    self.input_dir / "constraints.geojson"
                )
            )
            
            # Guardar resultados
            results = {
                'model_performance': {
                    'rmse': rmse,
                    'feature_importance': dict(
                        zip(
                            features.columns,
                            model.stages[-1].featureImportances
                        )
                    )
                },
                'rebalancing_plan': rebalancing_plan,
                'station_recommendations': recommendations.to_dict('records'),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'prediction_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("✅ Pipeline de predicción completado")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

def main():
    """Función principal para ejecutar predicciones."""
    predictor = BikeSharingPredictor()
    results = predictor.process()
    
    logger.info("\n📊 Resumen de resultados:")
    logger.info(f"RMSE: {results['model_performance']['rmse']:.2f}")
    logger.info("\nTop 5 features más importantes:")
    for feature, importance in sorted(
        results['model_performance']['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:
        logger.info(f"- {feature}: {importance:.3f}")
    
    logger.info("\nPlan de rebalanceo generado para {len(results['rebalancing_plan'])} vehículos")
    logger.info(f"Nuevas ubicaciones recomendadas: {len(results['station_recommendations'])}")

if __name__ == "__main__":
    main()