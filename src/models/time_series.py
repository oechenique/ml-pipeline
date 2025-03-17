import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, date_format, window, avg, stddev,
    sum, count, when, explode, sequence, to_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    TimestampType, DoubleType, IntegerType
)
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BikeTimeSeriesAnalyzer:
    """Análisis y predicción de series temporales para bike-sharing."""
    
    def __init__(
        self,
        input_dir: str = "data/processed",
        output_dir: str = "data/time_series",
        experiment_name: str = "bike_time_series"
    ):
        """
        Inicializa el analizador de series temporales.
        
        Args:
            input_dir: Directorio con datos procesados
            output_dir: Directorio para resultados
            experiment_name: Nombre del experimento MLflow
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar MLflow
        mlflow.set_experiment(experiment_name)
        
        # Inicializar Spark
        self.spark = self._init_spark()
    
    def _init_spark(self) -> SparkSession:
        """Inicializa Spark con configuración optimizada."""
        return SparkSession.builder \
            .appName("Bike-Time-Series") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
    
    def prepare_time_series(
        self,
        trips_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        events_df: Optional[pd.DataFrame] = None,
        resample_freq: str = '1H'
    ) -> pd.DataFrame:
        """
        Prepara datos para análisis de series temporales.
        
        Args:
            trips_df: DataFrame con viajes
            weather_df: DataFrame opcional con datos climáticos
            events_df: DataFrame opcional con eventos
            resample_freq: Frecuencia de remuestreo
            
        Returns:
            DataFrame preparado para series temporales
        """
        logger.info("🔄 Preparando series temporales")
        
        # Agrupar viajes por tiempo
        ts_df = trips_df.set_index('start_time') \
            .groupby('start_station_id') \
            .resample(resample_freq)['trip_id'] \
            .count() \
            .reset_index() \
            .rename(columns={'trip_id': 'trip_count'})
        
        # Agregar variables temporales
        ts_df['hour'] = ts_df['start_time'].dt.hour
        ts_df['day_of_week'] = ts_df['start_time'].dt.dayofweek
        ts_df['month'] = ts_df['start_time'].dt.month
        ts_df['is_weekend'] = ts_df['day_of_week'].isin([5, 6]).astype(int)
        ts_df['is_holiday'] = self._is_holiday(ts_df['start_time'])
        
        # Agregar clima si está disponible
        if weather_df is not None:
            ts_df = pd.merge_asof(
                ts_df,
                weather_df,
                left_on='start_time',
                right_on='timestamp',
                direction='nearest'
            )
        
        # Agregar eventos si están disponibles
        if events_df is not None:
            ts_df = self._add_event_features(ts_df, events_df)
        
        logger.info(f"✅ Series preparadas: {len(ts_df)} registros")
        return ts_df
    
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Identifica días festivos."""
        # TODO: Implementar calendario de festivos
        return pd.Series(0, index=dates.index)
    
    def _add_event_features(
        self,
        ts_df: pd.DataFrame,
        events_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Agrega features de eventos."""
        # Merge con ventana de tiempo
        events_df['event_window_start'] = events_df['start_time'] - timedelta(hours=1)
        events_df['event_window_end'] = events_df['end_time'] + timedelta(hours=1)
        
        # Para cada timestamp, ver si hay eventos activos
        ts_df['has_event'] = ts_df['start_time'].apply(
            lambda x: events_df[
                (events_df['event_window_start'] <= x) &
                (events_df['event_window_end'] >= x)
            ].shape[0] > 0
        ).astype(int)
        
        return ts_df
    
    def train_prophet_model(
        self,
        ts_df: pd.DataFrame,
        station_id: str,
        add_regressors: bool = True
    ) -> Tuple[Prophet, Dict[str, float]]:
        """
        Entrena modelo Prophet para una estación.
        
        Args:
            ts_df: DataFrame con series temporales
            station_id: ID de la estación
            add_regressors: Si agregar regresores adicionales
            
        Returns:
            Modelo entrenado y métricas
        """
        # Preparar datos para Prophet
        station_data = ts_df[ts_df['start_station_id'] == station_id].copy()
        prophet_df = pd.DataFrame({
            'ds': station_data['start_time'],
            'y': station_data['trip_count']
        })
        
        # Configurar modelo
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        
        # Agregar regresores
        if add_regressors:
            if 'temperature' in station_data.columns:
                model.add_regressor('temperature')
                prophet_df['temperature'] = station_data['temperature']
            
            if 'has_event' in station_data.columns:
                model.add_regressor('has_event')
                prophet_df['has_event'] = station_data['has_event']
        
        # Entrenar y evaluar
        with mlflow.start_run(run_name=f"prophet_{station_id}"):
            # Entrenar
            model.fit(prophet_df)
            
            # Cross-validation
            cv_results = cross_validation(
                model,
                initial='30 days',
                period='7 days',
                horizon='7 days'
            )
            
            # Calcular métricas
            metrics = performance_metrics(cv_results)
            
            # Log métricas
            mlflow.log_metrics({
                'rmse': metrics['rmse'].mean(),
                'mae': metrics['mae'].mean(),
                'mape': metrics['mape'].mean()
            })
            
            # Log modelo
            mlflow.prophet.log_model(model, "model")
        
        return model, metrics.to_dict('records')
    
    def train_xgboost_model(
        self,
        ts_df: pd.DataFrame,
        station_id: str,
        forecast_horizon: int = 24
    ) -> Tuple[xgb.Booster, Dict[str, float]]:
        """
        Entrena modelo XGBoost para una estación.
        
        Args:
            ts_df: DataFrame con series temporales
            station_id: ID de la estación
            forecast_horizon: Horizonte de predicción en horas
            
        Returns:
            Modelo entrenado y métricas
        """
        # Preparar datos
        station_data = ts_df[ts_df['start_station_id'] == station_id].copy()
        
        # Crear features con lag
        for i in [1, 2, 3, 6, 12, 24]:
            station_data[f'lag_{i}h'] = station_data['trip_count'].shift(i)
        
        # Crear features con rolling statistics
        for window in [3, 6, 12, 24]:
            station_data[f'rolling_mean_{window}h'] = station_data['trip_count'].rolling(window).mean()
            station_data[f'rolling_std_{window}h'] = station_data['trip_count'].rolling(window).std()
        
        # Preparar features y target
        feature_cols = [c for c in station_data.columns if 'lag_' in c or 'rolling_' in c]
        if 'temperature' in station_data.columns:
            feature_cols.append('temperature')
        
        # Split train/test
        train_size = len(station_data) - forecast_horizon
        train_data = station_data.iloc[:train_size]
        test_data = station_data.iloc[train_size:]
        
        X_train = train_data[feature_cols].dropna()
        y_train = train_data['trip_count'].iloc[len(X_train.index)]
        X_test = test_data[feature_cols]
        y_test = test_data['trip_count']
        
        # Entrenar modelo
        with mlflow.start_run(run_name=f"xgboost_{station_id}"):
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1
            )
            
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Generar predicciones
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            metrics = {
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'mae': mean_absolute_error(y_test, y_pred)
            }
            
            # Log métricas
            mlflow.log_metrics(metrics)
            
            # Log modelo
            mlflow.xgboost.log_model(model, "model")
        
        return model, metrics
    
    def analyze_patterns(
        self,
        ts_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Analiza patrones temporales en los datos.
        
        Args:
            ts_df: DataFrame con series temporales
            
        Returns:
            Diccionario con análisis de patrones
        """
        logger.info("🔄 Analizando patrones temporales")
        
        patterns = {}
        
        # Patrones por hora
        hourly = ts_df.groupby('hour')['trip_count'].agg(['mean', 'std']).to_dict()
        patterns['hourly'] = {
            'peak_hours': sorted(
                hourly['mean'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'statistics': hourly
        }
        
        # Patrones por día de la semana
        daily = ts_df.groupby('day_of_week')['trip_count'].agg(['mean', 'std']).to_dict()
        patterns['daily'] = {
            'busiest_days': sorted(
                daily['mean'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'statistics': daily
        }
        
        # Patrones por mes
        monthly = ts_df.groupby('month')['trip_count'].agg(['mean', 'std']).to_dict()
        patterns['monthly'] = {
            'peak_months': sorted(
                monthly['mean'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'statistics': monthly
        }
        
        # Efectos especiales
        patterns['effects'] = {
            'weekend_effect': float(
                ts_df[ts_df['is_weekend'] == 1]['trip_count'].mean() /
                ts_df[ts_df['is_weekend'] == 0]['trip_count'].mean()
            )
        }
        
        if 'temperature' in ts_df.columns:
            patterns['weather'] = {
                'temperature_correlation': float(
                    ts_df['trip_count'].corr(ts_df['temperature'])
                )
            }
        
        if 'has_event' in ts_df.columns:
            patterns['events'] = {
                'event_impact': float(
                    ts_df[ts_df['has_event'] == 1]['trip_count'].mean() /
                    ts_df[ts_df['has_event'] == 0]['trip_count'].mean()
                )
            }
        
        logger.info("✅ Análisis de patrones completado")
        return patterns
    
    def visualize_patterns(
        self,
        ts_df: pd.DataFrame,
        patterns: Dict
    ) -> None:
        """
        Genera visualizaciones de patrones temporales.
        
        Args:
            ts_df: DataFrame con series temporales
            patterns: Diccionario con análisis de patrones
        """
        logger.info("🔄 Generando visualizaciones")
        
        # Crear directorio
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Patrones horarios
        fig_hourly = px.line(
            ts_df.groupby('hour')['trip_count'].mean().reset_index(),
            x='hour',
            y='trip_count',
            title='Patrón de Uso por Hora'
        )
        fig_hourly.write_html(viz_dir / 'hourly_pattern.html')
        
        # 2. Patrones semanales
        fig_weekly = px.box(
            ts_df,
            x='day_of_week',
            y='trip_count',
            title='Distribución de Viajes por Día'
        )
        fig_weekly.write_html(viz_dir / 'weekly_pattern.html')
        
        # 3. Patrones mensuales
        fig_monthly = px.line(
            ts_df.groupby('month')['trip_count'].mean().reset_index(),
            x='month',
            y='trip_count',
            title='Patrón de Uso por Mes'
        )
        fig_monthly.write_html(viz_dir / 'monthly_pattern.html')
        
        # 4. Efectos del clima y eventos
        if 'temperature' in ts_df.columns:
            fig_weather = px.scatter(
                ts_df,
                x='temperature',
                y='trip_count',
                title='Impacto de la Temperatura en el Uso'
            )
            fig_weather.write_html(viz_dir / 'weather_impact.html')
        
        if 'has_event' in ts_df.columns:
            fig_events = px.box(
                ts_df,
                x='has_event',
                y='trip_count',
                title='Impacto de Eventos en el Uso'
            )
            fig_events.write_html(viz_dir / 'event_impact.html')
        
        logger.info(f"✅ Visualizaciones guardadas en {viz_dir}")
    
    def process(self) -> Dict:
        """Ejecuta pipeline completo de análisis temporal."""
        try:
            # 1. Cargar y preparar datos
            trips_df = pd.read_parquet(self.input_dir / "bike_sharing/trips.parquet")
            weather_df = pd.read_parquet(self.input_dir / "weather/weather.parquet")
            events_df = pd.read_parquet(self.input_dir / "events/events.parquet")
            
            ts_df = self.prepare_time_series(
                trips_df,
                weather_df,
                events_df
            )
            
            # 2. Analizar patrones
            patterns = self.analyze_patterns(ts_df)
            
            # 3. Entrenar modelos por estación
            station_predictions = {}
            for station_id in ts_df['start_station_id'].unique():
                # Prophet para tendencias y estacionalidad
                prophet_model, prophet_metrics = self.train_prophet_model(
                    ts_df,
                    station_id
                )
                
                # XGBoost para predicciones a corto plazo
                xgb_model, xgb_metrics = self.train_xgboost_model(
                    ts_df,
                    station_id
                )
                
                station_predictions[station_id] = {
                    'prophet': prophet_metrics,
                    'xgboost': xgb_metrics
                }
            
            # 4. Generar visualizaciones
            self.visualize_patterns(ts_df, patterns)
            
            # 5. Guardar resultados
            results = {
                'temporal_patterns': patterns,
                'station_predictions': station_predictions,
                'model_performance': {
                    'prophet_avg_rmse': np.mean([
                        p['prophet']['rmse']
                        for p in station_predictions.values()
                    ]),
                    'xgboost_avg_rmse': np.mean([
                        p['xgboost']['rmse']
                        for p in station_predictions.values()
                    ])
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / 'time_series_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info("✅ Análisis temporal completado")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en análisis temporal: {str(e)}")
            raise
        
        finally:
            self.spark.stop()

    def main():
        """Función principal para ejecutar análisis temporal."""
        analyzer = BikeTimeSeriesAnalyzer()
        results = analyzer.process()
        
        logger.info("\n📊 Resumen de resultados:")
        logger.info("Patrones temporales:")
        logger.info(f"- Horas pico: {results['temporal_patterns']['hourly']['peak_hours']}")
        logger.info(f"- Efecto fin de semana: {results['temporal_patterns']['effects']['weekend_effect']:.2f}x")
        
        if 'weather' in results['temporal_patterns']:
            logger.info(f"- Correlación con temperatura: {results['temporal_patterns']['weather']['temperature_correlation']:.2f}")
        
        logger.info("\nRendimiento de modelos:")
        logger.info(f"- Prophet RMSE promedio: {results['model_performance']['prophet_avg_rmse']:.2f}")
        logger.info(f"- XGBoost RMSE promedio: {results['model_performance']['xgboost_avg_rmse']:.2f}")

if __name__ == "__main__":
    main()