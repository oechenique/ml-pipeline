from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import json
from pathlib import Path
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_cross_validation_metric
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from airflow import DAG
from airflow.operators.python import PythonOperator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_time_series',
    default_args=default_args,
    description='Time series analysis for bike-sharing',
    schedule_interval=None,  # Triggered by master DAG
    catchup=False,
)

def run_time_series_analysis():
    """
    Run time series analysis directly using real data from PostgreSQL.
    """
    try:
        print("Starting time series analysis with real data from PostgreSQL")
        
        # Connection configuration
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        # 1. Extract historical trip data from PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        print("Extracting historical trip data for time series analysis...")
        
        # Check available data
        count_query = """
        SELECT COUNT(*), MIN(start_time), MAX(start_time) FROM bike_trips
        """
        cursor.execute(count_query)
        result = cursor.fetchone()
        total_trips, min_date, max_date = result
        
        print(f"Total trips in database: {total_trips:,}")
        print(f"Date range: {min_date} to {max_date}")
        
        # Query to get daily and hourly aggregates for time series
        # This uses all available data
        ts_query = """
        -- Daily trip counts for overall trend analysis
        WITH daily_trips AS (
            SELECT 
                date_trunc('day', start_time) as trip_date,
                COUNT(*) as trip_count,
                COUNT(DISTINCT start_station_id) as active_stations,
                AVG(duration_sec) as avg_duration
            FROM bike_trips
            GROUP BY date_trunc('day', start_time)
            ORDER BY date_trunc('day', start_time)
        ),
        -- Hourly patterns for daily seasonality
        hourly_patterns AS (
            SELECT 
                EXTRACT(DOW FROM start_time) as day_of_week,
                EXTRACT(HOUR FROM start_time) as hour_of_day,
                COUNT(*) as trip_count
            FROM bike_trips
            GROUP BY 
                EXTRACT(DOW FROM start_time),
                EXTRACT(HOUR FROM start_time)
            ORDER BY 
                EXTRACT(DOW FROM start_time),
                EXTRACT(HOUR FROM start_time)
        ),
        -- Weekly patterns for weekly seasonality
        weekly_patterns AS (
            SELECT 
                date_trunc('week', start_time) as week_start,
                COUNT(*) as trip_count
            FROM bike_trips
            GROUP BY date_trunc('week', start_time)
            ORDER BY date_trunc('week', start_time)
        ),
        -- Monthly patterns for monthly/seasonal trends
        monthly_patterns AS (
            SELECT 
                date_trunc('month', start_time) as month_start,
                COUNT(*) as trip_count,
                COUNT(DISTINCT start_station_id) as active_stations
            FROM bike_trips
            GROUP BY date_trunc('month', start_time)
            ORDER BY date_trunc('month', start_time)
        )
        -- Return all aggregated data
        SELECT 
            'daily' as aggregation_type,
            trip_date as date,
            trip_count,
            active_stations,
            avg_duration,
            NULL::float as day_of_week,
            NULL::float as hour_of_day,
            NULL::timestamp as week_start,
            NULL::timestamp as month_start
        FROM daily_trips
        
        UNION ALL
        
        SELECT 
            'hourly' as aggregation_type,
            NULL as date,
            trip_count,
            NULL as active_stations,
            NULL as avg_duration,
            NULL::float as day_of_week,
            NULL::float as hour_of_day,
            NULL::timestamp as week_start,
            NULL::timestamp as month_start
        FROM hourly_patterns
        
        UNION ALL
        
        SELECT 
            'weekly' as aggregation_type,
            NULL as date,
            trip_count,
            NULL as active_stations,
            NULL as avg_duration,
            NULL::float as day_of_week,
            NULL::float as hour_of_day,
            week_start,
            NULL::timestamp as month_start
        FROM weekly_patterns
        
        UNION ALL
        
        SELECT 
            'monthly' as aggregation_type,
            NULL as date,
            trip_count,
            active_stations,
            NULL as avg_duration,
            NULL::float as day_of_week,
            NULL::float as hour_of_day,
            NULL::timestamp as week_start,
            month_start
        FROM monthly_patterns
        """
        
        cursor.execute(ts_query)
        ts_rows = cursor.fetchall()
        
        if not ts_rows:
            raise ValueError("No time series data found in bike_trips")
            
        print(f"Retrieved {len(ts_rows)} time series data points")
        
        # Create DataFrame with correct column names
        column_names = [
            'aggregation_type', 'date', 'trip_count', 'active_stations', 
            'avg_duration', 'day_of_week', 'hour_of_day', 'week_start', 'month_start'
        ]
        
        ts_df = pd.DataFrame(ts_rows, columns=column_names)
        
        # Separate dataframes by aggregation type
        daily_df = ts_df[ts_df['aggregation_type'] == 'daily'].copy()
        hourly_df = ts_df[ts_df['aggregation_type'] == 'hourly'].copy()
        weekly_df = ts_df[ts_df['aggregation_type'] == 'weekly'].copy()
        monthly_df = ts_df[ts_df['aggregation_type'] == 'monthly'].copy()
        
        print(f"Data separated into: {len(daily_df)} daily records, {len(hourly_df)} hourly patterns, " 
              f"{len(weekly_df)} weekly records, {len(monthly_df)} monthly records")
        
        # Close database connection
        cursor.close()
        conn.close()

# 2. Statistical analysis of time series data
        print("Performing statistical analysis of time series data...")
        
        # Create directory for results
        output_dir = Path("/app/data/time_series")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 2.1. Daily trend analysis
        if len(daily_df) > 0:
            # Sort by date for time series analysis
            daily_df.sort_values('date', inplace=True)
            
            # Check for stationarity (Augmented Dickey-Fuller test)
            adf_result = adfuller(daily_df['trip_count'].dropna())
            is_stationary = adf_result[1] < 0.05  # p-value less than 0.05 means stationary
            
            # Seasonal decomposition to identify trend, seasonal, and residual components
            if len(daily_df) >= 14:  # Need at least 2x period length
                # Use a 7-day period for weekly seasonality
                try:
                    result = seasonal_decompose(daily_df['trip_count'], model='additive', period=7)
                    
                    # Create plots
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                    
                    result.observed.plot(ax=ax1)
                    ax1.set_title('Observed')
                    ax1.set_ylabel('Trip Count')
                    
                    result.trend.plot(ax=ax2)
                    ax2.set_title('Trend')
                    ax2.set_ylabel('Trip Count')
                    
                    result.seasonal.plot(ax=ax3)
                    ax3.set_title('Seasonal (Weekly)')
                    ax3.set_ylabel('Trip Count')
                    
                    result.resid.plot(ax=ax4)
                    ax4.set_title('Residual')
                    ax4.set_ylabel('Trip Count')
                    
                    plt.tight_layout()
                    
                    # Save the figure
                    decomposition_path = output_dir / "daily_decomposition.png"
                    plt.savefig(decomposition_path)
                    plt.close()
                    
                    print(f"Daily decomposition saved to {decomposition_path}")
                except Exception as e:
                    print(f"Could not perform seasonal decomposition: {str(e)}")
            
            # Store daily time series data for Prophet
            daily_prophet_df = daily_df[['date', 'trip_count']].rename(columns={'date': 'ds', 'trip_count': 'y'})
            daily_prophet_df.to_csv(output_dir / "daily_data_for_prophet.csv", index=False)
        
        # 2.2. Hourly pattern analysis
        if len(hourly_df) > 0:
            # Create heatmap data
            try:
                pivot_df = hourly_df.pivot_table(
                    index='day_of_week', 
                    columns='hour_of_day', 
                    values='trip_count',
                    aggfunc='mean'
                ).fillna(0)
                
                # Verificar que tenemos datos para los 7 días y 24 horas
                if not pivot_df.empty and len(pivot_df.columns) > 0:
                    # Asegurar que todos los días de la semana están presentes
                    for day in range(7):
                        if day not in pivot_df.index:
                            pivot_df.loc[day] = [0] * len(pivot_df.columns)
                    pivot_df = pivot_df.sort_index()
                else:
                    # Si no hay datos suficientes, crear un DataFrame base con ceros
                    print("No se encontraron suficientes datos para el heatmap, creando base vacía")
                    pivot_df = pd.DataFrame(
                        np.zeros((7, 24)),
                        index=range(7),
                        columns=range(24)
                    )
                
                # Create and save heatmap with Plotly
                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Hour of Day", y="Day of Week (0=Sunday)", color="Avg. Trip Count"),
                    x=[str(h) for h in range(24)],
                    y=[str(d) for d in range(7)],
                    title="Weekly Hourly Heat Map of Trip Counts",
                    color_continuous_scale="Viridis"
                )
                
                fig.update_layout(
                    width=800,
                    height=600,
                    xaxis=dict(tickmode='linear'),
                    yaxis=dict(tickmode='linear')
                )
                
                # Save as HTML (interactive)
                heatmap_path = output_dir / "hourly_heatmap.html"
                fig.write_html(str(heatmap_path))
                
                # Also save as image (static)
                fig.write_image(str(output_dir / "hourly_heatmap.png"))
                
                print(f"Hourly heatmap saved to {heatmap_path}")
            
                # Calculate peak hours
                hourly_avg = hourly_df.groupby('hour_of_day')['trip_count'].mean()
                peak_hours = hourly_avg.nlargest(3).index.tolist()
                
                # Calculate peak days
                daily_avg = hourly_df.groupby('day_of_week')['trip_count'].mean()
                peak_days = daily_avg.nlargest(3).index.tolist()
                
                day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                
                hourly_stats = {
                    "peak_hours": peak_hours,
                    "peak_days": [day_names[int(day)] for day in peak_days],
                    "busiest_hours_by_day": {
                        day_names[int(day)]: int(hourly_df[hourly_df['day_of_week'] == day]['trip_count'].idxmax()) 
                        for day in range(7)
                    }
                }
            
                # Save hourly stats as JSON
                with open(output_dir / "hourly_stats.json", 'w') as f:
                    json.dump(hourly_stats, f, indent=2)
        
            except Exception as e:
                print(f"Error creating hourly heatmap: {str(e)}")
        
        # 2.3. Weekly patterns analysis
        if len(weekly_df) > 0:
            # Format week dates for better visibility
            weekly_df['week_label'] = weekly_df['week_start'].dt.strftime('%Y-%m-%d')
            
            # Create line chart
            fig = px.line(
                weekly_df,
                x='week_label',
                y='trip_count',
                title="Weekly Trend of Trip Counts",
                labels={'week_label': 'Week Starting', 'trip_count': 'Trip Count'}
            )
            
            # Improve layout
            fig.update_layout(
                width=900,
                height=400,
                xaxis=dict(tickangle=45)
            )
            
            # Save as HTML
            weekly_path = output_dir / "weekly_trend.html"
            fig.write_html(str(weekly_path))
            
            # Also save as image
            fig.write_image(str(output_dir / "weekly_trend.png"))
            
            print(f"Weekly trend chart saved to {weekly_path}")
        
        # 2.4. Monthly patterns analysis
        if len(monthly_df) > 0:
            # Format month dates
            monthly_df['month_label'] = monthly_df['month_start'].dt.strftime('%Y-%m')
            
            # Create dual-axis chart (trips and active stations)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add trip counts
            fig.add_trace(
                go.Bar(
                    x=monthly_df['month_label'],
                    y=monthly_df['trip_count'],
                    name="Trip Count",
                    marker_color='royalblue'
                ),
                secondary_y=False
            )
            
            # Add active stations
            fig.add_trace(
                go.Scatter(
                    x=monthly_df['month_label'],
                    y=monthly_df['active_stations'],
                    name="Active Stations",
                    marker_color='firebrick',
                    mode='lines+markers'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title_text="Monthly Trend of Trips and Active Stations",
                width=900,
                height=500,
                xaxis=dict(tickangle=45)
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Trip Count", secondary_y=False)
            fig.update_yaxes(title_text="Active Stations", secondary_y=True)
            
            # Save as HTML
            monthly_path = output_dir / "monthly_trend.html"
            fig.write_html(str(monthly_path))
            
            # Also save as image
            fig.write_image(str(output_dir / "monthly_trend.png"))
            
            print(f"Monthly trend chart saved to {monthly_path}")
            
        # 3. Create summary statistics
        time_series_summary = {
            "data_coverage": {
                "start_date": min_date.strftime('%Y-%m-%d') if min_date else None,
                "end_date": max_date.strftime('%Y-%m-%d') if max_date else None,
                "total_trips": int(total_trips) if total_trips else 0,
                "daily_records": len(daily_df),
                "hourly_patterns": len(hourly_df) // 24 if len(hourly_df) > 0 else 0,
                "weekly_records": len(weekly_df),
                "monthly_records": len(monthly_df)
            },
            "stationarity": {
                "is_stationary": bool(is_stationary),
                "adf_pvalue": float(adf_result[1]) if 'adf_result' in locals() else None
            },
            "patterns": {
                "peak_hours": hourly_stats["peak_hours"] if 'hourly_stats' in locals() else [],
                "peak_days": hourly_stats["peak_days"] if 'hourly_stats' in locals() else [],
                "has_weekly_seasonality": True if len(daily_df) >= 14 else False
            }
        }
        
        # Save summary as JSON
        with open(output_dir / "time_series_summary.json", 'w') as f:
            json.dump(time_series_summary, f, indent=2)
            
        print(f"Time series summary saved to {output_dir / 'time_series_summary.json'}")

        # 4. Prophet forecasting model
        print("Training Prophet forecasting model...")
        
        if len(daily_df) >= 14:  # Need at least 2 weeks for meaningful forecasting
            # Prophet requires 'ds' and 'y' columns
            prophet_df = daily_df[['date', 'trip_count']].rename(columns={'date': 'ds', 'trip_count': 'y'})
            
            # Add potential holiday effects
            holidays = pd.DataFrame({
                'holiday': 'weekend',
                'ds': prophet_df['ds'][prophet_df['ds'].dt.dayofweek >= 5],
                'lower_window': 0,
                'upper_window': 1
            })
            
            # Configure and train Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                holidays=holidays,
                seasonality_mode='multiplicative'  # Usually better for demand patterns
            )
            
            # Add additional regressors if available
            if 'active_stations' in daily_df.columns and daily_df['active_stations'].notna().all():
                model.add_regressor('active_stations')
                prophet_df['active_stations'] = daily_df['active_stations'].values
            
            # Fit the model
            model.fit(prophet_df)
            
            # Create future dataframe for prediction (90 days)
            future = model.make_future_dataframe(periods=90)
            
            # Add regressor values for future dates if needed
            if 'active_stations' in prophet_df.columns:
                # Use mean value for prediction
                future['active_stations'] = prophet_df['active_stations'].mean()
            
            # Make predictions
            forecast = model.predict(future)
            
            # Create forecast visualization
            fig_forecast = model.plot(forecast)
            plt.title('Prophet Forecast for Trip Demand')
            plt.tight_layout()
            
            # Save forecast plot
            forecast_path = output_dir / "prophet_forecast.png"
            plt.savefig(forecast_path)
            plt.close()
            
            # Create components visualization
            fig_components = model.plot_components(forecast)
            plt.tight_layout()
            
            # Save components plot
            components_path = output_dir / "prophet_components.png"
            plt.savefig(components_path)
            plt.close()
            
            # Evaluate model on historical data
            # Use cross-validation with expanding windows
            from prophet.diagnostics import cross_validation, performance_metrics
            
            # Calculate initial training period based on data size
            initial = int(len(prophet_df) * 0.5)  # Use 50% of data for initial training
            if initial < 14:
                initial = 14  # Minimum 2 weeks for initial training
                
            # Cross-validation
            try:
                cv_results = cross_validation(
                    model=model,
                    initial=f"{initial} days",
                    period=f"7 days",
                    horizon=f"14 days"
                )
                
                # Calculate performance metrics
                cv_metrics = performance_metrics(cv_results)
                
                # Create metrics visualization
                fig_metrics = plot_cross_validation_metric(cv_results, metric='mape')
                plt.title('Prophet Model MAPE (Mean Absolute Percentage Error)')
                plt.tight_layout()
                
                # Save metrics plot
                metrics_path = output_dir / "prophet_cv_metrics.png"
                plt.savefig(metrics_path)
                plt.close()
                
                print(f"Prophet model evaluation metrics:")
                print(f"  - MAE: {cv_metrics['mae'].mean():.2f}")
                print(f"  - RMSE: {cv_metrics['rmse'].mean():.2f}")
                print(f"  - MAPE: {cv_metrics['mape'].mean():.2f}%")
                
                # Save forecast data to CSV for future use
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
                    output_dir / "prophet_forecast.csv", 
                    index=False
                )
                
                # Save cross-validation results
                cv_results.to_csv(output_dir / "prophet_cv_results.csv", index=False)
                cv_metrics.to_csv(output_dir / "prophet_cv_metrics.csv", index=False)
                
                # Store predictions in PostgreSQL
                print("Storing Prophet predictions in PostgreSQL...")
                
                # Connect to PostgreSQL
                conn = psycopg2.connect(**conn_params)
                cursor = conn.cursor()
                
                # Create table if it doesn't exist
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS bike_demand_forecasts (
                    id SERIAL PRIMARY KEY,
                    forecast_date DATE NOT NULL,
                    forecast_value FLOAT NOT NULL,
                    lower_bound FLOAT,
                    upper_bound FLOAT,
                    model_type VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Index for faster lookups
                CREATE INDEX IF NOT EXISTS idx_bike_demand_forecasts_date 
                ON bike_demand_forecasts(forecast_date);
                """)
                
                # Format forecast data for insertion
                forecast_data = []
                future_dates = forecast[forecast['ds'] > max_date][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                
                for _, row in future_dates.iterrows():
                    forecast_data.append((
                        row['ds'].date(),
                        float(row['yhat']),
                        float(row['yhat_lower']),
                        float(row['yhat_upper']),
                        'prophet'
                    ))
                
                # Clear previous forecasts
                cursor.execute("DELETE FROM bike_demand_forecasts WHERE model_type = 'prophet'")
                
                # Insert new forecasts
                from psycopg2.extras import execute_values
                execute_values(
                    cursor,
                    """
                    INSERT INTO bike_demand_forecasts
                    (forecast_date, forecast_value, lower_bound, upper_bound, model_type)
                    VALUES %s
                    """,
                    forecast_data
                )
                
                # Commit changes
                conn.commit()
                cursor.close()
                conn.close()
                
                print(f"Stored {len(forecast_data)} Prophet forecasts in PostgreSQL")
                
            except Exception as e:
                print(f"Error during Prophet cross-validation: {str(e)}")
            
            print(f"Prophet forecasting completed. Results saved to {output_dir}")
            
        else:
            print("Insufficient data for Prophet forecasting (need at least 2 weeks)")
        
        # 5. XGBoost forecasting model (time-series specific approach)
        if len(daily_df) >= 30:  # Need reasonable amount of data
            try:
                print("Training XGBoost forecasting model...")
                import xgboost as xgb
                from sklearn.model_selection import TimeSeriesSplit
                
                # Prepare features for XGBoost
                # Create lag features and rolling statistics
                ts_df = daily_df.sort_values('date').copy()
                ts_df.set_index('date', inplace=True)
                
                # Create target variable
                ts_df['target'] = ts_df['trip_count']
                
                # Create lag features (previous n days)
                for lag in range(1, 8):  # Use 1 to 7 day lags
                    ts_df[f'lag_{lag}'] = ts_df['trip_count'].shift(lag)
                
                # Add rolling means for different windows
                for window in [3, 7, 14]:
                    ts_df[f'rolling_mean_{window}'] = ts_df['trip_count'].rolling(window=window).mean()
                    ts_df[f'rolling_std_{window}'] = ts_df['trip_count'].rolling(window=window).std()
                
                # Add day of week
                ts_df['dayofweek'] = ts_df.index.dayofweek
                
                # Create one-hot encoding for day of week
                ts_df = pd.get_dummies(ts_df, columns=['dayofweek'], prefix='dow')
                
                # Add month and year
                ts_df['month'] = ts_df.index.month
                ts_df['year'] = ts_df.index.year
                
                # Drop NaN values from lag features
                ts_df = ts_df.dropna()
                
                # Split features and target
                features = ts_df.drop(['target', 'trip_count', 'active_stations', 'avg_duration'], axis=1, errors='ignore')
                target = ts_df['target']
                
                # Use time series CV
                tscv = TimeSeriesSplit(n_splits=5)
                
                # Configure XGBoost model
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                
                # Train model with cross-validation
                cv_scores = []
                
                for train_idx, test_idx in tscv.split(features):
                    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                    
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    
                    cv_scores.append({
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    })
                
                # Calculate average metrics
                avg_metrics = {
                    'rmse': np.mean([score['rmse'] for score in cv_scores]),
                    'mae': np.mean([score['mae'] for score in cv_scores]),
                    'r2': np.mean([score['r2'] for score in cv_scores])
                }
                
                print(f"XGBoost time series model evaluation:")
                print(f"  - MAE: {avg_metrics['mae']:.2f}")
                print(f"  - RMSE: {avg_metrics['rmse']:.2f}")
                print(f"  - R2: {avg_metrics['r2']:.4f}")
                
                # Train final model on all data
                model.fit(features, target)
                
                # Generate future dates for forecasting
                last_date = ts_df.index.max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
                
                # Create features for future dates
                future_df = pd.DataFrame(index=future_dates)
                future_df['dayofweek'] = future_df.index.dayofweek
                future_df = pd.get_dummies(future_df, columns=['dayofweek'], prefix='dow')
                future_df['month'] = future_df.index.month
                future_df['year'] = future_df.index.year
                
                # Ensure we have all necessary columns
                for col in features.columns:
                    if col not in future_df.columns:
                        future_df[col] = 0
                
                # For lag features, use the most recent available data
                # This is a simple approach - more complex models would use recursive forecasting
                for lag in range(1, 8):
                    future_df[f'lag_{lag}'] = ts_df['trip_count'].iloc[-lag]
                
                # For rolling statistics, use the last values from historical data
                for window in [3, 7, 14]:
                    future_df[f'rolling_mean_{window}'] = ts_df[f'rolling_mean_{window}'].iloc[-1]
                    future_df[f'rolling_std_{window}'] = ts_df[f'rolling_std_{window}'].iloc[-1]
                
                # Align columns with training data
                future_features = future_df[features.columns]
                
                # Make predictions
                xgb_predictions = model.predict(future_features)
                
                # Create results dataframe
                xgb_forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': xgb_predictions
                })
                
                # Save XGBoost predictions
                xgb_forecast.to_csv(output_dir / "xgboost_forecast.csv", index=False)
                
                # Save feature importance plot
                xgb.plot_importance(model, max_num_features=15)
                plt.tight_layout()
                plt.savefig(output_dir / "xgboost_feature_importance.png")
                plt.close()
                
                # Store XGBoost predictions in PostgreSQL
                print("Storing XGBoost predictions in PostgreSQL...")
                
                # Connect to PostgreSQL
                conn = psycopg2.connect(**conn_params)
                cursor = conn.cursor()
                
                # Format forecast data for insertion
                xgb_forecast_data = []
                
                for _, row in xgb_forecast.iterrows():
                    # Apply a simple uncertainty estimate (20% interval)
                    predicted_value = float(row['yhat'])
                    lower_bound = max(0, predicted_value * 0.8)  # Ensure non-negative
                    upper_bound = predicted_value * 1.2
                    
                    xgb_forecast_data.append((
                        row['ds'].date(),
                        predicted_value,
                        lower_bound,
                        upper_bound,
                        'xgboost'
                    ))
                
                # Clear previous XGBoost forecasts
                cursor.execute("DELETE FROM bike_demand_forecasts WHERE model_type = 'xgboost'")
                
                # Insert new forecasts
                execute_values(
                    cursor,
                    """
                    INSERT INTO bike_demand_forecasts
                    (forecast_date, forecast_value, lower_bound, upper_bound, model_type)
                    VALUES %s
                    """,
                    xgb_forecast_data
                )
                
                # Commit changes
                conn.commit()
                cursor.close()
                conn.close()
                
                print(f"Stored {len(xgb_forecast_data)} XGBoost forecasts in PostgreSQL")
                
                # Create comparison visualization if Prophet is also available
                if 'forecast' in locals():
                    # Combine forecasts for visualization
                    prophet_future = forecast[forecast['ds'] > max_date][['ds', 'yhat']]
                    prophet_future = prophet_future.rename(columns={'yhat': 'prophet_forecast'})
                    
                    xgb_future = xgb_forecast.rename(columns={'yhat': 'xgboost_forecast'})
                    
                    combined = pd.merge(prophet_future, xgb_future, on='ds')
                    
                    # Create comparison plot
                    plt.figure(figsize=(12, 6))
                    plt.plot(combined['ds'], combined['prophet_forecast'], label='Prophet Forecast')
                    plt.plot(combined['ds'], combined['xgboost_forecast'], label='XGBoost Forecast')
                    plt.title('Forecast Comparison: Prophet vs XGBoost')
                    plt.xlabel('Date')
                    plt.ylabel('Predicted Trip Count')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save comparison
                    plt.savefig(output_dir / "forecast_comparison.png")
                    plt.close()
            
            except Exception as e:
                print(f"Error in XGBoost forecasting: {str(e)}")
        else:
            print("Insufficient data for XGBoost forecasting (need at least 30 days)")
        
        # 6. Generate HTML report
        print("Generating time series HTML report...")
        
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bike Sharing Time Series Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; }}
                .summary {{ background-color: #f8f9fa; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart-container {{ margin: 20px 0; }}
                .forecast {{ background-color: #e6f7ff; }}
                .patterns {{ background-color: #f0f9e8; }}
            </style>
        </head>
        <body>
            <h1>Bike Sharing Time Series Analysis</h1>
            <p>Analysis date: {datetime.now().strftime('%Y-%m-%d')}</p>
            
            <div class="section summary">
                <h2>Data Summary</h2>
                <table>
                    <tr><th>Start Date</th><td>{min_date.strftime('%Y-%m-%d') if min_date else 'N/A'}</td></tr>
                    <tr><th>End Date</th><td>{max_date.strftime('%Y-%m-%d') if max_date else 'N/A'}</td></tr>
                    <tr><th>Total Trips</th><td>{total_trips:,}</td></tr>
                    <tr><th>Daily Records</th><td>{len(daily_df)}</td></tr>
                    <tr><th>Monthly Records</th><td>{len(monthly_df)}</td></tr>
                    <tr><th>Is Stationary</th><td>{bool(is_stationary) if 'is_stationary' in locals() else 'Unknown'}</td></tr>
                </table>
            </div>
            
            <div class="section patterns">
                <h2>Temporal Patterns</h2>
                
                <h3>Hourly Patterns</h3>
                <p>Peak hours: {', '.join(str(h) for h in hourly_stats["peak_hours"]) if 'hourly_stats' in locals() else 'N/A'}</p>
                <p>Peak days: {', '.join(hourly_stats["peak_days"]) if 'hourly_stats' in locals() else 'N/A'}</p>
                <div class="chart-container">
                    <img src="hourly_heatmap.png" alt="Hourly Heatmap" style="max-width:100%;">
                </div>
                
                <h3>Daily Decomposition</h3>
                <div class="chart-container">
                    <img src="daily_decomposition.png" alt="Daily Decomposition" style="max-width:100%;">
                </div>
                
                <h3>Monthly Trends</h3>
                <div class="chart-container">
                    <img src="monthly_trend.png" alt="Monthly Trends" style="max-width:100%;">
                </div>
            </div>
            
            <div class="section forecast">
                <h2>Demand Forecast</h2>
                
                <h3>Prophet Model</h3>
                <div class="chart-container">
                    <img src="prophet_forecast.png" alt="Prophet Forecast" style="max-width:100%;">
                </div>
                <div class="chart-container">
                    <img src="prophet_components.png" alt="Prophet Components" style="max-width:100%;">
                </div>
                
                <h3>XGBoost Model</h3>
                <div class="chart-container">
                    <img src="xgboost_feature_importance.png" alt="XGBoost Feature Importance" style="max-width:100%;">
                </div>
                
                <h3>Model Comparison</h3>
                <div class="chart-container">
                    <img src="forecast_comparison.png" alt="Forecast Comparison" style="max-width:100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_dir / "time_series_report.html", 'w') as f:
            f.write(report_html)
        
        print(f"Time series HTML report saved to {output_dir / 'time_series_report.html'}")
        
        # Return success with metrics
        return {
            'status': 'success',
            'data_coverage': {
                'start_date': min_date.strftime('%Y-%m-%d') if min_date else None,
                'end_date': max_date.strftime('%Y-%m-%d') if max_date else None,
                'total_trips': int(total_trips) if total_trips else 0
            },
            'metrics': {
                'prophet_rmse': float(cv_metrics['rmse'].mean()) if 'cv_metrics' in locals() else None,
                'xgboost_rmse': avg_metrics['rmse'] if 'avg_metrics' in locals() else None
            },
            'output_dir': str(output_dir)
        }
    
    except Exception as e:
        print(f"Error in time series analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
        
# Tarea para analizar patrones temporales
time_series_analysis_task = PythonOperator(
    task_id='time_series_analysis',
    python_callable=run_time_series_analysis,
    dag=dag,
)

# Tarea para comparar resultados con predicciones
def compare_with_predictions(**kwargs):
    """
    Compare time series forecasts with demand predictions from prediction DAG.
    """
    try:
        print("Comparing time series forecasts with demand predictions...")
        
        # Connection configuration
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query to get time series forecasts
        forecast_query = """
        SELECT 
            forecast_date,
            AVG(CASE WHEN model_type = 'prophet' THEN forecast_value ELSE NULL END) as prophet_forecast,
            AVG(CASE WHEN model_type = 'xgboost' THEN forecast_value ELSE NULL END) as xgboost_forecast
        FROM bike_demand_forecasts
        GROUP BY forecast_date
        ORDER BY forecast_date
        """
        
        cursor.execute(forecast_query)
        forecast_rows = cursor.fetchall()
        
        if not forecast_rows:
            print("No time series forecasts found for comparison")
            return {
                'status': 'warning',
                'message': 'No time series forecasts available'
            }
            
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecast_rows, columns=[
            'forecast_date', 'prophet_forecast', 'xgboost_forecast'
        ])
        
        # Query to get aggregated daily demand predictions
        demand_query = """
        SELECT 
            prediction_date,
            SUM(predicted_demand) as total_predicted_demand
        FROM bike_demand_predictions
        GROUP BY prediction_date
        ORDER BY prediction_date
        """
        
        cursor.execute(demand_query)
        demand_rows = cursor.fetchall()
        
        # Close connection
        cursor.close()
        conn.close()
        
        if not demand_rows:
            print("No demand predictions found for comparison")
            return {
                'status': 'warning',
                'message': 'No demand predictions available'
            }
            
        # Create demand DataFrame
        demand_df = pd.DataFrame(demand_rows, columns=[
            'prediction_date', 'total_predicted_demand'
        ])
        
        # Merge data for comparison
        combined_df = pd.merge(
            forecast_df,
            demand_df,
            left_on='forecast_date',
            right_on='prediction_date',
            how='inner'
        )
        
        if len(combined_df) == 0:
            print("No overlapping dates between forecasts and predictions")
            return {
                'status': 'warning',
                'message': 'No overlapping dates for comparison'
            }
            
        print(f"Found {len(combined_df)} days with both forecasts and demand predictions")
        
        # Calculate correlation and difference metrics
        prophet_corr = combined_df['prophet_forecast'].corr(combined_df['total_predicted_demand'])
        xgboost_corr = combined_df['xgboost_forecast'].corr(combined_df['total_predicted_demand'])
        
        prophet_mape = np.mean(np.abs((combined_df['prophet_forecast'] - combined_df['total_predicted_demand']) / combined_df['total_predicted_demand'])) * 100
        xgboost_mape = np.mean(np.abs((combined_df['xgboost_forecast'] - combined_df['total_predicted_demand']) / combined_df['total_predicted_demand'])) * 100
        
        # Create comparison visualization
        plt.figure(figsize=(12, 6))
        plt.plot(combined_df['forecast_date'], combined_df['prophet_forecast'], label='Prophet Forecast')
        plt.plot(combined_df['forecast_date'], combined_df['xgboost_forecast'], label='XGBoost Forecast')
        plt.plot(combined_df['forecast_date'], combined_df['total_predicted_demand'], label='Station-Level Predictions')
        plt.title('Comparison: Time Series vs. Station-Level Predictions')
        plt.xlabel('Date')
        plt.ylabel('Total Trip Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create output directory
        output_dir = Path("/app/data/time_series")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save comparison plot
        comparison_path = output_dir / "forecast_vs_predictions.png"
        plt.savefig(comparison_path)
        plt.close()
        
        # Save comparison data
        combined_df.to_csv(output_dir / "forecast_vs_predictions.csv", index=False)
        
        # Create HTML report for comparison
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast vs. Predictions Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .comparison {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; background-color: #f5f5f5; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Forecast vs. Predictions Comparison</h1>
            <p>Comparison date: {datetime.now().strftime('%Y-%m-%d')}</p>
            
            <div class="comparison">
                <h2>Correlation Analysis</h2>
                <table>
                    <tr><th>Metric</th><th>Prophet</th><th>XGBoost</th></tr>
                    <tr><td>Correlation with Station Predictions</td><td>{prophet_corr:.4f}</td><td>{xgboost_corr:.4f}</td></tr>
                    <tr><td>MAPE</td><td>{prophet_mape:.2f}%</td><td>{xgboost_mape:.2f}%</td></tr>
                </table>
                
                <h2>Visual Comparison</h2>
                <div class="chart">
                    <img src="forecast_vs_predictions.png" alt="Comparison Chart" style="max-width:100%;">
                </div>
                
                <h3>Interpretation</h3>
                <p>This comparison shows how time series forecasts at the system level relate to 
                   the sum of individual station-level predictions. A high correlation indicates 
                   alignment between approaches.</p>
                <p>The {prophet_corr > xgboost_corr and 'Prophet' or 'XGBoost'} model shows better 
                   alignment with station-level predictions.</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_dir / "forecast_vs_predictions.html", 'w') as f:
            f.write(html_output)
        
        print(f"Comparison report saved to {output_dir / 'forecast_vs_predictions.html'}")
        
        return {
            'status': 'success',
            'metrics': {
                'prophet_correlation': float(prophet_corr),
                'xgboost_correlation': float(xgboost_corr),
                'prophet_mape': float(prophet_mape),
                'xgboost_mape': float(xgboost_mape)
            },
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        print(f"Error in comparison task: {str(e)}")
        raise

comparison_task = PythonOperator(
    task_id='compare_with_predictions',
    python_callable=compare_with_predictions,
    provide_context=True,
    dag=dag,
)

# Tarea para notificar resultados
def send_results_notification(**kwargs):
    """
    Send notification with time series analysis results.
    """
    try:
        print("Preparing time series results notification...")
        
        # Get results from tasks
        ts_results = kwargs['ti'].xcom_pull(task_ids='time_series_analysis')
        comparison_results = kwargs['ti'].xcom_pull(task_ids='compare_with_predictions')
        
        # Create output directory if needed
        output_dir = Path("/app/data/time_series")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Combine results
        notification = {
            'timestamp': datetime.now().isoformat(),
            'time_series_analysis': ts_results,
            'comparison_results': comparison_results,
            'status': 'success' if ts_results and ts_results.get('status') == 'success' else 'warning'
        }
        
        # Save notification as JSON
        with open(output_dir / "time_series_results.json", 'w') as f:
            json.dump(notification, f, indent=2)
        
        # Connect to PostgreSQL to store metadata
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Create metadata table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_type VARCHAR(50) NOT NULL,
            run_date TIMESTAMP NOT NULL,
            metrics JSONB,
            status VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Format metrics
        ts_metrics = {}
        if ts_results and 'metrics' in ts_results:
            ts_metrics = ts_results['metrics']
        
        comp_metrics = {}
        if comparison_results and 'metrics' in comparison_results:
            comp_metrics = comparison_results['metrics']
        
        # Combine all metrics
        all_metrics = {**ts_metrics, **comp_metrics}
        
        # Store time series metadata
        cursor.execute(
            """
            INSERT INTO model_metadata
            (model_type, run_date, metrics, status)
            VALUES (%s, %s, %s, %s)
            """,
            ('time_series', datetime.now(), json.dumps(all_metrics), notification['status'])
        )
        
        # Commit changes
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Time series results notification prepared and stored")
        return notification
        
    except Exception as e:
        print(f"Error in notification task: {str(e)}")
        raise

notification_task = PythonOperator(
    task_id='send_results_notification',
    python_callable=send_results_notification,
    provide_context=True,
    dag=dag,
)

# Definir dependencias
time_series_analysis_task >> comparison_task >> notification_task