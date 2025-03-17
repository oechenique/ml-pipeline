from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np
import h3
import psycopg2
import psycopg2.extras
import json
from pathlib import Path
import logging
import decimal
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    'bike_sharing_prediction',
    default_args=default_args,
    description='Demand prediction for bike-sharing stations',
    schedule_interval=None,  # Triggered by master DAG
    catchup=False,
)

def run_prediction_direct():
    """
    Run prediction directly using real data from PostgreSQL.
    """
    try:
        print(" Starting direct prediction from PostgreSQL with real data")
        
        # Connection configuration
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        # 1. Extract station data with H3 from PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        print(" Querying station and cluster data...")
        query = """
        SELECT 
            bs.station_id, 
            bs.name, 
            bs.city,
            bs.latitude, 
            bs.longitude, 
            bs.h3_index,
            bs.capacity,
            ST_AsText(CASE WHEN bs.h3_polygon IS NOT NULL THEN bs.h3_polygon ELSE bs.geom END) as geometry_wkt,
            COALESCE(sc.cluster, -1) as cluster
        FROM bike_stations bs
        LEFT JOIN station_clusters sc ON bs.station_id = sc.station_id
        WHERE bs.h3_index IS NOT NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            raise ValueError("No stations with H3 indices found in database")
            
        # Create DataFrame with explicit column names
        column_names = [
            'station_id', 'name', 'city', 'latitude', 'longitude', 
            'h3_index', 'capacity', 'geometry_wkt', 'cluster'
        ]
        
        df = pd.DataFrame(rows, columns=column_names)
        print(f" Extracted {len(df)} stations with H3 indices")
        print(f"Columns available: {df.columns.tolist()}")
        
        # 2. Extract clustering statistics if available
        try:
            print(" Querying clustering statistics...")
            cursor.execute("""
            SELECT 
                cluster_analysis
            FROM clustering_stats
            ORDER BY created_at DESC
            LIMIT 1;
            """)
            stats_row = cursor.fetchone()
            
            if stats_row and stats_row[0]:
                # JSONB comes as a Python dictionary, no need for json.loads
                cluster_stats = stats_row[0]
                print(f" Clustering statistics loaded successfully")
                
                # Show cluster information
                for cluster_key, stats in cluster_stats.items():
                    if 'size' in stats:
                        print(f"  - Cluster {cluster_key}: {stats['size']} stations")
            else:
                print(" No detailed clustering statistics found")
                cluster_stats = {}
        except Exception as e:
            print(f" Error querying clustering statistics: {str(e)}")
            cluster_stats = {}
        
        # 3. Extract historical trip data - USING REAL DATA
        print(" Querying real historical trip data...")
        
        # Check amount of available data
        count_query = """
        SELECT COUNT(*) FROM bike_trips
        """
        cursor.execute(count_query)
        total_trips = cursor.fetchone()[0]
        print(f" Total trips in database: {total_trips:,}")
        
        # Query to get historical trip data
        # This query uses all available data without date filtering
        historic_query = """
        -- Outgoing trips from each station (aggregated by day and hour)
        WITH outgoing AS (
            SELECT 
                start_station_id as station_id,
                date_trunc('day', start_time) as day,
                EXTRACT(HOUR FROM start_time) as hour,
                COUNT(*) as outgoing_trips,
                AVG(duration_sec) as avg_duration
            FROM bike_trips
            WHERE start_station_id IS NOT NULL
            GROUP BY 
                start_station_id, 
                date_trunc('day', start_time),
                EXTRACT(HOUR FROM start_time)
        ),
        -- Incoming trips to each station
        incoming AS (
            SELECT 
                end_station_id as station_id,
                date_trunc('day', end_time) as day,
                EXTRACT(HOUR FROM end_time) as hour,
                COUNT(*) as incoming_trips
            FROM bike_trips
            WHERE end_station_id IS NOT NULL
            GROUP BY 
                end_station_id, 
                date_trunc('day', end_time),
                EXTRACT(HOUR FROM end_time)
        )
        -- Join both datasets
        SELECT 
            COALESCE(o.station_id, i.station_id) as station_id,
            COALESCE(o.day, i.day) as day,
            COALESCE(o.hour, i.hour) as hour,
            COALESCE(o.outgoing_trips, 0) as outgoing_trips,
            COALESCE(i.incoming_trips, 0) as incoming_trips,
            o.avg_duration
        FROM outgoing o
        FULL OUTER JOIN incoming i 
            ON o.station_id = i.station_id 
            AND o.day = i.day 
            AND o.hour = i.hour
        ORDER BY 
            station_id, day, hour
        LIMIT 100000  -- Adjust if needed
        """
        
        cursor.execute(historic_query)
        historic_rows = cursor.fetchall()
        
        if historic_rows:
            historic_df = pd.DataFrame(historic_rows, columns=[
                'station_id', 'day', 'hour', 'outgoing_trips', 'incoming_trips', 'avg_duration'
            ])
            print(f" Extracted {len(historic_df):,} REAL historical records")
            
            # Sample for debugging
            print("\n Sample of historical data:")
            print(historic_df.head())
            print(f"\nOutgoing trips statistics:")
            print(f"  Min: {historic_df['outgoing_trips'].min()}")
            print(f"  Max: {historic_df['outgoing_trips'].max()}")
            print(f"  Avg: {historic_df['outgoing_trips'].mean():.2f}")
            
            # Close cursor and connection 
            cursor.close()
            conn.close()
        else:
            cursor.close()
            conn.close()
            raise ValueError("No historical trip data found in bike_trips")
            
# 4. Prepare features for prediction model
        print(" Preparing features for prediction...")
        
        # 4.1 Add day of week and temporal variables
        historic_df['is_weekend'] = historic_df['day'].dt.dayofweek >= 5
        historic_df['day_of_week'] = historic_df['day'].dt.dayofweek
        historic_df['month'] = historic_df['day'].dt.month
        
        # 4.2 Handle null values in avg_duration
        # Instead of filling with mean, we'll identify where nulls exist
        null_avg_duration = historic_df['avg_duration'].isna().sum()
        if null_avg_duration > 0:
            print(f" Detected {null_avg_duration} null values in avg_duration ({null_avg_duration/len(historic_df)*100:.2f}%)")
        
        # 4.3 Add station information
        # Ensure cluster is a numeric value
        station_features = df[['station_id', 'latitude', 'longitude', 'capacity', 'cluster']].copy()
        station_features['cluster'] = pd.to_numeric(station_features['cluster'], errors='coerce').fillna(-1)
        
        # 4.4 Limit to stations that exist in our dataset
        valid_stations = set(station_features['station_id'].unique())
        historic_df = historic_df[historic_df['station_id'].isin(valid_stations)]
        
        if len(historic_df) == 0:
            raise ValueError("No historical data for available stations")
            
        print(f" After filtering, {len(historic_df):,} historical records remain with valid stations")
        
        # 4.5 Join historical data with station features
        model_df = pd.merge(historic_df, station_features, on='station_id', how='left')
        
        # 4.6 Create additional features: aggregates by station
        station_aggs = historic_df.groupby('station_id').agg({
            'outgoing_trips': ['mean', 'std', 'max'],
            'incoming_trips': ['mean', 'std', 'max']
        }).reset_index()
        
        # Flatten MultiIndex columns
        station_aggs.columns = ['_'.join(col).strip() for col in station_aggs.columns.values]
        station_aggs.rename(columns={'station_id_': 'station_id'}, inplace=True)
        
        # Join with main DataFrame
        model_df = pd.merge(model_df, station_aggs, on='station_id', how='left')
        
        # 4.7 Add correlation between incoming and outgoing by station
        def calc_station_correlation(group):
            if len(group) > 5:  # Only if enough data
                return group['outgoing_trips'].corr(group['incoming_trips'])
            return 0
            
        corr_df = historic_df.groupby('station_id').apply(calc_station_correlation).reset_index()
        corr_df.columns = ['station_id', 'io_correlation']
        model_df = pd.merge(model_df, corr_df, on='station_id', how='left')
        
        # 4.8 Categorical variables: One-hot encoding for day of week and hour
        model_df = pd.get_dummies(model_df, columns=['day_of_week', 'hour'], prefix=['dow', 'h'])
        
        # 4.9 Prepare target variables
        # For this model we'll predict outgoing_trips (demand)
        X = model_df.drop(['outgoing_trips', 'incoming_trips', 'day', 'station_id', 'avg_duration'], axis=1)
        y = model_df['outgoing_trips']
        
        # 4.10 Check and report NaN values in X
        nan_columns = X.columns[X.isna().any()].tolist()
        if nan_columns:
            print(f" Detected columns with NaN values: {nan_columns}")
            for col in nan_columns:
                print(f"  - {col}: {X[col].isna().sum()} null values ({X[col].isna().sum()/len(X)*100:.2f}%)")
        
        print(f" Features prepared: {X.shape[1]} variables for {X.shape[0]} records")
        
        # 5. Handle NaN values using SimpleImputer
        print(" Applying imputation for null values...")
        
        # Create and apply imputer for numeric values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Verify no NaN values remain
        if X_imputed.isna().sum().sum() > 0:
            raise ValueError("NaN values still remain after imputation")
        
        print(" Imputation completed successfully - all NaN values have been handled")

        # 6. Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        
        # 7. Train model
        print(" Training prediction model...")
        
        # Normalize numeric features
        numeric_features = ['latitude', 'longitude', 'capacity', 
                           'outgoing_trips_mean', 'outgoing_trips_std', 'outgoing_trips_max',
                           'incoming_trips_mean', 'incoming_trips_std', 'incoming_trips_max',
                           'io_correlation']
        
        scaler = StandardScaler()
        X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])
        
        # Use HistGradientBoostingRegressor instead of RandomForest
        # This model handles large datasets better and has better performance
        print(" Using HistGradientBoostingRegressor for better data handling")
        model = HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=15,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # 8. Evaluate model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f" Model evaluation:")
        print(f"  - RMSE on training: {train_rmse:.2f}")
        print(f"  - RMSE on test: {test_rmse:.2f}")
        print(f"  - R2 on training: {train_r2:.4f}")
        print(f"  - R2 on test: {test_r2:.4f}")
        
        # 9. Generate feature importance (if available)
        try:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(" Top 10 most important features:")
            print(feature_importance.head(10))
        except:
            print(" Feature importance not available for this model")
            # Create an empty list for use in visualizations
            feature_importance = pd.DataFrame(columns=['feature', 'importance'])

        # 10. Generate predictions for next 24 hours for each station
        print(" Generating predictions for next 24 hours...")
        
        # Get unique stations with real data
        stations_with_data = historic_df['station_id'].unique()
        print(f" Generating predictions for {len(stations_with_data)} stations with historical data")
        
        # Base date for prediction (next day after the last available)
        prediction_base = datetime.now()
        prediction_date = prediction_base.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate data for predictions
        prediction_data = []
        
        for station_id in stations_with_data:
            # Filter station data
            station_rows = df[df['station_id'] == station_id]
            if len(station_rows) == 0:
                continue
                
            station_info = station_rows.iloc[0]
            
            # Check that we have aggregations for this station
            if station_id not in station_aggs['station_id'].values:
                continue
                
            # Extract aggregations
            station_agg = station_aggs[station_aggs['station_id'] == station_id].iloc[0]
            
            # Check correlation
            station_corr = 0
            if station_id in corr_df['station_id'].values:
                station_corr = corr_df[corr_df['station_id'] == station_id]['io_correlation'].iloc[0]
            
            for hour in range(24):
                # Create base record for prediction
                pred_record = {
                    'station_id': station_id,
                    'prediction_date': prediction_date,
                    'prediction_hour': hour,
                    'latitude': station_info['latitude'],
                    'longitude': station_info['longitude'],
                    'capacity': station_info['capacity'],
                    'cluster': station_info['cluster'],
                    'is_weekend': int(prediction_date.weekday() >= 5),
                    # Add aggregation values
                    'outgoing_trips_mean': station_agg['outgoing_trips_mean'],
                    'outgoing_trips_std': station_agg['outgoing_trips_std'],
                    'outgoing_trips_max': station_agg['outgoing_trips_max'],
                    'incoming_trips_mean': station_agg['incoming_trips_mean'],
                    'incoming_trips_std': station_agg['incoming_trips_std'],
                    'incoming_trips_max': station_agg['incoming_trips_max'],
                    'io_correlation': station_corr
                }
                
                # Add temporal one-hot variables
                dow = prediction_date.weekday()
                for i in range(7):
                    pred_record[f'dow_{i}'] = 1 if i == dow else 0
                
                for i in range(24):
                    pred_record[f'h_{i}'] = 1 if i == hour else 0
                
                prediction_data.append(pred_record)
        
        # Create DataFrame for predictions
        pred_df = pd.DataFrame(prediction_data)
        
        if len(pred_df) == 0:
            raise ValueError("Could not generate predictions with available data")
            
        print(f" Prediction DataFrame created with {len(pred_df)} records")
        
        # Ensure all necessary columns are present
        for col in X.columns:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        # Adjust column order to match model
        pred_X = pred_df[X.columns]
        
        # Apply the same imputation to prediction data if needed
        pred_X = pd.DataFrame(imputer.transform(pred_X), columns=pred_X.columns)
        
        # Normalize the same features as in training
        pred_X[numeric_features] = scaler.transform(pred_X[numeric_features])
        
        # Make predictions
        pred_df['predicted_demand'] = model.predict(pred_X)
        
        # Truncate negative predictions to 0
        pred_df['predicted_demand'] = pred_df['predicted_demand'].apply(lambda x: max(0, x))
        
        print(f" Predicted demand statistics:")
        print(f"  Min: {pred_df['predicted_demand'].min():.2f}")
        print(f"  Max: {pred_df['predicted_demand'].max():.2f}")
        print(f"  Avg: {pred_df['predicted_demand'].mean():.2f}")
        
        # 11. Store predictions in PostgreSQL
        print(" Storing predictions in PostgreSQL...")
        
        # Select relevant columns for storage
        predictions_to_store = pred_df[[
            'station_id', 'prediction_date', 'prediction_hour', 
            'predicted_demand', 'is_weekend', 'cluster'
        ]].copy()
        
        # Convert predicted_demand to integer for simplicity
        predictions_to_store['predicted_demand'] = predictions_to_store['predicted_demand'].round().astype(int)
        
        try:
            # Reconnect to database
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS bike_demand_predictions (
                id SERIAL PRIMARY KEY,
                station_id VARCHAR,
                prediction_date DATE,
                prediction_hour INTEGER,
                predicted_demand INTEGER,
                is_weekend BOOLEAN,
                cluster INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indices for fast lookups
            CREATE INDEX IF NOT EXISTS idx_predictions_station 
            ON bike_demand_predictions(station_id);
            
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON bike_demand_predictions(prediction_date);
            """)
            
            # Clear previous predictions for same date/stations
            cursor.execute("""
            DELETE FROM bike_demand_predictions 
            WHERE prediction_date = %s
            """, (prediction_date,))
            
            # Insert new predictions in batches for better performance
            from psycopg2.extras import execute_values
            
            # Prepare data for bulk insertion
            insert_data = [
                (
                    row['station_id'],
                    row['prediction_date'],
                    row['prediction_hour'],
                    row['predicted_demand'],
                    bool(row['is_weekend']),
                    int(row['cluster'])
                )
                for _, row in predictions_to_store.iterrows()
            ]
            
            # Insert in batches
            execute_values(
                cursor,
                """
                INSERT INTO bike_demand_predictions 
                (station_id, prediction_date, prediction_hour, predicted_demand, is_weekend, cluster)
                VALUES %s
                """,
                insert_data,
                page_size=1000  # Adjust as needed
            )
            
            # Commit changes
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f" Stored {len(predictions_to_store)} predictions for {len(stations_with_data)} stations")
            
        except Exception as e:
            print(f" Error storing predictions: {str(e)}")
            raise

        # 12. Generate HTML visualization
        print(" Generating prediction visualization...")
        
        # Create directory for results
        output_dir = Path("/app/data/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data for visualization
        # Summarize by hour of day
        hour_summary = pred_df.groupby(['prediction_hour'])[['predicted_demand']].mean().reset_index()
        
        # Summarize by station and cluster
        station_summary = pred_df.groupby(['station_id', 'cluster', 'latitude', 'longitude'])[['predicted_demand']].agg([
            'mean', 'max'
        ]).reset_index()
        
        # Flatten MultiIndex
        station_summary.columns = [
            col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
            for col in station_summary.columns
        ]
        
        # Top 10 stations with highest demand
        top_stations = station_summary.sort_values('predicted_demand_mean', ascending=False).head(10)
        
        # Top 10 stations with lowest demand
        bottom_stations = station_summary.sort_values('predicted_demand_mean').head(10)
        
        # Create HTML for visualization
        html_output = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Demand Predictions - Bike Sharing</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .prediction-card {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; }}
                .summary {{ background-color: #f7f7f7; }}
                .high-demand {{ background-color: #f9e0e0; }}
                .low-demand {{ background-color: #e0f0ff; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart-container {{ height: 300px; margin: 20px 0; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Demand Predictions - Bike Sharing</h1>
            <p>Prediction date: {prediction_date.strftime('%Y-%m-%d')}</p>
            
            <div class="prediction-card summary">
                <h2>Model Summary</h2>
                <p>Trained with <strong>{len(historic_df):,}</strong> real historical records</p>
                <p>RMSE: {test_rmse:.2f}</p>
                <p>R2: {test_r2:.4f}</p>
                <p>Stations analyzed: {len(stations_with_data)}</p>
                <p>Clusters: {len(df['cluster'].unique())}</p>
            </div>
            
            <div class="prediction-card">
                <h2>Hourly Demand Distribution</h2>
                <div class="chart-container">
                    <canvas id="hourlyChart"></canvas>
                </div>
                <script>
                var ctx = document.getElementById('hourlyChart').getContext('2d');
                var hourlyChart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {hour_summary['prediction_hour'].tolist()},
                        datasets: [{{
                            label: 'Average Demand',
                            data: {hour_summary['predicted_demand'].tolist()},
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: 'Predicted Trips'
                                }}
                            }},
                            x: {{
                                title: {{
                                    display: true,
                                    text: 'Hour of Day'
                                }}
                            }}
                        }}
                    }}
                }});
                </script>
            </div>
            
            <div class="prediction-card high-demand">
                <h2>Stations with Highest Predicted Demand</h2>
                <table>
                    <tr>
                        <th>Station</th>
                        <th>Cluster</th>
                        <th>Average Demand</th>
                        <th>Maximum Demand</th>
                    </tr>
        """
        
        for _, row in top_stations.iterrows():
            html_output += f"""
                    <tr>
                        <td>{row['station_id']}</td>
                        <td>{int(row['cluster'])}</td>
                        <td>{row['predicted_demand_mean']:.1f}</td>
                        <td>{row['predicted_demand_max']:.1f}</td>
                    </tr>
            """
            
        html_output += """
                </table>
            </div>
            
            <div class="prediction-card low-demand">
                <h2>Stations with Lowest Predicted Demand</h2>
                <table>
                    <tr>
                        <th>Station</th>
                        <th>Cluster</th>
                        <th>Average Demand</th>
                        <th>Maximum Demand</th>
                    </tr>
        """
        
        for _, row in bottom_stations.iterrows():
            html_output += f"""
                    <tr>
                        <td>{row['station_id']}</td>
                        <td>{int(row['cluster'])}</td>
                        <td>{row['predicted_demand_mean']:.1f}</td>
                        <td>{row['predicted_demand_max']:.1f}</td>
                    </tr>
            """
            
        # Add important features if available
        if len(feature_importance) > 0:
            html_output += """
                </table>
            </div>
            
            <div class="prediction-card">
                <h2>Most Important Features</h2>
                <div class="chart-container">
                    <canvas id="featureChart"></canvas>
                </div>
                <script>
                var ctxFeature = document.getElementById('featureChart').getContext('2d');
                var featureChart = new Chart(ctxFeature, {
                    type: 'bar',
                    data: {
                        labels: """ + str([str(x) for x in feature_importance.head(10)['feature'].tolist()]) + """,
                        datasets: [{
                            label: 'Importance',
                            data: """ + str(feature_importance.head(10)['importance'].tolist()) + """,
                            backgroundColor: 'rgba(153, 102, 255, 0.5)',
                            borderColor: 'rgba(153, 102, 255, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Importance'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Feature'
                                }
                            }
                        }
                    }
                });
                </script>
            </div>
            """
        else:
            html_output += """
                </table>
            </div>
            """
        
        html_output += """
        </body>
        </html>
        """
        
        # Save HTML
        html_path = output_dir / f"prediction_report_{prediction_date.strftime('%Y%m%d')}.html"
        with open(html_path, 'w') as f:
            f.write(html_output)
        
        # Also save predictions as CSV
        csv_path = output_dir / f"station_predictions_{prediction_date.strftime('%Y%m%d')}.csv"
        predictions_to_store.to_csv(csv_path, index=False)
        
        # Also save model details as JSON
        model_details = {
            'metrics': {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2)
            },
            'top_features': {
                row['feature']: float(row['importance']) 
                for _, row in feature_importance.head(20).iterrows()
            } if len(feature_importance) > 0 else {},
            'model_type': type(model).__name__,
            'timestamp': datetime.now().isoformat(),
            'prediction_date': prediction_date.isoformat(),
            'n_stations': len(stations_with_data),
            'n_predictions': len(predictions_to_store),
            'data_source': 'historical_real_data'
        }
        
        # Save model details
        with open(output_dir / f"model_details_{prediction_date.strftime('%Y%m%d')}.json", 'w') as f:
            json.dump(model_details, f, indent=2)
        
        print(f" Visualizations and data saved in {output_dir}")
        
        return {
            'status': 'success',
            'metrics': {
                'rmse': float(test_rmse),
                'r2': float(test_r2)
            },
            'data_source': 'historical_real_data',
            'n_predictions': len(predictions_to_store),
            'n_stations': len(stations_with_data),
            'prediction_date': prediction_date.isoformat(),
            'output_dir': str(output_dir)
        }
        
    except Exception as e:
        print(f" Error in direct prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def run_rebalancing_optimization(**kwargs):
    """
    Executes rebalancing optimization based on predictions.
    """
    try:
        # Get result from prediction task
        prediction_result = kwargs['ti'].xcom_pull(task_ids='run_prediction')
        
        if not prediction_result:
            raise ValueError("No prediction results found")
        
        print(" Starting rebalancing optimization based on predictions")
        
        # Connect to PostgreSQL
        conn_params = {
            'host': 'db_service',
            'port': 5432,
            'database': 'geo_db',
            'user': 'geo_user',
            'password': 'NekoSakamoto448'
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Extract predictions
        prediction_date = prediction_result.get('prediction_date')
        if not prediction_date:
            # Use current date as fallback
            prediction_date = datetime.now().strftime('%Y-%m-%d')
        
        # Query to get predictions grouped by peak hours
        query = """
        WITH peak_hours AS (
            -- Define peak hours (morning: 7-9, evening: 16-19)
            SELECT 
                station_id,
                SUM(CASE WHEN prediction_hour BETWEEN 7 AND 9 THEN predicted_demand ELSE 0 END) as morning_demand,
                SUM(CASE WHEN prediction_hour BETWEEN 16 AND 19 THEN predicted_demand ELSE 0 END) as evening_demand,
                SUM(predicted_demand) as total_demand
            FROM bike_demand_predictions
            WHERE prediction_date = %s
            GROUP BY station_id
        ),
        station_data AS (
            -- Get station capacity data
            SELECT 
                bs.station_id,
                bs.capacity,
                COALESCE(bs.h3_index, '') as h3_index,
                sc.cluster
            FROM bike_stations bs
            LEFT JOIN station_clusters sc ON bs.station_id = sc.station_id
        )
        SELECT 
            p.station_id,
            p.morning_demand,
            p.evening_demand,
            p.total_demand,
            s.capacity,
            s.h3_index,
            s.cluster,
            -- Calculate deficit/surplus during peak hours
            (p.morning_demand - s.capacity * 0.5) as morning_deficit,
            (p.evening_demand - s.capacity * 0.5) as evening_deficit
        FROM peak_hours p
        JOIN station_data s ON p.station_id = s.station_id
        ORDER BY 
            ABS(p.morning_demand - s.capacity * 0.5) + 
            ABS(p.evening_demand - s.capacity * 0.5) DESC
        """
        
        cursor.execute(query, (prediction_date,))
        rebalance_data = cursor.fetchall()
        
        # Close cursor
        cursor.close()
        conn.close()
        
        if not rebalance_data:
            print(" No data found for rebalancing optimization")
            return {
                'status': 'warning',
                'message': 'Not enough data for optimization'
            }
        
        # Create DataFrame for analysis
        rebalance_df = pd.DataFrame(rebalance_data, columns=[
            'station_id', 'morning_demand', 'evening_demand', 'total_demand',
            'capacity', 'h3_index', 'cluster', 'morning_deficit', 'evening_deficit'
        ])
        
        # Handle null values in cluster
        rebalance_df['cluster'] = rebalance_df['cluster'].fillna(-1).astype(int)
        
        print(f" Rebalancing analysis for {len(rebalance_df)} stations")
        
        # Identify stations needing rebalancing
        # 1. Stations with significant deficit (more than 30% of capacity)
        deficit_stations = rebalance_df[
            (rebalance_df['morning_deficit'] > rebalance_df['capacity'] * 0.3) |
            (rebalance_df['evening_deficit'] > rebalance_df['capacity'] * 0.3)
        ]
        
        # 2. Stations with significant surplus (available space)
        surplus_stations = rebalance_df[
            (rebalance_df['morning_deficit'] < -rebalance_df['capacity'] * 0.3) |
            (rebalance_df['evening_deficit'] < -rebalance_df['capacity'] * 0.3)
        ]
        
        # Create rebalancing plan
        rebalance_plan = []
        
        # Generate rebalancing recommendations by pairs of nearby stations
        if len(deficit_stations) > 0 and len(surplus_stations) > 0:
            for idx, deficit_row in deficit_stations.head(min(10, len(deficit_stations))).iterrows():
                # Use cluster to find neighbors
                if deficit_row['cluster'] != -1:
                    # Find surplus by cluster
                    matching_surplus = surplus_stations[surplus_stations['cluster'] == deficit_row['cluster']]
                else:
                    # Fallback to all surplus stations
                    matching_surplus = surplus_stations
                
                if len(matching_surplus) > 0:
                    # Take first surplus station from same cluster
                    surplus_row = matching_surplus.iloc[0]
                    
                    # Determine critical time (morning or evening)
                    if deficit_row['morning_deficit'] > deficit_row['evening_deficit']:
                        peak_time = "Morning (7-9h)"
                        bikes_needed = int(max(3, min(10, deficit_row['morning_deficit'])))
                    else:
                        peak_time = "Evening (16-19h)"
                        bikes_needed = int(max(3, min(10, deficit_row['evening_deficit'])))
                    
                    rebalance_plan.append({
                        'origin_station': surplus_row['station_id'],
                        'destination_station': deficit_row['station_id'],
                        'bikes_to_move': bikes_needed,
                        'peak_time': peak_time,
                        'deficit': max(deficit_row['morning_deficit'], deficit_row['evening_deficit']),
                        'cluster': int(deficit_row['cluster'])
                    })
                    
                    # Remove this surplus station to avoid reusing it
                    surplus_stations = surplus_stations[surplus_stations['station_id'] != surplus_row['station_id']]
        
        # Create directory for results
        output_dir = Path("/app/data/predictions/rebalancing")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save rebalancing plan
        if rebalance_plan:
            rebalance_plan_df = pd.DataFrame(rebalance_plan)
            csv_path = output_dir / f"rebalance_plan_{datetime.now().strftime('%Y%m%d')}.csv"
            rebalance_plan_df.to_csv(csv_path, index=False)
            
            # También guardar como JSON - AQUÍ ESTÁ EL PROBLEMA
            # Necesitamos convertir los valores Decimal a float
            json_path = output_dir / f"rebalance_plan_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Solución 1: Usando una función de conversión
            def decimal_default(obj):
                if isinstance(obj, decimal.Decimal):
                    return float(obj)
                raise TypeError
            
            with open(json_path, 'w') as f:
                json.dump(rebalance_plan, f, indent=2, default=decimal_default)
            
            print(f" Rebalancing plan generated with {len(rebalance_plan)} actions")
            
            # Create HTML for visualization
            html_output = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Rebalancing Plan - Bike Sharing</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .rebalance-card {{ border: 1px solid #ccc; margin: 10px; padding: 15px; border-radius: 5px; }}
                    .summary {{ background-color: #f7f7f7; }}
                    .action {{ background-color: #e6f7ff; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Rebalancing Plan - Bike Sharing</h1>
                <p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>
                
                <div class="rebalance-card summary">
                    <h2>Summary</h2>
                    <p>Stations with deficit: {len(deficit_stations)}</p>
                    <p>Stations with surplus: {len(surplus_stations)}</p>
                    <p>Recommended rebalancing actions: {len(rebalance_plan)}</p>
                </div>
                
                <div class="rebalance-card">
                    <h2>Recommended Rebalancing Actions</h2>
                    <table>
                        <tr>
                            <th>Origin</th>
                            <th>Destination</th>
                            <th>Bikes</th>
                            <th>Critical Time</th>
                            <th>Cluster</th>
                        </tr>
            """
            
            for action in rebalance_plan:
                html_output += f"""
                        <tr>
                            <td>{action['origin_station']}</td>
                            <td>{action['destination_station']}</td>
                            <td>{action['bikes_to_move']}</td>
                            <td>{action['peak_time']}</td>
                            <td>{action['cluster']}</td>
                        </tr>
                """
                
            html_output += """
                    </table>
                </div>
                
                <div class="rebalance-card">
                    <h2>Top 10 Stations with Highest Deficit</h2>
                    <table>
                        <tr>
                            <th>Station</th>
                            <th>Capacity</th>
                            <th>Morning Deficit</th>
                            <th>Evening Deficit</th>
                            <th>Cluster</th>
                        </tr>
            """
            
            for _, row in deficit_stations.head(10).iterrows():
                html_output += f"""
                        <tr>
                            <td>{row['station_id']}</td>
                            <td>{int(row['capacity'])}</td>
                            <td>{row['morning_deficit']:.1f}</td>
                            <td>{row['evening_deficit']:.1f}</td>
                            <td>{int(row['cluster'])}</td>
                        </tr>
                """
                
            html_output += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            # Save HTML
            html_path = output_dir / f"rebalance_plan_{datetime.now().strftime('%Y%m%d')}.html"
            with open(html_path, 'w') as f:
                f.write(html_output)
                
            return {
                'status': 'success',
                'actions': len(rebalance_plan),
                'deficit_stations': len(deficit_stations),
                'surplus_stations': len(surplus_stations),
                'output_dir': str(output_dir)
            }
        else:
            print(" No rebalancing plan generated, no actions needed")
            return {
                'status': 'info',
                'message': 'No rebalancing actions needed',
                'deficit_stations': len(deficit_stations),
                'surplus_stations': len(surplus_stations)
            }
    
    except Exception as e:
        print(f" Error in rebalancing optimization: {str(e)}")
        raise

# Define task with Python operator to run prediction
run_prediction_task = PythonOperator(
    task_id='run_prediction',
    python_callable=run_prediction_direct,
    dag=dag,
)

# Task for rebalancing optimization
rebalancing_task = PythonOperator(
    task_id='run_rebalancing',
    python_callable=run_rebalancing_optimization,
    provide_context=True,
    dag=dag,
)

# Define dependencies
run_prediction_task >> rebalancing_task