from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.models.feature_engineering import BikeFeatureEngineer

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'bike_sharing_feature_engineering',
    default_args=default_args,
    description='Feature engineering para bike-sharing',
    schedule_interval=None,  # Triggered by master DAG
    catchup=False,
)

def run_feature_engineering(**kwargs):
    h3_resolution = kwargs.get('h3_resolution', 9)
    engineer = BikeFeatureEngineer(h3_resolution=h3_resolution)
    features = engineer.process()
    return f"Procesados {len(features.columns)} features para {len(features)} estaciones"

feature_engineering_task = PythonOperator(
    task_id='run_feature_engineering',
    python_callable=run_feature_engineering,
    op_kwargs={'h3_resolution': 9},
    dag=dag,
)