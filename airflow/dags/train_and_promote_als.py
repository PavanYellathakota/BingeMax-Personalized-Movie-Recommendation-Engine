# File: airflow/dags/train_and_promote_als.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os
# Fix for Airflow import context
sys.path.append(os.path.abspath('/opt/airflow/src'))
from src.models.als.train_als import train_and_log_model
from src.utils.mlflow.mlflow_utils import promote_model_version
import mlflow

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

def train_model():
    rmse, run_id = train_and_log_model()
    return {"rmse": rmse, "run_id": run_id}

def promote_if_good(**context):
    rmse = context['ti'].xcom_pull(task_ids='train_model')['rmse']
    run_id = context['ti'].xcom_pull(task_ids='train_model')['run_id']

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    version = client.search_model_versions(f"run_id='{run_id}'")[0].version

    if rmse < 0.9:
        promote_model_version(version=version, stage="Production")
        print(f"✅ Promoted version {version} to Production")
    else:
        print(f"❌ RMSE too high: {rmse}. Not promoting.")

with DAG("als_train_promote_pipeline", schedule_interval="@weekly", default_args=default_args, catchup=False) as dag:
    
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    promote = PythonOperator(
        task_id="promote_if_good",
        python_callable=promote_if_good,
        provide_context=True
    )

    train >> promote
