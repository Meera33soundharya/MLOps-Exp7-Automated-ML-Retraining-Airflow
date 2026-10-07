"""Daily Airflow workflow for training and promoting a classifier."""

from datetime import datetime
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.train import evaluate_candidate, promote_model, train_model  # noqa: E402


with DAG(
    dag_id="retrain_classification_model",
    description="Train, validate, and promote a classification model each day.",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "retraining"],
) as dag:
    train_candidate = PythonOperator(
        task_id="train_candidate",
        python_callable=train_model,
    )
    evaluate_candidate_task = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=evaluate_candidate,
    )
    promote_candidate = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    train_candidate >> evaluate_candidate_task >> promote_candidate
