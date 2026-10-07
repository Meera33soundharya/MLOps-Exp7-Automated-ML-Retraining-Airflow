# EXP-7 - Automated ML Retraining Pipeline with Apache Airflow

A small end-to-end MLOps example that trains a classifier from CSV data, checks
its test accuracy against a minimum threshold and the current production model,
and promotes it only when it passes evaluation. Apache Airflow schedules these
steps; a PowerShell script runs the same workflow once for local testing.

## Project layout

```text
.
|-- airflow/
|   `-- dags/
|       `-- retraining_dag.py
|-- data/
|   `-- training.csv
|-- src/
|   `-- __init__.py
|   `-- train.py
|-- .env.example
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- run_exp7.ps1
```

## Requirements

- Python 3.10-3.12
- The packages in `requirements.txt` for training and evaluating the model
- Apache Airflow 2.10.5 to schedule the DAG. Airflow is intended to run on
  Linux/WSL or in a Linux container; the PowerShell runner works on Windows.

Create and activate a virtual environment, then install the model packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a reproducible Airflow install, use Airflow's constraints file for your
Python version from its official installation documentation:

```bash
pip install "apache-airflow==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"
```

## Run once on Windows

From the project root, run:

```powershell
.\run_exp7.ps1
```

The runner calls the same train, evaluate, and promote functions used by the
DAG. A successful run writes generated files under `artifacts/`:

- `candidate_model.joblib` and `candidate_metrics.json` are the latest run.
- `production_model.joblib` and `production_metrics.json` are updated only
  when evaluation passes.

The sample data is included so the pipeline can be exercised immediately.
Replace `data/training.csv` with your data using numeric feature columns and a
classification target column. Set `DATA_PATH` and `TARGET_COLUMN` in your
environment (or in your local `.env` file) to point the pipeline at your data.
Do not commit `.env` files or private data.

## Run with Apache Airflow

Set `AIRFLOW_HOME` and make the project root available to the Airflow process.
Initialize the Airflow database and create an administrator using the commands
for your Airflow installation, then start the webserver and scheduler. Ensure
the Airflow process can read the project data and write to the configured model
directory. The DAG is `retrain_classification_model`, runs daily, and has
`catchup=False`.

The DAG runs three ordered tasks:

1. `train_candidate` trains and saves a candidate model and its metrics.
2. `evaluate_candidate` requires the candidate to meet `MIN_ACCURACY` and not
   score below the currently promoted model.
3. `promote_model` replaces the production model and metrics after a passing
   evaluation.

Use a shared `MODEL_DIR` when Airflow tasks run on separate workers so each task
can access the candidate and production artifacts.

## Configuration

Copy `.env.example` to `.env` to keep local settings together. The Python
pipeline reads environment variables directly; `.env` is not loaded
automatically, so export the variables in your shell or configure them in
Airflow:

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_PATH` | `data/training.csv` | Input CSV path |
| `TARGET_COLUMN` | `passed` | Classification label column |
| `MODEL_DIR` | `artifacts` | Candidate and production artifact directory |
| `MIN_ACCURACY` | `0.70` | Minimum candidate test accuracy, from 0 to 1 |

The evaluation fails explicitly if the data is invalid, the threshold is
invalid, or the candidate does not meet the promotion criteria.
