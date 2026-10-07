"""Train, evaluate, and promote a classification model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: str) -> Path:
    value = Path(os.environ.get(variable, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _model_paths() -> dict[str, Path]:
    model_dir = _configured_path("MODEL_DIR", "artifacts")
    return {
        "candidate_model": model_dir / "candidate_model.joblib",
        "candidate_metrics": model_dir / "candidate_metrics.json",
        "production_model": model_dir / "production_model.joblib",
        "production_metrics": model_dir / "production_metrics.json",
    }


def train_model() -> dict[str, Any]:
    """Fit a candidate model and save it with held-out test metrics."""
    data_path = _configured_path("DATA_PATH", "data/training.csv")
    target_column = os.environ.get("TARGET_COLUMN", "passed")
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV does not exist: {data_path}")

    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(
            f"Target column {target_column!r} is missing from {data_path}. "
            f"Available columns: {', '.join(map(str, data.columns))}"
        )
    if data.empty:
        raise ValueError(f"Training CSV contains no rows: {data_path}")
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Training data contains missing values in: {missing}")

    features = data.drop(columns=[target_column])
    if features.empty:
        raise ValueError("Training data must include at least one feature column.")
    non_numeric = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Features must be numeric. Non-numeric columns: {non_numeric}"
        )
    labels = data[target_column]
    if labels.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each target class must have at least two rows for splitting.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(x_test)))

    paths = _model_paths()
    paths["candidate_model"].parent.mkdir(parents=True, exist_ok=True)
    model_temp = paths["candidate_model"].with_suffix(".joblib.tmp")
    metrics_temp = paths["candidate_metrics"].with_suffix(".json.tmp")
    metrics = {
        "accuracy": accuracy,
        "target_column": target_column,
        "training_rows": int(len(data)),
        "test_rows": int(len(x_test)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
    }

    try:
        joblib.dump(model, model_temp)
        metrics_temp.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        model_temp.replace(paths["candidate_model"])
        metrics_temp.replace(paths["candidate_metrics"])
    finally:
        model_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)

    print(f"Candidate trained: accuracy={accuracy:.3f}")
    return metrics


def evaluate_candidate() -> dict[str, Any]:
    """Check candidate accuracy and ensure it does not regress production."""
    paths = _model_paths()
    if not paths["candidate_model"].is_file():
        raise FileNotFoundError(
            f"Candidate model is missing: {paths['candidate_model']}. "
            "Run training before evaluation."
        )
    if not paths["candidate_metrics"].is_file():
        raise FileNotFoundError(
            f"Candidate metrics are missing: {paths['candidate_metrics']}. "
            "Run training before evaluation."
        )

    metrics = json.loads(paths["candidate_metrics"].read_text(encoding="utf-8"))
    minimum_accuracy = float(os.environ.get("MIN_ACCURACY", "0.70"))
    if not 0.0 <= minimum_accuracy <= 1.0:
        raise ValueError("MIN_ACCURACY must be between 0 and 1.")
    accuracy = float(metrics["accuracy"])
    if accuracy < minimum_accuracy:
        raise ValueError(
            f"Candidate accuracy {accuracy:.3f} is below MIN_ACCURACY "
            f"{minimum_accuracy:.3f}; production model was not changed."
        )

    if paths["production_metrics"].is_file():
        production = json.loads(
            paths["production_metrics"].read_text(encoding="utf-8")
        )
        production_accuracy = float(production["accuracy"])
        if accuracy < production_accuracy:
            raise ValueError(
                f"Candidate accuracy {accuracy:.3f} is below production accuracy "
                f"{production_accuracy:.3f}; production model was not changed."
            )

    print(f"Candidate accepted: accuracy={accuracy:.3f}")
    return metrics


def promote_model() -> dict[str, str]:
    """Copy an evaluated candidate into the production artifact paths."""
    evaluate_candidate()
    paths = _model_paths()
    for key in ("candidate_model", "candidate_metrics"):
        if not paths[key].is_file():
            raise FileNotFoundError(
                f"Cannot promote; required candidate file is missing: {paths[key]}"
            )

    paths["production_model"].parent.mkdir(parents=True, exist_ok=True)
    model_temp = paths["production_model"].with_suffix(".joblib.tmp")
    metrics_temp = paths["production_metrics"].with_suffix(".json.tmp")
    try:
        shutil.copy2(paths["candidate_model"], model_temp)
        shutil.copy2(paths["candidate_metrics"], metrics_temp)
        model_temp.replace(paths["production_model"])
        metrics_temp.replace(paths["production_metrics"])
    finally:
        model_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)

    print(f"Promoted model to {paths['production_model']}")
    return {
        "model_path": str(paths["production_model"]),
        "metrics_path": str(paths["production_metrics"]),
    }


def run_pipeline() -> dict[str, Any]:
    """Run all stages in order, promoting only an accepted candidate."""
    metrics = train_model()
    artifacts = promote_model()
    return {"metrics": metrics, "artifacts": artifacts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("train", "evaluate", "promote", "run"),
        help="Pipeline stage to execute",
    )
    action = parser.parse_args().action
    if action == "train":
        train_model()
    elif action == "evaluate":
        evaluate_candidate()
    elif action == "promote":
        promote_model()
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
