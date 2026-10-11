"""Retrain the frozen FD001 LSTM on all training engines and evaluate the official test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from industrial_maintenance.classical import regression_metrics
from industrial_maintenance.data import (
    SENSOR_COLUMNS,
    SensorPreprocessor,
    add_rul,
    load_cmapss,
    load_rul_targets,
    make_windows,
)
from industrial_maintenance.lstm import (
    LSTMConfig,
    fit_lstm_fixed_epochs,
    load_lstm_checkpoint,
    predict_lstm,
    save_lstm_checkpoint,
)
from industrial_maintenance.train import engine_balancing_weights, nasa_score


@dataclass(frozen=True)
class FinalEvaluationConfig:
    """Resolved paths and frozen-selection assertions for the official evaluation."""

    project_root: Path
    config_path: Path
    train_trajectories: Path
    test_trajectories: Path
    test_targets: Path
    selection_summary: Path
    selection_summary_sha256: str
    selected_model: str
    fixed_epochs: int
    artifact_directory: Path
    report_directory: Path
    tracking_directory: Path
    experiment_name: str
    num_threads: int
    parameters: dict[str, Any]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty project-relative path")
    result = (root / value).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError(f"{name} must stay below the project root")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reported_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(project_root).as_posix()
        if resolved.is_relative_to(project_root)
        else resolved.as_posix()
    )


def read_final_config(path: str | Path = "configs/fd001_final.toml") -> FinalEvaluationConfig:
    """Read the final-evaluation config without opening training or test data."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        parameters = tomllib.load(handle)
    root = config_path.parent.parent
    try:
        train_trajectories = _project_path(
            root, parameters["data"]["train_trajectories"], "data.train_trajectories"
        )
        test_trajectories = _project_path(
            root, parameters["data"]["test_trajectories"], "data.test_trajectories"
        )
        test_targets = _project_path(root, parameters["data"]["test_targets"], "data.test_targets")
        selection_summary = _project_path(
            root, parameters["selection"]["summary"], "selection.summary"
        )
        expected_hash = parameters["selection"]["summary_sha256"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("selection.summary_sha256 must be a lowercase SHA-256")
        selected_model = parameters["selection"]["model"]
        if selected_model != "lstm":
            raise ValueError("selection.model must be the validation-selected 'lstm'")
        fixed_epochs = _positive_integer(
            parameters["selection"]["fixed_epochs"], "selection.fixed_epochs"
        )
        artifact_directory = _project_path(
            root, parameters["output"]["artifact_directory"], "output.artifact_directory"
        )
        report_directory = _project_path(
            root, parameters["output"]["report_directory"], "output.report_directory"
        )
        tracking_directory = _project_path(
            root, parameters["tracking"]["directory"], "tracking.directory"
        )
        experiment_name = parameters["tracking"]["experiment_name"]
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("tracking.experiment_name must be a non-empty string")
        num_threads = _positive_integer(parameters["runtime"]["num_threads"], "runtime.num_threads")
    except KeyError as exc:
        raise ValueError(f"Missing configuration key: {exc}") from exc

    inputs = (train_trajectories, test_trajectories, test_targets, selection_summary)
    outputs = (artifact_directory, report_directory, tracking_directory)
    for output in outputs:
        if any(output == item or output.is_relative_to(item) for item in inputs):
            raise ValueError("Output directories must not overlap evaluation inputs")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Output directories must not overlap")
    return FinalEvaluationConfig(
        project_root=root,
        config_path=config_path,
        train_trajectories=train_trajectories,
        test_trajectories=test_trajectories,
        test_targets=test_targets,
        selection_summary=selection_summary,
        selection_summary_sha256=expected_hash,
        selected_model=selected_model,
        fixed_epochs=fixed_epochs,
        artifact_directory=artifact_directory,
        report_directory=report_directory,
        tracking_directory=tracking_directory,
        experiment_name=experiment_name,
        num_threads=num_threads,
        parameters=parameters,
    )


def _read_frozen_selection(config: FinalEvaluationConfig) -> dict[str, Any]:
    if not config.selection_summary.is_file():
        raise ValueError(f"Missing validation summary: {config.selection_summary}")
    actual_hash = _sha256(config.selection_summary)
    if actual_hash != config.selection_summary_sha256:
        raise ValueError("Validation summary hash differs from the frozen selection evidence")
    try:
        summary = json.loads(config.selection_summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Validation summary is not valid JSON") from exc
    if summary.get("official_test_read") is not False:
        raise ValueError("Selection evidence must predate official-test evaluation")
    if summary.get("selection", {}).get("model") != config.selected_model:
        raise ValueError("Configured final model differs from validation selection")
    if summary.get("selection", {}).get("primary_metric") != "rmse":
        raise ValueError("The frozen primary metric must be RMSE")
    if summary.get("lstm", {}).get("best_epoch") != config.fixed_epochs:
        raise ValueError("Fixed final epochs differ from the validation-selected epoch")
    experiment = summary.get("config", {}).get("experiment", {})
    if experiment.get("balance_engines") is not True:
        raise ValueError("Frozen selection must balance engines")
    if experiment.get("exclude_failure_windows") is not True:
        raise ValueError("Frozen selection must exclude failure windows")
    return summary


def _complete_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = regression_metrics(y_true, y_pred)
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    result.update(
        bias=float(errors.mean()),
        p90_absolute_error=float(np.quantile(np.abs(errors), 0.9)),
        overestimate_rate=float(np.mean(errors > 0.0)),
        nasa_score=nasa_score(y_true, y_pred),
    )
    return result


def _configure_mlflow(directory: Path, experiment_name: str) -> tuple[str, str]:
    database = directory / "mlflow.db"
    artifact_directory = directory / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_directory.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri, experiment_id


def _save_figure(predictions: pd.DataFrame, destination: Path) -> Path:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    lower = float(min(predictions["true_rul"].min(), predictions["predicted_rul"].min()))
    upper = float(max(predictions["true_rul"].max(), predictions["predicted_rul"].max()))
    with plt.rc_context({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(predictions["true_rul"], predictions["predicted_rul"], alpha=0.75)
        axes[0].plot([lower, upper], [lower, upper], color="0.35", linewidth=1)
        axes[0].set(xlabel="RUL réelle (cycles)", ylabel="RUL prédite (cycles)")
        axes[1].scatter(predictions["true_rul"], predictions["error"], alpha=0.75)
        axes[1].axhline(0.0, color="0.35", linewidth=1)
        axes[1].set(xlabel="RUL réelle (cycles)", ylabel="Erreur prédite - réelle (cycles)")
        figure.suptitle("FD001 — évaluation finale sur 100 moteurs officiels de test")
        figure.tight_layout()
        figure.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(figure)
    return destination


def _report_markdown(
    metrics: dict[str, float],
    validation_metrics: dict[str, float],
    data: dict[str, Any],
    fixed_epochs: int,
) -> str:
    return f"""# FD001 — Évaluation finale sur le test officiel

Cette évaluation a été exécutée après gel du modèle sur la validation interne. Le test officiel
n'a servi ni à choisir l'architecture, ni à fixer les hyperparamètres, ni à déterminer le nombre
d'époques.

## Protocole final

- LSTM sélectionné sur la validation interne avec une RMSE de {validation_metrics["rmse"]:.3f} cycles.
- Prétraitement réajusté sur les {data["training_units"]} moteurs complets de `train_FD001.txt`.
- {data["eligible_training_windows"]} fenêtres admissibles, avec poids total identique par moteur.
- Entraînement fixé à {fixed_epochs} époques, sans arrêt anticipé sur le test.
- Une dernière fenêtre de {data["window_length"]} cycles par moteur officiel de test.
- {data["test_units"]} cibles lues dans `RUL_FD001.txt`, alignées par identifiant de moteur.

## Résultat officiel

| MAE | RMSE | Biais | Erreur absolue P90 | Surestimation | Score NASA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| {metrics["mae"]:.3f} | {metrics["rmse"]:.3f} | {metrics["bias"]:.3f} | {metrics["p90_absolute_error"]:.3f} | {100 * metrics["overestimate_rate"]:.1f} % | {metrics["nasa_score"]:.3f} |

![Prédictions et résidus du test officiel](figures/official_test_predictions.png)

Les résultats détaillés se trouvent dans [official_test_predictions.csv](official_test_predictions.csv).
Ils constituent le bilan du modèle gelé. Toute amélioration ultérieure devra être conçue sur un
nouveau protocole de validation et présentée comme une nouvelle version, sans transformer ce test
en jeu de réglage.
"""


def run_official_evaluation(
    config_path: str | Path = "configs/fd001_final.toml",
) -> dict[str, Any]:
    """Run the frozen one-time FD001 evaluation and create the deployment bundle."""
    started = time.perf_counter()
    config = read_final_config(config_path)
    selection = _read_frozen_selection(config)
    for required in (config.train_trajectories, config.test_trajectories, config.test_targets):
        if not required.is_file():
            raise ValueError(f"Missing official FD001 input: {required}")

    experiment = selection["config"]["experiment"]
    lstm_parameters = selection["config"]["lstm"]
    window_length = int(selection["data"]["window_length"])
    minimum_end_cycle = int(experiment["minimum_end_cycle"])
    seed = int(experiment["seed"])

    train_raw = load_cmapss(config.train_trajectories)
    test_raw = load_cmapss(config.test_trajectories)
    train_labeled = add_rul(train_raw)
    preprocessor = SensorPreprocessor().fit(train_labeled)
    train_scaled = preprocessor.transform(train_labeled)
    test_scaled = preprocessor.transform(test_raw)
    feature_names = tuple(preprocessor.feature_names_)

    train_windows = make_windows(
        train_scaled,
        feature_names=feature_names,
        window_length=window_length,
        last_only=False,
    )
    if train_windows.y is None:  # pragma: no cover - add_rul guarantees labels
        raise RuntimeError("Final training windows unexpectedly lack RUL targets")
    eligible = (train_windows.end_cycles >= minimum_end_cycle) & (train_windows.y > 0)
    if not eligible.any():
        raise ValueError("Frozen training filter retained no windows")
    train_x = train_windows.X[eligible]
    train_y = train_windows.y[eligible]
    train_unit_ids = train_windows.unit_ids[eligible]
    train_lengths = train_windows.observed_lengths[eligible]
    weights = engine_balancing_weights(train_unit_ids)

    test_windows = make_windows(
        test_scaled,
        feature_names=feature_names,
        window_length=window_length,
        last_only=True,
    )
    if test_windows.y is not None:
        raise RuntimeError("Official test trajectories must not contain reconstructed labels")
    expected_ids = np.arange(1, len(test_windows.unit_ids) + 1, dtype=np.int64)
    if not np.array_equal(test_windows.unit_ids, expected_ids):
        raise ValueError("Official test windows are not ordered one per contiguous engine ID")
    test_y = load_rul_targets(config.test_targets, expected_count=len(test_windows.unit_ids))

    config.artifact_directory.mkdir(parents=True, exist_ok=True)
    config.report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = config.report_directory / "figures"
    config.tracking_directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.num_threads)
    lstm_config = LSTMConfig(
        input_size=len(feature_names),
        hidden_size=int(lstm_parameters["hidden_size"]),
        num_layers=int(lstm_parameters["num_layers"]),
        dropout=float(lstm_parameters["dropout"]),
        batch_size=int(lstm_parameters["batch_size"]),
        learning_rate=float(lstm_parameters["learning_rate"]),
        weight_decay=float(lstm_parameters["weight_decay"]),
        max_epochs=config.fixed_epochs,
        patience=int(lstm_parameters["patience"]),
        min_delta=float(lstm_parameters["min_delta"]),
        gradient_clip_norm=float(lstm_parameters["gradient_clip"]),
        seed=seed,
    )

    training_started = time.perf_counter()
    result = fit_lstm_fixed_epochs(
        train_x,
        train_y,
        lstm_config,
        train_sample_weights=weights,
        train_observed_lengths=train_lengths,
        device="cpu",
    )
    training_seconds = time.perf_counter() - training_started

    model_path = config.artifact_directory / "lstm.pt"
    preprocessor_path = config.artifact_directory / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    save_lstm_checkpoint(
        model_path,
        result.model,
        result.target_scaler,
        feature_names,
        metadata={
            "stage": "final_fit_on_all_train_FD001_engines",
            "fixed_epochs": config.fixed_epochs,
            "seed": seed,
            "selection_summary_sha256": config.selection_summary_sha256,
            "official_test_used_for_training": False,
        },
    )

    inference_started = time.perf_counter()
    predictions = predict_lstm(
        result.model,
        test_windows.X,
        result.target_scaler,
        observed_lengths=test_windows.observed_lengths,
        batch_size=lstm_config.batch_size,
        device="cpu",
    )
    inference_seconds = time.perf_counter() - inference_started
    restored = load_lstm_checkpoint(model_path)
    restored_predictions = predict_lstm(
        restored.model,
        test_windows.X,
        restored.target_scaler,
        observed_lengths=test_windows.observed_lengths,
        batch_size=lstm_config.batch_size,
        device="cpu",
    )
    np.testing.assert_array_equal(restored_predictions, predictions)
    restored_preprocessor = joblib.load(preprocessor_path)
    if tuple(restored_preprocessor.feature_names_) != feature_names:
        raise RuntimeError("Reloaded preprocessor feature order changed")

    metrics = _complete_metrics(test_y, predictions)
    prediction_frame = pd.DataFrame(
        {
            "unit_id": test_windows.unit_ids,
            "end_cycle": test_windows.end_cycles,
            "observed_length": test_windows.observed_lengths,
            "true_rul": test_y,
            "predicted_rul": predictions,
            "error": predictions - test_y,
            "absolute_error": np.abs(predictions - test_y),
        }
    )
    predictions_path = config.report_directory / "official_test_predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)
    history = pd.DataFrame([asdict(row) for row in result.history])
    history_path = config.report_directory / "final_training_history.csv"
    history.to_csv(history_path, index=False)
    figure_path = _save_figure(prediction_frame, figure_directory / "official_test_predictions.png")

    validation_metrics = {
        key: float(selection["selection"][key])
        for key in ("mae", "rmse", "bias", "p90_absolute_error", "overestimate_rate", "nasa_score")
    }
    data_summary = {
        "training_units": int(train_raw["unit_id"].nunique()),
        "test_units": int(test_raw["unit_id"].nunique()),
        "eligible_training_windows": int(eligible.sum()),
        "test_samples": len(test_y),
        "window_length": window_length,
        "sensor_count": len(feature_names),
        "test_target_min": float(test_y.min()),
        "test_target_max": float(test_y.max()),
        "test_targets_at_most_30": int((test_y <= 30).sum()),
    }
    report_path = config.report_directory / "OFFICIAL_TEST_EVALUATION.md"
    report_path.write_text(
        _report_markdown(metrics, validation_metrics, data_summary, config.fixed_epochs),
        encoding="utf-8",
    )
    metrics_path = config.report_directory / "official_test_metrics.json"
    _write_json(metrics_path, metrics)

    manifest_path = config.artifact_directory / "final_model.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "model": "lstm",
            "model_artifact": model_path.name,
            "model_sha256": _sha256(model_path),
            "preprocessor_artifact": preprocessor_path.name,
            "preprocessor_sha256": _sha256(preprocessor_path),
            "selection_evidence": {
                "path": config.selection_summary.relative_to(config.project_root).as_posix(),
                "sha256": config.selection_summary_sha256,
                "validation_rmse": validation_metrics["rmse"],
            },
            "training": {
                "source": config.train_trajectories.relative_to(config.project_root).as_posix(),
                "units": data_summary["training_units"],
                "fixed_epochs": config.fixed_epochs,
                "seed": seed,
                "official_test_used": False,
            },
            "input_contract": {
                "required_columns": ["unit_id", "cycle", *feature_names],
                "accepted_cmapss_sensor_columns": list(SENSOR_COLUMNS),
                "feature_names": list(feature_names),
                "window_length": window_length,
                "padding": "left_repeat_first_observation",
                "observed_length_required": True,
            },
            "official_test_metrics": metrics,
        },
    )

    tracking_uri, experiment_id = _configure_mlflow(
        config.tracking_directory, config.experiment_name
    )
    implementation_paths = [
        Path(__file__).with_name(name) for name in ("data.py", "lstm.py", "evaluate.py")
    ]
    artifact_paths = [
        model_path,
        preprocessor_path,
        manifest_path,
        predictions_path,
        metrics_path,
        history_path,
        report_path,
        figure_path,
    ]
    with mlflow.start_run(run_name="lstm_official_test_frozen") as run:
        mlflow.log_params(
            {
                "model": "lstm",
                "seed": seed,
                "fixed_epochs": config.fixed_epochs,
                "window_length": window_length,
                "sensor_count": len(feature_names),
                "eligible_training_windows": int(eligible.sum()),
                "engine_balanced": True,
                "official_test_used_for_selection": False,
                "selection_summary_sha256": config.selection_summary_sha256,
            }
        )
        mlflow.log_metrics(
            {
                **metrics,
                "validation_reference_rmse": validation_metrics["rmse"],
                "training_seconds": training_seconds,
                "inference_ms_per_engine": 1000.0 * inference_seconds / len(test_y),
                "model_size_bytes": float(model_path.stat().st_size),
            }
        )
        for row in result.history:
            mlflow.log_metrics(
                {
                    "final_fit_train_rmse": row.train_rmse,
                    "final_fit_train_mse_standardized": row.train_mse_standardized,
                },
                step=row.epoch,
            )
        summary: dict[str, Any] = {
            "schema_version": 1,
            "stage": "FD001 frozen-model official test evaluation",
            "official_test_read": True,
            "official_test_used_for_selection": False,
            "hyperparameters_frozen_before_test": True,
            "config": config.parameters,
            "config_sha256": _sha256(config.config_path),
            "selection": {
                "summary": config.selection_summary.relative_to(config.project_root).as_posix(),
                "summary_sha256": config.selection_summary_sha256,
                "model": config.selected_model,
                "validation_metrics": validation_metrics,
                "fixed_epochs": config.fixed_epochs,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "torch",
                        "mlflow",
                        "matplotlib",
                        "joblib",
                    )
                },
                "device": "cpu",
                "torch_threads": torch.get_num_threads(),
            },
            "data": {
                **data_summary,
                "train_sha256": _sha256(config.train_trajectories),
                "test_sha256": _sha256(config.test_trajectories),
                "test_targets_sha256": _sha256(config.test_targets),
                "feature_names": list(feature_names),
            },
            "official_test_metrics": metrics,
            "runtime": {
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "inference_ms_per_engine": 1000.0 * inference_seconds / len(test_y),
                "model_size_bytes": model_path.stat().st_size,
                "total_seconds": time.perf_counter() - started,
            },
            "mlflow": {
                "backend": "sqlite",
                "database": (config.tracking_directory / "mlflow.db")
                .relative_to(config.project_root)
                .as_posix(),
                "tracking_uri": tracking_uri,
                "experiment_name": config.experiment_name,
                "experiment_id": experiment_id,
                "run_id": run.info.run_id,
            },
            "implementation": [
                {
                    "path": _reported_path(path, config.project_root),
                    "sha256": _sha256(path),
                }
                for path in implementation_paths
            ],
            "artifacts": [
                {
                    "path": path.relative_to(config.project_root).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in artifact_paths
            ],
        }
        summary_path = config.report_directory / "official_test_summary.json"
        _write_json(summary_path, summary)
        for artifact in [*artifact_paths, summary_path]:
            mlflow.log_artifact(str(artifact))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fd001_final.toml"))
    args = parser.parse_args()
    try:
        summary = run_official_evaluation(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Official evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "official_test_metrics": summary["official_test_metrics"],
                "fixed_epochs": summary["selection"]["fixed_epochs"],
                "official_test_used_for_selection": summary["official_test_used_for_selection"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
