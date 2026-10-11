"""Train and compare FD001 RUL models without reading the official test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from industrial_maintenance.classical import (
    MeanRULBaseline,
    PreparedWindows,
    engineer_window_features,
    fit_random_forest,
    load_prepared_windows,
    regression_metrics,
)
from industrial_maintenance.data import SENSOR_COLUMNS
from industrial_maintenance.lstm import (
    LSTMConfig,
    fit_lstm,
    load_lstm_checkpoint,
    predict_lstm,
    save_lstm_checkpoint,
)


@dataclass(frozen=True)
class ForestCandidate:
    """One deliberately small Random Forest validation candidate."""

    name: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved milestone-2 configuration."""

    project_root: Path
    config_path: Path
    learning_windows: Path
    validation_windows: Path
    preprocessor: Path
    artifact_directory: Path
    report_directory: Path
    tracking_directory: Path
    experiment_name: str
    seed: int
    balance_engines: bool
    primary_metric: str
    minimum_end_cycle: int
    exclude_failure_windows: bool
    forest_candidates: tuple[ForestCandidate, ...]
    lstm_parameters: dict[str, Any]
    parameters: dict[str, Any]


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return result


def _project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty project-relative path")
    result = (root / value).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError(f"{name} must stay below the project root")
    return result


def read_config(path: str | Path = "configs/fd001_models.toml") -> ExperimentConfig:
    """Read and validate the validation-only experiment configuration."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        parameters = tomllib.load(handle)
    root = config_path.parent.parent
    try:
        data_directory = _project_path(root, parameters["data"]["directory"], "data.directory")
        learning = _project_path(
            data_directory, parameters["data"]["learning_windows"], "data.learning_windows"
        )
        validation = _project_path(
            data_directory,
            parameters["data"]["validation_windows"],
            "data.validation_windows",
        )
        preprocessor = _project_path(
            data_directory, parameters["data"]["preprocessor"], "data.preprocessor"
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
        experiment = parameters["experiment"]
        seed = _positive_integer(experiment["seed"], "experiment.seed")
        balance_engines = experiment["balance_engines"]
        if not isinstance(balance_engines, bool):
            raise ValueError("experiment.balance_engines must be a boolean")
        primary_metric = experiment["primary_metric"]
        if primary_metric != "rmse":
            raise ValueError("experiment.primary_metric must be 'rmse' for this milestone")
        minimum_end_cycle = _positive_integer(
            experiment["minimum_end_cycle"], "experiment.minimum_end_cycle"
        )
        exclude_failure_windows = experiment["exclude_failure_windows"]
        if not isinstance(exclude_failure_windows, bool):
            raise ValueError("experiment.exclude_failure_windows must be a boolean")

        candidates: list[ForestCandidate] = []
        for raw in parameters["random_forest"]["candidates"]:
            name = raw["name"]
            if (
                not isinstance(name, str)
                or not name
                or any(item.name == name for item in candidates)
            ):
                raise ValueError("Random Forest candidate names must be non-empty and unique")
            candidates.append(
                ForestCandidate(
                    name=name,
                    n_estimators=_positive_integer(raw["n_estimators"], f"{name}.n_estimators"),
                    max_depth=_positive_integer(raw["max_depth"], f"{name}.max_depth"),
                    min_samples_leaf=_positive_integer(
                        raw["min_samples_leaf"], f"{name}.min_samples_leaf"
                    ),
                    max_features=_finite_float(
                        raw["max_features"],
                        f"{name}.max_features",
                        minimum=0.000001,
                        maximum=1.0,
                    ),
                )
            )
        if not candidates:
            raise ValueError("At least one Random Forest candidate is required")

        lstm = parameters["lstm"]
        lstm_parameters = {
            "hidden_size": _positive_integer(lstm["hidden_size"], "lstm.hidden_size"),
            "num_layers": _positive_integer(lstm["num_layers"], "lstm.num_layers"),
            "dropout": _finite_float(
                lstm["dropout"], "lstm.dropout", minimum=0.0, maximum=0.999999
            ),
            "batch_size": _positive_integer(lstm["batch_size"], "lstm.batch_size"),
            "learning_rate": _finite_float(
                lstm["learning_rate"], "lstm.learning_rate", minimum=0.0000001
            ),
            "weight_decay": _finite_float(lstm["weight_decay"], "lstm.weight_decay", minimum=0.0),
            "max_epochs": _positive_integer(lstm["max_epochs"], "lstm.max_epochs"),
            "patience": _positive_integer(lstm["patience"], "lstm.patience"),
            "min_delta": _finite_float(lstm["min_delta"], "lstm.min_delta", minimum=0.0),
            "gradient_clip_norm": _finite_float(
                lstm["gradient_clip"], "lstm.gradient_clip", minimum=0.000001
            ),
            "num_threads": _positive_integer(lstm["num_threads"], "lstm.num_threads"),
        }
    except KeyError as exc:
        raise ValueError(f"Missing configuration key: {exc}") from exc

    outputs = (artifact_directory, report_directory, tracking_directory)
    for output in outputs:
        if output.is_relative_to(data_directory) or data_directory.is_relative_to(output):
            raise ValueError("Experiment outputs must not overlap prepared input data")
    for index, left in enumerate(outputs):
        for right in outputs[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                raise ValueError("Experiment output directories must not overlap")
    return ExperimentConfig(
        project_root=root,
        config_path=config_path,
        learning_windows=learning,
        validation_windows=validation,
        preprocessor=preprocessor,
        artifact_directory=artifact_directory,
        report_directory=report_directory,
        tracking_directory=tracking_directory,
        experiment_name=experiment_name,
        seed=seed,
        balance_engines=balance_engines,
        primary_metric=primary_metric,
        minimum_end_cycle=minimum_end_cycle,
        exclude_failure_windows=exclude_failure_windows,
        forest_candidates=tuple(candidates),
        lstm_parameters=lstm_parameters,
        parameters=parameters,
    )


def _subset_windows(windows: PreparedWindows, mask: np.ndarray) -> PreparedWindows:
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != (len(windows.y),) or not mask.any():
        raise ValueError("Window selection mask must retain at least one sample")
    return PreparedWindows(
        X=windows.X[mask],
        y=windows.y[mask],
        unit_ids=windows.unit_ids[mask],
        end_cycles=windows.end_cycles[mask],
        observed_lengths=windows.observed_lengths[mask],
        feature_names=windows.feature_names,
    )


def eligible_learning_windows(
    windows: PreparedWindows,
    minimum_end_cycle: int,
    exclude_failure_windows: bool,
) -> PreparedWindows:
    """Match the learning cutoff support to the simulated validation protocol."""
    mask = windows.end_cycles >= minimum_end_cycle
    if exclude_failure_windows:
        mask &= windows.y > 0
    return _subset_windows(windows, mask)


def engine_balancing_weights(unit_ids: np.ndarray) -> np.ndarray:
    """Give every engine equal total weight, normalized to mean sample weight one."""
    ids = np.asarray(unit_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("unit_ids must be a non-empty one-dimensional array")
    unique, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    if unique.size < 1 or (counts < 1).any():  # pragma: no cover - defensive
        raise ValueError("Cannot compute engine weights")
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float64)


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the standard asymmetric C-MAPSS score (lower is better)."""
    truth = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != predictions.shape or truth.size == 0:
        raise ValueError("NASA score inputs must be equal non-empty vectors")
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("NASA score inputs must be finite")
    errors = predictions - truth
    penalties = np.empty_like(errors)
    early = errors < 0.0
    penalties[early] = np.expm1(-errors[early] / 13.0)
    penalties[~early] = np.expm1(errors[~early] / 10.0)
    if not np.isfinite(penalties).all():
        raise ValueError("NASA score overflowed")
    return float(penalties.sum())


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = regression_metrics(y_true, y_pred)
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    result["bias"] = float(errors.mean())
    result["p90_absolute_error"] = float(np.quantile(np.abs(errors), 0.9))
    result["overestimate_rate"] = float(np.mean(errors > 0.0))
    result["nasa_score"] = nasa_score(y_true, y_pred)
    return result


def _configure_mlflow(config: ExperimentConfig) -> tuple[str, str]:
    """Use a local SQLite backend and a separate local artifact directory."""
    database = config.tracking_directory / "mlflow.db"
    artifact_directory = config.tracking_directory / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.experiment_name,
            artifact_location=artifact_directory.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return tracking_uri, experiment_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reported_path(path: Path, project_root: Path) -> str:
    """Prefer a project-relative path, while supporting isolated integration tests."""
    resolved = path.resolve()
    return (
        resolved.relative_to(project_root).as_posix()
        if resolved.is_relative_to(project_root)
        else resolved.as_posix()
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _log_run(
    name: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    artifacts: list[Path],
    history: tuple[Any, ...] = (),
) -> str:
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        for record in history:
            mlflow.log_metrics(
                {
                    "train_rmse": record.train_rmse,
                    "validation_rmse": record.validation_rmse,
                },
                step=record.epoch,
            )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id


def _prediction_rows(
    name: str,
    validation: PreparedWindows,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": name,
            "unit_id": validation.unit_ids,
            "end_cycle": validation.end_cycles,
            "observed_length": validation.observed_lengths,
            "true_rul": validation.y,
            "predicted_rul": predictions,
            "error": predictions - validation.y,
            "absolute_error": np.abs(predictions - validation.y),
        }
    )


def _save_figures(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    history: pd.DataFrame,
    best_epoch: int,
    destination: Path,
) -> list[Path]:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with plt.rc_context({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        lower = float(min(predictions["true_rul"].min(), predictions["predicted_rul"].min()))
        upper = float(max(predictions["true_rul"].max(), predictions["predicted_rul"].max()))
        for name, group in predictions.groupby("model", sort=False):
            axes[0].scatter(group["true_rul"], group["predicted_rul"], label=name, alpha=0.75)
            axes[1].scatter(group["true_rul"], group["error"], label=name, alpha=0.75)
        axes[0].plot([lower, upper], [lower, upper], color="0.35", linewidth=1)
        axes[0].set(xlabel="RUL réelle (cycles)", ylabel="RUL prédite (cycles)")
        axes[1].axhline(0.0, color="0.35", linewidth=1)
        axes[1].set(xlabel="RUL réelle (cycles)", ylabel="Erreur prédite - réelle (cycles)")
        axes[0].legend()
        figure.suptitle("FD001 — prédictions sur 20 moteurs de validation")
        figure.tight_layout()
        path = destination / "validation_predictions.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

        figure, axis = plt.subplots(figsize=(8, 4.5))
        ordered = comparison.sort_values("rmse", ascending=True)
        axis.barh(ordered["model"], ordered["rmse"])
        axis.set(xlabel="RMSE de validation (cycles)", title="Comparaison des modèles")
        figure.tight_layout()
        path = destination / "validation_rmse.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

        top = importance.nlargest(20, "importance").sort_values("importance")
        figure, axis = plt.subplots(figsize=(9, 6))
        axis.barh(top["feature"], top["importance"])
        axis.set(xlabel="Importance du Random Forest", title="20 caractéristiques principales")
        figure.tight_layout()
        path = destination / "random_forest_importance.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(history["epoch"], history["train_rmse"], label="apprentissage")
        axis.plot(history["epoch"], history["validation_rmse"], label="validation")
        best = history.loc[history["epoch"].eq(best_epoch)].iloc[0]
        axis.scatter([best["epoch"]], [best["validation_rmse"]], label="meilleure époque")
        axis.set(xlabel="Époque", ylabel="RMSE (cycles)", title="Apprentissage du LSTM")
        axis.legend()
        figure.tight_layout()
        path = destination / "lstm_history.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _markdown_report(summary: dict[str, Any], comparison: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {row.model} | {row.mae:.3f} | {row.rmse:.3f} | {row.bias:.3f} | "
        f"{row.p90_absolute_error:.3f} | {100 * row.overestimate_rate:.1f} % | "
        f"{row.nasa_score:.3f} |"
        for row in comparison.sort_values("rmse").itertuples()
    )
    data = summary["data"]
    selected = summary["selection"]
    lstm = summary["lstm"]
    weighting = (
        "chaque moteur possède le même poids total"
        if data["engine_balanced"]
        else "chaque fenêtre possède le même poids"
    )
    failure_filter = "et strictement avant la panne" if data["exclude_failure_windows"] else ""
    return f"""# FD001 — Comparaison des premiers modèles de RUL

Ce rapport présente uniquement les résultats sur la validation interne issue de
`train_FD001.txt`. Le test officiel et `RUL_FD001.txt` n'ont pas été lus.

## Protocole

- {data["learning_units"]} moteurs d'apprentissage et {data["validation_units"]} moteurs de validation disjoints.
- Fenêtres de {data["window_length"]} cycles et {data["sensor_count"]} capteurs, préparées au jalon 1.
- Apprentissage limité aux coupures de cycle >= {data["minimum_end_cycle"]} {failure_filter}.
- {data["eligible_learning_windows"]} fenêtres admissibles ; {weighting}.
- Une dernière fenêtre par moteur de validation, soit {data["validation_samples"]} prédictions.
- Cible RUL brute, non plafonnée. Sélection principale par RMSE de validation.

## Résultats de validation

| Modèle | MAE | RMSE | Biais | Erreur absolue P90 | Surestimation | Score NASA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Le modèle sélectionné sur ce protocole est **{selected["model"]}**, avec une RMSE de
**{selected["rmse"]:.3f} cycles** et une MAE de **{selected["mae"]:.3f} cycles**.

![Comparaison RMSE](figures/validation_rmse.png)

![Prédictions et résidus](figures/validation_predictions.png)

## Modèles

La référence constante prédit la moyenne pondérée des cibles admissibles. Le Random Forest
utilise sept statistiques causales par capteur — dernière valeur, moyenne, écart-type,
minimum, maximum, différence et pente — plus la longueur réellement observée. Les statistiques
ignorent le remplissage à gauche.

![Importance du Random Forest](figures/random_forest_importance.png)

Le LSTM compact utilise directement les séquences `30 × 15`. Sa cible est standardisée à partir
de l'apprentissage pondéré puis reconvertie en cycles. L'arrêt anticipé a restauré l'époque
{lstm["best_epoch"]} sur {lstm["epochs_ran"]} exécutées. Les longueurs réellement observées sont
fournies au réseau afin que le remplissage à gauche des historiques courts soit ignoré.

![Historique du LSTM](figures/lstm_history.png)

## Lecture et limites

Ces chiffres servent à choisir un modèle dans le protocole actuel. La validation ne contient que
20 moteurs et une coupure aléatoire fixée par moteur ; l'incertitude autour des métriques reste
donc importante. Les fenêtres d'un même moteur se chevauchent, mais aucun moteur n'est partagé
entre apprentissage et validation. La RUL brute est difficile à estimer loin de la panne lorsque
les capteurs décrivent peu l'âge initial.

Les fichiers [comparison.csv](comparison.csv), [validation_predictions.csv](validation_predictions.csv)
et [experiment_summary.json](experiment_summary.json) permettent de vérifier les valeurs. Les
artefacts rechargeables sont enregistrés hors Git dans `artifacts/models/FD001/`, et les runs
MLflow locaux dans une base SQLite sous `mlruns/`. Le manifeste du modèle sélectionné référence
aussi le préprocesseur, l'ordre des capteurs et la règle de fenêtrage nécessaires à l'inférence.

Le test officiel reste réservé au bilan final, après gel du protocole et du modèle.
"""


def run_experiment(config_path: str | Path = "configs/fd001_models.toml") -> dict[str, Any]:
    """Run baseline, Random Forest candidates, and one compact LSTM on validation."""
    started = time.perf_counter()
    config = read_config(config_path)
    learning_all = load_prepared_windows(config.learning_windows)
    validation = load_prepared_windows(config.validation_windows)
    if learning_all.feature_names != validation.feature_names:
        raise ValueError("Learning and validation feature orders differ")
    if learning_all.X.shape[1:] != validation.X.shape[1:]:
        raise ValueError("Learning and validation window shapes differ")
    if set(learning_all.unit_ids).intersection(validation.unit_ids):
        raise ValueError("Learning and validation engines overlap")
    if len(validation.y) != np.unique(validation.unit_ids).size:
        raise ValueError("Validation must contain exactly one window per engine")
    if (validation.y <= 0).any():
        raise ValueError("Validation targets must remain strictly positive after truncation")
    if not config.preprocessor.is_file():
        raise ValueError(f"Missing fitted preprocessor: {config.preprocessor}")

    learning = eligible_learning_windows(
        learning_all, config.minimum_end_cycle, config.exclude_failure_windows
    )
    weights = (
        engine_balancing_weights(learning.unit_ids)
        if config.balance_engines
        else np.ones(len(learning.y), dtype=np.float64)
    )
    config.artifact_directory.mkdir(parents=True, exist_ok=True)
    config.report_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = config.report_directory / "figures"
    config.tracking_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri, experiment_id = _configure_mlflow(config)

    learning_features = engineer_window_features(learning)
    validation_features = engineer_window_features(validation)
    if learning_features.names != validation_features.names:
        raise ValueError("Engineered feature orders differ")
    feature_spec_path = config.artifact_directory / "random_forest_features.json"
    _write_json(
        feature_spec_path,
        {
            "extractor": "padding_aware_window_statistics_v1",
            "source_feature_names": list(learning.feature_names),
            "engineered_feature_names": list(learning_features.names),
            "observed_lengths_required": True,
        },
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    run_ids: dict[str, str] = {}

    baseline = MeanRULBaseline().fit(learning_features.values, learning.y, sample_weight=weights)
    baseline_predictions = baseline.predict(validation_features.values)
    baseline_metrics = _metrics(validation.y, baseline_predictions)
    baseline_path = config.artifact_directory / "baseline.joblib"
    joblib.dump(baseline, baseline_path)
    reloaded_baseline = joblib.load(baseline_path)
    np.testing.assert_array_equal(
        reloaded_baseline.predict(validation_features.values), baseline_predictions
    )
    metric_rows.append({"model": "baseline_mean", **baseline_metrics})
    prediction_frames.append(_prediction_rows("baseline_mean", validation, baseline_predictions))
    run_ids["baseline_mean"] = _log_run(
        "baseline_mean",
        {
            "strategy": "engine_weighted_mean" if config.balance_engines else "window_mean",
            "eligible_windows": len(learning.y),
            "minimum_end_cycle": config.minimum_end_cycle,
        },
        baseline_metrics,
        [baseline_path, config.preprocessor],
    )

    forests: dict[str, Any] = {}
    forest_paths: list[Path] = []
    for candidate in config.forest_candidates:
        model = fit_random_forest(
            learning_features.values,
            learning.y,
            seed=config.seed,
            n_estimators=candidate.n_estimators,
            max_depth=candidate.max_depth,
            min_samples_leaf=candidate.min_samples_leaf,
            max_features=candidate.max_features,
            sample_weight=weights,
        )
        predictions = model.predict(validation_features.values)
        metrics = _metrics(validation.y, predictions)
        path = config.artifact_directory / f"{candidate.name}.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)
        np.testing.assert_array_equal(restored.predict(validation_features.values), predictions)
        forests[candidate.name] = model
        forest_paths.append(path)
        metric_rows.append({"model": candidate.name, **metrics})
        prediction_frames.append(_prediction_rows(candidate.name, validation, predictions))
        run_ids[candidate.name] = _log_run(
            candidate.name,
            {**asdict(candidate), "seed": config.seed, "engine_balanced": config.balance_engines},
            metrics,
            [path, feature_spec_path, config.preprocessor],
        )

    comparison_before_lstm = pd.DataFrame(metric_rows)
    forest_names = [candidate.name for candidate in config.forest_candidates]
    best_forest_name = str(
        comparison_before_lstm.loc[comparison_before_lstm["model"].isin(forest_names)]
        .sort_values([config.primary_metric, "mae", "model"])
        .iloc[0]["model"]
    )
    best_forest = forests[best_forest_name]
    best_forest_path = config.artifact_directory / "random_forest.joblib"
    joblib.dump(best_forest, best_forest_path)

    torch.set_num_threads(config.lstm_parameters["num_threads"])
    lstm_config = LSTMConfig(
        input_size=learning.X.shape[2],
        hidden_size=config.lstm_parameters["hidden_size"],
        num_layers=config.lstm_parameters["num_layers"],
        dropout=config.lstm_parameters["dropout"],
        batch_size=config.lstm_parameters["batch_size"],
        learning_rate=config.lstm_parameters["learning_rate"],
        weight_decay=config.lstm_parameters["weight_decay"],
        max_epochs=config.lstm_parameters["max_epochs"],
        patience=config.lstm_parameters["patience"],
        min_delta=config.lstm_parameters["min_delta"],
        gradient_clip_norm=config.lstm_parameters["gradient_clip_norm"],
        seed=config.seed,
    )
    lstm_result = fit_lstm(
        learning.X,
        learning.y,
        validation.X,
        validation.y,
        lstm_config,
        train_sample_weights=weights,
        validation_sample_weights=np.ones(len(validation.y), dtype=np.float32),
        train_observed_lengths=learning.observed_lengths,
        validation_observed_lengths=validation.observed_lengths,
        device="cpu",
    )
    lstm_predictions = predict_lstm(
        lstm_result.model,
        validation.X,
        lstm_result.target_scaler,
        observed_lengths=validation.observed_lengths,
        batch_size=lstm_config.batch_size,
        device="cpu",
    )
    lstm_metrics = _metrics(validation.y, lstm_predictions)
    lstm_path = config.artifact_directory / "lstm.pt"
    save_lstm_checkpoint(
        lstm_path,
        lstm_result.model,
        lstm_result.target_scaler,
        learning.feature_names,
        metadata={
            "best_epoch": lstm_result.best_epoch,
            "eligible_windows": len(learning.y),
            "engine_balanced": config.balance_engines,
        },
    )
    restored_lstm = load_lstm_checkpoint(lstm_path)
    reloaded_predictions = predict_lstm(
        restored_lstm.model,
        validation.X,
        restored_lstm.target_scaler,
        observed_lengths=validation.observed_lengths,
        batch_size=lstm_config.batch_size,
    )
    np.testing.assert_array_equal(reloaded_predictions, lstm_predictions)
    history = pd.DataFrame([asdict(row) for row in lstm_result.history])
    history_path = config.report_directory / "lstm_history.csv"
    history.to_csv(history_path, index=False)
    metric_rows.append({"model": "lstm", **lstm_metrics})
    prediction_frames.append(_prediction_rows("lstm", validation, lstm_predictions))
    run_ids["lstm"] = _log_run(
        "lstm",
        {**asdict(lstm_config), "engine_balanced": config.balance_engines},
        {**lstm_metrics, "best_epoch": float(lstm_result.best_epoch)},
        [lstm_path, history_path, config.preprocessor],
        lstm_result.history,
    )

    comparison = pd.DataFrame(metric_rows).sort_values([config.primary_metric, "mae", "model"])
    comparison_path = config.report_directory / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = config.report_directory / "validation_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    importance = pd.DataFrame(
        {"feature": learning_features.names, "importance": best_forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_path = config.report_directory / "random_forest_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    figure_paths = _save_figures(
        comparison,
        predictions,
        importance,
        history,
        lstm_result.best_epoch,
        figure_directory,
    )

    selected_row = comparison.iloc[0]
    selected_model = str(selected_row["model"])
    selected_artifact = (
        lstm_path
        if selected_model == "lstm"
        else baseline_path
        if selected_model == "baseline_mean"
        else best_forest_path
    )
    selected_manifest_path = config.artifact_directory / "selected_model.json"
    model_input = (
        "engineered_statistics"
        if selected_model in forest_names
        else "scaled_sequence"
        if selected_model == "lstm"
        else "sample_count_only"
    )
    _write_json(
        selected_manifest_path,
        {
            "schema_version": 1,
            "model": selected_model,
            "artifact": selected_artifact.name,
            "artifact_sha256": _sha256(selected_artifact),
            "model_input": model_input,
            "preprocessor": {
                "path": config.preprocessor.relative_to(config.project_root).as_posix(),
                "sha256": _sha256(config.preprocessor),
            },
            "input_contract": {
                "required_columns": ["unit_id", "cycle", *learning.feature_names],
                "accepted_cmapss_sensor_columns": list(SENSOR_COLUMNS),
                "feature_names": list(learning.feature_names),
                "window_length": int(learning.X.shape[1]),
                "padding": "left_repeat_first_observation",
                "observed_length_required": selected_model in forest_names
                or selected_model == "lstm",
                "random_forest_feature_spec": feature_spec_path.name
                if selected_model in forest_names
                else None,
                "random_forest_feature_spec_sha256": _sha256(feature_spec_path)
                if selected_model in forest_names
                else None,
            },
            "validation_metrics": {
                key: float(selected_row[key])
                for key in [
                    "mae",
                    "rmse",
                    "bias",
                    "p90_absolute_error",
                    "overestimate_rate",
                    "nasa_score",
                ]
            },
            "selection_metric": config.primary_metric,
            "official_test_evaluated": False,
        },
    )

    implementation_paths = [
        Path(__file__).with_name(name) for name in ["classical.py", "lstm.py", "train.py"]
    ]
    summary: dict[str, Any] = {
        "schema_version": 2,
        "stage": "FD001 validation comparison; official test not evaluated",
        "official_test_read": False,
        "config": config.parameters,
        "config_sha256": _sha256(config.config_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "torch",
                    "mlflow",
                    "matplotlib",
                    "joblib",
                ]
            },
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
        },
        "data": {
            "learning_artifact": config.learning_windows.relative_to(
                config.project_root
            ).as_posix(),
            "learning_sha256": _sha256(config.learning_windows),
            "validation_artifact": config.validation_windows.relative_to(
                config.project_root
            ).as_posix(),
            "validation_sha256": _sha256(config.validation_windows),
            "preprocessor_artifact": config.preprocessor.relative_to(
                config.project_root
            ).as_posix(),
            "preprocessor_sha256": _sha256(config.preprocessor),
            "learning_units": int(np.unique(learning.unit_ids).size),
            "validation_units": int(np.unique(validation.unit_ids).size),
            "all_learning_windows": len(learning_all.y),
            "eligible_learning_windows": len(learning.y),
            "validation_samples": len(validation.y),
            "validation_target_min": float(validation.y.min()),
            "validation_target_max": float(validation.y.max()),
            "validation_targets_at_most_30": int((validation.y <= 30).sum()),
            "minimum_end_cycle": config.minimum_end_cycle,
            "exclude_failure_windows": config.exclude_failure_windows,
            "window_length": int(learning.X.shape[1]),
            "sensor_count": int(learning.X.shape[2]),
            "engine_balanced": config.balance_engines,
        },
        "metrics": comparison.to_dict(orient="records"),
        "selection": {
            "model": selected_model,
            "primary_metric": config.primary_metric,
            "mae": float(selected_row["mae"]),
            "rmse": float(selected_row["rmse"]),
            "bias": float(selected_row["bias"]),
            "p90_absolute_error": float(selected_row["p90_absolute_error"]),
            "overestimate_rate": float(selected_row["overestimate_rate"]),
            "nasa_score": float(selected_row["nasa_score"]),
            "artifact": selected_artifact.relative_to(config.project_root).as_posix(),
        },
        "random_forest": {
            "selected_candidate": best_forest_name,
            "engineered_feature_count": len(learning_features.names),
        },
        "lstm": {
            "best_epoch": lstm_result.best_epoch,
            "epochs_ran": len(lstm_result.history),
            "stopped_early": lstm_result.stopped_early,
            "target_mean": lstm_result.target_scaler.mean,
            "target_scale": lstm_result.target_scaler.scale,
        },
        "mlflow": {
            "backend": "sqlite",
            "database": (config.tracking_directory / "mlflow.db")
            .relative_to(config.project_root)
            .as_posix(),
            "tracking_directory": config.tracking_directory.relative_to(
                config.project_root
            ).as_posix(),
            "tracking_uri": tracking_uri,
            "experiment_name": config.experiment_name,
            "experiment_id": experiment_id,
            "run_ids": run_ids,
        },
        "implementation": [
            {
                "path": _reported_path(path, config.project_root),
                "sha256": _sha256(path),
            }
            for path in implementation_paths
        ],
        "duration_seconds": time.perf_counter() - started,
        "artifacts": [],
    }
    report_path = config.report_directory / "MODEL_COMPARISON.md"
    report_path.write_text(_markdown_report(summary, comparison), encoding="utf-8")
    summary_path = config.report_directory / "experiment_summary.json"
    artifact_paths = [
        baseline_path,
        *forest_paths,
        best_forest_path,
        feature_spec_path,
        lstm_path,
        selected_manifest_path,
        comparison_path,
        predictions_path,
        importance_path,
        history_path,
        report_path,
        *figure_paths,
    ]
    summary["artifacts"] = [
        {
            "path": path.relative_to(config.project_root).as_posix(),
            "sha256": _sha256(path),
        }
        for path in artifact_paths
    ]
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fd001_models.toml"))
    args = parser.parse_args()
    try:
        summary = run_experiment(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "selection": summary["selection"],
                "duration_seconds": summary["duration_seconds"],
                "official_test_read": summary["official_test_read"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
