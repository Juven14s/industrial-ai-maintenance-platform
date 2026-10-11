"""Safety-oriented FD001 model selection without consulting the official test.

The validation protocol is grouped by engine and evaluates deterministic RUL
horizons. Overestimating RUL is explicitly more expensive than underestimating
it, while conventional accuracy metrics remain visible to prevent a trivial
always-zero solution.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from industrial_maintenance.classical import (
    PreparedWindows,
    engineer_window_features,
    fit_random_forest,
    validate_prepared_windows,
)
from industrial_maintenance.data import SensorPreprocessor, add_rul, load_cmapss, make_windows
from industrial_maintenance.train import engine_balancing_weights


@dataclass(frozen=True)
class QuantileCandidate:
    name: str
    quantile: float
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float


@dataclass(frozen=True)
class SafetyConfig:
    project_root: Path
    config_path: Path
    train_path: Path
    artifact_directory: Path
    report_directory: Path
    tracking_directory: Path
    experiment_name: str
    seed: int
    folds: int
    window_length: int
    minimum_end_cycle: int
    horizons: tuple[int, ...]
    overestimate_cost: float
    underestimate_cost: float
    maximum_overestimate_rate: float
    dangerous_margin: float
    random_forest: dict[str, Any]
    candidates: tuple[QuantileCandidate, ...]


def _project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must remain inside the project root") from exc
    return candidate


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def read_safety_config(path: str | Path = "configs/fd001_safety_v2.toml") -> SafetyConfig:
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent.resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    validation = raw["validation"]
    risk = raw["risk"]
    folds = _positive_int(validation["folds"], "validation.folds", minimum=2)
    horizons = tuple(_positive_int(value, "validation.horizons") for value in validation["horizons"])
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("validation.horizons must be non-empty, unique, and increasing")
    maximum_rate = float(risk["maximum_overestimate_rate"])
    if not np.isfinite(maximum_rate) or not 0 < maximum_rate < 1:
        raise ValueError("risk.maximum_overestimate_rate must lie strictly between 0 and 1")

    candidates: list[QuantileCandidate] = []
    for item in raw["quantile_boosting"]["candidates"]:
        quantile = float(item["quantile"])
        if not 0 < quantile < 1:
            raise ValueError("Each boosting quantile must lie strictly between 0 and 1")
        candidates.append(
            QuantileCandidate(
                name=str(item["name"]),
                quantile=quantile,
                learning_rate=_positive_float(item["learning_rate"], "learning_rate"),
                max_iter=_positive_int(item["max_iter"], "max_iter"),
                max_leaf_nodes=_positive_int(item["max_leaf_nodes"], "max_leaf_nodes", minimum=2),
                min_samples_leaf=_positive_int(item["min_samples_leaf"], "min_samples_leaf"),
                l2_regularization=_positive_float(
                    item["l2_regularization"], "l2_regularization"
                ),
            )
        )
    names = [candidate.name for candidate in candidates]
    if not candidates or len(names) != len(set(names)):
        raise ValueError("Quantile candidates must have unique non-empty names")

    return SafetyConfig(
        project_root=project_root,
        config_path=config_path,
        train_path=_project_path(
            project_root, raw["data"]["train_trajectories"], "data.train_trajectories"
        ),
        artifact_directory=_project_path(
            project_root, raw["output"]["artifact_directory"], "output.artifact_directory"
        ),
        report_directory=_project_path(
            project_root, raw["output"]["report_directory"], "output.report_directory"
        ),
        tracking_directory=_project_path(
            project_root, raw["tracking"]["directory"], "tracking.directory"
        ),
        experiment_name=str(raw["tracking"]["experiment_name"]),
        seed=_positive_int(validation["seed"], "validation.seed", minimum=0),
        folds=folds,
        window_length=_positive_int(validation["window_length"], "validation.window_length"),
        minimum_end_cycle=_positive_int(
            validation["minimum_end_cycle"], "validation.minimum_end_cycle"
        ),
        horizons=horizons,
        overestimate_cost=_positive_float(risk["overestimate_cost"], "overestimate_cost"),
        underestimate_cost=_positive_float(risk["underestimate_cost"], "underestimate_cost"),
        maximum_overestimate_rate=maximum_rate,
        dangerous_margin=_positive_float(
            risk["dangerous_overestimate_margin"], "dangerous_overestimate_margin"
        ),
        random_forest=dict(raw["random_forest_reference"]),
        candidates=tuple(candidates),
    )


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("values and weights must be non-empty one-dimensional arrays")
    if not 0 <= quantile <= 1 or not np.isfinite(values).all():
        raise ValueError("Invalid weighted quantile inputs")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and have a positive sum")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    threshold = quantile * cumulative[-1]
    return float(sorted_values[min(np.searchsorted(cumulative, threshold, side="left"), len(values) - 1)])


def safety_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
    *,
    overestimate_cost: float,
    underestimate_cost: float,
    dangerous_margin: float,
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if truth.ndim != 1 or predictions.shape != truth.shape or weights.shape != truth.shape:
        raise ValueError("Metric arrays must be one-dimensional and have equal shapes")
    if truth.size == 0 or not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite, non-empty arrays")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Metric weights are invalid")
    errors = predictions - truth
    absolute = np.abs(errors)
    positive = np.maximum(errors, 0.0)
    costs = np.where(errors > 0, overestimate_cost, underestimate_cost) * absolute
    nasa = np.where(errors < 0, np.expm1(-errors / 13.0), np.expm1(errors / 10.0))
    return {
        "mae": float(np.average(absolute, weights=weights)),
        "rmse": float(np.sqrt(np.average(np.square(errors), weights=weights))),
        "bias": float(np.average(errors, weights=weights)),
        "asymmetric_mae": float(np.average(costs, weights=weights)),
        "overestimate_rate": float(np.average(errors > 0, weights=weights)),
        "dangerous_overestimate_rate": float(
            np.average(errors > dangerous_margin, weights=weights)
        ),
        "mean_overestimate": float(np.average(positive, weights=weights)),
        "p90_overestimate": weighted_quantile(positive, 0.90, weights),
        "maximum_overestimate": float(positive.max()),
        "mean_nasa_score": float(np.average(nasa, weights=weights)),
    }


def select_safety_candidate(
    comparison: pd.DataFrame, maximum_overestimate_rate: float
) -> tuple[pd.Series, bool]:
    required = {"model", "overestimate_rate", "asymmetric_mae", "rmse"}
    if comparison.empty or not required.issubset(comparison.columns):
        raise ValueError("Comparison table is empty or incomplete")
    eligible = comparison[comparison["overestimate_rate"] <= maximum_overestimate_rate]
    constraint_met = not eligible.empty
    pool = eligible if constraint_met else comparison
    selected = pool.sort_values(
        ["asymmetric_mae", "dangerous_overestimate_rate", "rmse", "model"]
    ).iloc[0]
    return selected, constraint_met


def _to_prepared(batch: Any, feature_names: tuple[str, ...]) -> PreparedWindows:
    if batch.y is None:
        raise ValueError("Supervised windows require RUL targets")
    return validate_prepared_windows(
        batch.X,
        batch.y,
        batch.unit_ids,
        batch.end_cycles,
        batch.observed_lengths,
        feature_names,
    )


def _subset(windows: PreparedWindows, mask: np.ndarray) -> PreparedWindows:
    return validate_prepared_windows(
        windows.X[mask],
        windows.y[mask],
        windows.unit_ids[mask],
        windows.end_cycles[mask],
        windows.observed_lengths[mask],
        windows.feature_names,
    )


def _training_windows(frame: pd.DataFrame, preprocessor: SensorPreprocessor, config: SafetyConfig):
    normalized = preprocessor.transform(frame)
    windows = _to_prepared(
        make_windows(normalized, preprocessor.feature_names_, config.window_length),
        preprocessor.feature_names_,
    )
    mask = (windows.end_cycles >= config.minimum_end_cycle) & (windows.y > 0)
    return _subset(windows, mask)


def _validation_windows(frame: pd.DataFrame, preprocessor: SensorPreprocessor, config: SafetyConfig):
    normalized = preprocessor.transform(frame)
    windows = _to_prepared(
        make_windows(normalized, preprocessor.feature_names_, config.window_length),
        preprocessor.feature_names_,
    )
    rounded = windows.y.astype(np.int64)
    mask = (
        (windows.end_cycles >= config.minimum_end_cycle)
        & (windows.y > 0)
        & np.isin(rounded, np.asarray(config.horizons))
        & np.isclose(windows.y, rounded)
    )
    return _subset(windows, mask)


def _quantile_model(candidate: QuantileCandidate, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=candidate.quantile,
        learning_rate=candidate.learning_rate,
        max_iter=candidate.max_iter,
        max_leaf_nodes=candidate.max_leaf_nodes,
        min_samples_leaf=candidate.min_samples_leaf,
        l2_regularization=candidate.l2_regularization,
        random_state=seed,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _save_figures(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    destination: Path,
    selected_name: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    ordered = comparison.sort_values("asymmetric_mae")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["model"], ordered["asymmetric_mae"])
    axis.set(xlabel="MAE asymétrique (surestimation × 5)", title="Sélection sécurité — CV moteur")
    axis.invert_yaxis()
    figure.tight_layout()
    figure.savefig(destination / "asymmetric_cost.png", dpi=160)
    plt.close(figure)

    selected = predictions[predictions["model"] == selected_name]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(selected["true_rul"], selected["predicted_rul"], alpha=0.55)
    limit = float(max(selected["true_rul"].max(), selected["predicted_rul"].max()))
    axis.plot([0, limit], [0, limit], color="black", linewidth=1)
    axis.set(xlabel="RUL réelle", ylabel="RUL prédite", title=f"Prédictions hors pli — {selected_name}")
    figure.tight_layout()
    figure.savefig(destination / "selected_oof_predictions.png", dpi=160)
    plt.close(figure)


def _report(summary: dict[str, Any], comparison: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {row.model} | {row.mae:.3f} | {row.rmse:.3f} | {row.asymmetric_mae:.3f} | "
        f"{100 * row.overestimate_rate:.1f} % | "
        f"{100 * row.dangerous_overestimate_rate:.1f} % | {row.maximum_overestimate:.3f} |"
        for row in comparison.itertuples(index=False)
    )
    selection = summary["selection"]
    constraint = "respectée" if selection["safety_constraint_met"] else "non atteinte"
    return f"""# FD001 — sélection orientée sécurité, version 2

Cette expérience n'ouvre ni `test_FD001.txt` ni `RUL_FD001.txt`. Elle corrige l'objectif de
sélection : une surestimation coûte {summary['risk']['overestimate_cost']:.1f} fois une
sous-estimation de même amplitude.

## Protocole

- Validation croisée à {summary['validation']['folds']} plis, avec moteurs entièrement disjoints.
- Prétraitement réajusté dans chaque pli, sans information des moteurs évalués.
- Horizons déterministes : {summary['validation']['horizons']} cycles de RUL.
- Pondération donnant le même poids total à chaque moteur.
- Contrainte visée : au plus {100 * summary['risk']['maximum_overestimate_rate']:.0f} % de surestimations.
- Une surestimation supérieure à {summary['risk']['dangerous_margin']:.0f} cycles est comptée séparément.

## Prédictions hors pli

| Modèle | MAE | RMSE | MAE asymétrique | Surestimation | Surestimation > marge | Maximum positif |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Le modèle retenu est **{selection['model']}**. Sa contrainte de surestimation est **{constraint}**.
Le modèle final est réentraîné sur les 100 moteurs de `train_FD001.txt`. Aucun nouveau score de
test officiel n'est calculé : les cibles officielles déjà consultées pour la V1 ne servent jamais
à choisir ou corriger cette V2.

![Coût asymétrique](figures/asymmetric_cost.png)

![Prédictions hors pli](figures/selected_oof_predictions.png)
"""


def run_safety_experiment(
    config_path: str | Path = "configs/fd001_safety_v2.toml",
) -> dict[str, Any]:
    started = time.perf_counter()
    config = read_safety_config(config_path)
    raw = add_rul(load_cmapss(config.train_path))
    engine_ids = np.asarray(sorted(raw["unit_id"].unique()), dtype=np.int64)
    if config.folds > len(engine_ids):
        raise ValueError("Number of folds cannot exceed the number of engines")

    model_names = ["rf_squared_error", *[candidate.name for candidate in config.candidates]]
    prediction_rows: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    splitter = KFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    feature_names: tuple[str, ...] | None = None

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(engine_ids), start=1):
        train_ids = engine_ids[train_indices]
        validation_ids = engine_ids[validation_indices]
        train_frame = raw[raw["unit_id"].isin(train_ids)].copy()
        validation_frame = raw[raw["unit_id"].isin(validation_ids)].copy()
        preprocessor = SensorPreprocessor().fit(train_frame)
        train_windows = _training_windows(train_frame, preprocessor, config)
        validation_windows = _validation_windows(validation_frame, preprocessor, config)
        train_features = engineer_window_features(train_windows)
        validation_features = engineer_window_features(validation_windows)
        if train_features.names != validation_features.names:
            raise RuntimeError("Training and validation feature contracts differ")
        feature_names = train_features.names
        train_weights = engine_balancing_weights(train_windows.unit_ids)
        validation_weights = engine_balancing_weights(validation_windows.unit_ids)

        forest = fit_random_forest(
            train_features.values,
            train_windows.y,
            seed=config.seed + fold,
            sample_weight=train_weights,
            **config.random_forest,
        )
        models: dict[str, Any] = {"rf_squared_error": forest}
        models.update(
            {
                candidate.name: _quantile_model(candidate, config.seed + fold).fit(
                    train_features.values, train_windows.y, sample_weight=train_weights
                )
                for candidate in config.candidates
            }
        )
        for model_name, model in models.items():
            predictions = np.maximum(model.predict(validation_features.values), 0.0)
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "fold": fold,
                        "model": model_name,
                        "unit_id": validation_windows.unit_ids,
                        "end_cycle": validation_windows.end_cycles,
                        "true_rul": validation_windows.y,
                        "predicted_rul": predictions,
                        "sample_weight": validation_weights,
                        "error": predictions - validation_windows.y,
                    }
                )
            )
        fold_records.append(
            {
                "fold": fold,
                "train_units": train_ids.tolist(),
                "validation_units": validation_ids.tolist(),
                "train_windows": len(train_windows.y),
                "validation_windows": len(validation_windows.y),
            }
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metric_rows = []
    for model_name in model_names:
        subset = predictions[predictions["model"] == model_name]
        metric_rows.append(
            {
                "model": model_name,
                **safety_metrics(
                    subset["true_rul"].to_numpy(),
                    subset["predicted_rul"].to_numpy(),
                    subset["sample_weight"].to_numpy(),
                    overestimate_cost=config.overestimate_cost,
                    underestimate_cost=config.underestimate_cost,
                    dangerous_margin=config.dangerous_margin,
                ),
            }
        )
    comparison = pd.DataFrame(metric_rows).sort_values(
        ["asymmetric_mae", "dangerous_overestimate_rate", "rmse", "model"]
    )
    selected, constraint_met = select_safety_candidate(
        comparison, config.maximum_overestimate_rate
    )
    selected_name = str(selected["model"])

    final_preprocessor = SensorPreprocessor().fit(raw)
    final_windows = _training_windows(raw, final_preprocessor, config)
    final_features = engineer_window_features(final_windows)
    final_weights = engine_balancing_weights(final_windows.unit_ids)
    if selected_name == "rf_squared_error":
        final_model = fit_random_forest(
            final_features.values,
            final_windows.y,
            seed=config.seed,
            sample_weight=final_weights,
            **config.random_forest,
        )
        selected_parameters = {"kind": "random_forest", **config.random_forest}
    else:
        candidate = next(item for item in config.candidates if item.name == selected_name)
        final_model = _quantile_model(candidate, config.seed).fit(
            final_features.values, final_windows.y, sample_weight=final_weights
        )
        selected_parameters = {"kind": "hist_gradient_boosting_quantile", **asdict(candidate)}

    config.artifact_directory.mkdir(parents=True, exist_ok=True)
    config.report_directory.mkdir(parents=True, exist_ok=True)
    model_path = config.artifact_directory / "model.joblib"
    preprocessor_path = config.artifact_directory / "preprocessor.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(final_preprocessor, preprocessor_path)
    reloaded = joblib.load(model_path)
    probe = final_features.values[: min(16, len(final_features.values))]
    np.testing.assert_array_equal(reloaded.predict(probe), final_model.predict(probe))

    predictions_path = config.report_directory / "oof_predictions.csv"
    comparison_path = config.report_directory / "comparison.csv"
    predictions.to_csv(predictions_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    _save_figures(
        comparison,
        predictions,
        config.report_directory / "figures",
        selected_name,
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "FD001 safety-oriented cross-validation; official test excluded",
        "official_test_read": False,
        "official_test_used_for_selection": False,
        "config": str(config.config_path.relative_to(config.project_root)).replace("\\", "/"),
        "config_sha256": _sha256(config.config_path),
        "data": {
            "source": str(config.train_path.relative_to(config.project_root)).replace("\\", "/"),
            "sha256": _sha256(config.train_path),
            "engines": len(engine_ids),
            "final_training_windows": len(final_windows.y),
        },
        "validation": {
            "folds": config.folds,
            "seed": config.seed,
            "window_length": config.window_length,
            "minimum_end_cycle": config.minimum_end_cycle,
            "horizons": list(config.horizons),
            "fold_records": fold_records,
        },
        "risk": {
            "overestimate_cost": config.overestimate_cost,
            "underestimate_cost": config.underestimate_cost,
            "maximum_overestimate_rate": config.maximum_overestimate_rate,
            "dangerous_margin": config.dangerous_margin,
        },
        "metrics": comparison.to_dict(orient="records"),
        "selection": {
            "model": selected_name,
            "safety_constraint_met": constraint_met,
            "metrics": {key: float(selected[key]) for key in comparison.columns if key != "model"},
            "parameters": selected_parameters,
        },
        "final_model": {
            "trained_on_all_100_engines": len(engine_ids) == 100,
            "feature_names": list(final_preprocessor.feature_names_),
            "engineered_feature_names": list(feature_names or final_features.names),
            "model_path": str(model_path.relative_to(config.project_root)).replace("\\", "/"),
            "preprocessor_path": str(preprocessor_path.relative_to(config.project_root)).replace(
                "\\", "/"
            ),
            "model_sha256": _sha256(model_path),
            "preprocessor_sha256": _sha256(preprocessor_path),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = config.report_directory / "SAFETY_SELECTION.md"
    report_path.write_text(_report(summary, comparison), encoding="utf-8")
    summary_path = config.report_directory / "experiment_summary.json"
    _write_json(summary_path, summary)

    manifest = {
        "schema_version": 1,
        "model": selected_name,
        "parameters": selected_parameters,
        "selection_evidence": str(summary_path.relative_to(config.project_root)).replace("\\", "/"),
        "official_test_used": False,
        "input_contract": {
            "window_length": config.window_length,
            "sensor_names": list(final_preprocessor.feature_names_),
            "engineered_feature_names": list(final_features.names),
            "prediction_floor": 0.0,
        },
        "artifacts": {
            "model": model_path.name,
            "preprocessor": preprocessor_path.name,
        },
    }
    _write_json(config.artifact_directory / "model_manifest.json", manifest)

    config.tracking_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{(config.tracking_directory / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="safety_v2_grouped_cv") as run:
        mlflow.log_params(
            {
                "folds": config.folds,
                "overestimate_cost": config.overestimate_cost,
                "maximum_overestimate_rate": config.maximum_overestimate_rate,
                "selected_model": selected_name,
                "official_test_used": False,
            }
        )
        mlflow.log_metrics({f"selected_{key}": float(selected[key]) for key in comparison.columns if key != "model"})
        mlflow.log_artifacts(str(config.report_directory), artifact_path="safety_v2_report")
        summary["mlflow_run_id"] = run.info.run_id
    _write_json(summary_path, summary)
    return summary
