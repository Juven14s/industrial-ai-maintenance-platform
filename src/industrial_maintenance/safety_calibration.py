"""Cross-fitted safety-margin calibration for the FD001 conservative model."""

from __future__ import annotations

import json
import time
import tomllib
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from industrial_maintenance.classical import engineer_window_features
from industrial_maintenance.data import SensorPreprocessor, add_rul, load_cmapss
from industrial_maintenance.postprocessing import apply_safety_margin
from industrial_maintenance.safety import (
    _quantile_model,
    _sha256,
    _training_windows,
    _write_json,
    read_safety_config,
    safety_metrics,
    weighted_quantile,
)
from industrial_maintenance.train import engine_balancing_weights

REQUIRED_COLUMNS = {
    "fold",
    "model",
    "unit_id",
    "end_cycle",
    "true_rul",
    "predicted_rul",
    "sample_weight",
    "error",
}


def cross_fitted_margin_predictions(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    target_overestimate_rate: float,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Apply to each fold a margin estimated only from all other folds."""
    if not REQUIRED_COLUMNS.issubset(predictions.columns):
        raise ValueError("OOF predictions are missing required columns")
    if not 0 < target_overestimate_rate < 1:
        raise ValueError("target_overestimate_rate must lie strictly between 0 and 1")
    source = predictions[predictions["model"] == model_name].copy()
    folds = sorted(int(value) for value in source["fold"].unique())
    if len(folds) < 2:
        raise ValueError("Cross-fitted calibration requires at least two folds")
    outputs: list[pd.DataFrame] = []
    margins: dict[int, float] = {}
    for fold in folds:
        calibration = source[source["fold"] != fold]
        evaluation = source[source["fold"] == fold].copy()
        if calibration.empty or evaluation.empty:
            raise ValueError("Every calibration fold must contain predictions")
        margin = max(
            0.0,
            weighted_quantile(
                calibration["error"].to_numpy(dtype=np.float64),
                1.0 - target_overestimate_rate,
                calibration["sample_weight"].to_numpy(dtype=np.float64),
            ),
        )
        evaluation["uncalibrated_prediction"] = evaluation["predicted_rul"]
        evaluation["predicted_rul"] = apply_safety_margin(
            evaluation["predicted_rul"].to_numpy(dtype=np.float64), margin
        )
        evaluation["error"] = evaluation["predicted_rul"] - evaluation["true_rul"]
        evaluation["safety_margin"] = margin
        evaluation["base_model"] = model_name
        evaluation["calibration_target_rate"] = target_overestimate_rate
        outputs.append(evaluation)
        margins[fold] = margin
    return pd.concat(outputs, ignore_index=True), margins


def _weighted_rate(frame: pd.DataFrame, threshold: float = 0.0) -> float:
    weights = frame["sample_weight"].to_numpy(dtype=np.float64)
    errors = frame["error"].to_numpy(dtype=np.float64)
    return float(np.average(errors > threshold, weights=weights))


def calibration_candidate_metrics(
    predictions: pd.DataFrame,
    *,
    overestimate_cost: float,
    underestimate_cost: float,
    dangerous_margin: float,
) -> dict[str, float]:
    metrics = safety_metrics(
        predictions["true_rul"].to_numpy(dtype=np.float64),
        predictions["predicted_rul"].to_numpy(dtype=np.float64),
        predictions["sample_weight"].to_numpy(dtype=np.float64),
        overestimate_cost=overestimate_cost,
        underestimate_cost=underestimate_cost,
        dangerous_margin=dangerous_margin,
    )
    fold_rates = [_weighted_rate(group) for _, group in predictions.groupby("fold")]
    horizon_rates = [_weighted_rate(group) for _, group in predictions.groupby("true_rul")]
    fold_danger = [
        _weighted_rate(group, dangerous_margin) for _, group in predictions.groupby("fold")
    ]
    metrics.update(
        {
            "maximum_fold_overestimate_rate": max(fold_rates),
            "maximum_horizon_overestimate_rate": max(horizon_rates),
            "maximum_fold_dangerous_rate": max(fold_danger),
        }
    )
    return metrics


def select_calibrated_candidate(
    comparison: pd.DataFrame,
    *,
    maximum_overestimate_rate: float,
    maximum_fold_overestimate_rate: float,
    maximum_horizon_overestimate_rate: float,
) -> tuple[pd.Series, bool]:
    eligible = comparison[
        (comparison["overestimate_rate"] <= maximum_overestimate_rate)
        & (
            comparison["maximum_fold_overestimate_rate"]
            <= maximum_fold_overestimate_rate
        )
        & (
            comparison["maximum_horizon_overestimate_rate"]
            <= maximum_horizon_overestimate_rate
        )
    ]
    constraint_met = not eligible.empty
    pool = eligible if constraint_met else comparison
    selected = pool.sort_values(
        [
            "asymmetric_mae",
            "dangerous_overestimate_rate",
            "maximum_fold_overestimate_rate",
            "rmse",
            "candidate",
        ]
    ).iloc[0]
    return selected, constraint_met


def _configuration(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    calibration = raw.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Missing [calibration] configuration")
    rates = tuple(float(value) for value in calibration["target_overestimate_rates"])
    if not rates or any(not 0 < value < 1 for value in rates):
        raise ValueError("Calibration target rates must lie strictly between 0 and 1")
    project_root = path.parent.parent.resolve()
    safety_config = (project_root / raw["base"]["safety_config"]).resolve()
    try:
        safety_config.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("base.safety_config must remain inside the project root") from exc
    return {
        "safety_config": safety_config,
        "target_rates": rates,
        "maximum_fold_rate": float(calibration["maximum_fold_overestimate_rate"]),
        "maximum_horizon_rate": float(calibration["maximum_horizon_overestimate_rate"]),
    }


def _markdown(summary: dict[str, Any], comparison: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {row.candidate} | {row.mae:.3f} | {row.rmse:.3f} | "
        f"{row.asymmetric_mae:.3f} | {100 * row.overestimate_rate:.1f} % | "
        f"{100 * row.dangerous_overestimate_rate:.1f} % | "
        f"{100 * row.maximum_fold_overestimate_rate:.1f} % | "
        f"{100 * row.maximum_horizon_overestimate_rate:.1f} % |"
        for row in comparison.itertuples(index=False)
    )
    selection = summary["selection"]
    status = "respectées" if selection["constraints_met"] else "non atteintes"
    return f"""# FD001 — calibration croisée de la marge de sécurité

La marge appliquée à chaque pli est estimée uniquement avec les erreurs hors pli des quatre autres
plis. `test_FD001.txt` et `RUL_FD001.txt` ne sont jamais lus. Une marge positive est soustraite à
la prédiction, avec un plancher à zéro cycle.

| Candidat | MAE | RMSE | MAE asymétrique | Surestimation | > 10 cycles | Pire pli | Pire horizon |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Le candidat retenu est **{selection['candidate']}** et les contraintes sont **{status}**. Sa marge
finale de **{selection['final_safety_margin']:.3f} cycles** est calculée sur toutes les erreurs hors
pli du modèle de base. Le compromis est volontaire : une estimation plus prudente augmente la MAE
et la RMSE classiques, mais réduit le risque de repousser une maintenance à cause d'une RUL trop
optimiste.
"""


def run_safety_calibration(
    config_path: str | Path = "configs/fd001_safety_calibration.toml",
) -> dict[str, Any]:
    started = time.perf_counter()
    calibration_config_path = Path(config_path).resolve()
    calibration = _configuration(calibration_config_path)
    config = read_safety_config(calibration["safety_config"])
    predictions_path = config.report_directory / "oof_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError("Run the safety cross-validation before calibration")
    predictions = pd.read_csv(predictions_path)
    available_models = sorted(
        name for name in predictions["model"].unique() if str(name).startswith("hgb_quantile_")
    )
    if not available_models:
        raise ValueError("No quantile model is available for safety calibration")

    adjusted_frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    fold_margins: dict[str, dict[str, float]] = {}
    for model_name in available_models:
        for target_rate in calibration["target_rates"]:
            adjusted, margins = cross_fitted_margin_predictions(
                predictions,
                model_name=model_name,
                target_overestimate_rate=target_rate,
            )
            candidate_name = f"{model_name}_calibrated_{int(100 * target_rate):02d}pct"
            adjusted["candidate"] = candidate_name
            adjusted_frames.append(adjusted)
            fold_margins[candidate_name] = {str(key): value for key, value in margins.items()}
            rows.append(
                {
                    "candidate": candidate_name,
                    "base_model": model_name,
                    "target_overestimate_rate": target_rate,
                    **calibration_candidate_metrics(
                        adjusted,
                        overestimate_cost=config.overestimate_cost,
                        underestimate_cost=config.underestimate_cost,
                        dangerous_margin=config.dangerous_margin,
                    ),
                }
            )

    adjusted_predictions = pd.concat(adjusted_frames, ignore_index=True)
    comparison = pd.DataFrame(rows)
    selected, constraints_met = select_calibrated_candidate(
        comparison,
        maximum_overestimate_rate=config.maximum_overestimate_rate,
        maximum_fold_overestimate_rate=calibration["maximum_fold_rate"],
        maximum_horizon_overestimate_rate=calibration["maximum_horizon_rate"],
    )
    selected_candidate = str(selected["candidate"])
    selected_model = str(selected["base_model"])
    selected_target = float(selected["target_overestimate_rate"])
    base_errors = predictions[predictions["model"] == selected_model]
    final_margin = max(
        0.0,
        weighted_quantile(
            base_errors["error"].to_numpy(dtype=np.float64),
            1.0 - selected_target,
            base_errors["sample_weight"].to_numpy(dtype=np.float64),
        ),
    )

    candidate_config = next(
        candidate for candidate in config.candidates if candidate.name == selected_model
    )
    raw = add_rul(load_cmapss(config.train_path))
    preprocessor = SensorPreprocessor().fit(raw)
    windows = _training_windows(raw, preprocessor, config)
    features = engineer_window_features(windows)
    weights = engine_balancing_weights(windows.unit_ids)
    model = _quantile_model(candidate_config, config.seed).fit(
        features.values, windows.y, sample_weight=weights
    )
    config.artifact_directory.mkdir(parents=True, exist_ok=True)
    model_path = config.artifact_directory / "model.joblib"
    preprocessor_path = config.artifact_directory / "preprocessor.joblib"
    calibration_artifact_path = config.artifact_directory / "calibration.json"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    reloaded = joblib.load(model_path)
    probe = features.values[: min(16, len(features.values))]
    np.testing.assert_array_equal(reloaded.predict(probe), model.predict(probe))
    _write_json(
        calibration_artifact_path,
        {
            "method": "cross_fitted_one_sided_residual_margin",
            "base_model": selected_model,
            "target_overestimate_rate": selected_target,
            "safety_margin_cycles": final_margin,
            "prediction_rule": "max(base_prediction - safety_margin_cycles, 0)",
            "official_test_used": False,
        },
    )

    config.report_directory.mkdir(parents=True, exist_ok=True)
    adjusted_path = config.report_directory / "calibrated_oof_predictions.csv"
    comparison_path = config.report_directory / "calibrated_comparison.csv"
    adjusted_predictions.to_csv(adjusted_path, index=False)
    comparison.sort_values("asymmetric_mae").to_csv(comparison_path, index=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "stage": "FD001 cross-fitted safety calibration; official test excluded",
        "official_test_read": False,
        "official_test_used_for_selection": False,
        "calibration_config": str(
            calibration_config_path.relative_to(config.project_root)
        ).replace("\\", "/"),
        "calibration_config_sha256": _sha256(calibration_config_path),
        "safety_config": str(config.config_path.relative_to(config.project_root)).replace(
            "\\", "/"
        ),
        "safety_config_sha256": _sha256(config.config_path),
        "source_predictions": str(predictions_path.relative_to(config.project_root)).replace(
            "\\", "/"
        ),
        "source_predictions_sha256": _sha256(predictions_path),
        "constraints": {
            "maximum_overestimate_rate": config.maximum_overestimate_rate,
            "maximum_fold_overestimate_rate": calibration["maximum_fold_rate"],
            "maximum_horizon_overestimate_rate": calibration["maximum_horizon_rate"],
            "dangerous_margin_cycles": config.dangerous_margin,
        },
        "comparison": comparison.to_dict(orient="records"),
        "fold_margins": fold_margins,
        "selection": {
            "candidate": selected_candidate,
            "base_model": selected_model,
            "target_overestimate_rate": selected_target,
            "final_safety_margin": final_margin,
            "constraints_met": constraints_met,
            "metrics": {
                key: float(selected[key])
                for key in comparison.columns
                if key not in {"candidate", "base_model"}
            },
        },
        "artifacts": {
            "model": str(model_path.relative_to(config.project_root)).replace("\\", "/"),
            "preprocessor": str(preprocessor_path.relative_to(config.project_root)).replace(
                "\\", "/"
            ),
            "calibration": str(
                calibration_artifact_path.relative_to(config.project_root)
            ).replace("\\", "/"),
            "model_sha256": _sha256(model_path),
            "preprocessor_sha256": _sha256(preprocessor_path),
            "calibration_sha256": _sha256(calibration_artifact_path),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = config.report_directory / "SAFETY_CALIBRATION.md"
    report_path.write_text(_markdown(summary, comparison.sort_values("asymmetric_mae")), encoding="utf-8")
    summary_path = config.report_directory / "calibration_summary.json"
    _write_json(summary_path, summary)

    manifest_path = config.artifact_directory / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "model": selected_model,
            "parameters": {"kind": "hist_gradient_boosting_quantile", **candidate_config.__dict__},
            "selection_evidence": str(summary_path.relative_to(config.project_root)).replace(
                "\\", "/"
            ),
            "safety_calibration": {
                "artifact": calibration_artifact_path.name,
                "margin_cycles": final_margin,
                "prediction_rule": "max(base_prediction - margin, 0)",
            },
            "official_test_used": False,
        }
    )
    _write_json(manifest_path, manifest)

    tracking_uri = f"sqlite:///{(config.tracking_directory / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name="safety_v2_cross_fitted_calibration") as run:
        mlflow.log_params(
            {
                "base_model": selected_model,
                "target_overestimate_rate": selected_target,
                "safety_margin_cycles": final_margin,
                "official_test_used": False,
            }
        )
        mlflow.log_metrics(
            {
                f"selected_{key}": float(selected[key])
                for key in comparison.columns
                if key not in {"candidate", "base_model"}
            }
        )
        mlflow.log_artifacts(str(config.report_directory), artifact_path="safety_calibration")
        summary["mlflow_run_id"] = run.info.run_id
    _write_json(summary_path, summary)
    return summary
