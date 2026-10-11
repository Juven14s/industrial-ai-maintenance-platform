"""Self-contained tests for loading and running the dual inference bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from industrial_maintenance.classical import (  # noqa: E402
    MeanRULBaseline,
    engineer_inference_features,
)
from industrial_maintenance.data import (  # noqa: E402
    SENSOR_COLUMNS,
    SensorPreprocessor,
    make_windows,
)
from industrial_maintenance.inference import (  # noqa: E402
    ArtifactLoadError,
    DualRULPredictor,
    InferenceConfig,
    InferenceInputError,
    read_inference_config,
)
from industrial_maintenance.lstm import (  # noqa: E402
    LSTMConfig,
    LSTMRegressor,
    TargetScaler,
    predict_lstm,
    save_lstm_checkpoint,
)
from industrial_maintenance.postprocessing import apply_safety_margin  # noqa: E402

WINDOW_LENGTH = 5
FEATURE_NAMES = ("sensor_2", "sensor_3")
SAFETY_MARGIN = 3.0


@dataclass(frozen=True)
class SyntheticBundles:
    config: InferenceConfig
    central_model: LSTMRegressor
    central_scaler: TargetScaler
    central_preprocessor: SensorPreprocessor
    safety_model: MeanRULBaseline
    safety_preprocessor: SensorPreprocessor
    central_model_path: Path
    safety_model_path: Path
    calibration_path: Path
    central_manifest_path: Path
    safety_manifest_path: Path
    safety_summary_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _preprocessor_training_frame() -> pd.DataFrame:
    rows: dict[str, np.ndarray] = {
        "unit_id": np.ones(8, dtype=np.int64),
        "cycle": np.arange(1, 9, dtype=np.int64),
    }
    for index, sensor in enumerate(SENSOR_COLUMNS, start=1):
        if sensor == "sensor_2":
            rows[sensor] = np.linspace(10.0, 17.0, 8)
        elif sensor == "sensor_3":
            rows[sensor] = np.linspace(30.0, 23.0, 8)
        else:
            rows[sensor] = np.full(8, float(index))
    return pd.DataFrame(rows)


def _history(cycles: np.ndarray | list[int]) -> pd.DataFrame:
    cycle_values = np.asarray(cycles, dtype=np.int64)
    return pd.DataFrame(
        {
            "unit_id": np.full(len(cycle_values), 42, dtype=np.int64),
            "cycle": cycle_values,
            "sensor_2": 11.0 + 0.4 * cycle_values,
            "sensor_3": 29.0 - 0.3 * cycle_values,
        }
    )


def _build_bundles(tmp_path: Path) -> SyntheticBundles:
    project_root = tmp_path.resolve()
    central_directory = project_root / "bundles" / "central"
    safety_directory = project_root / "bundles" / "safety"
    central_directory.mkdir(parents=True)
    safety_directory.mkdir(parents=True)

    fit_frame = _preprocessor_training_frame()
    central_preprocessor = SensorPreprocessor().fit(fit_frame)
    safety_preprocessor = SensorPreprocessor().fit(fit_frame)
    assert central_preprocessor.feature_names_ == FEATURE_NAMES
    assert safety_preprocessor.feature_names_ == FEATURE_NAMES

    central_preprocessor_path = central_directory / "preprocessor.joblib"
    safety_preprocessor_path = safety_directory / "preprocessor.joblib"
    joblib.dump(central_preprocessor, central_preprocessor_path)
    joblib.dump(safety_preprocessor, safety_preprocessor_path)

    central_model = LSTMRegressor(
        LSTMConfig(
            input_size=len(FEATURE_NAMES),
            hidden_size=4,
            batch_size=8,
            max_epochs=1,
            patience=1,
            seed=9,
        )
    )
    for parameter in central_model.parameters():
        torch.nn.init.zeros_(parameter)
    central_model.expects_observed_lengths_ = True
    central_scaler = TargetScaler(mean=40.0, scale=2.0)
    central_model_path = save_lstm_checkpoint(
        central_directory / "lstm.pt",
        central_model,
        central_scaler,
        FEATURE_NAMES,
        metadata={"fixture": "synthetic"},
    )

    engineered_names = engineer_inference_features(
        np.zeros((1, WINDOW_LENGTH, len(FEATURE_NAMES)), dtype=np.float32),
        np.array([WINDOW_LENGTH]),
        FEATURE_NAMES,
    ).names
    safety_model = MeanRULBaseline().fit(
        np.zeros((2, len(engineered_names))),
        np.array([22.0, 22.0]),
    )
    safety_model_path = safety_directory / "model.joblib"
    joblib.dump(safety_model, safety_model_path)

    calibration_path = safety_directory / "calibration.json"
    _write_json(
        calibration_path,
        {
            "base_model": "synthetic_safety_model",
            "safety_margin_cycles": SAFETY_MARGIN,
            "prediction_rule": "max(base_prediction - safety_margin_cycles, 0)",
            "official_test_used": False,
        },
    )

    central_manifest_path = central_directory / "final_model.json"
    _write_json(
        central_manifest_path,
        {
            "schema_version": 1,
            "model": "lstm",
            "model_artifact": central_model_path.name,
            "model_sha256": _sha256(central_model_path),
            "preprocessor_artifact": central_preprocessor_path.name,
            "preprocessor_sha256": _sha256(central_preprocessor_path),
            "input_contract": {
                "feature_names": list(FEATURE_NAMES),
                "window_length": WINDOW_LENGTH,
                "observed_length_required": True,
            },
            "official_test_metrics": {"rmse": 1.0},
        },
    )

    safety_summary_path = project_root / "reports" / "safety_summary.json"
    _write_json(
        safety_summary_path,
        {
            "schema_version": 1,
            "official_test_used_for_selection": False,
            "selection": {
                "base_model": "synthetic_safety_model",
                "constraints_met": True,
                "metrics": {"overestimate_rate": 0.05},
            },
            "artifacts": {
                "model_sha256": _sha256(safety_model_path),
                "preprocessor_sha256": _sha256(safety_preprocessor_path),
                "calibration_sha256": _sha256(calibration_path),
            },
        },
    )

    safety_manifest_path = safety_directory / "model_manifest.json"
    _write_json(
        safety_manifest_path,
        {
            "schema_version": 1,
            "model": "synthetic_safety_model",
            "official_test_used": False,
            "selection_evidence": safety_summary_path.relative_to(project_root).as_posix(),
            "input_contract": {
                "window_length": WINDOW_LENGTH,
                "sensor_names": list(FEATURE_NAMES),
                "engineered_feature_names": list(engineered_names),
                "prediction_floor": 0.0,
            },
            "artifacts": {
                "model": safety_model_path.name,
                "preprocessor": safety_preprocessor_path.name,
            },
            "safety_calibration": {
                "artifact": calibration_path.name,
                "margin_cycles": SAFETY_MARGIN,
                "prediction_rule": "max(base_prediction - margin, 0)",
            },
        },
    )

    config_path = project_root / "configs" / "api.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# synthetic inference fixture\n", encoding="utf-8")
    config = InferenceConfig(
        project_root=project_root,
        config_path=config_path,
        central_directory=central_directory,
        safety_directory=safety_directory,
        host="127.0.0.1",
        port=8000,
        minimum_end_cycle=3,
    )
    return SyntheticBundles(
        config=config,
        central_model=central_model,
        central_scaler=central_scaler,
        central_preprocessor=central_preprocessor,
        safety_model=safety_model,
        safety_preprocessor=safety_preprocessor,
        central_model_path=central_model_path,
        safety_model_path=safety_model_path,
        calibration_path=calibration_path,
        central_manifest_path=central_manifest_path,
        safety_manifest_path=safety_manifest_path,
        safety_summary_path=safety_summary_path,
    )


def _local_predictions(bundle: SyntheticBundles, frame: pd.DataFrame) -> tuple[float, float, float]:
    central_scaled = bundle.central_preprocessor.transform(frame)
    central_windows = make_windows(
        central_scaled,
        FEATURE_NAMES,
        window_length=WINDOW_LENGTH,
        last_only=True,
    )
    central = predict_lstm(
        bundle.central_model,
        central_windows.X,
        bundle.central_scaler,
        observed_lengths=central_windows.observed_lengths,
        batch_size=8,
    )

    safety_scaled = bundle.safety_preprocessor.transform(frame)
    safety_windows = make_windows(
        safety_scaled,
        FEATURE_NAMES,
        window_length=WINDOW_LENGTH,
        last_only=True,
    )
    tabular = engineer_inference_features(
        safety_windows.X,
        safety_windows.observed_lengths,
        FEATURE_NAMES,
    )
    safety_base = bundle.safety_model.predict(tabular.values)
    conservative = apply_safety_margin(safety_base, SAFETY_MARGIN)
    return float(central[0]), float(safety_base[0]), float(conservative[0])


def test_integrity_checked_bundles_match_local_pipelines_for_short_history(tmp_path):
    bundle = _build_bundles(tmp_path)
    predictor = DualRULPredictor.load(bundle.config)
    frame = _history([1, 2, 3])

    expected_central, expected_base, expected_conservative = _local_predictions(bundle, frame)
    prediction = predictor.predict_frame(frame)

    assert prediction.unit_id == 42
    assert prediction.history_start_cycle == 1
    assert prediction.last_cycle == 3
    assert prediction.observed_cycles == 3
    assert prediction.window_cycles_used == 3
    assert prediction.rul_estimate == pytest.approx(expected_central)
    assert prediction.safety_base_prediction == pytest.approx(expected_base)
    assert prediction.rul_conservative == pytest.approx(expected_conservative)
    assert prediction.decision_value == prediction.rul_conservative
    assert prediction.safety_margin == SAFETY_MARGIN
    assert predictor.central_checkpoint.expects_observed_lengths


def test_long_history_uses_the_same_last_window_as_an_explicit_tail(tmp_path):
    bundle = _build_bundles(tmp_path)
    predictor = DualRULPredictor.load(bundle.config)
    full_history = _history(np.arange(1, 9))
    last_window = full_history.tail(WINDOW_LENGTH).reset_index(drop=True)

    full_prediction = predictor.predict_frame(full_history)
    tail_prediction = predictor.predict_frame(last_window)

    assert full_prediction.observed_cycles == 8
    assert full_prediction.window_cycles_used == WINDOW_LENGTH
    assert tail_prediction.history_start_cycle == 4
    assert full_prediction.rul_estimate == pytest.approx(tail_prediction.rul_estimate)
    assert full_prediction.safety_base_prediction == pytest.approx(
        tail_prediction.safety_base_prediction
    )
    assert full_prediction.rul_conservative == pytest.approx(tail_prediction.rul_conservative)


@pytest.mark.parametrize(
    ("artifact_name", "error_label"),
    [("central", "central model"), ("safety", "safety model")],
)
def test_loading_rejects_tampered_model_artifact_hashes(
    tmp_path, artifact_name: str, error_label: str
):
    bundle = _build_bundles(tmp_path)
    artifact = (
        bundle.central_model_path if artifact_name == "central" else bundle.safety_model_path
    )
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ArtifactLoadError, match=rf"SHA-256 mismatch for {error_label}"):
        DualRULPredictor.load(bundle.config)


def test_loading_rejects_feature_order_inconsistent_with_checkpoint(tmp_path):
    bundle = _build_bundles(tmp_path)
    manifest = json.loads(bundle.central_manifest_path.read_text(encoding="utf-8"))
    manifest["input_contract"]["feature_names"] = list(reversed(FEATURE_NAMES))
    _write_json(bundle.central_manifest_path, manifest)

    with pytest.raises(ArtifactLoadError, match="checkpoint feature order differs"):
        DualRULPredictor.load(bundle.config)


def test_loading_rejects_margin_inconsistent_with_calibration_artifact(tmp_path):
    bundle = _build_bundles(tmp_path)
    calibration = json.loads(bundle.calibration_path.read_text(encoding="utf-8"))
    calibration["safety_margin_cycles"] = SAFETY_MARGIN + 1.0
    _write_json(bundle.calibration_path, calibration)

    summary = json.loads(bundle.safety_summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["calibration_sha256"] = _sha256(bundle.calibration_path)
    _write_json(bundle.safety_summary_path, summary)

    with pytest.raises(ArtifactLoadError, match="margin differs"):
        DualRULPredictor.load(bundle.config)


def test_short_partial_history_is_rejected_instead_of_being_misleadingly_padded(tmp_path):
    predictor = DualRULPredictor.load(_build_bundles(tmp_path).config)

    with pytest.raises(InferenceInputError, match="must start at cycle 1"):
        predictor.predict_frame(_history([8, 9, 10]))


def test_history_before_minimum_supported_cycle_is_rejected(tmp_path):
    predictor = DualRULPredictor.load(_build_bundles(tmp_path).config)

    with pytest.raises(InferenceInputError, match="last cycle must be at least 3"):
        predictor.predict_frame(_history([1, 2]))


def test_configuration_without_model_directories_has_a_clear_error(tmp_path):
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    config_path = config_directory / "api.toml"
    config_path.write_text(
        '[server]\nhost = "127.0.0.1"\nport = 8000\n\n'
        "[input]\nminimum_end_cycle = 20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid API configuration"):
        read_inference_config(config_path)


def test_configuration_rejects_artifact_directories_outside_project(tmp_path):
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    config_path = config_directory / "api.toml"
    config_path.write_text(
        '[models]\ncentral_directory = "../outside"\n'
        'safety_directory = "artifacts/safety"\n\n'
        '[server]\nhost = "127.0.0.1"\nport = 8000\n\n'
        "[input]\nminimum_end_cycle = 20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must remain inside the project root"):
        read_inference_config(config_path)
