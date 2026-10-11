"""Reload and run the central and conservative FD001 inference bundles."""

from __future__ import annotations

import hashlib
import json
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from industrial_maintenance.classical import engineer_inference_features
from industrial_maintenance.data import make_windows
from industrial_maintenance.lstm import load_lstm_checkpoint, predict_lstm
from industrial_maintenance.postprocessing import apply_safety_margin


class ArtifactLoadError(RuntimeError):
    """Raised when an inference bundle is missing, changed, or inconsistent."""


class InferenceInputError(ValueError):
    """Raised when a trajectory cannot be used safely for inference."""


@dataclass(frozen=True)
class InferenceConfig:
    project_root: Path
    config_path: Path
    central_directory: Path
    safety_directory: Path
    host: str
    port: int
    minimum_end_cycle: int


@dataclass(frozen=True)
class RULPrediction:
    unit_id: int
    history_start_cycle: int
    last_cycle: int
    observed_cycles: int
    window_cycles_used: int
    rul_estimate: float
    rul_conservative: float
    safety_base_prediction: float
    safety_margin: float
    decision_value: float
    central_model_version: str
    safety_model_version: str


def _inside(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must remain inside the project root") from exc
    return path


def read_inference_config(path: str | Path = "configs/api.toml") -> InferenceConfig:
    source = Path(path).resolve()
    project_root = source.parent.parent.resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
        host = raw["server"]["host"]
        port = raw["server"]["port"]
        minimum_end_cycle = raw["input"]["minimum_end_cycle"]
        central_directory = raw["models"]["central_directory"]
        safety_directory = raw["models"]["safety_directory"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid API configuration {source}: {exc}") from exc
    if not isinstance(host, str) or not host.strip():
        raise ValueError("server.host must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be an integer between 1 and 65535")
    if (
        isinstance(minimum_end_cycle, bool)
        or not isinstance(minimum_end_cycle, int)
        or minimum_end_cycle < 1
    ):
        raise ValueError("input.minimum_end_cycle must be a positive integer")
    return InferenceConfig(
        project_root=project_root,
        config_path=source,
        central_directory=_inside(
            project_root, central_directory, "models.central_directory"
        ),
        safety_directory=_inside(
            project_root, safety_directory, "models.safety_directory"
        ),
        host=host,
        port=port,
        minimum_end_cycle=minimum_end_cycle,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactLoadError(f"Cannot read artifact: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactLoadError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactLoadError(f"{label} must contain a JSON object")
    return value


def _artifact_path(directory: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactLoadError(f"Missing {label} artifact name")
    path = (directory / value).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise ArtifactLoadError(f"{label} artifact escapes its bundle directory") from exc
    if not path.is_file():
        raise ArtifactLoadError(f"Missing {label} artifact: {path}")
    return path


def _project_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArtifactLoadError(f"Missing {label} path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactLoadError(f"{label} path escapes the project root") from exc
    if not path.is_file():
        raise ArtifactLoadError(f"Missing {label}: {path}")
    return path


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ArtifactLoadError(f"Missing or invalid SHA-256 for {label}")
    actual = _sha256(path)
    if actual != expected.lower():
        raise ArtifactLoadError(f"SHA-256 mismatch for {label}: {path}")
    return actual


def _feature_names(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ArtifactLoadError(f"{label} must be a non-empty string list")
    names = tuple(value)
    if len(names) != len(set(names)):
        raise ArtifactLoadError(f"{label} contains duplicate names")
    return names


def _positive_window_length(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArtifactLoadError(f"{label} must be a positive integer")
    return value


class DualRULPredictor:
    """Thread-safe predictor exposing central and conservative RUL estimates."""

    def __init__(
        self,
        *,
        config: InferenceConfig,
        central_manifest: dict[str, Any],
        safety_manifest: dict[str, Any],
        safety_summary: dict[str, Any],
        central_checkpoint: Any,
        central_preprocessor: Any,
        safety_model: Any,
        safety_preprocessor: Any,
        safety_margin: float,
        central_model_hash: str,
        safety_model_hash: str,
        calibration_hash: str,
    ) -> None:
        self.config = config
        self.central_manifest = central_manifest
        self.safety_manifest = safety_manifest
        self.safety_summary = safety_summary
        self.central_checkpoint = central_checkpoint
        self.central_preprocessor = central_preprocessor
        self.safety_model = safety_model
        self.safety_preprocessor = safety_preprocessor
        self.safety_margin = safety_margin
        self.central_model_hash = central_model_hash
        self.safety_model_hash = safety_model_hash
        self.calibration_hash = calibration_hash
        self.central_features = tuple(central_checkpoint.feature_names)
        self.safety_features = tuple(safety_preprocessor.feature_names_)
        self.central_window_length = _positive_window_length(
            central_manifest["input_contract"]["window_length"],
            "central input_contract.window_length",
        )
        self.safety_window_length = _positive_window_length(
            safety_manifest["input_contract"]["window_length"],
            "safety input_contract.window_length",
        )
        if self.central_window_length != self.safety_window_length:
            raise ArtifactLoadError("Central and safety bundles use different window lengths")
        self.window_length = self.central_window_length
        self.sensor_names = tuple(dict.fromkeys((*self.central_features, *self.safety_features)))
        self._lock = threading.Lock()

    @classmethod
    def load(cls, config: InferenceConfig) -> DualRULPredictor:
        central_manifest_path = config.central_directory / "final_model.json"
        central_manifest = _read_json(central_manifest_path, "central manifest")
        if central_manifest.get("schema_version") != 1 or central_manifest.get("model") != "lstm":
            raise ArtifactLoadError("Unsupported central model manifest")
        central_model_path = _artifact_path(
            config.central_directory, central_manifest.get("model_artifact"), "central model"
        )
        central_preprocessor_path = _artifact_path(
            config.central_directory,
            central_manifest.get("preprocessor_artifact"),
            "central preprocessor",
        )
        central_model_hash = _verify_hash(
            central_model_path, central_manifest.get("model_sha256"), "central model"
        )
        _verify_hash(
            central_preprocessor_path,
            central_manifest.get("preprocessor_sha256"),
            "central preprocessor",
        )
        try:
            central_checkpoint = load_lstm_checkpoint(central_model_path)
            central_preprocessor = joblib.load(central_preprocessor_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise ArtifactLoadError("Cannot reload central inference artifacts") from exc
        central_contract = central_manifest.get("input_contract", {})
        central_features = _feature_names(
            central_contract.get("feature_names"), "central input_contract.feature_names"
        )
        if central_features != tuple(central_checkpoint.feature_names):
            raise ArtifactLoadError("Central checkpoint feature order differs from its manifest")
        if tuple(getattr(central_preprocessor, "feature_names_", ())) != central_features:
            raise ArtifactLoadError("Central preprocessor feature order differs from its manifest")
        if central_contract.get("observed_length_required") is not True:
            raise ArtifactLoadError("Central manifest must require observed window lengths")
        if central_checkpoint.expects_observed_lengths is not True:
            raise ArtifactLoadError("Central checkpoint does not use observed window lengths")

        safety_manifest_path = config.safety_directory / "model_manifest.json"
        safety_manifest = _read_json(safety_manifest_path, "safety manifest")
        if safety_manifest.get("schema_version") != 1:
            raise ArtifactLoadError("Unsupported safety model manifest")
        if safety_manifest.get("official_test_used") is not False:
            raise ArtifactLoadError("Safety manifest does not prove official-test exclusion")
        evidence_path = _project_path(
            config.project_root, safety_manifest.get("selection_evidence"), "safety evidence"
        )
        safety_summary = _read_json(evidence_path, "safety calibration summary")
        if safety_summary.get("official_test_used_for_selection") is not False:
            raise ArtifactLoadError("Safety summary does not prove official-test exclusion")
        if safety_summary.get("selection", {}).get("constraints_met") is not True:
            raise ArtifactLoadError("Selected safety model does not meet its constraints")

        safety_artifacts = safety_manifest.get("artifacts", {})
        safety_model_path = _artifact_path(
            config.safety_directory, safety_artifacts.get("model"), "safety model"
        )
        safety_preprocessor_path = _artifact_path(
            config.safety_directory,
            safety_artifacts.get("preprocessor"),
            "safety preprocessor",
        )
        calibration_name = safety_manifest.get("safety_calibration", {}).get("artifact")
        calibration_path = _artifact_path(
            config.safety_directory, calibration_name, "safety calibration"
        )
        summary_artifacts = safety_summary.get("artifacts", {})
        safety_model_hash = _verify_hash(
            safety_model_path, summary_artifacts.get("model_sha256"), "safety model"
        )
        _verify_hash(
            safety_preprocessor_path,
            summary_artifacts.get("preprocessor_sha256"),
            "safety preprocessor",
        )
        calibration_hash = _verify_hash(
            calibration_path,
            summary_artifacts.get("calibration_sha256"),
            "safety calibration",
        )
        calibration = _read_json(calibration_path, "safety calibration")
        try:
            safety_model = joblib.load(safety_model_path)
            safety_preprocessor = joblib.load(safety_preprocessor_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise ArtifactLoadError("Cannot reload safety inference artifacts") from exc
        if not callable(getattr(safety_model, "predict", None)):
            raise ArtifactLoadError("Safety artifact is not a fitted predictor")
        safety_contract = safety_manifest.get("input_contract", {})
        safety_features = _feature_names(
            safety_contract.get("sensor_names"), "safety input_contract.sensor_names"
        )
        if tuple(getattr(safety_preprocessor, "feature_names_", ())) != safety_features:
            raise ArtifactLoadError("Safety preprocessor feature order differs from its manifest")
        engineered_names = _feature_names(
            safety_contract.get("engineered_feature_names"),
            "safety input_contract.engineered_feature_names",
        )
        n_features = getattr(safety_model, "n_features_in_", len(engineered_names))
        if int(n_features) != len(engineered_names):
            raise ArtifactLoadError("Safety model input size differs from its manifest")
        try:
            safety_margin = float(calibration["safety_margin_cycles"])
            manifest_margin = float(safety_manifest["safety_calibration"]["margin_cycles"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactLoadError("Invalid safety margin metadata") from exc
        if not np.isfinite(safety_margin) or safety_margin < 0:
            raise ArtifactLoadError("Safety margin must be finite and non-negative")
        if not np.isclose(safety_margin, manifest_margin, rtol=0.0, atol=1e-12):
            raise ArtifactLoadError("Safety margin differs between manifest and artifact")
        if calibration.get("official_test_used") is not False:
            raise ArtifactLoadError("Safety calibration does not prove official-test exclusion")
        if calibration.get("base_model") != safety_manifest.get("model"):
            raise ArtifactLoadError("Safety calibration base model differs from its manifest")
        if safety_summary.get("selection", {}).get("base_model") != safety_manifest.get("model"):
            raise ArtifactLoadError("Safety selection base model differs from its manifest")

        return cls(
            config=config,
            central_manifest=central_manifest,
            safety_manifest=safety_manifest,
            safety_summary=safety_summary,
            central_checkpoint=central_checkpoint,
            central_preprocessor=central_preprocessor,
            safety_model=safety_model,
            safety_preprocessor=safety_preprocessor,
            safety_margin=safety_margin,
            central_model_hash=central_model_hash,
            safety_model_hash=safety_model_hash,
            calibration_hash=calibration_hash,
        )

    @property
    def central_version(self) -> str:
        return f"lstm-{self.central_model_hash[:12]}"

    @property
    def safety_version(self) -> str:
        return f"{self.safety_manifest['model']}-{self.safety_model_hash[:12]}"

    def model_info(self) -> dict[str, Any]:
        central_metrics = self.central_manifest.get("official_test_metrics", {})
        safety_selection = self.safety_summary.get("selection", {})
        return {
            "window_length": self.window_length,
            "minimum_end_cycle": self.config.minimum_end_cycle,
            "required_sensors": list(self.sensor_names),
            "central": {
                "name": "lstm",
                "version": self.central_version,
                "role": "point_estimate",
                "evaluation": "official_test_fd001_v1",
                "metrics": central_metrics,
            },
            "conservative": {
                "name": self.safety_manifest["model"],
                "version": self.safety_version,
                "role": "maintenance_decision",
                "evaluation": "grouped_cross_validation_fd001_train_only",
                "safety_margin_cycles": self.safety_margin,
                "calibration_hash": self.calibration_hash,
                "metrics": safety_selection.get("metrics", {}),
                "constraints_met": safety_selection.get("constraints_met"),
            },
        }

    def predict_frame(self, frame: pd.DataFrame) -> RULPrediction:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise InferenceInputError("History must contain at least one observation")
        if "unit_id" not in frame or "cycle" not in frame:
            raise InferenceInputError("History must contain unit_id and cycle")
        missing = [name for name in self.sensor_names if name not in frame.columns]
        if missing:
            raise InferenceInputError(f"Missing required sensors: {', '.join(missing)}")
        try:
            metadata = frame.loc[:, ["unit_id", "cycle"]].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InferenceInputError("unit_id and cycle must be numeric") from exc
        if (
            not np.isfinite(metadata).all()
            or (metadata <= 0).any()
            or (metadata != np.floor(metadata)).any()
        ):
            raise InferenceInputError("unit_id and cycle must be finite positive integers")
        unit_ids = metadata[:, 0].astype(np.int64)
        cycles = metadata[:, 1].astype(np.int64)
        if len(np.unique(unit_ids)) != 1:
            raise InferenceInputError("One request must contain exactly one engine")
        if len(cycles) > 1 and np.any(np.diff(cycles) != 1):
            raise InferenceInputError(
                "Cycles must be ordered and consecutive, without duplicates or gaps"
            )
        if int(cycles[-1]) < self.config.minimum_end_cycle:
            raise InferenceInputError(
                f"The last cycle must be at least {self.config.minimum_end_cycle}; "
                "earlier cycles were excluded during training"
            )
        if len(frame) < self.window_length and int(cycles[0]) != 1:
            raise InferenceInputError(
                f"Histories shorter than {self.window_length} cycles must start at cycle 1"
            )
        unit_id = int(unit_ids[0])

        try:
            central_scaled = self.central_preprocessor.transform(frame)
            central_windows = make_windows(
                central_scaled,
                self.central_features,
                window_length=self.window_length,
                last_only=True,
            )
            safety_scaled = self.safety_preprocessor.transform(frame)
            safety_windows = make_windows(
                safety_scaled,
                self.safety_features,
                window_length=self.window_length,
                last_only=True,
            )
            safety_tabular = engineer_inference_features(
                safety_windows.X,
                safety_windows.observed_lengths,
                self.safety_features,
            )
        except ValueError as exc:
            raise InferenceInputError(str(exc)) from exc
        expected_engineered = tuple(
            self.safety_manifest["input_contract"]["engineered_feature_names"]
        )
        if safety_tabular.names != expected_engineered:
            raise ArtifactLoadError("Runtime safety feature order differs from its manifest")

        with self._lock:
            central_prediction = predict_lstm(
                self.central_checkpoint.model,
                central_windows.X,
                self.central_checkpoint.target_scaler,
                observed_lengths=central_windows.observed_lengths,
                batch_size=self.central_checkpoint.model.config.batch_size,
                device="cpu",
            )
            safety_base = np.asarray(
                self.safety_model.predict(safety_tabular.values), dtype=np.float64
            )
        conservative = apply_safety_margin(safety_base, self.safety_margin)
        if central_prediction.shape != (1,) or safety_base.shape != (1,):
            raise ArtifactLoadError("Bundles returned an unexpected prediction shape")
        values = np.asarray([central_prediction[0], safety_base[0], conservative[0]])
        if not np.isfinite(values).all():
            raise ArtifactLoadError("Bundles returned a non-finite prediction")
        return RULPrediction(
            unit_id=unit_id,
            history_start_cycle=int(cycles[0]),
            last_cycle=int(cycles[-1]),
            observed_cycles=len(frame),
            window_cycles_used=min(len(frame), self.window_length),
            rul_estimate=float(central_prediction[0]),
            rul_conservative=float(conservative[0]),
            safety_base_prediction=float(safety_base[0]),
            safety_margin=self.safety_margin,
            decision_value=float(conservative[0]),
            central_model_version=self.central_version,
            safety_model_version=self.safety_version,
        )
