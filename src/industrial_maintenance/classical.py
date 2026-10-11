"""Classical RUL baselines built from the prepared causal window artifacts.

This module only consumes ``.npz`` files produced by the FD001 preparation
stage.  It does not know about, or open, the official C-MAPSS test files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

WINDOW_STATISTICS: tuple[str, ...] = (
    "last",
    "mean",
    "std",
    "min",
    "max",
    "delta",
    "slope",
)


@dataclass(frozen=True)
class PreparedWindows:
    """A validated supervised window artifact produced by milestone 1."""

    X: np.ndarray
    y: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    observed_lengths: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TabularFeatures:
    """Causal tabular representation used by the classical models."""

    values: np.ndarray
    names: tuple[str, ...]


def _one_dimensional_finite(values: np.ndarray, name: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},)")
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} must contain only finite values")
    return numeric


def _validated_window_inputs(
    X: np.ndarray,
    observed_lengths: np.ndarray,
    feature_names: tuple[str, ...] | list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate the target-independent part of a causal window batch."""
    inputs = np.asarray(X)
    if inputs.ndim != 3:
        raise ValueError("X must have shape (n_windows, window_length, n_features)")
    n_windows, window_length, n_features = inputs.shape
    if n_windows < 1 or window_length < 1 or n_features < 1:
        raise ValueError("X dimensions must all be non-zero")
    try:
        numeric_inputs = inputs.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric values") from exc
    if not np.isfinite(numeric_inputs).all():
        raise ValueError("X must contain only finite values")

    numeric_lengths = _one_dimensional_finite(
        observed_lengths, "observed_lengths", n_windows
    )
    if (numeric_lengths != np.floor(numeric_lengths)).any():
        raise ValueError("observed_lengths must contain integers")
    if (numeric_lengths < 1).any() or (numeric_lengths > window_length).any():
        raise ValueError(f"observed_lengths must lie between 1 and {window_length}")

    names_array = np.asarray(feature_names)
    if names_array.ndim != 1 or len(names_array) != n_features:
        raise ValueError(f"feature_names must contain exactly {n_features} names")
    names = tuple(str(name) for name in names_array.tolist())
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    return numeric_inputs, numeric_lengths.astype(np.int64), names


def validate_prepared_windows(
    X: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
    end_cycles: np.ndarray,
    observed_lengths: np.ndarray,
    feature_names: tuple[str, ...] | list[str] | np.ndarray,
) -> PreparedWindows:
    """Validate arrays before feature extraction or model fitting.

    ``observed_lengths`` identifies the real suffix of every window. Rows to
    its left are padding and must never influence the engineered statistics.
    """
    numeric_inputs, numeric_lengths, names = _validated_window_inputs(
        X, observed_lengths, feature_names
    )
    n_windows = numeric_inputs.shape[0]

    targets = _one_dimensional_finite(y, "y", n_windows)
    if (targets < 0).any():
        raise ValueError("y must contain non-negative RUL values")
    numeric_unit_ids = _one_dimensional_finite(unit_ids, "unit_ids", n_windows)
    numeric_end_cycles = _one_dimensional_finite(end_cycles, "end_cycles", n_windows)
    for array, name in (
        (numeric_unit_ids, "unit_ids"),
        (numeric_end_cycles, "end_cycles"),
    ):
        if (array != np.floor(array)).any():
            raise ValueError(f"{name} must contain integers")
    if (numeric_unit_ids <= 0).any() or (numeric_end_cycles <= 0).any():
        raise ValueError("unit_ids and end_cycles must be positive")

    return PreparedWindows(
        X=numeric_inputs,
        y=targets,
        unit_ids=numeric_unit_ids.astype(np.int64),
        end_cycles=numeric_end_cycles.astype(np.int64),
        observed_lengths=numeric_lengths.astype(np.int64),
        feature_names=names,
    )


def load_prepared_windows(path: str | Path) -> PreparedWindows:
    """Load one supervised milestone-1 ``.npz`` artifact without pickle."""
    source = Path(path)
    required = {
        "X",
        "y",
        "unit_ids",
        "end_cycles",
        "observed_lengths",
        "feature_names",
    }
    try:
        with np.load(source, allow_pickle=False) as artifact:
            missing = sorted(required.difference(artifact.files))
            if missing:
                raise ValueError(f"Missing arrays in {source}: {', '.join(missing)}")
            arrays = {name: artifact[name] for name in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Missing arrays"):
            raise
        raise ValueError(f"Cannot load prepared windows from {source}: {exc}") from exc
    return validate_prepared_windows(**arrays)


def engineer_window_features(windows: PreparedWindows) -> TabularFeatures:
    """Extract padding-aware statistics from each causal sensor window.

    Seven statistics are produced per sensor: last value, mean, population
    standard deviation, minimum, maximum, last-minus-first delta and ordinary
    least-squares slope. A final ``observed_length`` column tells the model how
    much real history was available for windows shorter than their fixed size.
    """
    validated = validate_prepared_windows(
        windows.X,
        windows.y,
        windows.unit_ids,
        windows.end_cycles,
        windows.observed_lengths,
        windows.feature_names,
    )
    return _engineer_validated_windows(
        validated.X, validated.observed_lengths, validated.feature_names
    )


def engineer_inference_features(
    X: np.ndarray,
    observed_lengths: np.ndarray,
    feature_names: tuple[str, ...] | list[str] | np.ndarray,
) -> TabularFeatures:
    """Extract the training-time statistics from unlabeled inference windows."""
    inputs, lengths, names = _validated_window_inputs(X, observed_lengths, feature_names)
    return _engineer_validated_windows(inputs, lengths, names)


def _engineer_validated_windows(
    inputs: np.ndarray,
    observed_lengths: np.ndarray,
    feature_names: tuple[str, ...],
) -> TabularFeatures:
    n_windows, _, n_sensors = inputs.shape
    n_statistics = len(WINDOW_STATISTICS)
    output = np.empty((n_windows, n_sensors * n_statistics + 1), dtype=np.float64)

    for row_index, observed_length in enumerate(observed_lengths):
        segment = inputs[row_index, -int(observed_length) :, :]
        sensor_statistics = np.empty((n_sensors, n_statistics), dtype=np.float64)
        sensor_statistics[:, 0] = segment[-1]
        sensor_statistics[:, 1] = segment.mean(axis=0)
        sensor_statistics[:, 2] = segment.std(axis=0, ddof=0)
        sensor_statistics[:, 3] = segment.min(axis=0)
        sensor_statistics[:, 4] = segment.max(axis=0)
        sensor_statistics[:, 5] = segment[-1] - segment[0]

        if observed_length == 1:
            sensor_statistics[:, 6] = 0.0
        else:
            time = np.arange(observed_length, dtype=np.float64)
            centered_time = time - time.mean()
            centered_values = segment - segment.mean(axis=0)
            sensor_statistics[:, 6] = (centered_time[:, None] * centered_values).sum(
                axis=0
            ) / np.square(centered_time).sum()

        output[row_index, :-1] = sensor_statistics.reshape(-1)
        output[row_index, -1] = observed_length

    names = tuple(
        f"{sensor}__{statistic}"
        for sensor in feature_names
        for statistic in WINDOW_STATISTICS
    ) + ("observed_length",)
    if not np.isfinite(output).all():
        raise ValueError("Engineered window features contain non-finite values")
    return TabularFeatures(values=output, names=names)


def _validated_model_inputs(X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    features = np.asarray(X)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("Model features must have shape (n_samples, n_features)")
    try:
        numeric_features = features.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model features must be numeric") from exc
    if not np.isfinite(numeric_features).all():
        raise ValueError("Model features must contain only finite values")
    if y is None:
        return (numeric_features,)
    targets = _one_dimensional_finite(y, "y", features.shape[0])
    if (targets < 0).any():
        raise ValueError("y must contain non-negative RUL values")
    return numeric_features, targets


def _validated_sample_weight(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weights = _one_dimensional_finite(sample_weight, "sample_weight", n_samples)
    if (weights < 0).any():
        raise ValueError("sample_weight must be non-negative")
    if not (weights > 0).any():
        raise ValueError("sample_weight must contain at least one positive value")
    return weights


class MeanRULBaseline:
    """Regressor that always predicts the mean training RUL."""

    constant_: float
    n_features_in_: int

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> MeanRULBaseline:
        features, targets = _validated_model_inputs(X, y)
        weights = _validated_sample_weight(sample_weight, features.shape[0])
        self.constant_ = float(
            targets.mean() if weights is None else np.average(targets, weights=weights)
        )
        self.n_features_in_ = int(features.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "constant_"):
            raise ValueError("MeanRULBaseline must be fitted before predict")
        (features,) = _validated_model_inputs(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} model features, got {features.shape[1]}"
            )
        return np.full(features.shape[0], self.constant_, dtype=np.float64)


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    max_features: float | str | None = 1.0,
    sample_weight: np.ndarray | None = None,
) -> RandomForestRegressor:
    """Fit a deterministic Random Forest on engineered causal features."""
    features, targets = _validated_model_inputs(X, y)
    weights = _validated_sample_weight(sample_weight, features.shape[0])
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion="squared_error",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(features, targets, sample_weight=weights)
    return model


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE and RMSE in RUL cycles."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if truth.ndim != 1 or predictions.ndim != 1 or truth.shape != predictions.shape:
        raise ValueError("y_true and y_pred must be one-dimensional arrays with equal shapes")
    if truth.size < 1:
        raise ValueError("Metrics require at least one prediction")
    try:
        truth = truth.astype(np.float64, copy=False)
        predictions = predictions.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_pred must be numeric") from exc
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("y_true and y_pred must contain only finite values")
    errors = predictions - truth
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def model_parameters(model: RandomForestRegressor) -> dict[str, Any]:
    """Return the serializable parameters that define a fitted forest."""
    if not hasattr(model, "estimators_"):
        raise ValueError("RandomForestRegressor must be fitted first")
    keys = (
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "criterion",
        "random_state",
        "n_jobs",
    )
    parameters = model.get_params(deep=False)
    return {key: parameters[key] for key in keys}
