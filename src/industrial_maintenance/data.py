"""Load C-MAPSS trajectories and prepare causal, engine-separated ML inputs.

The official test set is never opened implicitly. RUL construction and validation
truncation are intended for complete run-to-failure training trajectories only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SETTING_COLUMNS: tuple[str, ...] = tuple(f"setting_{i}" for i in range(1, 4))
SENSOR_COLUMNS: tuple[str, ...] = tuple(f"sensor_{i}" for i in range(1, 22))
DATA_COLUMNS: tuple[str, ...] = ("unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS)


def _numeric_values(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Return a finite float64 matrix, with useful errors for invalid inputs."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Columns must contain numeric values: {list(columns)}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"NaN or infinite values in columns: {list(columns)}")
    return values


def _validated_copy(frame: pd.DataFrame, *, raw: bool = False) -> pd.DataFrame:
    """Validate row order without silently sorting; preserve the input index."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Expected a pandas DataFrame")
    if frame.empty:
        raise ValueError("A trajectory DataFrame must not be empty")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    values = _numeric_values(frame, ("unit_id", "cycle"))
    if (values <= 0).any() or (values != np.floor(values)).any():
        raise ValueError("unit_id and cycle must be positive integers")
    if (values >= float(np.iinfo(np.int64).max)).any():
        raise ValueError("unit_id and cycle must fit in int64")

    result = frame.copy()
    result["unit_id"] = values[:, 0].astype(np.int64)
    result["cycle"] = values[:, 1].astype(np.int64)
    if result.duplicated(["unit_id", "cycle"]).any():
        raise ValueError("Duplicate (unit_id, cycle) observations are not allowed")

    for unit_id, trajectory in result.groupby("unit_id", sort=False):
        cycles = trajectory["cycle"].to_numpy()
        if np.any(np.diff(cycles) != 1):
            raise ValueError(
                f"Engine {unit_id}: cycles must be ordered and consecutive, without gaps"
            )
        if raw and cycles[0] != 1:
            raise ValueError(f"Engine {unit_id}: raw trajectory must start at cycle 1")

    if raw:
        ids = result["unit_id"].to_numpy()
        unique_ids = np.unique(ids)
        if not np.array_equal(unique_ids, np.arange(1, len(unique_ids) + 1)):
            raise ValueError("Raw engine IDs must be contiguous, starting at 1")
        if np.any(np.diff(ids) < 0):
            raise ValueError("Raw rows must be ordered by engine ID and then cycle")
    return result


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_rul(frame: pd.DataFrame) -> None:
    values = _numeric_values(frame, ("rul",))
    if (values < 0).any():
        raise ValueError("rul must be non-negative")


def load_cmapss(path: Path) -> pd.DataFrame:
    """Read one raw train/test text file and validate its 26-column schema.

    Whitespace separates fields, and trailing spaces are allowed. Identifiers
    must form 1..N; each engine's observations must form cycles 1..T, in order.
    This function does not attach targets or access any other file.
    """
    try:
        frame = pd.read_csv(
            Path(path), sep=r"\s+", header=None, dtype=np.float64, skip_blank_lines=False
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"C-MAPSS file is empty: {path}") from exc
    except (pd.errors.ParserError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric C-MAPSS data in {path}: {exc}") from exc
    if frame.shape[1] != len(DATA_COLUMNS):
        raise ValueError(f"Expected {len(DATA_COLUMNS)} columns in {path}, got {frame.shape[1]}")
    frame.columns = DATA_COLUMNS
    _numeric_values(frame, DATA_COLUMNS)
    return _validated_copy(frame, raw=True)


def load_rul_targets(path: Path, *, expected_count: int | None = None) -> np.ndarray:
    """Load one non-negative integer RUL target per official test engine."""
    try:
        frame = pd.read_csv(
            Path(path), sep=r"\s+", header=None, dtype=np.float64, skip_blank_lines=False
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"RUL file is empty: {path}") from exc
    except (pd.errors.ParserError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric RUL data in {path}: {exc}") from exc
    if frame.shape[1] != 1 or frame.empty:
        raise ValueError(f"Expected exactly one RUL column in {path}")
    values = frame.iloc[:, 0].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Official RUL targets must be finite non-negative integers")
    if expected_count is not None:
        count = _positive_integer(expected_count, "expected_count")
        if len(values) != count:
            raise ValueError(f"Expected {count} RUL targets, got {len(values)}")
    return values.astype(np.float32)


def add_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy complete run-to-failure trajectories and attach RUL = T - cycle.

    Never call this on truncated validation or official test trajectories: their
    final observed cycle is not their failure cycle. RUL is uncapped here.
    """
    result = _validated_copy(frame)
    failure_cycles = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = failure_cycles - result["cycle"]
    return result


def split_by_engine(
    frame: pd.DataFrame,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split entire engines before any preprocessing or window construction.

    Sorted engine IDs are permuted using ``np.random.default_rng(seed)``; the
    first ceil(N * validation_fraction) engines form validation. Row order is
    preserved within each returned copy, and their row indices are reset.
    """
    result = _validated_copy(frame)
    if isinstance(validation_fraction, (bool, np.bool_)):
        raise ValueError("validation_fraction must be a number strictly between 0 and 1")
    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("validation_fraction must be strictly between 0 and 1") from exc
    if not np.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    engine_ids = np.sort(result["unit_id"].unique())
    n_validation = ceil(len(engine_ids) * fraction)
    if n_validation >= len(engine_ids):
        raise ValueError("Engine split must leave at least one engine in train and validation")
    shuffled_ids = np.random.default_rng(seed).permutation(engine_ids)
    is_validation = result["unit_id"].isin(shuffled_ids[:n_validation])
    train = result.loc[~is_validation].copy().reset_index(drop=True)
    validation = result.loc[is_validation].copy().reset_index(drop=True)
    return train, validation


def truncate_validation(
    frame_with_rul: pd.DataFrame,
    seed: int = 43,
    min_observed_cycles: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate histories stopped before failure using held-out train engines.

    Input trajectories must be complete and have RUL attached *before* this
    call. For each sorted engine ID with failure cycle T, sample uniformly from
    [min(min_observed_cycles, T-1), T-1]. Existing RUL values are preserved;
    recomputing RUL after truncation would incorrectly make final targets zero.

    Return the observed prefixes and a one-row-per-engine cutoff table.
    """
    minimum = _positive_integer(min_observed_cycles, "min_observed_cycles")
    result = _validated_copy(frame_with_rul)
    _validate_rul(result)
    rng = np.random.default_rng(seed)
    cutoffs: list[dict[str, int | float]] = []
    cutoff_by_id: dict[int, int] = {}
    for unit_id, trajectory in result.groupby("unit_id", sort=True):
        cycles = trajectory["cycle"].to_numpy()
        if cycles[0] != 1:
            raise ValueError(f"Engine {unit_id}: complete trajectory must start at cycle 1")
        failure_cycle = int(cycles[-1])
        if failure_cycle < 2:
            raise ValueError(f"Engine {unit_id}: truncation requires at least 2 cycles")
        if not np.array_equal(trajectory["rul"].to_numpy(), failure_cycle - cycles):
            raise ValueError(
                f"Engine {unit_id}: expected uncapped RUL from a complete trajectory; "
                "do not truncate an already truncated history"
            )
        cutoff = int(rng.integers(min(minimum, failure_cycle - 1), failure_cycle))
        target = trajectory.loc[trajectory["cycle"].eq(cutoff), "rul"].iloc[0]
        cutoff_by_id[int(unit_id)] = cutoff
        cutoffs.append(
            {
                "unit_id": int(unit_id),
                "cutoff_cycle": cutoff,
                "failure_cycle": failure_cycle,
                "rul": float(target),
            }
        )
    retained = result["cycle"] <= result["unit_id"].map(cutoff_by_id)
    observed = result.loc[retained].copy().reset_index(drop=True)
    cutoff_frame = pd.DataFrame(
        cutoffs, columns=["unit_id", "cutoff_cycle", "failure_cycle", "rul"]
    )
    return observed, cutoff_frame


class SensorPreprocessor:
    """Drop exactly constant training sensors and standardize remaining sensors.

    Fit on the learning engines only. Settings, cycle and targets never enter
    the scaler. The object and its fitted scaler can be saved with joblib.
    """

    feature_names_: tuple[str, ...]
    dropped_features_: tuple[str, ...]
    scaler_: StandardScaler

    def fit(self, frame: pd.DataFrame) -> SensorPreprocessor:
        result = _validated_copy(frame)
        values = _numeric_values(result, SENSOR_COLUMNS)
        numeric_sensors = pd.DataFrame(values, columns=SENSOR_COLUMNS)
        unique_counts = numeric_sensors.nunique(dropna=False)
        selected = tuple(name for name in SENSOR_COLUMNS if unique_counts[name] > 1)
        dropped = tuple(name for name in SENSOR_COLUMNS if unique_counts[name] <= 1)
        if not selected:
            raise ValueError("All training sensors are constant; no features remain")
        scaler = StandardScaler()
        scaler.fit(numeric_sensors.loc[:, list(selected)])
        self.feature_names_ = selected
        self.dropped_features_ = dropped
        self.scaler_ = scaler
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "scaler_"):
            raise ValueError("SensorPreprocessor must be fitted before transform")
        result = _validated_copy(frame)
        values = _numeric_values(result, self.feature_names_)
        metadata_columns = ["unit_id", "cycle"]
        if "rul" in result.columns:
            _validate_rul(result)
            metadata_columns.append("rul")
        metadata = result.loc[:, metadata_columns].copy()
        numeric_sensors = pd.DataFrame(values, columns=self.feature_names_, index=result.index)
        scaled = self.scaler_.transform(numeric_sensors)
        if not np.isfinite(scaled).all():
            raise ValueError("Sensor scaling produced non-finite values")
        normalized = pd.DataFrame(scaled, columns=self.feature_names_, index=result.index)
        return pd.concat([metadata, normalized], axis=1)


@dataclass
class WindowBatch:
    """Causal inputs, optional end-of-window targets, and engine metadata.

    X is float32 (n_windows, window_length, n_features), y is float32 (n_windows,)
    when available, and all metadata arrays are int64 (n_windows,).
    ``observed_lengths`` gives the number of real rows before left padding.
    """

    X: np.ndarray
    y: np.ndarray | None
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    observed_lengths: np.ndarray


def make_windows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    window_length: int = 30,
    last_only: bool = False,
) -> WindowBatch:
    """Create one causal window per cycle, or only the last per engine.

    Short histories are padded on the left by repeating their first available
    observation. No future observation is used. This retains short engines and
    gives last_only evaluation exactly one prediction per observed engine.
    """
    length = _positive_integer(window_length, "window_length")
    if not isinstance(last_only, (bool, np.bool_)):
        raise ValueError("last_only must be a boolean")
    if isinstance(feature_names, str):
        raise ValueError("feature_names must be a sequence of column names, not a string")
    try:
        features = tuple(feature_names)
    except TypeError as exc:
        raise ValueError("feature_names must be a sequence of column names") from exc
    if not features or any(not isinstance(name, str) for name in features):
        raise ValueError("feature_names must contain at least one string column name")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names are not allowed")
    forbidden = {"unit_id", "cycle", "rul"}.intersection(features)
    if forbidden:
        raise ValueError(f"Metadata and target columns cannot be features: {sorted(forbidden)}")
    result = _validated_copy(frame)
    _numeric_values(result, features)
    has_targets = "rul" in result.columns
    if has_targets:
        _validate_rul(result)

    count = result["unit_id"].nunique() if last_only else len(result)
    inputs = np.empty((count, length, len(features)), dtype=np.float32)
    targets = np.empty(count, dtype=np.float32) if has_targets else None
    unit_ids = np.empty(count, dtype=np.int64)
    end_cycles = np.empty(count, dtype=np.int64)
    observed_lengths = np.empty(count, dtype=np.int64)
    output_index = 0
    for unit_id, trajectory in result.groupby("unit_id", sort=False):
        with np.errstate(over="ignore", invalid="ignore"):
            values = trajectory.loc[:, list(features)].to_numpy(dtype=np.float32)
            rul = trajectory["rul"].to_numpy(dtype=np.float32) if has_targets else None
        if not np.isfinite(values).all() or (rul is not None and not np.isfinite(rul).all()):
            raise ValueError("Features and targets must fit in finite float32 values")
        cycles = trajectory["cycle"].to_numpy(dtype=np.int64)
        endpoints = [len(trajectory) - 1] if last_only else range(len(trajectory))
        for endpoint in endpoints:
            start = max(0, endpoint - length + 1)
            observed_length = endpoint - start + 1
            padding = length - observed_length
            if padding:
                inputs[output_index, :padding] = values[0]
            inputs[output_index, padding:] = values[start : endpoint + 1]
            if targets is not None and rul is not None:
                targets[output_index] = rul[endpoint]
            unit_ids[output_index] = unit_id
            end_cycles[output_index] = cycles[endpoint]
            observed_lengths[output_index] = observed_length
            output_index += 1
    return WindowBatch(inputs, targets, unit_ids, end_cycles, observed_lengths)
