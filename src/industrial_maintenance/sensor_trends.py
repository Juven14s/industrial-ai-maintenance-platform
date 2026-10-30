"""Deterministic, non-diagnostic summaries of engine sensor trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


class SensorTrendInputError(ValueError):
    """Raised when a trajectory cannot be summarized safely."""


@dataclass(frozen=True)
class SensorTrend:
    """Unit-preserving and dimensionless trend measures for one sensor."""

    sensor: str
    first_value: float
    last_value: float
    mean: float
    standard_deviation: float
    delta: float
    recent_delta: float
    global_slope_per_cycle: float
    recent_slope_per_cycle: float
    global_relative_change: float
    recent_relative_change: float
    global_direction: str
    recent_direction: str
    trend_score: float


@dataclass(frozen=True)
class SensorTrendSummary:
    """Serializable trend summary for exactly one engine history."""

    unit_id: int
    history_start_cycle: int
    last_cycle: int
    observed_cycles: int
    recent_window_used: int
    strongest_changes: tuple[str, ...]
    sensors: tuple[SensorTrend, ...]
    method_version: str = "sensor-trends-v1"
    limitation: str = "Descriptive trends only; they do not identify a fault, a cause, or a repair."

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without losing field names."""

        value = asdict(self)
        value["strongest_changes"] = list(self.strongest_changes)
        value["sensors"] = [asdict(item) for item in self.sensors]
        return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sensor_names(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("sensor_names must be a non-empty sequence of unique names")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("sensor_names must be a non-empty sequence of unique names")
    if len(names) != len(set(names)):
        raise ValueError("sensor_names must not contain duplicates")
    return names


def _slope(cycles: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    centered_cycles = cycles - np.mean(cycles)
    denominator = float(np.dot(centered_cycles, centered_cycles))
    if denominator <= 0.0:
        return 0.0
    centered_values = values - np.mean(values)
    return float(np.dot(centered_cycles, centered_values) / denominator)


def _direction(relative_effect: float, tolerance: float) -> str:
    if abs(relative_effect) <= tolerance:
        return "stable"
    return "increasing" if relative_effect > 0.0 else "decreasing"


def _validated_history(
    frame: pd.DataFrame, sensor_names: tuple[str, ...]
) -> tuple[pd.DataFrame, int, np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SensorTrendInputError("History must contain at least one observation")
    if not frame.columns.is_unique:
        raise SensorTrendInputError("Duplicate column names are not allowed")
    required = ("unit_id", "cycle", *sensor_names)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise SensorTrendInputError(f"Missing required columns: {', '.join(missing)}")

    try:
        metadata = frame.loc[:, ["unit_id", "cycle"]].to_numpy(dtype=np.float64)
        sensor_values = frame.loc[:, list(sensor_names)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SensorTrendInputError("History columns must contain numeric values") from exc
    if not np.isfinite(metadata).all():
        raise SensorTrendInputError("unit_id and cycle must contain finite values")
    if (metadata <= 0).any() or (metadata != np.floor(metadata)).any():
        raise SensorTrendInputError("unit_id and cycle must be positive integers")
    if (metadata >= float(np.iinfo(np.int64).max)).any():
        raise SensorTrendInputError("unit_id and cycle must fit in int64")
    if not np.isfinite(sensor_values).all():
        raise SensorTrendInputError("Sensor values must be finite")

    unit_ids = metadata[:, 0].astype(np.int64)
    cycles = metadata[:, 1].astype(np.int64)
    unique_units = np.unique(unit_ids)
    if len(unique_units) != 1:
        raise SensorTrendInputError("One history must contain exactly one engine")
    if len(cycles) > 1 and np.any(np.diff(cycles) != 1):
        raise SensorTrendInputError(
            "Cycles must be ordered and consecutive, without duplicates or gaps"
        )
    return frame, int(unique_units[0]), cycles, sensor_values


def summarize_sensor_trends(
    frame: pd.DataFrame,
    sensor_names: Sequence[str],
    *,
    recent_window: int = 10,
    strongest_count: int = 5,
    stable_relative_tolerance: float = 1e-3,
) -> SensorTrendSummary:
    """Summarize global and recent changes without inferring a physical cause.

    Slopes retain each sensor's native unit per cycle. Relative effects divide the
    fitted change across the corresponding interval by a scale derived from that
    sensor's mean and standard deviation. They are intended for ranking only and
    are not anomaly scores calibrated against a reference population.
    """

    names = _sensor_names(sensor_names)
    recent_window = _positive_integer(recent_window, "recent_window")
    strongest_count = _positive_integer(strongest_count, "strongest_count")
    if (
        isinstance(stable_relative_tolerance, (bool, np.bool_))
        or not isinstance(stable_relative_tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(float(stable_relative_tolerance))
        or float(stable_relative_tolerance) < 0.0
    ):
        raise ValueError("stable_relative_tolerance must be finite and non-negative")
    tolerance = float(stable_relative_tolerance)

    _, unit_id, cycles, matrix = _validated_history(frame, names)
    recent_count = min(len(cycles), recent_window)
    recent_cycles = cycles[-recent_count:].astype(np.float64)
    full_cycles = cycles.astype(np.float64)
    trends: list[SensorTrend] = []

    for index, sensor in enumerate(names):
        values = matrix[:, index]
        recent_values = values[-recent_count:]
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=0))
        reference_scale = max(abs(mean), standard_deviation, np.finfo(np.float64).eps)
        global_slope = _slope(full_cycles, values)
        recent_slope = _slope(recent_cycles, recent_values)
        global_span = float(full_cycles[-1] - full_cycles[0]) if len(full_cycles) > 1 else 0.0
        recent_span = float(recent_cycles[-1] - recent_cycles[0]) if len(recent_cycles) > 1 else 0.0
        global_relative_change = global_slope * global_span / reference_scale
        recent_relative_change = recent_slope * recent_span / reference_scale
        trends.append(
            SensorTrend(
                sensor=sensor,
                first_value=float(values[0]),
                last_value=float(values[-1]),
                mean=mean,
                standard_deviation=standard_deviation,
                delta=float(values[-1] - values[0]),
                recent_delta=float(recent_values[-1] - recent_values[0]),
                global_slope_per_cycle=global_slope,
                recent_slope_per_cycle=recent_slope,
                global_relative_change=float(global_relative_change),
                recent_relative_change=float(recent_relative_change),
                global_direction=_direction(global_relative_change, tolerance),
                recent_direction=_direction(recent_relative_change, tolerance),
                trend_score=float(max(abs(global_relative_change), abs(recent_relative_change))),
            )
        )

    ranked = sorted(trends, key=lambda item: (-item.trend_score, item.sensor))
    strongest_changes = tuple(item.sensor for item in ranked[: min(strongest_count, len(ranked))])
    return SensorTrendSummary(
        unit_id=unit_id,
        history_start_cycle=int(cycles[0]),
        last_cycle=int(cycles[-1]),
        observed_cycles=len(cycles),
        recent_window_used=recent_count,
        strongest_changes=strongest_changes,
        sensors=tuple(trends),
    )


class SensorTrendService:
    """Configured facade suitable for API, graph, or MCP dependency injection."""

    def __init__(
        self,
        sensor_names: Sequence[str],
        *,
        recent_window: int = 10,
        strongest_count: int = 5,
        stable_relative_tolerance: float = 1e-3,
    ) -> None:
        self.sensor_names = _sensor_names(sensor_names)
        self.recent_window = _positive_integer(recent_window, "recent_window")
        self.strongest_count = _positive_integer(strongest_count, "strongest_count")
        if (
            isinstance(stable_relative_tolerance, (bool, np.bool_))
            or not isinstance(stable_relative_tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(float(stable_relative_tolerance))
            or float(stable_relative_tolerance) < 0.0
        ):
            raise ValueError("stable_relative_tolerance must be finite and non-negative")
        self.stable_relative_tolerance = float(stable_relative_tolerance)

    @property
    def version(self) -> str:
        return "sensor-trends-v1"

    def summarize(self, frame: pd.DataFrame) -> SensorTrendSummary:
        """Apply this service's fixed, auditable trend configuration."""

        return summarize_sensor_trends(
            frame,
            self.sensor_names,
            recent_window=self.recent_window,
            strongest_count=self.strongest_count,
            stable_relative_tolerance=self.stable_relative_tolerance,
        )
