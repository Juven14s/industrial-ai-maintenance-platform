"""Tests for deterministic, non-diagnostic sensor trend summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from industrial_maintenance.sensor_trends import (
    SensorTrendInputError,
    SensorTrendService,
    summarize_sensor_trends,
)

SENSORS = ("sensor_up", "sensor_down", "sensor_flat")


def _history(cycles: list[int] | np.ndarray) -> pd.DataFrame:
    cycle_values = np.asarray(cycles, dtype=np.int64)
    return pd.DataFrame(
        {
            "unit_id": np.full(len(cycle_values), 17, dtype=np.int64),
            "cycle": cycle_values,
            "sensor_up": 10.0 + 2.0 * cycle_values,
            "sensor_down": 50.0 - cycle_values,
            "sensor_flat": np.full(len(cycle_values), 3.5),
        }
    )


def test_summary_preserves_units_and_classifies_global_and_recent_directions():
    summary = summarize_sensor_trends(_history([1, 2, 3, 4, 5]), SENSORS, recent_window=3)
    by_name = {item.sensor: item for item in summary.sensors}

    assert summary.unit_id == 17
    assert summary.history_start_cycle == 1
    assert summary.last_cycle == 5
    assert summary.observed_cycles == 5
    assert summary.recent_window_used == 3
    assert by_name["sensor_up"].global_slope_per_cycle == pytest.approx(2.0)
    assert by_name["sensor_up"].recent_slope_per_cycle == pytest.approx(2.0)
    assert by_name["sensor_up"].delta == pytest.approx(8.0)
    assert by_name["sensor_up"].recent_delta == pytest.approx(4.0)
    assert by_name["sensor_up"].global_direction == "increasing"
    assert by_name["sensor_up"].recent_direction == "increasing"
    assert by_name["sensor_down"].global_slope_per_cycle == pytest.approx(-1.0)
    assert by_name["sensor_down"].global_direction == "decreasing"
    assert by_name["sensor_down"].recent_direction == "decreasing"
    assert by_name["sensor_flat"].global_slope_per_cycle == pytest.approx(0.0)
    assert by_name["sensor_flat"].global_direction == "stable"
    assert by_name["sensor_flat"].recent_direction == "stable"
    assert summary.strongest_changes[0] == "sensor_up"
    assert "do not identify a fault" in summary.limitation


def test_one_observation_is_a_valid_stable_summary_and_serializes_cleanly():
    summary = summarize_sensor_trends(_history([1]), SENSORS, recent_window=10)

    assert summary.recent_window_used == 1
    assert all(item.delta == 0.0 for item in summary.sensors)
    assert all(item.global_slope_per_cycle == 0.0 for item in summary.sensors)
    assert all(item.recent_direction == "stable" for item in summary.sensors)
    serialized = summary.to_dict()
    assert serialized["strongest_changes"] == list(summary.strongest_changes)
    assert isinstance(serialized["sensors"], list)
    assert serialized["sensors"][0]["sensor"] == "sensor_up"


@pytest.mark.parametrize(
    "cycles",
    [
        [1, 3],
        [1, 1],
        [2, 1],
    ],
)
def test_invalid_cycle_sequences_are_rejected(cycles: list[int]):
    with pytest.raises(SensorTrendInputError, match="ordered and consecutive"):
        summarize_sensor_trends(_history(cycles), SENSORS)


def test_multiple_engines_are_rejected():
    frame = _history([1, 2, 3])
    frame.loc[2, "unit_id"] = 18

    with pytest.raises(SensorTrendInputError, match="exactly one engine"):
        summarize_sensor_trends(frame, SENSORS)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_non_finite_sensor_values_are_rejected(invalid_value: float):
    frame = _history([1, 2, 3])
    frame.loc[1, "sensor_up"] = invalid_value

    with pytest.raises(SensorTrendInputError, match="Sensor values must be finite"):
        summarize_sensor_trends(frame, SENSORS)


def test_missing_sensor_is_reported():
    frame = _history([1, 2, 3]).drop(columns="sensor_down")

    with pytest.raises(SensorTrendInputError, match="sensor_down"):
        summarize_sensor_trends(frame, SENSORS)


@pytest.mark.parametrize("recent_window", [0, -1, True, 1.5])
def test_recent_window_must_be_a_positive_integer(recent_window):
    with pytest.raises(ValueError, match="recent_window must be a positive integer"):
        summarize_sensor_trends(_history([1, 2]), SENSORS, recent_window=recent_window)


def test_configured_service_delegates_with_stable_metadata():
    service = SensorTrendService(SENSORS, recent_window=2, strongest_count=1)

    summary = service.summarize(_history([1, 2, 3]))

    assert service.version == "sensor-trends-v1"
    assert service.sensor_names == SENSORS
    assert summary.method_version == service.version
    assert summary.recent_window_used == 2
    assert len(summary.strongest_changes) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_names": ("sensor_up", "sensor_up")}, "must not contain duplicates"),
        ({"sensor_names": SENSORS, "strongest_count": 0}, "strongest_count"),
        (
            {"sensor_names": SENSORS, "stable_relative_tolerance": np.nan},
            "stable_relative_tolerance",
        ),
    ],
)
def test_service_configuration_is_validated(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        SensorTrendService(**kwargs)
