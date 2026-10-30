"""Tests for the CSV convenience entry point of the maintenance agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from industrial_maintenance.agent_cli import request_from_csv


def _write_history(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "unit_id,cycle,sensor_2,sensor_11\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_request_from_csv_selects_latest_consecutive_window(tmp_path: Path) -> None:
    source = _write_history(
        tmp_path / "history.csv",
        [f"3,{cycle},{640 + cycle / 10},{47 - cycle / 100}" for cycle in range(1, 36)],
    )

    request = request_from_csv(
        source,
        incident_id="demo-csv-003",
        message="Analyse les tendances et la RUL.",
        requested_actions=["sensor_trends", "predict_rul"],
        window_size=30,
    )

    assert request.history is not None
    assert request.history.unit_id == 3
    assert [row.cycle for row in request.history.observations] == list(range(6, 36))
    assert request.history.observations[-1].sensors == {
        "sensor_2": 643.5,
        "sensor_11": 46.65,
    }
    assert request.requested_actions == ["sensor_trends", "predict_rul"]


def test_request_from_csv_can_select_one_unit(tmp_path: Path) -> None:
    source = _write_history(
        tmp_path / "fleet.csv",
        [
            "1,1,640.0,47.0",
            "2,1,641.0,46.0",
            "2,2,642.0,45.0",
        ],
    )

    request = request_from_csv(
        source,
        incident_id="unit-2",
        message="Analyse ce moteur.",
        unit_id=2,
    )

    assert request.history is not None
    assert [row.cycle for row in request.history.observations] == [1, 2]


def test_request_from_csv_requires_unit_selection_for_a_fleet(tmp_path: Path) -> None:
    source = _write_history(
        tmp_path / "fleet.csv",
        ["1,1,640.0,47.0", "2,1,641.0,46.0"],
    )

    with pytest.raises(ValueError, match="--unit-id"):
        request_from_csv(source, incident_id="ambiguous", message="Analyse la flotte.")


def test_request_from_csv_rejects_non_positive_window(tmp_path: Path) -> None:
    source = _write_history(tmp_path / "history.csv", ["1,1,640.0,47.0"])

    with pytest.raises(ValueError, match="window_size"):
        request_from_csv(
            source,
            incident_id="bad-window",
            message="Analyse ce moteur.",
            window_size=0,
        )
