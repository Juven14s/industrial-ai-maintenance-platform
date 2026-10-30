"""Tests for the Streamlit dashboard's data and HTTP boundary helpers.

Most tests exercise the framework-independent functions in
``industrial_maintenance.dashboard``.  One lightweight Streamlit smoke test
also ensures that the complete page, including human review, renders without
requiring a separately running API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from industrial_maintenance.dashboard import (
    REQUIRED_SENSORS,
    DashboardInputError,
    DashboardServiceError,
    MaintenanceApiClient,
    build_agent_history_payload,
    build_prediction_payload,
    normalize_incident_id,
    prediction_cutoffs,
    prepare_engine_history,
    validate_source_frame,
)


def _source_frame(
    *,
    units: tuple[int, ...] = (1, 2),
    cycles: range = range(1, 36),
) -> pd.DataFrame:
    """Build a compact, deterministic C-MAPSS-like frame."""

    rows: list[dict[str, float | int]] = []
    for unit_id in units:
        for cycle in cycles:
            row: dict[str, float | int] = {
                "unit_id": unit_id,
                "cycle": cycle,
                # An irrelevant column must never leak into the API payload.
                "rul": 200 - cycle,
            }
            for offset, sensor in enumerate(REQUIRED_SENSORS, start=1):
                row[sensor] = unit_id * 1000.0 + cycle + offset / 100.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_validate_source_frame_accepts_valid_data_without_mutating_it() -> None:
    frame = _source_frame()
    original = frame.copy(deep=True)

    validated = validate_source_frame(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert isinstance(validated, pd.DataFrame)
    assert set(("unit_id", "cycle", *REQUIRED_SENSORS)).issubset(validated.columns)


@pytest.mark.parametrize("missing", ["unit_id", "cycle", "sensor_2"])
def test_validate_source_frame_rejects_missing_required_columns(missing: str) -> None:
    frame = _source_frame().drop(columns=[missing])

    with pytest.raises(DashboardInputError, match=missing):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_empty_data() -> None:
    with pytest.raises(DashboardInputError, match="(?i)vide|aucune|empty"):
        validate_source_frame(_source_frame().iloc[0:0])


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("unit_id", "moteur-A"),
        ("cycle", "premier"),
        ("sensor_2", "inconnu"),
        ("sensor_3", np.nan),
        ("sensor_4", np.inf),
        ("sensor_6", -np.inf),
    ],
)
def test_validate_source_frame_rejects_non_numeric_or_non_finite_values(
    column: str,
    bad_value: object,
) -> None:
    frame = _source_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = bad_value

    with pytest.raises(DashboardInputError, match=column):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_non_integer_identifiers_and_cycles() -> None:
    for column in ("unit_id", "cycle"):
        frame = _source_frame()
        frame[column] = frame[column].astype(float)
        frame.loc[0, column] = 1.5

        with pytest.raises(DashboardInputError, match=column):
            validate_source_frame(frame)


def test_validate_source_frame_rejects_duplicate_cycles_for_an_engine() -> None:
    frame = _source_frame(units=(1,))
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(DashboardInputError, match="(?i)dupli|unique"):
        validate_source_frame(frame)


def test_validate_source_frame_rejects_non_consecutive_cycles() -> None:
    frame = _source_frame(units=(1,)).query("cycle != 17")

    with pytest.raises(DashboardInputError, match="(?i)cons.cut|continu"):
        validate_source_frame(frame)


def test_prepare_engine_history_selects_one_engine_and_requested_cutoff() -> None:
    frame = _source_frame()

    history = prepare_engine_history(frame, unit_id=2, end_cycle=32)

    assert history["unit_id"].unique().tolist() == [2]
    assert history["cycle"].tolist() == list(range(1, 33))
    assert int(history["cycle"].max()) == 32


def test_prepare_engine_history_keeps_all_cycles_up_to_late_cutoff() -> None:
    frame = _source_frame(cycles=range(1, 51))

    history = prepare_engine_history(frame, unit_id=1, end_cycle=45)

    assert history["cycle"].tolist() == list(range(1, 46))
    assert len(history) == 45


def test_prepare_engine_history_keeps_complete_short_history_from_cycle_one() -> None:
    frame = _source_frame(cycles=range(1, 30))

    history = prepare_engine_history(frame, unit_id=1, end_cycle=24)

    assert history["cycle"].tolist() == list(range(1, 25))


def test_prediction_cutoffs_allow_cycle_twenty_for_history_starting_at_one() -> None:
    cutoffs = prediction_cutoffs(_source_frame(), unit_id=1)

    assert cutoffs == list(range(20, 36))


def test_prediction_cutoffs_wait_for_thirty_rows_when_history_is_a_suffix() -> None:
    frame = _source_frame(cycles=range(10, 45))

    cutoffs = prediction_cutoffs(frame, unit_id=1)

    assert cutoffs == list(range(39, 45))


@pytest.mark.parametrize(
    ("unit_id", "end_cycle", "message"),
    [
        (999, 30, "(?i)moteur|unit"),
        (1, 999, "(?i)cycle"),
    ],
)
def test_prepare_engine_history_rejects_unknown_selection(
    unit_id: int,
    end_cycle: int,
    message: str,
) -> None:
    with pytest.raises(DashboardInputError, match=message):
        prepare_engine_history(_source_frame(), unit_id=unit_id, end_cycle=end_cycle)


def test_build_prediction_payload_contains_only_required_api_fields() -> None:
    history = prepare_engine_history(_source_frame(), unit_id=2, end_cycle=35)

    payload = build_prediction_payload(history)

    assert payload["unit_id"] == 2
    assert len(payload["observations"]) == 35
    assert [observation["cycle"] for observation in payload["observations"]] == list(range(1, 36))
    expected_observation_keys = {"cycle", *REQUIRED_SENSORS}
    assert all(
        set(observation) == expected_observation_keys for observation in payload["observations"]
    )
    assert all("rul" not in observation for observation in payload["observations"])


def test_build_prediction_payload_returns_native_json_numbers() -> None:
    frame = _source_frame(units=(1,), cycles=range(1, 31))
    frame["unit_id"] = frame["unit_id"].astype(np.int64)
    frame["cycle"] = frame["cycle"].astype(np.int64)
    frame[list(REQUIRED_SENSORS)] = frame[list(REQUIRED_SENSORS)].astype(np.float64)

    payload = build_prediction_payload(frame)

    # This catches NumPy scalar leakage before it reaches an HTTP client.
    json.dumps(payload)
    assert type(payload["unit_id"]) is int
    assert all(type(item["cycle"]) is int for item in payload["observations"])
    assert all(
        type(item[sensor]) is float
        for item in payload["observations"]
        for sensor in REQUIRED_SENSORS
    )


def test_build_agent_history_payload_nests_only_required_sensors() -> None:
    history = prepare_engine_history(_source_frame(), unit_id=2, end_cycle=30)

    payload = build_agent_history_payload(history)

    assert payload["unit_id"] == 2
    assert len(payload["observations"]) == 30
    assert payload["observations"][0]["cycle"] == 1
    assert set(payload["observations"][0]) == {"cycle", "sensors"}
    assert set(payload["observations"][0]["sensors"]) == set(REQUIRED_SENSORS)
    assert "rul" not in payload["observations"][0]["sensors"]
    json.dumps(payload)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: Any = None,
        raw_body: bytes | None = None,
    ) -> None:
        self.status = status_code
        self._body = raw_body if raw_body is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_api_client_predict_posts_payload_to_normalized_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}
    expected = {
        "unit_id": 1,
        "rul_estimate": 51.5,
        "rul_conservative": 40.0,
        "unit": "cycles",
    }

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        sent.update(
            url=request.full_url,
            method=request.get_method(),
            json=json.loads(request.data.decode("utf-8")),
            timeout=timeout,
            content_type=request.get_header("Content-type"),
        )
        return _FakeResponse(status_code=200, payload=expected)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload = {"unit_id": 1, "observations": []}

    result = MaintenanceApiClient("http://api.test/", timeout=4.5).predict(payload)

    assert result == expected
    assert sent["url"] == "http://api.test/predict/rul"
    assert sent["method"] == "POST"
    assert sent["json"] == payload
    assert sent["timeout"] == 4.5
    assert str(sent["content_type"]).startswith("application/json")


def test_api_client_posts_document_question(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}
    expected = {
        "answerable": False,
        "answer": "Aucune preuve suffisante.",
        "citations": [],
    }

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        sent.update(
            url=request.full_url,
            method=request.get_method(),
            json=json.loads(request.data.decode("utf-8")),
            timeout=timeout,
        )
        return _FakeResponse(status_code=200, payload=expected)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = MaintenanceApiClient("http://api.test", timeout=120).ask_documents(
        "  Quelle réparation faut-il effectuer ?  "
    )

    assert result == expected
    assert sent == {
        "url": "http://api.test/documents/ask",
        "method": "POST",
        "json": {"question": "Quelle réparation faut-il effectuer ?"},
        "timeout": 120.0,
    }


def test_api_client_posts_maintenance_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}
    expected = {
        "scenario_id": "demo",
        "heuristic": {"status": "feasible"},
        "optimized": {"status": "optimal"},
    }

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        sent.update(
            url=request.full_url,
            method=request.get_method(),
            json=json.loads(request.data.decode("utf-8")),
            timeout=timeout,
        )
        return _FakeResponse(status_code=200, payload=expected)

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    payload = {
        "scenario_id": "demo",
        "planning_horizon": 10,
        "technicians": 2,
        "workshop_capacity": 1,
        "engines": [],
    }

    result = MaintenanceApiClient("http://api.test", timeout=20).plan_maintenance(payload)

    assert result == expected
    assert sent == {
        "url": "http://api.test/maintenance/plan",
        "method": "POST",
        "json": payload,
        "timeout": 20.0,
    }


def test_api_client_starts_reads_and_reviews_supervised_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    responses = [
        {"result": {"status": "pending_approval"}, "interrupts": [{}]},
        {"result": {"status": "pending_approval"}, "interrupts": [{}]},
        {"result": {"status": "approved"}, "interrupts": []},
    ]

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        sent.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "json": (
                    None if request.data is None else json.loads(request.data.decode("utf-8"))
                ),
                "timeout": timeout,
            }
        )
        return _FakeResponse(status_code=200, payload=responses[len(sent) - 1])

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = MaintenanceApiClient("http://api.test", timeout=30)
    start_payload = {
        "incident_id": "incident:7-fragile",
        "message": "Analyse ce moteur.",
        "requested_actions": ["sensor_trends", "predict_rul"],
    }

    started = client.start_agent(start_payload)
    current = client.get_agent("incident:7-fragile")
    reviewed = client.review_agent(
        "incident:7-fragile",
        {"action": "approve", "comment": "Vérifié"},
    )

    assert started["result"]["status"] == "pending_approval"
    assert current["result"]["status"] == "pending_approval"
    assert reviewed["result"]["status"] == "approved"
    encoded_path = "/agent/runs/incident%3A7-fragile"
    assert sent == [
        {
            "url": "http://api.test/agent/runs",
            "method": "POST",
            "json": start_payload,
            "timeout": 30.0,
        },
        {
            "url": "http://api.test" + encoded_path,
            "method": "GET",
            "json": None,
            "timeout": 30.0,
        },
        {
            "url": "http://api.test" + encoded_path + "/review",
            "method": "POST",
            "json": {"action": "approve", "comment": "Vérifié"},
            "timeout": 30.0,
        },
    ]


def test_api_client_rejects_blank_agent_incident() -> None:
    client = MaintenanceApiClient("http://api.test")

    with pytest.raises(ValueError, match="identifiant"):
        client.get_agent("   ")


@pytest.mark.parametrize("value", ["incident/7", "incident 7", "-incident"])
def test_incident_identifier_rejects_path_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError, match="uniquement"):
        normalize_incident_id(value)


def test_streamlit_app_renders_supervised_agent_section() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src/industrial_maintenance/dashboard_app.py"
    app = AppTest.from_file(
        app_path,
        default_timeout=20,
    ).run()

    assert len(app.exception) == 0
    assert "6. Lancer une analyse supervisée de l'incident" in [
        item.value for item in app.subheader
    ]


def test_api_client_health_uses_get_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        sent.update(
            url=request.full_url,
            method=request.get_method(),
            data=request.data,
            timeout=timeout,
        )
        return _FakeResponse(status_code=200, payload={"status": "ok"})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = MaintenanceApiClient("http://api.test", timeout=2.0).health()

    assert result == {"status": "ok"}
    assert sent == {
        "url": "http://api.test/health",
        "method": "GET",
        "data": None,
        "timeout": 2.0,
    }


def test_api_client_models_uses_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, *, timeout: float) -> _FakeResponse:
        seen.update(url=request.full_url, method=request.get_method(), timeout=timeout)
        return _FakeResponse(status_code=200, payload={"window_size": 30})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = MaintenanceApiClient("http://api.test", timeout=3.0).models()

    assert result == {"window_size": 30}
    assert seen == {
        "url": "http://api.test/models",
        "method": "GET",
        "timeout": 3.0,
    }


def test_api_client_translates_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request

    def fail_urlopen(*args: Any, **kwargs: Any) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(DashboardServiceError, match="(?i)api|connexion|joindre"):
        MaintenanceApiClient("http://api.test").predict({"unit_id": 1, "observations": []})


def test_api_client_exposes_api_validation_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request
    from io import BytesIO

    def invalid_request(request: Any, *, timeout: float) -> None:
        body = json.dumps({"detail": {"message": "Cycles non consécutifs"}}).encode("utf-8")
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            hdrs=None,
            fp=BytesIO(body),
        )

    monkeypatch.setattr(urllib.request, "urlopen", invalid_request)

    with pytest.raises(DashboardServiceError, match="Cycles non consécutifs") as exc_info:
        MaintenanceApiClient("http://api.test").predict({"unit_id": 1, "observations": []})

    assert exc_info.value.status_code == 422


def test_api_client_rejects_invalid_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def invalid_json(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            raw_body=b"not-json",
        )

    monkeypatch.setattr(urllib.request, "urlopen", invalid_json)

    with pytest.raises(DashboardServiceError, match="(?i)json|réponse"):
        MaintenanceApiClient("http://api.test").predict({"unit_id": 1, "observations": []})
