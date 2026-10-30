"""In-memory contract tests for the injected MCP maintenance facade."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import pytest
from mcp import Client, MCPError

from industrial_maintenance.inference import RULPrediction
from industrial_maintenance.mcp_server import create_mcp_server
from industrial_maintenance.planning import (
    MaintenanceAssignment,
    PlanningComparison,
    PlanningError,
    PlanResult,
)
from industrial_maintenance.rag_generation import ABSTENTION_MESSAGE, GroundedAnswer
from industrial_maintenance.rag_retrieval import SearchResult
from industrial_maintenance.sensor_trends import SensorTrendService

REQUIRED_SENSORS = (
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)


def _observation(cycle: int) -> dict[str, int | float]:
    return {
        "cycle": cycle,
        **{
            sensor: float(10 * sensor_index + cycle * (-1 if sensor_index == 2 else 1))
            for sensor_index, sensor in enumerate(REQUIRED_SENSORS, start=1)
        },
    }


def _history(*cycles: int) -> dict[str, Any]:
    return {"unit_id": 7, "observations": [_observation(cycle) for cycle in cycles]}


def _document_result() -> SearchResult:
    return SearchResult(
        rank=1,
        score=0.052,
        dense_score=0.84,
        lexical_score=0.21,
        chunk_id="readme-001",
        document_id="cmapss_dataset_readme",
        title="NASA C-MAPSS dataset README",
        section="Experimental Scenario",
        page_start=None,
        page_end=None,
        text="There are three operational settings.",
        source_url="https://example.test/readme",
        limitations="The settings are anonymized.",
    )


class FakePredictor:
    central_version = "lstm-central123456"
    safety_version = "hgb-guard123456"

    def __init__(self) -> None:
        self.received_frames: list[pd.DataFrame] = []

    def model_info(self) -> dict[str, Any]:
        return {
            "window_length": 30,
            "minimum_end_cycle": 20,
            "required_sensors": list(REQUIRED_SENSORS),
            "central": {
                "name": "lstm",
                "version": self.central_version,
                "checkpoint_path": r"C:\private\model.pt",
            },
            "conservative": {
                "name": "hgb_quantile_005",
                "version": self.safety_version,
                "secret_token": "must-not-leak",
            },
            "artifact_directory": r"C:\private",
        }

    def predict_frame(self, frame: pd.DataFrame) -> RULPrediction:
        self.received_frames.append(frame.copy(deep=True))
        return RULPrediction(
            unit_id=int(frame["unit_id"].iloc[0]),
            history_start_cycle=int(frame["cycle"].iloc[0]),
            last_cycle=int(frame["cycle"].iloc[-1]),
            observed_cycles=len(frame),
            window_cycles_used=min(len(frame), 30),
            rul_estimate=48.25,
            rul_conservative=27.5,
            safety_base_prediction=31.5,
            safety_margin=4.0,
            decision_value=27.5,
            central_model_version=self.central_version,
            safety_model_version=self.safety_version,
        )


class FakeDocumentSearch:
    version = "multilingual-e5-small-test123"

    def __init__(self) -> None:
        self.received_queries: list[tuple[str, int]] = []

    def info(self) -> dict[str, Any]:
        return {
            "retrieval_version": self.version,
            "model_name": "intfloat/multilingual-e5-small",
            "model_revision": "test-revision",
            "chunk_count": 192,
            "default_top_k": 5,
            "retrieval_method": "hybrid_rrf_dense_cosine_tfidf",
            "index_directory": r"C:\private\rag",
        }

    def search(self, query: str, *, top_k: int) -> list[SearchResult]:
        self.received_queries.append((query, top_k))
        return [_document_result()][:top_k]


class FakeAnswerService:
    version = "Qwen3-1.7B-test123"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str) -> GroundedAnswer:
        self.questions.append(question)
        if "inconnue" in question:
            return GroundedAnswer(
                question=question,
                answerable=False,
                answer=ABSTENTION_MESSAGE,
                citations=(),
                retrieval_version="multilingual-e5-small-test123",
                generation_version=self.version,
            )
        return GroundedAnswer(
            question=question,
            answerable=True,
            answer="Trois réglages opérationnels sont présents.",
            citations=(_document_result(),),
            retrieval_version="multilingual-e5-small-test123",
            generation_version=self.version,
        )


class FakePlanningService:
    version = "maintenance-planning-v1"

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def plan(self, payload: dict[str, Any]) -> PlanningComparison:
        self.payloads.append(payload)
        engines = payload["engines"]
        identifiers = [engine["unit_id"] for engine in engines]
        if len(identifiers) != len(set(identifiers)):
            raise PlanningError("Duplicate unit_id values are not allowed")
        engine = engines[0]
        duration = engine["maintenance_duration"]
        deadline = max(0, math.floor(engine["conservative_rul"]))
        cost = int(engine.get("preventive_cost", 500)) + 50 * duration
        assignment = MaintenanceAssignment(
            unit_id=engine["unit_id"],
            start_cycle=0,
            end_cycle=duration,
            deadline_cycle=deadline,
            maintenance_duration=duration,
            technicians_required=engine.get("technicians_required", 1),
            criticality=engine["criticality"],
            preventive_cost=int(engine.get("preventive_cost", 500)),
            earliness_cycles=max(deadline - duration, 0),
            tardiness_cycles=max(duration - deadline, 0),
            cost=cost,
        )
        heuristic = PlanResult(
            method="heuristic",
            status="feasible",
            objective_cost=cost,
            assignments=(assignment,),
            unscheduled_unit_ids=(),
            deferred_unit_ids=(),
            solve_time_seconds=0.001,
            solver_status="fake_heuristic",
        )
        optimized = PlanResult(
            method="cp_sat",
            status="optimal",
            objective_cost=cost,
            assignments=(assignment,),
            unscheduled_unit_ids=(),
            deferred_unit_ids=(),
            solve_time_seconds=0.002,
            solver_status="optimal",
        )
        return PlanningComparison(
            scenario_id=payload["scenario_id"],
            heuristic=heuristic,
            optimized=optimized,
            absolute_savings=0,
            relative_savings=0.0,
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def services() -> dict[str, Any]:
    return {
        "predictor": FakePredictor(),
        "document_search": FakeDocumentSearch(),
        "answer_service": FakeAnswerService(),
        "planning_service": FakePlanningService(),
        "sensor_trend_service": SensorTrendService(
            REQUIRED_SENSORS,
            recent_window=3,
            strongest_count=3,
        ),
    }


@pytest.fixture
async def client(services: dict[str, Any]):
    server = create_mcp_server(**services)
    async with Client(server, raise_exceptions=True) as connected:
        yield connected


@pytest.mark.anyio
async def test_server_lists_typed_tools_and_public_resources(client: Client) -> None:
    tools = await client.list_tools()
    resources = await client.list_resources()

    assert {tool.name for tool in tools.tools} == {
        "predict_rul",
        "summarize_sensor_trends",
        "search_documents",
        "answer_documents",
        "plan_maintenance",
    }
    assert all(tool.output_schema is not None for tool in tools.tools)
    assert {str(resource.uri) for resource in resources.resources} == {
        "maintenance://models",
        "maintenance://documents/status",
        "maintenance://planning/policy",
        "maintenance://sensors",
        "maintenance://service/status",
    }


@pytest.mark.anyio
async def test_prediction_and_sensor_tools_preserve_history_and_limit_claims(
    client: Client,
    services: dict[str, Any],
) -> None:
    history = _history(1, 2, 3, 4)

    prediction = await client.call_tool("predict_rul", history)
    trends = await client.call_tool("summarize_sensor_trends", history)

    assert prediction.is_error is False
    assert prediction.structured_content["rul_estimate"] == 48.25
    assert prediction.structured_content["rul_conservative"] == 27.5
    assert prediction.structured_content["decision_value"] == 27.5
    assert trends.is_error is False
    assert trends.structured_content["unit_id"] == 7
    assert trends.structured_content["recent_window_used"] == 3
    assert len(trends.structured_content["strongest_changes"]) == 3
    assert "do not identify a fault" in trends.structured_content["limitation"]
    received = services["predictor"].received_frames[0]
    assert received["cycle"].tolist() == [1, 2, 3, 4]
    assert "rul" not in received.columns


@pytest.mark.anyio
async def test_document_tools_return_provenance_and_explicit_abstention(
    client: Client,
    services: dict[str, Any],
) -> None:
    search = await client.call_tool(
        "search_documents",
        {"query": "Que représentent les réglages ?", "top_k": 3},
    )
    answer = await client.call_tool(
        "answer_documents",
        {"question": "Combien de réglages opérationnels sont présents ?"},
    )
    abstention = await client.call_tool(
        "answer_documents",
        {"question": "Quelle procédure inconnue faut-il appliquer ?"},
    )

    assert search.structured_content["results"][0]["document_id"] == ("cmapss_dataset_readme")
    assert search.structured_content["results"][0]["limitations"]
    assert answer.structured_content["status"] == "answered"
    assert answer.structured_content["citations"][0]["citation"].endswith("Experimental Scenario")
    assert abstention.structured_content["status"] == "abstained"
    assert abstention.structured_content["citations"] == []
    assert services["document_search"].received_queries == [("Que représentent les réglages ?", 3)]


@pytest.mark.anyio
async def test_planning_tool_uses_explicit_conservative_rul(
    client: Client,
    services: dict[str, Any],
) -> None:
    result = await client.call_tool(
        "plan_maintenance",
        {
            "scenario_id": "mcp-test",
            "planning_horizon": 10,
            "technicians": 2,
            "workshop_capacity": 1,
            "engines": [
                {
                    "unit_id": 7,
                    "conservative_rul": 4.5,
                    "maintenance_duration": 2,
                    "technicians_required": 1,
                    "criticality": "high",
                    "preventive_cost": 500,
                }
            ],
        },
    )

    assert result.is_error is False
    assert result.structured_content["scenario_id"] == "mcp-test"
    assert result.structured_content["optimized"]["status"] == "optimal"
    payload = services["planning_service"].payloads[0]
    assert payload["engines"][0]["conservative_rul"] == 4.5
    assert "rul_estimate" not in payload["engines"][0]


@pytest.mark.anyio
async def test_expected_input_failure_is_visible_as_tool_error(client: Client) -> None:
    result = await client.call_tool("predict_rul", _history(1, 3))

    assert result.is_error is True
    assert result.structured_content is None
    assert "ordered and consecutive" in result.content[0].text


@pytest.mark.anyio
async def test_missing_service_is_a_protocol_error() -> None:
    async with Client(create_mcp_server(), raise_exceptions=True) as empty_client:
        with pytest.raises(MCPError, match="RUL prediction service is unavailable"):
            await empty_client.call_tool("predict_rul", _history(1))


@pytest.mark.anyio
async def test_resources_expose_versions_without_local_paths_or_secrets(
    client: Client,
) -> None:
    resource_uris = (
        "maintenance://models",
        "maintenance://documents/status",
        "maintenance://planning/policy",
        "maintenance://sensors",
        "maintenance://service/status",
    )
    documents: dict[str, dict[str, Any]] = {}
    raw_texts: list[str] = []

    for uri in resource_uris:
        response = await client.read_resource(uri)
        text = response.contents[0].text
        raw_texts.append(text)
        documents[uri] = json.loads(text)

    combined = "\n".join(raw_texts).lower()
    assert "c:\\" not in combined
    assert "private" not in combined
    assert "secret" not in combined
    assert "token" not in combined
    assert documents["maintenance://models"]["central"]["version"] == ("lstm-central123456")
    assert documents["maintenance://documents/status"]["chunk_count"] == 192
    assert documents["maintenance://planning/policy"]["decision_input"] == ("conservative_rul")
    assert documents["maintenance://service/status"]["prediction"]["available"] is True
