"""Tests for the supervised maintenance graph without real models or network calls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from industrial_maintenance.agent_checkpoint import open_agent_graph, read_agent_config
from industrial_maintenance.agent_contracts import (
    HumanReviewDecision,
    MaintenanceAgentRequest,
    RouterDecision,
)
from industrial_maintenance.agent_runtime import (
    AgentRunConflictError,
    MaintenanceAgentRuntime,
)
from industrial_maintenance.agent_workflow import (
    AgentServices,
    AgentTransientToolError,
    KeywordRouter,
    build_maintenance_graph,
    initial_agent_state,
    result_from_state,
)
from industrial_maintenance.inference import RULPrediction


@dataclass(frozen=True)
class FakeTrendSummary:
    unit_id: int
    strongest_changes: tuple[str, ...]
    limitation: str = "Descriptive trends only; no fault diagnosis."

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "strongest_changes": list(self.strongest_changes),
            "limitation": self.limitation,
        }


@dataclass(frozen=True)
class FakeDocumentAnswer:
    question: str
    answerable: bool
    answer: str
    citations: tuple[dict[str, str], ...]
    retrieval_version: str = "retrieval-test"
    generation_version: str = "generation-test"


class FakePredictor:
    def __init__(self, *, transient_failures: int = 0) -> None:
        self.frames: list[pd.DataFrame] = []
        self.transient_failures = transient_failures
        self.attempts = 0

    def predict_frame(self, frame: pd.DataFrame) -> RULPrediction:
        self.attempts += 1
        if self.attempts <= self.transient_failures:
            raise AgentTransientToolError("temporary inference outage")
        self.frames.append(frame.copy(deep=True))
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
            central_model_version="central-test",
            safety_model_version="safety-test",
        )


class FakeTrends:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def summarize(self, frame: pd.DataFrame) -> FakeTrendSummary:
        self.frames.append(frame.copy(deep=True))
        return FakeTrendSummary(
            unit_id=int(frame["unit_id"].iloc[0]),
            strongest_changes=("sensor_2", "sensor_11"),
        )


class FakeAnswers:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str) -> FakeDocumentAnswer:
        self.questions.append(question)
        return FakeDocumentAnswer(
            question=question,
            answerable=True,
            answer="La documentation décrit une dégradation progressive.",
            citations=({"title": "C-MAPSS", "page": "6"},),
        )


class FakePlanner:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "scenario_id": payload["scenario_id"],
            "optimized": {"status": "optimal", "objective_cost": 1234},
        }


class ScriptedRouter:
    def __init__(self, *actions: str) -> None:
        self.actions = list(actions)
        self.requests: list[MaintenanceAgentRequest] = []

    def route(self, request: MaintenanceAgentRequest) -> RouterDecision:
        self.requests.append(request)
        return RouterDecision(actions=self.actions, rationale="scripted test route")


@pytest.fixture
def services():
    value = AgentServices(
        predictor=FakePredictor(),
        trends=FakeTrends(),
        answers=FakeAnswers(),
        planner=FakePlanner(),
    )
    return value


def _request(
    *,
    requested_actions: list[str] | None = None,
    include_planning: bool = False,
) -> MaintenanceAgentRequest:
    payload: dict[str, Any] = {
        "incident_id": "incident-007",
        "message": "Analyse la RUL et les tendances de ce moteur.",
        "question": "Que décrit la documentation sur la dégradation ?",
        "history": {
            "unit_id": 7,
            "observations": [
                {
                    "cycle": cycle,
                    "sensors": {
                        "sensor_2": float(500 + cycle),
                        "sensor_11": float(40 - cycle / 10),
                    },
                }
                for cycle in range(1, 21)
            ],
        },
    }
    if requested_actions is not None:
        payload["requested_actions"] = requested_actions
    if include_planning:
        payload["planning"] = {
            "scenario_id": "fleet-agent-test",
            "description": "Scenario de test",
            "planning_horizon": 30,
            "technicians": 3,
            "workshop_capacity": 2,
            "maintenance_duration": 3,
            "technicians_required": 2,
            "criticality": "high",
            "other_engines": [
                {
                    "unit_id": 9,
                    "conservative_rul": 12.0,
                    "maintenance_duration": 2,
                    "technicians_required": 1,
                    "criticality": "medium",
                }
            ],
        }
    return MaintenanceAgentRequest.model_validate(payload)


def _run_to_interrupt(graph, request: MaintenanceAgentRequest):
    config = {"configurable": {"thread_id": request.incident_id}}
    state = graph.invoke(initial_agent_state(request), config)
    assert "__interrupt__" in state
    assert state["status"] == "pending_approval"
    return state, config


def test_rul_only_route_calls_only_predictor_and_pauses_for_review(services):
    router = ScriptedRouter("predict_rul")
    graph = build_maintenance_graph(
        services,
        router=router,
        checkpointer=InMemorySaver(),
    )

    state, config = _run_to_interrupt(graph, _request())

    assert state["trace"] == [
        "validate_request",
        "route_request",
        "predict_rul",
        "compose_draft",
    ]
    assert len(services.predictor.frames) == 1
    assert services.trends.frames == []
    assert services.answers.questions == []
    assert services.planner.payloads == []
    completed = graph.invoke(Command(resume={"action": "approve"}), config)
    result = result_from_state(completed)
    assert result.status == "approved"
    assert result.final_recommendation == result.draft_recommendation
    assert result.trace[-2:] == ["human_review", "finalize"]


def test_full_route_is_sequential_and_plans_with_model_decision_value(services):
    router = ScriptedRouter(
        "plan_maintenance",
        "answer_documents",
        "sensor_trends",
    )
    graph = build_maintenance_graph(
        services,
        router=router,
        checkpointer=InMemorySaver(),
    )

    state, _ = _run_to_interrupt(graph, _request(include_planning=True))

    assert state["route"]["actions"] == [
        "sensor_trends",
        "predict_rul",
        "answer_documents",
        "plan_maintenance",
    ]
    assert state["trace"] == [
        "validate_request",
        "route_request",
        "sensor_trends",
        "predict_rul",
        "answer_documents",
        "plan_maintenance",
        "compose_draft",
    ]
    payload = services.planner.payloads[0]
    current_engine = payload["engines"][0]
    assert current_engine["unit_id"] == 7
    assert current_engine["conservative_rul"] == 27.5
    assert current_engine["conservative_rul"] == state["rul_prediction"]["decision_value"]
    assert payload["engines"][1]["unit_id"] == 9


@pytest.mark.parametrize(
    ("decision", "expected_status", "expected_final"),
    [
        ({"action": "reject", "comment": "Données à revoir"}, "rejected", None),
        (
            {
                "action": "edit",
                "comment": "Formulation validée après correction",
                "edited_recommendation": "Recommandation humaine corrigée.",
            },
            "edited",
            "Recommandation humaine corrigée.",
        ),
    ],
)
def test_human_can_reject_or_edit(services, decision, expected_status, expected_final):
    graph = build_maintenance_graph(
        services,
        router=ScriptedRouter("predict_rul"),
        checkpointer=InMemorySaver(),
    )
    _, config = _run_to_interrupt(graph, _request())

    completed = graph.invoke(Command(resume=decision), config)

    assert completed["status"] == expected_status
    assert completed.get("final_recommendation") == expected_final


def test_missing_history_or_planning_context_fails_without_calling_tools(services):
    request = MaintenanceAgentRequest(
        incident_id="missing-input",
        message="Planifie une intervention de maintenance.",
        requested_actions=["plan_maintenance"],
    )
    graph = build_maintenance_graph(
        services,
        router=KeywordRouter(),
        checkpointer=InMemorySaver(),
    )

    state = graph.invoke(
        initial_agent_state(request),
        {"configurable": {"thread_id": request.incident_id}},
    )

    assert state["status"] == "failed"
    assert state["errors"][0]["category"] == "missing_input"
    assert "history" in state["errors"][0]["message"]
    assert "planning" in state["errors"][0]["message"]
    assert services.predictor.frames == []
    assert "__interrupt__" not in state


def test_transient_predictor_failure_is_retried():
    predictor = FakePredictor(transient_failures=1)
    services = AgentServices(
        predictor=predictor,
        trends=FakeTrends(),
        answers=FakeAnswers(),
        planner=FakePlanner(),
    )
    graph = build_maintenance_graph(
        services,
        router=ScriptedRouter("predict_rul"),
        checkpointer=InMemorySaver(),
        max_retry_attempts=2,
    )

    _run_to_interrupt(graph, _request())

    assert predictor.attempts == 2
    assert len(predictor.frames) == 1


def test_keyword_router_is_auditable_and_adds_rul_before_planning():
    request = _request(include_planning=True)

    decision = KeywordRouter().route(request)

    assert decision.rationale == "Routage lexical déterministe maintenance-agent-v1."
    assert decision.actions == [
        "sensor_trends",
        "predict_rul",
        "answer_documents",
        "plan_maintenance",
    ]


def test_sqlite_checkpoint_survives_graph_reopening(tmp_path: Path, services):
    project = tmp_path / "project"
    config_directory = project / "configs"
    config_directory.mkdir(parents=True)
    config_path = config_directory / "agent.toml"
    config_path.write_text(
        """
[workflow]
policy_version = "maintenance-agent-v1"
max_retry_attempts = 2
human_approval_required = true

[checkpoint]
path = "data/processed/agent/checkpoints.sqlite3"
""".strip(),
        encoding="utf-8",
    )
    request = _request()
    router = ScriptedRouter("predict_rul")
    run_config = {"configurable": {"thread_id": request.incident_id}}

    with open_agent_graph(services, router=router, config_path=config_path) as graph:
        interrupted = graph.invoke(initial_agent_state(request), run_config)
        assert "__interrupt__" in interrupted

    with open_agent_graph(services, router=router, config_path=config_path) as graph:
        completed = graph.invoke(Command(resume={"action": "approve"}), run_config)

    assert completed["status"] == "approved"
    assert read_agent_config(config_path).checkpoint_path.is_file()


def test_request_rejects_inconsistent_sensor_sets():
    with pytest.raises(ValueError, match="same sensor names"):
        MaintenanceAgentRequest.model_validate(
            {
                "incident_id": "bad-history",
                "message": "Analyse ce moteur",
                "history": {
                    "unit_id": 1,
                    "observations": [
                        {"cycle": 1, "sensors": {"sensor_2": 1.0}},
                        {"cycle": 2, "sensors": {"sensor_3": 2.0}},
                    ],
                },
            }
        )


def test_request_rejects_incident_identifier_that_cannot_be_used_in_api_path():
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        MaintenanceAgentRequest.model_validate(
            {
                "incident_id": "incident/unsafe",
                "message": "Analyse ce moteur",
            }
        )


def test_invalid_tool_output_fails_as_structured_state_without_human_review(services):
    class CitationlessAnswers:
        def ask(self, question: str) -> FakeDocumentAnswer:
            return FakeDocumentAnswer(
                question=question,
                answerable=True,
                answer="Réponse sans preuve.",
                citations=(),
            )

    graph = build_maintenance_graph(
        AgentServices(
            predictor=services.predictor,
            trends=services.trends,
            answers=CitationlessAnswers(),
            planner=services.planner,
        ),
        router=ScriptedRouter("answer_documents"),
        checkpointer=InMemorySaver(),
    )
    request = _request()

    state = graph.invoke(
        initial_agent_state(request),
        {"configurable": {"thread_id": request.incident_id}},
    )

    assert state["status"] == "failed"
    assert state["errors"][0]["node"] == "answer_documents"
    assert "citation" in state["errors"][0]["message"]
    assert "__interrupt__" not in state


def test_non_object_tool_output_fails_as_structured_state(services):
    class InvalidTrends:
        def summarize(self, frame: pd.DataFrame) -> list[str]:
            return ["sensor_2"]

    graph = build_maintenance_graph(
        AgentServices(
            predictor=services.predictor,
            trends=InvalidTrends(),
            answers=services.answers,
            planner=services.planner,
        ),
        router=ScriptedRouter("sensor_trends"),
        checkpointer=InMemorySaver(),
    )

    state = graph.invoke(
        initial_agent_state(_request()),
        {"configurable": {"thread_id": "incident-007"}},
    )

    assert state["status"] == "failed"
    assert state["errors"][0]["node"] == "sensor_trends"
    assert "JSON object" in state["errors"][0]["message"]
    assert "__interrupt__" not in state


def test_invalid_router_output_fails_as_structured_state(services):
    class InvalidRouter:
        def route(self, request: MaintenanceAgentRequest) -> dict[str, Any]:
            return {"actions": ["delete_everything"], "rationale": "invalid"}

    graph = build_maintenance_graph(
        services,
        router=InvalidRouter(),
        checkpointer=InMemorySaver(),
    )

    state = graph.invoke(
        initial_agent_state(_request()),
        {"configurable": {"thread_id": "incident-007"}},
    )

    assert state["status"] == "failed"
    assert state["errors"][0]["category"] == "invalid_router_output"
    assert state["trace"] == ["validate_request", "route_request"]
    assert "__interrupt__" not in state


def test_new_run_on_same_thread_clears_results_and_recomputes_tools(services):
    graph = build_maintenance_graph(
        services,
        router=ScriptedRouter("predict_rul"),
        checkpointer=InMemorySaver(),
    )
    first_request = _request()
    run_config = {"configurable": {"thread_id": first_request.incident_id}}
    _, _ = _run_to_interrupt(graph, first_request)
    graph.invoke(Command(resume={"action": "approve"}), run_config)

    payload = first_request.model_dump(mode="json")
    payload["history"]["observations"][-1]["sensors"]["sensor_2"] = 999.0
    second_request = MaintenanceAgentRequest.model_validate(payload)
    second_state = graph.invoke(initial_agent_state(second_request), run_config)

    assert "__interrupt__" in second_state
    assert services.predictor.attempts == 2
    assert services.predictor.frames[-1]["sensor_2"].iloc[-1] == 999.0
    assert second_state["trace"] == [
        "validate_request",
        "route_request",
        "predict_rul",
        "compose_draft",
    ]


def test_runtime_exposes_start_status_and_single_human_review(tmp_path: Path, services):
    project = tmp_path / "runtime-project"
    config_directory = project / "configs"
    config_directory.mkdir(parents=True)
    config_path = config_directory / "agent.toml"
    config_path.write_text(
        """
[workflow]
policy_version = "maintenance-agent-v1"
max_retry_attempts = 2
human_approval_required = true

[checkpoint]
path = "data/processed/agent/checkpoints.sqlite3"
""".strip(),
        encoding="utf-8",
    )

    with MaintenanceAgentRuntime(
        services,
        router=ScriptedRouter("predict_rul"),
        config_path=config_path,
    ) as runtime:
        started = runtime.start(_request())
        current = runtime.status("incident-007")

        assert started.result.status == "pending_approval"
        assert current.result == started.result
        assert current.interrupts[0]["kind"] == "maintenance_recommendation_review"
        with pytest.raises(AgentRunConflictError, match="already waiting"):
            runtime.start(_request())

        reviewed = runtime.review(
            "incident-007",
            HumanReviewDecision(action="approve"),
        )

        assert reviewed.result.status == "approved"
        assert reviewed.interrupts == []
        with pytest.raises(AgentRunConflictError, match="not waiting"):
            runtime.review(
                "incident-007",
                HumanReviewDecision(action="reject"),
            )
