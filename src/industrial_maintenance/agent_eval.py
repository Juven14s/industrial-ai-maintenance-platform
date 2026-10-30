"""Deterministic, versioned evaluation harness for the supervised agent workflow.

This module deliberately uses lightweight fake domain services.  It evaluates the
orchestration contract independently from model quality, network access, or local
artifacts.  Reported measurements therefore describe routing and safety invariants,
not predictive accuracy or industrial performance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from industrial_maintenance.agent_contracts import (
    AgentAction,
    AgentStatus,
    MaintenanceAgentRequest,
)
from industrial_maintenance.agent_workflow import (
    AGENT_POLICY_VERSION,
    AgentServices,
    KeywordRouter,
    build_maintenance_graph,
    initial_agent_state,
)

EVALUATION_SCHEMA_VERSION = 1
DEFAULT_SCENARIOS_PATH = Path("evals/agent_scenarios.json")


class AgentEvaluationScenario(BaseModel):
    """One compact scenario expanded into a valid agent request at runtime."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_id: str = Field(
        min_length=1,
        max_length=95,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    message: str = Field(min_length=3, max_length=2000)
    question: str | None = Field(default=None, min_length=3, max_length=500)
    include_history: bool = False
    include_planning: bool = False
    requested_actions: list[AgentAction] | None = None
    expected_actions: list[AgentAction] = Field(min_length=1)
    expected_initial_status: Literal["pending_approval", "failed"]
    expected_interrupt: bool
    review_action: Literal["approve", "reject"] | None = None
    expected_final_status: AgentStatus
    expected_answerable: bool | None = None
    expect_planning_safety: bool = False

    @field_validator("requested_actions", "expected_actions")
    @classmethod
    def unique_actions(cls, value: list[AgentAction] | None) -> list[AgentAction] | None:
        if value is None:
            return None
        if len(value) != len(set(value)):
            raise ValueError("actions must not contain duplicates")
        return value


class AgentEvaluationSuite(BaseModel):
    """Versioned collection of orchestration scenarios."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    suite_id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    scenarios: list[AgentEvaluationScenario] = Field(min_length=1)

    @field_validator("scenarios")
    @classmethod
    def unique_scenario_ids(
        cls, value: list[AgentEvaluationScenario]
    ) -> list[AgentEvaluationScenario]:
        identifiers = [scenario.scenario_id for scenario in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scenario_id values must be unique")
        return value


class ScenarioEvaluation(BaseModel):
    """Auditable checks and timings for one executed scenario."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    scenario_id: str
    passed: bool
    expected_actions: list[AgentAction]
    actual_actions: list[AgentAction]
    route_exact: bool
    expected_initial_status: str
    actual_initial_status: str
    initial_status_correct: bool
    expected_interrupt: bool
    interrupt_observed: bool
    interrupt_correct: bool
    citation_policy_applicable: bool
    citation_policy_pass: bool | None
    planning_safety_applicable: bool
    planning_safety_pass: bool | None
    expected_final_status: str
    actual_final_status: str
    final_status_correct: bool
    initial_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
    failures: list[str]


class AgentEvaluationReport(BaseModel):
    """JSON-serializable report with aggregate metrics and scenario evidence."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal[1]
    suite_id: str
    workflow_policy_version: str
    router: str
    service_mode: str
    metrics: dict[str, Any]
    scenarios: list[ScenarioEvaluation]


class _DeterministicPredictor:
    def __init__(self) -> None:
        self.predictions: list[dict[str, Any]] = []

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        prediction = {
            "unit_id": int(frame["unit_id"].iloc[0]),
            "history_start_cycle": int(frame["cycle"].iloc[0]),
            "last_cycle": int(frame["cycle"].iloc[-1]),
            "observed_cycles": len(frame),
            "window_cycles_used": min(len(frame), 30),
            "rul_estimate": 42.0,
            "rul_conservative": 18.0,
            "safety_base_prediction": 21.0,
            "safety_margin": 3.0,
            "decision_value": 18.0,
            "central_model_version": "eval-central-v1",
            "safety_model_version": "eval-safety-v1",
        }
        self.predictions.append(prediction)
        return prediction


class _DeterministicTrends:
    def summarize(self, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "unit_id": int(frame["unit_id"].iloc[0]),
            "strongest_changes": ["sensor_2", "sensor_11"],
            "limitation": "Descriptive only; no fault diagnosis.",
        }


class _DeterministicAnswers:
    def ask(self, question: str) -> dict[str, Any]:
        answerable = "inconnue" not in question.casefold()
        return {
            "question": question,
            "answerable": answerable,
            "answer": (
                "La documentation décrit une dégradation progressive."
                if answerable
                else "Les sources disponibles ne permettent pas de répondre."
            ),
            "citations": (
                [{"title": "C-MAPSS", "section": "Damage propagation", "page": 6}]
                if answerable
                else []
            ),
            "retrieval_version": "eval-retrieval-v1",
            "generation_version": "eval-generation-v1",
        }


class _DeterministicPlanner:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "scenario_id": payload["scenario_id"],
            "optimized": {"status": "optimal", "objective_cost": 1000},
        }


def load_evaluation_suite(
    path: str | Path = DEFAULT_SCENARIOS_PATH,
) -> AgentEvaluationSuite:
    """Load and strictly validate a versioned scenario file."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
        return AgentEvaluationSuite.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid agent evaluation suite {source}: {exc}") from exc


def _history() -> dict[str, Any]:
    return {
        "unit_id": 7,
        "observations": [
            {
                "cycle": cycle,
                "sensors": {
                    "sensor_2": float(640 + cycle / 10),
                    "sensor_11": float(48 - cycle / 100),
                },
            }
            for cycle in range(1, 21)
        ],
    }


def _planning() -> dict[str, Any]:
    return {
        "scenario_id": "agent-eval-fleet",
        "description": "Deterministic orchestration evaluation",
        "planning_horizon": 30,
        "technicians": 2,
        "workshop_capacity": 1,
        "maintenance_duration": 3,
        "technicians_required": 1,
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


def _request_for(scenario: AgentEvaluationScenario) -> MaintenanceAgentRequest:
    payload: dict[str, Any] = {
        "incident_id": f"eval-{scenario.scenario_id}",
        "message": scenario.message,
    }
    if scenario.question is not None:
        payload["question"] = scenario.question
    if scenario.include_history:
        payload["history"] = _history()
    if scenario.include_planning:
        payload["planning"] = _planning()
    if scenario.requested_actions is not None:
        payload["requested_actions"] = scenario.requested_actions
    return MaintenanceAgentRequest.model_validate(payload)


def _interrupt_observed(state: dict[str, Any]) -> bool:
    values = state.get("__interrupt__", ())
    return bool(values)


def _citation_check(
    scenario: AgentEvaluationScenario,
    state: dict[str, Any],
) -> tuple[bool, bool | None]:
    applicable = "answer_documents" in scenario.expected_actions
    if not applicable:
        return False, None
    answer = state.get("document_answer")
    if not isinstance(answer, dict):
        return True, False
    answerable = answer.get("answerable")
    citations = answer.get("citations")
    if not isinstance(answerable, bool) or not isinstance(citations, list):
        return True, False
    if scenario.expected_answerable is not None and answerable != scenario.expected_answerable:
        return True, False
    return True, bool(citations) if answerable else not citations


def _planning_safety_check(
    scenario: AgentEvaluationScenario,
    state: dict[str, Any],
    predictor: _DeterministicPredictor,
    planner: _DeterministicPlanner,
) -> tuple[bool, bool | None]:
    if not scenario.expect_planning_safety:
        return False, None
    if not predictor.predictions or not planner.payloads:
        return True, False
    prediction = predictor.predictions[-1]
    engines = planner.payloads[-1].get("engines")
    if not isinstance(engines, list) or not engines:
        return True, False
    current = engines[0]
    if not isinstance(current, dict):
        return True, False
    conservative = current.get("conservative_rul")
    passed = (
        conservative == prediction["decision_value"]
        and conservative == prediction["rul_conservative"]
        and "rul_estimate" not in current
        and "decision_value" not in current
        and state.get("planning_result") is not None
    )
    return True, passed


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _rate(passed: int, total: int) -> float | None:
    return passed / total if total else None


def _aggregate(results: list[ScenarioEvaluation]) -> dict[str, Any]:
    scenario_count = len(results)
    citation_results = [item for item in results if item.citation_policy_applicable]
    planning_results = [item for item in results if item.planning_safety_applicable]
    expected_completion = [item for item in results if item.expected_final_status != "failed"]
    total_latencies = [item.total_latency_ms for item in results]
    return {
        "scenario_count": scenario_count,
        "scenario_pass_count": sum(item.passed for item in results),
        "scenario_pass_rate": _rate(sum(item.passed for item in results), scenario_count),
        "exact_tool_route_count": sum(item.route_exact for item in results),
        "exact_tool_route_rate": _rate(sum(item.route_exact for item in results), scenario_count),
        "initial_status_count": sum(item.initial_status_correct for item in results),
        "initial_status_rate": _rate(
            sum(item.initial_status_correct for item in results), scenario_count
        ),
        "interrupt_policy_count": sum(item.interrupt_correct for item in results),
        "interrupt_policy_rate": _rate(
            sum(item.interrupt_correct for item in results), scenario_count
        ),
        "expected_outcome_count": sum(item.final_status_correct for item in results),
        "expected_outcome_rate": _rate(
            sum(item.final_status_correct for item in results), scenario_count
        ),
        "task_completion_checked": len(expected_completion),
        "task_completion_count": sum(item.final_status_correct for item in expected_completion),
        "task_completion_rate": _rate(
            sum(item.final_status_correct for item in expected_completion),
            len(expected_completion),
        ),
        "citation_policy_checked": len(citation_results),
        "citation_policy_count": sum(
            item.citation_policy_pass is True for item in citation_results
        ),
        "citation_policy_rate": _rate(
            sum(item.citation_policy_pass is True for item in citation_results),
            len(citation_results),
        ),
        "planning_safety_checked": len(planning_results),
        "planning_safety_count": sum(
            item.planning_safety_pass is True for item in planning_results
        ),
        "planning_safety_rate": _rate(
            sum(item.planning_safety_pass is True for item in planning_results),
            len(planning_results),
        ),
        "latency_ms": {
            "mean": statistics.fmean(total_latencies) if total_latencies else 0.0,
            "p95_nearest_rank": _percentile_95(total_latencies),
            "maximum": max(total_latencies, default=0.0),
        },
    }


def evaluate_agent_suite(
    suite: AgentEvaluationSuite,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentEvaluationReport:
    """Execute every scenario and compute evidence-backed orchestration metrics."""

    results: list[ScenarioEvaluation] = []
    for scenario in suite.scenarios:
        predictor = _DeterministicPredictor()
        planner = _DeterministicPlanner()
        services = AgentServices(
            predictor=predictor,
            trends=_DeterministicTrends(),
            answers=_DeterministicAnswers(),
            planner=planner,
        )
        graph = build_maintenance_graph(
            services,
            router=KeywordRouter(),
            checkpointer=InMemorySaver(),
        )
        request = _request_for(scenario)
        config = {"configurable": {"thread_id": request.incident_id}}

        started_at = clock()
        initial = graph.invoke(initial_agent_state(request), config)
        initial_finished_at = clock()

        actual_actions = list((initial.get("route") or {}).get("actions", []))
        route_exact = actual_actions == scenario.expected_actions
        actual_initial_status = str(initial.get("status", "missing"))
        initial_status_correct = actual_initial_status == scenario.expected_initial_status
        observed_interrupt = _interrupt_observed(initial)
        interrupt_correct = observed_interrupt == scenario.expected_interrupt
        citation_applicable, citation_pass = _citation_check(scenario, initial)
        planning_applicable, planning_pass = _planning_safety_check(
            scenario,
            initial,
            predictor,
            planner,
        )

        final = initial
        if scenario.review_action is not None and observed_interrupt:
            final = graph.invoke(
                Command(
                    resume={
                        "action": scenario.review_action,
                        "comment": "Deterministic evaluation decision",
                    }
                ),
                config,
            )
        finished_at = clock()
        actual_final_status = str(final.get("status", "missing"))
        final_status_correct = actual_final_status == scenario.expected_final_status

        checks = {
            "route": route_exact,
            "initial_status": initial_status_correct,
            "interrupt": interrupt_correct,
            "final_status": final_status_correct,
        }
        if citation_applicable:
            checks["citations"] = citation_pass is True
        if planning_applicable:
            checks["planning_safety"] = planning_pass is True
        failures = [name for name, passed in checks.items() if not passed]
        results.append(
            ScenarioEvaluation(
                scenario_id=scenario.scenario_id,
                passed=not failures,
                expected_actions=scenario.expected_actions,
                actual_actions=actual_actions,
                route_exact=route_exact,
                expected_initial_status=scenario.expected_initial_status,
                actual_initial_status=actual_initial_status,
                initial_status_correct=initial_status_correct,
                expected_interrupt=scenario.expected_interrupt,
                interrupt_observed=observed_interrupt,
                interrupt_correct=interrupt_correct,
                citation_policy_applicable=citation_applicable,
                citation_policy_pass=citation_pass,
                planning_safety_applicable=planning_applicable,
                planning_safety_pass=planning_pass,
                expected_final_status=scenario.expected_final_status,
                actual_final_status=actual_final_status,
                final_status_correct=final_status_correct,
                initial_latency_ms=round(
                    max(0.0, initial_finished_at - started_at) * 1000,
                    3,
                ),
                total_latency_ms=round(max(0.0, finished_at - started_at) * 1000, 3),
                failures=failures,
            )
        )

    return AgentEvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        suite_id=suite.suite_id,
        workflow_policy_version=AGENT_POLICY_VERSION,
        router="KeywordRouter",
        service_mode="deterministic_fakes_no_model_or_network",
        metrics=_aggregate(results),
        scenarios=results,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = evaluate_agent_suite(load_evaluation_suite(arguments.scenarios))
    rendered = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
    if arguments.output is not None:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.metrics["scenario_pass_count"] == report.metrics["scenario_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
