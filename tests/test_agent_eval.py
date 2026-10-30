"""Tests for the deterministic agent-orchestration evaluation harness."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from industrial_maintenance.agent_eval import (
    AgentEvaluationScenario,
    AgentEvaluationSuite,
    evaluate_agent_suite,
    load_evaluation_suite,
)

SCENARIOS_PATH = Path("evals/agent_scenarios.json")


def _deterministic_clock(step: float = 0.01):
    values = itertools.count(start=0.0, step=step)
    return lambda: next(values)


def test_versioned_suite_loads_with_unique_scenarios() -> None:
    suite = load_evaluation_suite(SCENARIOS_PATH)

    assert suite.schema_version == 1
    assert suite.suite_id == "maintenance-agent-routing-safety-v1"
    assert len(suite.scenarios) == 6
    assert len({scenario.scenario_id for scenario in suite.scenarios}) == 6


def test_evaluation_executes_routes_safety_and_human_review() -> None:
    suite = load_evaluation_suite(SCENARIOS_PATH)

    report = evaluate_agent_suite(suite, clock=_deterministic_clock())

    metrics = report.metrics
    assert metrics["scenario_count"] == 6
    assert metrics["scenario_pass_count"] == 6
    assert metrics["exact_tool_route_count"] == 6
    assert metrics["interrupt_policy_count"] == 6
    assert metrics["expected_outcome_count"] == 6
    assert metrics["task_completion_checked"] == 5
    assert metrics["task_completion_count"] == 5
    assert metrics["citation_policy_checked"] == 3
    assert metrics["citation_policy_count"] == 3
    assert metrics["planning_safety_checked"] == 2
    assert metrics["planning_safety_count"] == 2
    assert metrics["latency_ms"] == {
        "mean": pytest.approx(20.0),
        "p95_nearest_rank": pytest.approx(20.0),
        "maximum": pytest.approx(20.0),
    }
    assert all(item.initial_latency_ms == pytest.approx(10.0) for item in report.scenarios)
    assert all(item.total_latency_ms == pytest.approx(20.0) for item in report.scenarios)
    assert report.service_mode == "deterministic_fakes_no_model_or_network"


def test_report_is_json_serializable_and_contains_scenario_evidence() -> None:
    report = evaluate_agent_suite(
        load_evaluation_suite(SCENARIOS_PATH),
        clock=_deterministic_clock(),
    )

    rendered = json.dumps(report.model_dump(mode="json"), allow_nan=False)
    restored = json.loads(rendered)
    planning = next(
        item
        for item in restored["scenarios"]
        if item["scenario_id"] == "planning-with-conservative-rul"
    )
    abstention = next(
        item for item in restored["scenarios"] if item["scenario_id"] == "document-abstention"
    )

    assert planning["planning_safety_applicable"] is True
    assert planning["planning_safety_pass"] is True
    assert abstention["citation_policy_applicable"] is True
    assert abstention["citation_policy_pass"] is True
    assert planning["failures"] == []


def test_wrong_expectation_is_reported_instead_of_hidden() -> None:
    original = load_evaluation_suite(SCENARIOS_PATH).scenarios[0]
    payload = original.model_dump(mode="json")
    payload["expected_actions"] = ["predict_rul"]
    scenario = AgentEvaluationScenario.model_validate(payload)
    suite = AgentEvaluationSuite(
        schema_version=1,
        suite_id="intentional-failure",
        description="Vérifie que le harnais signale une attente incorrecte.",
        scenarios=[scenario],
    )

    report = evaluate_agent_suite(suite, clock=_deterministic_clock())

    assert report.metrics["scenario_pass_count"] == 0
    assert report.metrics["exact_tool_route_count"] == 0
    assert report.scenarios[0].passed is False
    assert report.scenarios[0].failures == ["route"]


def test_loader_rejects_duplicate_scenario_identifiers(tmp_path: Path) -> None:
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    raw["scenarios"] = [raw["scenarios"][0], raw["scenarios"][0]]
    invalid = tmp_path / "duplicates.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_evaluation_suite(invalid)


def test_loader_rejects_path_unsafe_scenario_identifier(tmp_path: Path) -> None:
    raw = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    raw["scenarios"][0]["scenario_id"] = "unsafe/scenario"
    invalid = tmp_path / "unsafe.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        load_evaluation_suite(invalid)
