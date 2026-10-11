from __future__ import annotations

from pathlib import Path

import pytest

from industrial_maintenance.planning import (
    PlanningError,
    compare_plans,
    load_scenario,
    read_planning_config,
    scenario_from_dict,
    validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "maintenance_planning.toml"
SCENARIO_DIRECTORY = ROOT / "data" / "scenarios" / "maintenance"


def _load(name: str):
    config = read_planning_config(CONFIG_PATH)
    return config, load_scenario(SCENARIO_DIRECTORY / name, config)


def test_normal_scenario_schedules_due_engines_and_defers_later_engine() -> None:
    config, scenario = _load("normal_load.json")

    comparison = compare_plans(scenario, config)

    assert comparison.optimized.status == "optimal"
    assert comparison.optimized.objective_cost == 2570
    assert comparison.optimized.deferred_unit_ids == (21,)
    assert {item.unit_id for item in comparison.optimized.assignments} == {3, 7, 12, 18}
    assert all(item.tardiness_cycles == 0 for item in comparison.optimized.assignments)


def test_cp_sat_improves_the_same_cost_function_under_constrained_capacity() -> None:
    config, scenario = _load("constrained_load.json")

    comparison = compare_plans(scenario, config)

    assert comparison.heuristic.status == "feasible"
    assert comparison.optimized.status == "optimal"
    assert comparison.heuristic.objective_cost == 7680
    assert comparison.optimized.objective_cost == 5110
    assert comparison.absolute_savings == 2570
    assert comparison.relative_savings == pytest.approx(0.3346354167)
    validate_plan(scenario, comparison.heuristic)
    validate_plan(scenario, comparison.optimized)


def test_overloaded_scenario_is_reported_infeasible() -> None:
    config, scenario = _load("infeasible_load.json")

    comparison = compare_plans(scenario, config)

    assert comparison.heuristic.status == "partial"
    assert comparison.heuristic.unscheduled_unit_ids == (203,)
    assert comparison.optimized.status == "infeasible"
    assert comparison.optimized.objective_cost is None
    assert comparison.optimized.unscheduled_unit_ids == (201, 202, 203)


def test_cost_uses_completion_deadline_and_criticality_weight() -> None:
    config, scenario = _load("constrained_load.json")

    comparison = compare_plans(scenario, config)
    engine_103 = next(
        item for item in comparison.optimized.assignments if item.unit_id == 103
    )

    assert engine_103.end_cycle == 10
    assert engine_103.deadline_cycle == 9
    assert engine_103.tardiness_cycles == 1
    assert engine_103.cost == 680 + 4 * 50 + 1 * 4 * 250


def test_scenario_rejects_duplicate_units_and_unknown_criticality() -> None:
    config = read_planning_config(CONFIG_PATH)
    base = {
        "schema_version": 1,
        "scenario_id": "invalid",
        "planning_horizon": 10,
        "technicians": 2,
        "workshop_capacity": 1,
        "engines": [
            {
                "unit_id": 1,
                "conservative_rul": 5,
                "maintenance_duration": 1,
                "criticality": "high",
            }
        ],
    }
    duplicate = {**base, "engines": [*base["engines"], dict(base["engines"][0])]}
    with pytest.raises(PlanningError, match="Duplicate unit_id"):
        scenario_from_dict(duplicate, config)

    invalid_criticality = {**base, "engines": [{**base["engines"][0], "criticality": "urgent"}]}
    with pytest.raises(PlanningError, match="criticality"):
        scenario_from_dict(invalid_criticality, config)
