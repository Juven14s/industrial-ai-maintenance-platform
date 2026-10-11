"""Maintenance scheduling baseline and CP-SAT optimizer for simulated fleets."""

from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

PLANNING_VERSION = "maintenance-planning-v1"
Criticality = Literal["low", "medium", "high"]
PlanStatus = Literal["optimal", "feasible", "partial", "infeasible", "unknown"]


class PlanningError(ValueError):
    """Raised when a planning configuration or scenario is invalid."""


@dataclass(frozen=True)
class PlanningCosts:
    preventive_default: int
    downtime_per_cycle: int
    early_per_cycle: int
    late_per_cycle: int
    unscheduled: int


@dataclass(frozen=True)
class PlanningConfig:
    default_horizon_cycles: int
    default_technicians: int
    default_workshop_capacity: int
    solver_time_limit_seconds: float
    solver_workers: int
    solver_random_seed: int
    costs: PlanningCosts
    criticality_weights: dict[str, int]
    scenario_directory: Path
    evaluation_json: Path
    evaluation_report: Path
    comparison_csv: Path
    gantt_figure: Path


@dataclass(frozen=True)
class EngineMaintenance:
    unit_id: int
    conservative_rul: float
    maintenance_duration: int
    technicians_required: int
    criticality: Criticality
    preventive_cost: int

    @property
    def deadline_cycle(self) -> int:
        return max(0, math.floor(self.conservative_rul))


@dataclass(frozen=True)
class PlanningScenario:
    scenario_id: str
    description: str
    planning_horizon: int
    technicians: int
    workshop_capacity: int
    engines: tuple[EngineMaintenance, ...]


@dataclass(frozen=True)
class MaintenanceAssignment:
    unit_id: int
    start_cycle: int
    end_cycle: int
    deadline_cycle: int
    maintenance_duration: int
    technicians_required: int
    criticality: Criticality
    preventive_cost: int
    earliness_cycles: int
    tardiness_cycles: int
    cost: int


@dataclass(frozen=True)
class PlanResult:
    method: Literal["heuristic", "cp_sat"]
    status: PlanStatus
    objective_cost: int | None
    assignments: tuple[MaintenanceAssignment, ...]
    unscheduled_unit_ids: tuple[int, ...]
    deferred_unit_ids: tuple[int, ...]
    solve_time_seconds: float
    solver_status: str
    planning_version: str = PLANNING_VERSION


@dataclass(frozen=True)
class PlanningComparison:
    scenario_id: str
    heuristic: PlanResult
    optimized: PlanResult
    absolute_savings: int | None
    relative_savings: float | None


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlanningError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanningError(f"{label} must be a non-negative integer")
    return value


def read_planning_config(path: str | Path) -> PlanningConfig:
    source = Path(path).resolve()
    root = source.parent.parent
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        costs = PlanningCosts(
            preventive_default=_non_negative_int(
                raw["costs"]["preventive_default"], "costs.preventive_default"
            ),
            downtime_per_cycle=_non_negative_int(
                raw["costs"]["downtime_per_cycle"], "costs.downtime_per_cycle"
            ),
            early_per_cycle=_non_negative_int(
                raw["costs"]["early_per_cycle"], "costs.early_per_cycle"
            ),
            late_per_cycle=_non_negative_int(
                raw["costs"]["late_per_cycle"], "costs.late_per_cycle"
            ),
            unscheduled=_non_negative_int(
                raw["costs"]["unscheduled"], "costs.unscheduled"
            ),
        )
        weights = {
            level: _positive_int(raw["criticality_weights"][level], f"weight {level}")
            for level in ("low", "medium", "high")
        }
        outputs = raw["outputs"]
        config = PlanningConfig(
            default_horizon_cycles=_positive_int(
                raw["default_horizon_cycles"], "default_horizon_cycles"
            ),
            default_technicians=_positive_int(
                raw["default_technicians"], "default_technicians"
            ),
            default_workshop_capacity=_positive_int(
                raw["default_workshop_capacity"], "default_workshop_capacity"
            ),
            solver_time_limit_seconds=float(raw["solver_time_limit_seconds"]),
            solver_workers=_positive_int(raw["solver_workers"], "solver_workers"),
            solver_random_seed=int(raw["solver_random_seed"]),
            costs=costs,
            criticality_weights=weights,
            scenario_directory=root / raw["scenario_directory"],
            evaluation_json=root / outputs["evaluation_json"],
            evaluation_report=root / outputs["evaluation_report"],
            comparison_csv=root / outputs["comparison_csv"],
            gantt_figure=root / outputs["gantt_figure"],
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise PlanningError(f"Invalid planning configuration {source}: {exc}") from exc
    if not math.isfinite(config.solver_time_limit_seconds) or (
        config.solver_time_limit_seconds <= 0
    ):
        raise PlanningError("solver_time_limit_seconds must be positive and finite")
    return config


def scenario_from_dict(
    raw: dict[str, Any],
    config: PlanningConfig,
) -> PlanningScenario:
    try:
        schema_version = raw["schema_version"]
        scenario_id = str(raw["scenario_id"]).strip()
        description = str(raw.get("description", "")).strip()
        horizon = _positive_int(raw["planning_horizon"], "planning_horizon")
        technicians = _positive_int(raw["technicians"], "technicians")
        capacity = _positive_int(raw["workshop_capacity"], "workshop_capacity")
        engine_rows = raw["engines"]
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanningError(f"Invalid planning scenario: {exc}") from exc
    if schema_version != 1:
        raise PlanningError(f"Unsupported planning scenario schema: {schema_version}")
    if not scenario_id:
        raise PlanningError("scenario_id must be non-empty")
    if not isinstance(engine_rows, list) or not engine_rows:
        raise PlanningError("engines must be a non-empty list")

    engines: list[EngineMaintenance] = []
    seen_units: set[int] = set()
    for position, row in enumerate(engine_rows, start=1):
        if not isinstance(row, dict):
            raise PlanningError(f"engine {position} must be an object")
        try:
            unit_id = _positive_int(row["unit_id"], f"engine {position} unit_id")
            rul = float(row["conservative_rul"])
            duration = _positive_int(
                row["maintenance_duration"],
                f"engine {unit_id} maintenance_duration",
            )
            required = _positive_int(
                row.get("technicians_required", 1),
                f"engine {unit_id} technicians_required",
            )
            criticality = str(row["criticality"])
            preventive_cost = _non_negative_int(
                row.get("preventive_cost", config.costs.preventive_default),
                f"engine {unit_id} preventive_cost",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanningError(f"Invalid engine at position {position}: {exc}") from exc
        if unit_id in seen_units:
            raise PlanningError(f"Duplicate unit_id: {unit_id}")
        if not math.isfinite(rul) or rul < 0:
            raise PlanningError(f"engine {unit_id} conservative_rul must be finite and >= 0")
        if criticality not in config.criticality_weights:
            raise PlanningError(
                f"engine {unit_id} criticality must be low, medium or high"
            )
        seen_units.add(unit_id)
        engines.append(
            EngineMaintenance(
                unit_id=unit_id,
                conservative_rul=rul,
                maintenance_duration=duration,
                technicians_required=required,
                criticality=criticality,  # type: ignore[arg-type]
                preventive_cost=preventive_cost,
            )
        )
    return PlanningScenario(
        scenario_id=scenario_id,
        description=description,
        planning_horizon=horizon,
        technicians=technicians,
        workshop_capacity=capacity,
        engines=tuple(engines),
    )


def load_scenario(path: str | Path, config: PlanningConfig) -> PlanningScenario:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanningError(f"Invalid scenario file {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PlanningError("Planning scenario root must be an object")
    return scenario_from_dict(raw, config)


def planning_config_from_api(path: str | Path) -> Path:
    source = Path(path).resolve()
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        configured = Path(raw["planning"]["config"])
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise PlanningError(f"Invalid planning API configuration {source}: {exc}") from exc
    if configured.is_absolute():
        return configured
    return source.parent.parent / configured


class MaintenancePlanningService:
    def __init__(self, config_path: str | Path) -> None:
        self.config = read_planning_config(config_path)

    @classmethod
    def from_api_config(cls, path: str | Path) -> MaintenancePlanningService:
        return cls(planning_config_from_api(path))

    @property
    def version(self) -> str:
        return PLANNING_VERSION

    def plan(self, payload: dict[str, Any]) -> PlanningComparison:
        scenario = scenario_from_dict(payload, self.config)
        return compare_plans(scenario, self.config)


def _split_engines(
    scenario: PlanningScenario,
) -> tuple[list[EngineMaintenance], tuple[int, ...]]:
    due = [
        engine
        for engine in scenario.engines
        if engine.deadline_cycle <= scenario.planning_horizon
    ]
    deferred = tuple(
        sorted(
            engine.unit_id
            for engine in scenario.engines
            if engine.deadline_cycle > scenario.planning_horizon
        )
    )
    return due, deferred


def _assignment(
    engine: EngineMaintenance,
    start_cycle: int,
    config: PlanningConfig,
) -> MaintenanceAssignment:
    end_cycle = start_cycle + engine.maintenance_duration
    earliness = max(engine.deadline_cycle - end_cycle, 0)
    tardiness = max(end_cycle - engine.deadline_cycle, 0)
    weight = config.criticality_weights[engine.criticality]
    cost = (
        engine.preventive_cost
        + config.costs.downtime_per_cycle * engine.maintenance_duration
        + config.costs.early_per_cycle * earliness
        + config.costs.late_per_cycle * weight * tardiness
    )
    return MaintenanceAssignment(
        unit_id=engine.unit_id,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
        deadline_cycle=engine.deadline_cycle,
        maintenance_duration=engine.maintenance_duration,
        technicians_required=engine.technicians_required,
        criticality=engine.criticality,
        preventive_cost=engine.preventive_cost,
        earliness_cycles=earliness,
        tardiness_cycles=tardiness,
        cost=cost,
    )


def heuristic_plan(
    scenario: PlanningScenario,
    config: PlanningConfig,
) -> PlanResult:
    started = time.perf_counter()
    due, deferred = _split_engines(scenario)
    workshop_use = [0] * scenario.planning_horizon
    technician_use = [0] * scenario.planning_horizon
    assignments: list[MaintenanceAssignment] = []
    unscheduled: list[int] = []
    ordered = sorted(
        due,
        key=lambda engine: (
            engine.deadline_cycle,
            -config.criticality_weights[engine.criticality],
            engine.unit_id,
        ),
    )
    for engine in ordered:
        candidates: list[MaintenanceAssignment] = []
        latest_start = scenario.planning_horizon - engine.maintenance_duration
        if engine.technicians_required <= scenario.technicians:
            for start in range(max(latest_start + 1, 0)):
                slots = range(start, start + engine.maintenance_duration)
                if all(
                    workshop_use[slot] < scenario.workshop_capacity
                    and technician_use[slot] + engine.technicians_required
                    <= scenario.technicians
                    for slot in slots
                ):
                    candidates.append(_assignment(engine, start, config))
        if not candidates:
            unscheduled.append(engine.unit_id)
            continue
        chosen = min(candidates, key=lambda item: (item.cost, item.start_cycle))
        assignments.append(chosen)
        for slot in range(chosen.start_cycle, chosen.end_cycle):
            workshop_use[slot] += 1
            technician_use[slot] += chosen.technicians_required

    unscheduled_cost = sum(
        config.costs.unscheduled * config.criticality_weights[engine.criticality]
        for engine in due
        if engine.unit_id in unscheduled
    )
    objective = sum(item.cost for item in assignments) + unscheduled_cost
    return PlanResult(
        method="heuristic",
        status="feasible" if not unscheduled else "partial",
        objective_cost=objective,
        assignments=tuple(sorted(assignments, key=lambda item: (item.start_cycle, item.unit_id))),
        unscheduled_unit_ids=tuple(sorted(unscheduled)),
        deferred_unit_ids=deferred,
        solve_time_seconds=time.perf_counter() - started,
        solver_status="greedy_deadline_priority",
    )


def optimize_plan(
    scenario: PlanningScenario,
    config: PlanningConfig,
) -> PlanResult:
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise PlanningError(
            "OR-Tools is required for optimized maintenance planning"
        ) from exc

    started = time.perf_counter()
    due, deferred = _split_engines(scenario)
    impossible = [
        engine.unit_id
        for engine in due
        if engine.maintenance_duration > scenario.planning_horizon
        or engine.technicians_required > scenario.technicians
    ]
    if impossible:
        return PlanResult(
            method="cp_sat",
            status="infeasible",
            objective_cost=None,
            assignments=(),
            unscheduled_unit_ids=tuple(sorted(impossible)),
            deferred_unit_ids=deferred,
            solve_time_seconds=time.perf_counter() - started,
            solver_status="precheck_infeasible",
        )

    model = cp_model.CpModel()
    starts: dict[int, Any] = {}
    ends: dict[int, Any] = {}
    intervals: list[Any] = []
    early_vars: dict[int, Any] = {}
    late_vars: dict[int, Any] = {}
    for engine in due:
        latest_start = scenario.planning_horizon - engine.maintenance_duration
        start = model.new_int_var(0, latest_start, f"start_{engine.unit_id}")
        end = model.new_int_var(
            engine.maintenance_duration,
            scenario.planning_horizon,
            f"end_{engine.unit_id}",
        )
        interval = model.new_interval_var(
            start,
            engine.maintenance_duration,
            end,
            f"maintenance_{engine.unit_id}",
        )
        early = model.new_int_var(0, scenario.planning_horizon, f"early_{engine.unit_id}")
        late = model.new_int_var(0, scenario.planning_horizon, f"late_{engine.unit_id}")
        model.add_max_equality(early, [engine.deadline_cycle - end, 0])
        model.add_max_equality(late, [end - engine.deadline_cycle, 0])
        starts[engine.unit_id] = start
        ends[engine.unit_id] = end
        early_vars[engine.unit_id] = early
        late_vars[engine.unit_id] = late
        intervals.append(interval)

    if intervals:
        model.add_cumulative(intervals, [1] * len(intervals), scenario.workshop_capacity)
        model.add_cumulative(
            intervals,
            [engine.technicians_required for engine in due],
            scenario.technicians,
        )

    constant_cost = sum(
        engine.preventive_cost
        + config.costs.downtime_per_cycle * engine.maintenance_duration
        for engine in due
    )
    variable_costs = []
    for engine in due:
        variable_costs.append(config.costs.early_per_cycle * early_vars[engine.unit_id])
        variable_costs.append(
            config.costs.late_per_cycle
            * config.criticality_weights[engine.criticality]
            * late_vars[engine.unit_id]
        )
    model.minimize(constant_cost + sum(variable_costs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = config.solver_time_limit_seconds
    solver.parameters.num_search_workers = config.solver_workers
    solver.parameters.random_seed = config.solver_random_seed
    status_code = solver.solve(model)
    status_name = solver.status_name(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status: PlanStatus = "infeasible" if status_code == cp_model.INFEASIBLE else "unknown"
        return PlanResult(
            method="cp_sat",
            status=status,
            objective_cost=None,
            assignments=(),
            unscheduled_unit_ids=tuple(sorted(engine.unit_id for engine in due)),
            deferred_unit_ids=deferred,
            solve_time_seconds=time.perf_counter() - started,
            solver_status=status_name,
        )

    assignments = tuple(
        sorted(
            (
                _assignment(engine, int(solver.value(starts[engine.unit_id])), config)
                for engine in due
            ),
            key=lambda item: (item.start_cycle, item.unit_id),
        )
    )
    return PlanResult(
        method="cp_sat",
        status="optimal" if status_code == cp_model.OPTIMAL else "feasible",
        objective_cost=sum(item.cost for item in assignments),
        assignments=assignments,
        unscheduled_unit_ids=(),
        deferred_unit_ids=deferred,
        solve_time_seconds=time.perf_counter() - started,
        solver_status=status_name,
    )


def validate_plan(scenario: PlanningScenario, result: PlanResult) -> None:
    """Independently verify assignment identity, bounds and resource capacities."""
    engines = {engine.unit_id: engine for engine in scenario.engines}
    seen: set[int] = set()
    workshop_use = [0] * scenario.planning_horizon
    technician_use = [0] * scenario.planning_horizon
    for assignment in result.assignments:
        if assignment.unit_id in seen:
            raise PlanningError(f"unit {assignment.unit_id} is assigned more than once")
        if assignment.unit_id not in engines:
            raise PlanningError(f"unknown assigned unit: {assignment.unit_id}")
        engine = engines[assignment.unit_id]
        if assignment.start_cycle < 0 or assignment.end_cycle > scenario.planning_horizon:
            raise PlanningError(f"unit {assignment.unit_id} is outside the planning horizon")
        if assignment.end_cycle - assignment.start_cycle != engine.maintenance_duration:
            raise PlanningError(f"unit {assignment.unit_id} has an invalid duration")
        if assignment.technicians_required != engine.technicians_required:
            raise PlanningError(f"unit {assignment.unit_id} has an invalid technician demand")
        seen.add(assignment.unit_id)
        for slot in range(assignment.start_cycle, assignment.end_cycle):
            workshop_use[slot] += 1
            technician_use[slot] += assignment.technicians_required
    if any(value > scenario.workshop_capacity for value in workshop_use):
        raise PlanningError("plan exceeds workshop capacity")
    if any(value > scenario.technicians for value in technician_use):
        raise PlanningError("plan exceeds technician capacity")


def compare_plans(
    scenario: PlanningScenario,
    config: PlanningConfig,
) -> PlanningComparison:
    heuristic = heuristic_plan(scenario, config)
    optimized = optimize_plan(scenario, config)
    validate_plan(scenario, heuristic)
    validate_plan(scenario, optimized)
    savings = None
    relative = None
    if optimized.objective_cost is not None and heuristic.status == "feasible":
        savings = heuristic.objective_cost - optimized.objective_cost
        relative = savings / heuristic.objective_cost if heuristic.objective_cost else 0.0
    return PlanningComparison(
        scenario_id=scenario.scenario_id,
        heuristic=heuristic,
        optimized=optimized,
        absolute_savings=savings,
        relative_savings=relative,
    )


def comparison_to_dict(comparison: PlanningComparison) -> dict[str, Any]:
    return asdict(comparison)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="Path to a simulated fleet scenario JSON")
    parser.add_argument("--config", default="configs/maintenance_planning.toml")
    arguments = parser.parse_args()
    try:
        config = read_planning_config(arguments.config)
        scenario = load_scenario(arguments.scenario, config)
        result = compare_plans(scenario, config)
    except PlanningError as exc:
        parser.error(str(exc))
    print(json.dumps(comparison_to_dict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
