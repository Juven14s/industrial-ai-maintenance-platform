"""Evaluate heuristic and CP-SAT maintenance plans on simulated scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from industrial_maintenance.planning import (
    MaintenanceAssignment,
    PlanningComparison,
    PlanningError,
    PlanningScenario,
    PlanResult,
    compare_plans,
    load_scenario,
    read_planning_config,
)


def _plan_metrics(plan: PlanResult) -> dict[str, Any]:
    tardy = [item for item in plan.assignments if item.tardiness_cycles > 0]
    return {
        "status": plan.status,
        "objective_cost": plan.objective_cost,
        "scheduled_count": len(plan.assignments),
        "unscheduled_count": len(plan.unscheduled_unit_ids),
        "deferred_count": len(plan.deferred_unit_ids),
        "tardy_count": len(tardy),
        "total_tardiness_cycles": sum(item.tardiness_cycles for item in tardy),
        "maximum_tardiness_cycles": max(
            (item.tardiness_cycles for item in tardy), default=0
        ),
        "solve_time_seconds": plan.solve_time_seconds,
        "solver_status": plan.solver_status,
    }


def _row(
    scenario: PlanningScenario,
    comparison: PlanningComparison,
    plan: PlanResult,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "planning_horizon": scenario.planning_horizon,
        "technicians": scenario.technicians,
        "workshop_capacity": scenario.workshop_capacity,
        "engine_count": len(scenario.engines),
        "method": plan.method,
        **_plan_metrics(plan),
        "absolute_savings": comparison.absolute_savings,
        "relative_savings": comparison.relative_savings,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gantt_axis(
    axis: Any,
    assignments: tuple[MaintenanceAssignment, ...],
    title: str,
) -> None:
    colors = {"low": "#4C78A8", "medium": "#F2A541", "high": "#D1495B"}
    ordered = sorted(assignments, key=lambda item: item.unit_id)
    positions = list(range(len(ordered)))
    for position, assignment in zip(positions, ordered):
        axis.barh(
            position,
            assignment.maintenance_duration,
            left=assignment.start_cycle,
            color=colors[assignment.criticality],
            edgecolor="black",
            linewidth=0.6,
        )
        axis.plot(
            [assignment.deadline_cycle, assignment.deadline_cycle],
            [position - 0.35, position + 0.35],
            color="black",
            linestyle="--",
            linewidth=1,
        )
    axis.set_yticks(positions, [f"Moteur {item.unit_id}" for item in ordered])
    axis.set_xlabel("Cycle de planification")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)


def _write_gantt(path: Path, comparison: PlanningComparison) -> None:
    if not comparison.heuristic.assignments or not comparison.optimized.assignments:
        return
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    _gantt_axis(
        axes[0],
        comparison.heuristic.assignments,
        f"Heuristique — coût {comparison.heuristic.objective_cost}",
    )
    _gantt_axis(
        axes[1],
        comparison.optimized.assignments,
        f"CP-SAT — coût {comparison.optimized.objective_cost}",
    )
    figure.suptitle(
        "Scénario contraint : barres = interventions, tirets = échéances de fin"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    scenarios: list[PlanningScenario],
    comparisons: list[PlanningComparison],
) -> None:
    table_rows: list[str] = []
    for scenario, comparison in zip(scenarios, comparisons):
        for plan in (comparison.heuristic, comparison.optimized):
            metrics = _plan_metrics(plan)
            cost = "—" if plan.objective_cost is None else str(plan.objective_cost)
            table_rows.append(
                f"| `{scenario.scenario_id}` | {plan.method} | {plan.status} | {cost} | "
                f"{metrics['tardy_count']} | {metrics['total_tardiness_cycles']} | "
                f"{metrics['unscheduled_count']} | {plan.solve_time_seconds:.4f} s |"
            )
    constrained = next(
        comparison
        for comparison in comparisons
        if comparison.scenario_id == "constrained_load"
    )
    relative = (constrained.relative_savings or 0.0) * 100
    report = "\n".join(
        [
            "# Évaluation de la planification de maintenance",
            "",
            f"Date : {date.today().isoformat()}  ",
            "Version : `maintenance-planning-v1`  ",
            "Solveur : OR-Tools CP-SAT, un seul worker, limite de 10 secondes",
            "",
            "## Résultat principal",
            "",
            "Sur le scénario contraint, l'heuristique obtient un coût simulé de "
            f"{constrained.heuristic.objective_cost}, contre "
            f"{constrained.optimized.objective_cost} pour CP-SAT, soit une réduction de "
            f"{constrained.absolute_savings} ({relative:.1f} %). Le scénario normal est "
            "résolu sans retard par les deux méthodes. Le scénario surchargé est déclaré "
            "irréalisable par CP-SAT car toutes les interventions obligatoires ne tiennent "
            "pas dans l'horizon.",
            "",
            "| Scénario | Méthode | Statut | Coût | Interventions tardives | Retard total | Non planifiées | Temps |",
            "|---|---|---|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "## Fonction de coût et convention",
            "",
            "L'échéance porte sur la **fin** de l'intervention et vaut la partie entière "
            "de la RUL conservatrice. Le coût additionne le coût préventif, 50 unités par "
            "cycle d'immobilisation, 10 unités par cycle d'anticipation et 250 unités par "
            "cycle de retard, multipliées par une criticité de 1, 2 ou 4. Les équipements "
            "dont l'échéance dépasse l'horizon sont différés et exclus du coût courant.",
            "",
            "## Portée",
            "",
            "Les coûts, durées, ressources et criticités sont des hypothèses simulées. La "
            "réduction mesurée n'est ni une économie industrielle observée ni une validation "
            "du planning sur une flotte réelle. L'intérêt de l'expérience est de vérifier "
            "la chaîne RUL conservatrice → échéance → contraintes → décision et de comparer "
            "les méthodes avec exactement les mêmes règles.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def evaluate(config_path: str | Path) -> dict[str, Any]:
    config = read_planning_config(config_path)
    scenario_paths = sorted(config.scenario_directory.glob("*.json"))
    if not scenario_paths:
        raise PlanningError(f"No scenario found in {config.scenario_directory}")
    scenarios = [load_scenario(path, config) for path in scenario_paths]
    comparisons = [compare_plans(scenario, config) for scenario in scenarios]
    rows = [
        _row(scenario, comparison, plan)
        for scenario, comparison in zip(scenarios, comparisons)
        for plan in (comparison.heuristic, comparison.optimized)
    ]
    payload = {
        "schema_version": 1,
        "evaluation_date": date.today().isoformat(),
        "planning_version": "maintenance-planning-v1",
        "costs": asdict(config.costs),
        "criticality_weights": config.criticality_weights,
        "scenarios": [
            {
                "scenario": asdict(scenario),
                "comparison": asdict(comparison),
            }
            for scenario, comparison in zip(scenarios, comparisons)
        ],
    }
    config.evaluation_json.parent.mkdir(parents=True, exist_ok=True)
    config.evaluation_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(config.comparison_csv, rows)
    _write_report(config.evaluation_report, scenarios, comparisons)
    constrained = next(
        comparison
        for comparison in comparisons
        if comparison.scenario_id == "constrained_load"
    )
    _write_gantt(config.gantt_figure, constrained)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/maintenance_planning.toml")
    arguments = parser.parse_args()
    try:
        payload = evaluate(arguments.config)
    except PlanningError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": "completed",
                "scenario_count": len(payload["scenarios"]),
                "report": str(read_planning_config(arguments.config).evaluation_report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
