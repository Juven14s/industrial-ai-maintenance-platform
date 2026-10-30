"""Run, inspect, and resume the supervised maintenance LangGraph workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from industrial_maintenance.agent_contracts import (
    AgentAction,
    HumanReviewDecision,
    MaintenanceAgentRequest,
    MaintenanceAgentRun,
)
from industrial_maintenance.agent_runtime import MaintenanceAgentRuntime
from industrial_maintenance.services import load_maintenance_services


def _read_request(path: str | Path) -> MaintenanceAgentRequest:
    source = Path(path)
    return MaintenanceAgentRequest.model_validate_json(source.read_text(encoding="utf-8"))


def request_from_csv(
    path: str | Path,
    *,
    incident_id: str,
    message: str,
    question: str | None = None,
    requested_actions: list[AgentAction] | None = None,
    unit_id: int | None = None,
    window_size: int = 30,
) -> MaintenanceAgentRequest:
    """Build one validated agent request from a flat C-MAPSS history CSV."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    source = Path(path)
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read history CSV {source}: {exc}") from exc
    required_columns = {"unit_id", "cycle"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(f"History CSV is missing columns: {', '.join(missing)}")
    unit_values = sorted(frame["unit_id"].dropna().unique().tolist())
    selected_unit = unit_id
    if selected_unit is None:
        if len(unit_values) != 1:
            raise ValueError("History CSV must contain one unit, or --unit-id must be provided")
        selected_unit = int(unit_values[0])
    selected = frame.loc[frame["unit_id"] == selected_unit].sort_values("cycle")
    if selected.empty:
        raise ValueError(f"Unit {selected_unit} is absent from history CSV")
    sensor_columns = [name for name in selected.columns if name.startswith("sensor_")]
    if not sensor_columns:
        raise ValueError("History CSV must contain at least one sensor_* column")
    selected = selected.tail(window_size)
    observations = [
        {
            "cycle": int(row["cycle"]),
            "sensors": {name: float(row[name]) for name in sensor_columns},
        }
        for row in selected.to_dict(orient="records")
    ]
    return MaintenanceAgentRequest.model_validate(
        {
            "incident_id": incident_id,
            "message": message,
            "history": {"unit_id": int(selected_unit), "observations": observations},
            "question": question,
            "requested_actions": requested_actions,
        }
    )


def _print_run(run: MaintenanceAgentRun) -> None:
    print(json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/api.toml")
    parser.add_argument("--agent-config", default="configs/agent.toml")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser(
        "run",
        help="Start an incident workflow (pending reviews cannot be overwritten)",
    )
    run_parser.add_argument("request_json")

    csv_parser = commands.add_parser(
        "run-csv",
        help="Start an incident directly from one flat C-MAPSS history CSV",
    )
    csv_parser.add_argument("history_csv")
    csv_parser.add_argument("--incident-id", required=True)
    csv_parser.add_argument(
        "--message",
        default="Analyse les tendances des capteurs et estime la RUL de ce moteur.",
    )
    csv_parser.add_argument("--question", default=None)
    csv_parser.add_argument("--unit-id", type=int, default=None)
    csv_parser.add_argument("--window-size", type=int, default=30)
    csv_parser.add_argument(
        "--actions",
        nargs="+",
        choices=(
            "sensor_trends",
            "predict_rul",
            "answer_documents",
        ),
        default=("sensor_trends", "predict_rul"),
    )

    status_parser = commands.add_parser("status", help="Read the latest checkpoint")
    status_parser.add_argument("incident_id")

    resume_parser = commands.add_parser("resume", help="Approve, edit, or reject a draft")
    resume_parser.add_argument("incident_id")
    resume_parser.add_argument("--action", choices=("approve", "edit", "reject"), required=True)
    resume_parser.add_argument("--comment", default="")
    resume_parser.add_argument("--edited-recommendation", default=None)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    bundle = load_maintenance_services(arguments.api_config)
    with MaintenanceAgentRuntime(
        bundle.as_agent_services(),
        config_path=arguments.agent_config,
    ) as runtime:
        if arguments.command == "run":
            request = _read_request(arguments.request_json)
            run = runtime.start(request)
        elif arguments.command == "run-csv":
            request = request_from_csv(
                arguments.history_csv,
                incident_id=arguments.incident_id,
                message=arguments.message,
                question=arguments.question,
                requested_actions=list(arguments.actions),
                unit_id=arguments.unit_id,
                window_size=arguments.window_size,
            )
            run = runtime.start(request)
        elif arguments.command == "status":
            run = runtime.status(arguments.incident_id)
        else:
            decision = HumanReviewDecision(
                action=arguments.action,
                comment=arguments.comment,
                edited_recommendation=arguments.edited_recommendation,
            )
            run = runtime.review(arguments.incident_id, decision)
        _print_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
