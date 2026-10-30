"""Thread-safe lifecycle and API facade for the persistent maintenance graph."""

from __future__ import annotations

import threading
from pathlib import Path
from types import TracebackType
from typing import Any

from langgraph.types import Command

from industrial_maintenance.agent_checkpoint import open_agent_graph
from industrial_maintenance.agent_contracts import (
    HumanReviewDecision,
    MaintenanceAgentRequest,
    MaintenanceAgentRun,
)
from industrial_maintenance.agent_workflow import (
    AgentServices,
    RouterProtocol,
    initial_agent_state,
    result_from_state,
)


class AgentRunNotFoundError(LookupError):
    """Raised when no checkpoint exists for an incident identifier."""


class AgentRunConflictError(RuntimeError):
    """Raised when an operation is incompatible with the current graph state."""


def _thread_config(incident_id: str) -> dict[str, dict[str, str]]:
    value = incident_id.strip()
    if not value:
        raise ValueError("incident_id must be non-empty")
    return {"configurable": {"thread_id": value}}


def _interrupt_payloads(values: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in values or ():
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            payloads.append(dict(value))
        else:
            payloads.append({"value": value})
    return payloads


class MaintenanceAgentRuntime:
    """Own one SQLite graph connection and serialize access to it."""

    def __init__(
        self,
        services: AgentServices,
        *,
        router: RouterProtocol | None = None,
        config_path: str | Path = "configs/agent.toml",
    ) -> None:
        self._context = open_agent_graph(
            services,
            router=router,
            config_path=config_path,
        )
        self.graph = self._context.__enter__()
        self._lock = threading.RLock()
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("maintenance agent runtime is closed")

    def start(self, request: MaintenanceAgentRequest) -> MaintenanceAgentRun:
        with self._lock:
            self._ensure_open()
            snapshot = self.graph.get_state(_thread_config(request.incident_id))
            if snapshot.values and snapshot.values.get("status") == "pending_approval":
                raise AgentRunConflictError(
                    f"Incident {request.incident_id} is already waiting for human approval"
                )
            state = self.graph.invoke(
                initial_agent_state(request),
                _thread_config(request.incident_id),
            )
            return MaintenanceAgentRun(
                result=result_from_state(state),
                interrupts=_interrupt_payloads(state.get("__interrupt__")),
            )

    def status(self, incident_id: str) -> MaintenanceAgentRun:
        with self._lock:
            self._ensure_open()
            snapshot = self.graph.get_state(_thread_config(incident_id))
            if not snapshot.values:
                raise AgentRunNotFoundError(
                    f"No maintenance agent checkpoint for incident {incident_id}"
                )
            return MaintenanceAgentRun(
                result=result_from_state(dict(snapshot.values)),
                interrupts=_interrupt_payloads(snapshot.interrupts),
            )

    def review(
        self,
        incident_id: str,
        decision: HumanReviewDecision,
    ) -> MaintenanceAgentRun:
        with self._lock:
            self._ensure_open()
            current = self.status(incident_id)
            if current.result.status != "pending_approval" or not current.interrupts:
                raise AgentRunConflictError(
                    f"Incident {incident_id} is not waiting for human approval"
                )
            state = self.graph.invoke(
                Command(resume=decision.model_dump(mode="json")),
                _thread_config(incident_id),
            )
            return MaintenanceAgentRun(
                result=result_from_state(state),
                interrupts=_interrupt_payloads(state.get("__interrupt__")),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._context.__exit__(None, None, None)
            self._closed = True

    def __enter__(self) -> MaintenanceAgentRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "AgentRunConflictError",
    "AgentRunNotFoundError",
    "MaintenanceAgentRuntime",
]
