"""Configuration and durable SQLite checkpoint lifecycle for the agent graph."""

from __future__ import annotations

import sqlite3
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from industrial_maintenance.agent_workflow import (
    AGENT_POLICY_VERSION,
    AgentServices,
    RouterProtocol,
    build_maintenance_graph,
)


@dataclass(frozen=True)
class AgentConfig:
    project_root: Path
    config_path: Path
    policy_version: str
    max_retry_attempts: int
    human_approval_required: bool
    checkpoint_path: Path


def read_agent_config(path: str | Path = "configs/agent.toml") -> AgentConfig:
    source = Path(path).resolve()
    project_root = source.parent.parent.resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
        policy_version = raw["workflow"]["policy_version"]
        max_retry_attempts = raw["workflow"]["max_retry_attempts"]
        approval_required = raw["workflow"]["human_approval_required"]
        checkpoint_value = raw["checkpoint"]["path"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid agent configuration {source}: {exc}") from exc
    if policy_version != AGENT_POLICY_VERSION:
        raise ValueError(f"Unsupported agent policy_version: {policy_version}")
    if (
        isinstance(max_retry_attempts, bool)
        or not isinstance(max_retry_attempts, int)
        or max_retry_attempts < 1
    ):
        raise ValueError("workflow.max_retry_attempts must be a positive integer")
    if approval_required is not True:
        raise ValueError("workflow.human_approval_required must be true")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError("checkpoint.path must be a non-empty path")
    checkpoint_path = (project_root / checkpoint_value).resolve()
    try:
        checkpoint_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("checkpoint.path must remain inside the project root") from exc
    return AgentConfig(
        project_root=project_root,
        config_path=source,
        policy_version=policy_version,
        max_retry_attempts=max_retry_attempts,
        human_approval_required=approval_required,
        checkpoint_path=checkpoint_path,
    )


@contextmanager
def open_agent_graph(
    services: AgentServices,
    *,
    router: RouterProtocol | None = None,
    config_path: str | Path = "configs/agent.toml",
) -> Iterator[object]:
    """Open a graph whose primitive-only state is persisted in a local SQLite file."""

    config = read_agent_config(config_path)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(config.checkpoint_path), check_same_thread=False)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=[])
    checkpointer = SqliteSaver(connection, serde=serializer)
    try:
        yield build_maintenance_graph(
            services,
            router=router,
            checkpointer=checkpointer,
            max_retry_attempts=config.max_retry_attempts,
        )
    finally:
        connection.close()
