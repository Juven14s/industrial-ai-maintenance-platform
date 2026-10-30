"""Injected MCP facade for the maintenance platform's existing domain services.

The module deliberately creates no global server and loads no model at import time.
Call :func:`create_mcp_server` with already constructed services so FastAPI,
LangGraph, and MCP can share the same business logic and model instances.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Annotated, Any, Literal, Protocol

import pandas as pd
from mcp import MCPError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import INTERNAL_ERROR
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from industrial_maintenance import __version__
from industrial_maintenance.api import (
    CycleObservation,
    DocumentAnswerResponse,
    DocumentPassage,
    DocumentSearchResponse,
    MaintenancePlanRequest,
    MaintenancePlanResponse,
    PlanningEngineRequest,
    RULRequest,
    RULResponse,
)
from industrial_maintenance.inference import (
    ArtifactLoadError,
    InferenceInputError,
    RULPrediction,
)
from industrial_maintenance.planning import (
    PlanningComparison,
    PlanningError,
    comparison_to_dict,
)
from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_generation import GenerationError, GroundedAnswer
from industrial_maintenance.rag_retrieval import SearchResult
from industrial_maintenance.sensor_trends import (
    SensorTrendInputError,
    SensorTrendSummary,
)

PositiveInt = Annotated[int, Field(gt=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class PredictorProtocol(Protocol):
    @property
    def central_version(self) -> str: ...

    @property
    def safety_version(self) -> str: ...

    def model_info(self) -> dict[str, Any]: ...

    def predict_frame(self, frame: pd.DataFrame) -> RULPrediction: ...


class DocumentSearchProtocol(Protocol):
    @property
    def version(self) -> str: ...

    def info(self) -> dict[str, Any]: ...

    def search(self, query: str, *, top_k: int) -> list[SearchResult]: ...


class AnswerServiceProtocol(Protocol):
    @property
    def version(self) -> str: ...

    def ask(self, question: str) -> GroundedAnswer: ...


class PlanningServiceProtocol(Protocol):
    @property
    def version(self) -> str: ...

    def plan(self, payload: dict[str, Any]) -> PlanningComparison: ...


class SensorTrendServiceProtocol(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def sensor_names(self) -> tuple[str, ...]: ...

    def summarize(self, frame: pd.DataFrame) -> SensorTrendSummary: ...


class SensorTrendItemResponse(BaseModel):
    """One descriptive sensor trend, with no diagnostic interpretation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sensor: str
    first_value: FiniteFloat
    last_value: FiniteFloat
    mean: FiniteFloat
    standard_deviation: FiniteFloat
    delta: FiniteFloat
    recent_delta: FiniteFloat
    global_slope_per_cycle: FiniteFloat
    recent_slope_per_cycle: FiniteFloat
    global_relative_change: FiniteFloat
    recent_relative_change: FiniteFloat
    global_direction: Literal["increasing", "decreasing", "stable"]
    recent_direction: Literal["increasing", "decreasing", "stable"]
    trend_score: FiniteFloat


class SensorTrendResponse(BaseModel):
    """Serializable response returned by ``summarize_sensor_trends``."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    unit_id: int
    history_start_cycle: int
    last_cycle: int
    observed_cycles: int
    recent_window_used: int
    strongest_changes: list[str]
    sensors: list[SensorTrendItemResponse]
    method_version: str
    limitation: str


def _history_frame(unit_id: int, observations: list[CycleObservation]) -> pd.DataFrame:
    request = RULRequest(unit_id=unit_id, observations=observations)
    return request.to_frame()


def _citation(result: SearchResult) -> str:
    if result.page_start is None:
        return f"{result.title} — {result.section}"
    if result.page_end is None or result.page_end == result.page_start:
        return f"{result.title}, p. {result.page_start} — {result.section}"
    return f"{result.title}, pp. {result.page_start}-{result.page_end} — {result.section}"


def _passage(result: SearchResult) -> DocumentPassage:
    return DocumentPassage(
        rank=result.rank,
        score=result.score,
        document_id=result.document_id,
        title=result.title,
        section=result.section,
        page_start=result.page_start,
        page_end=result.page_end,
        text=result.text,
        source_url=result.source_url,
        citation=_citation(result),
        limitations=result.limitations,
    )


def _unavailable(service_name: str) -> MCPError:
    return MCPError(
        code=INTERNAL_ERROR,
        message=f"The {service_name} service is unavailable",
    )


def _json_resource(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


_SENSITIVE_KEY_PARTS = (
    "credential",
    "directory",
    "password",
    "path",
    "secret",
    "token",
)


def _public_value(value: Any) -> Any:
    """Keep JSON metadata while dropping path- or credential-shaped fields."""

    if isinstance(value, dict):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if not any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS)
        }
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _models_resource(predictor: PredictorProtocol | None) -> dict[str, Any]:
    if predictor is None:
        return {"status": "unavailable"}
    raw = predictor.model_info()
    public_keys = (
        "window_length",
        "minimum_end_cycle",
        "required_sensors",
        "central",
        "conservative",
    )
    return {
        "status": "ok",
        **{key: _public_value(raw[key]) for key in public_keys if key in raw},
    }


def _documents_resource(search: DocumentSearchProtocol | None) -> dict[str, Any]:
    if search is None:
        return {"status": "unavailable"}
    raw = search.info()
    public_keys = (
        "retrieval_version",
        "model_name",
        "model_revision",
        "chunk_count",
        "default_top_k",
        "retrieval_method",
    )
    return {
        "status": "ok",
        **{key: _public_value(raw[key]) for key in public_keys if key in raw},
    }


def create_mcp_server(
    *,
    predictor: PredictorProtocol | None = None,
    document_search: DocumentSearchProtocol | None = None,
    answer_service: AnswerServiceProtocol | None = None,
    planning_service: PlanningServiceProtocol | None = None,
    sensor_trend_service: SensorTrendServiceProtocol | None = None,
) -> MCPServer[Any]:
    """Create a model-free MCP server around injected domain services."""

    server: MCPServer[Any] = MCPServer(
        name="industrial-ai-maintenance",
        title="Industrial AI Maintenance Tools",
        description="Typed access to RUL, sensor trends, documentation, and planning.",
        instructions=(
            "Use the conservative RUL for maintenance decisions. Sensor trends are "
            "descriptive only. Do not infer a repair without cited procedural evidence."
        ),
        version=__version__,
    )

    @server.tool(structured_output=True)
    def predict_rul(
        unit_id: PositiveInt,
        observations: Annotated[list[CycleObservation], Field(min_length=1, max_length=1000)],
    ) -> RULResponse:
        """Predict central and conservative RUL from one consecutive engine history."""

        if predictor is None:
            raise _unavailable("RUL prediction")
        try:
            result = predictor.predict_frame(_history_frame(unit_id, observations))
        except (InferenceInputError, ValidationError) as exc:
            raise ToolError(f"Invalid engine history: {exc}") from exc
        except ArtifactLoadError as exc:
            raise _unavailable("RUL prediction") from exc
        return RULResponse(**asdict(result))

    @server.tool(structured_output=True)
    def summarize_sensor_trends(
        unit_id: PositiveInt,
        observations: Annotated[list[CycleObservation], Field(min_length=1, max_length=1000)],
    ) -> SensorTrendResponse:
        """Summarize sensor changes without diagnosing a fault, cause, or repair."""

        if sensor_trend_service is None:
            raise _unavailable("sensor trend")
        try:
            summary = sensor_trend_service.summarize(_history_frame(unit_id, observations))
        except (SensorTrendInputError, ValidationError) as exc:
            raise ToolError(f"Invalid engine history: {exc}") from exc
        return SensorTrendResponse(**summary.to_dict())

    @server.tool(structured_output=True)
    def search_documents(
        query: Annotated[str, Field(min_length=3, max_length=500)],
        top_k: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> DocumentSearchResponse:
        """Retrieve ranked technical passages with provenance and usage limitations."""

        if document_search is None:
            raise _unavailable("document search")
        try:
            results = document_search.search(query, top_k=top_k)
        except (CorpusError, OSError, RuntimeError, ValueError) as exc:
            raise _unavailable("document search") from exc
        return DocumentSearchResponse(
            query=query,
            retrieval_version=document_search.version,
            results=[_passage(result) for result in results],
        )

    @server.tool(structured_output=True)
    def answer_documents(
        question: Annotated[str, Field(min_length=3, max_length=500)],
    ) -> DocumentAnswerResponse:
        """Answer from retrieved technical evidence or return an explicit abstention."""

        if answer_service is None:
            raise _unavailable("grounded answer")
        try:
            result = answer_service.ask(question)
        except (CorpusError, GenerationError, OSError, RuntimeError, ValueError) as exc:
            raise _unavailable("grounded answer") from exc
        return DocumentAnswerResponse(
            question=result.question,
            status="answered" if result.answerable else "abstained",
            answerable=result.answerable,
            answer=result.answer,
            retrieval_version=result.retrieval_version,
            generation_version=result.generation_version,
            citations=[_passage(citation) for citation in result.citations],
        )

    @server.tool(structured_output=True)
    def plan_maintenance(
        planning_horizon: Annotated[int, Field(gt=0, le=1000)],
        technicians: Annotated[int, Field(gt=0, le=100)],
        workshop_capacity: Annotated[int, Field(gt=0, le=100)],
        engines: Annotated[list[PlanningEngineRequest], Field(min_length=1, max_length=500)],
        scenario_id: Annotated[str, Field(min_length=1, max_length=100)] = "mcp_scenario",
        description: Annotated[str, Field(max_length=500)] = "",
        schema_version: Literal[1] = 1,
    ) -> MaintenancePlanResponse:
        """Plan maintenance using conservative RUL values supplied for each engine."""

        if planning_service is None:
            raise _unavailable("maintenance planning")
        try:
            request = MaintenancePlanRequest(
                schema_version=schema_version,
                scenario_id=scenario_id,
                description=description,
                planning_horizon=planning_horizon,
                technicians=technicians,
                workshop_capacity=workshop_capacity,
                engines=engines,
            )
            result = planning_service.plan(request.to_planning_payload())
        except (PlanningError, ValidationError) as exc:
            raise ToolError(f"Invalid planning scenario: {exc}") from exc
        return MaintenancePlanResponse(**comparison_to_dict(result))

    @server.resource(
        "maintenance://models",
        description="Public model roles, versions, input contract, and evaluation metadata.",
        mime_type="application/json",
    )
    def model_metadata() -> str:
        return _json_resource(_models_resource(predictor))

    @server.resource(
        "maintenance://documents/status",
        description="Public retrieval-index identity and corpus size.",
        mime_type="application/json",
    )
    def document_metadata() -> str:
        return _json_resource(_documents_resource(document_search))

    @server.resource(
        "maintenance://planning/policy",
        description="Decision rules applied to the simulated maintenance planner.",
        mime_type="application/json",
    )
    def planning_policy() -> str:
        return _json_resource(
            {
                "status": "ok" if planning_service is not None else "unavailable",
                "version": getattr(planning_service, "version", None),
                "decision_input": "conservative_rul",
                "deadline_rule": "floor(conservative_rul), applied to intervention end",
                "unit": "C-MAPSS cycles",
                "costs_are_simulated": True,
            }
        )

    @server.resource(
        "maintenance://sensors",
        description="Sensors supported by the descriptive trend service and its limitation.",
        mime_type="application/json",
    )
    def sensor_metadata() -> str:
        names = list(getattr(sensor_trend_service, "sensor_names", ()))
        return _json_resource(
            {
                "status": "ok" if sensor_trend_service is not None else "unavailable",
                "version": getattr(sensor_trend_service, "version", None),
                "sensor_names": names,
                "limitation": ("Descriptive trends only; no fault, cause, or repair is inferred."),
            }
        )

    @server.resource(
        "maintenance://service/status",
        description="Availability and public versions of the injected maintenance services.",
        mime_type="application/json",
    )
    def service_status() -> str:
        return _json_resource(
            {
                "api_version": __version__,
                "prediction": {
                    "available": predictor is not None,
                    "central_version": getattr(predictor, "central_version", None),
                    "safety_version": getattr(predictor, "safety_version", None),
                },
                "sensor_trends": {
                    "available": sensor_trend_service is not None,
                    "version": getattr(sensor_trend_service, "version", None),
                },
                "document_search": {
                    "available": document_search is not None,
                    "version": getattr(document_search, "version", None),
                },
                "grounded_answer": {
                    "available": answer_service is not None,
                    "version": getattr(answer_service, "version", None),
                },
                "planning": {
                    "available": planning_service is not None,
                    "version": getattr(planning_service, "version", None),
                },
            }
        )

    return server


def main() -> int:
    """Load the pinned services once and expose them over an MCP transport."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-config", default="configs/api.toml")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--path", default="/mcp")
    arguments = parser.parse_args()

    from industrial_maintenance.services import load_maintenance_services

    bundle = load_maintenance_services(arguments.api_config)
    server = create_mcp_server(
        predictor=bundle.predictor,
        document_search=bundle.document_search,
        answer_service=bundle.answer_service,
        planning_service=bundle.planning_service,
        sensor_trend_service=bundle.sensor_trend_service,
    )
    if arguments.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http",
            host=arguments.host,
            port=arguments.port,
            streamable_http_path=arguments.path,
        )
    return 0


__all__ = [
    "SensorTrendItemResponse",
    "SensorTrendResponse",
    "create_mcp_server",
]


if __name__ == "__main__":
    raise SystemExit(main())
