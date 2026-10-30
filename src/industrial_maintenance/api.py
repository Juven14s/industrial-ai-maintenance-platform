"""FastAPI service for prediction, grounded support, planning, and supervision."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from industrial_maintenance import __version__
from industrial_maintenance.agent_contracts import (
    HumanReviewDecision,
    MaintenanceAgentRequest,
    MaintenanceAgentRun,
)
from industrial_maintenance.agent_runtime import (
    AgentRunConflictError,
    AgentRunNotFoundError,
    MaintenanceAgentRuntime,
)
from industrial_maintenance.agent_workflow import AgentServices
from industrial_maintenance.document_search import (
    DocumentSearchService,
    read_document_search_config,
)
from industrial_maintenance.inference import (
    ArtifactLoadError,
    DualRULPredictor,
    InferenceInputError,
    RULPrediction,
    read_inference_config,
)
from industrial_maintenance.planning import (
    MaintenancePlanningService,
    PlanningComparison,
    PlanningError,
    comparison_to_dict,
)
from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_generation import (
    GenerationError,
    GroundedAnswer,
    GroundedAnswerService,
    LocalQwenGenerator,
    read_generation_settings,
)
from industrial_maintenance.rag_retrieval import SearchResult
from industrial_maintenance.sensor_trends import SensorTrendService

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class CycleObservation(BaseModel):
    """One ordered operating cycle; constant sensors may be omitted."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    cycle: PositiveInt
    sensor_1: FiniteFloat | None = None
    sensor_2: FiniteFloat
    sensor_3: FiniteFloat
    sensor_4: FiniteFloat
    sensor_5: FiniteFloat | None = None
    sensor_6: FiniteFloat
    sensor_7: FiniteFloat
    sensor_8: FiniteFloat
    sensor_9: FiniteFloat
    sensor_10: FiniteFloat | None = None
    sensor_11: FiniteFloat
    sensor_12: FiniteFloat
    sensor_13: FiniteFloat
    sensor_14: FiniteFloat
    sensor_15: FiniteFloat
    sensor_16: FiniteFloat | None = None
    sensor_17: FiniteFloat
    sensor_18: FiniteFloat | None = None
    sensor_19: FiniteFloat | None = None
    sensor_20: FiniteFloat
    sensor_21: FiniteFloat


class RULRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    unit_id: PositiveInt
    observations: Annotated[list[CycleObservation], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def validate_cycle_order(self) -> RULRequest:
        cycles = [row.cycle for row in self.observations]
        if any(current != previous + 1 for previous, current in zip(cycles, cycles[1:])):
            raise ValueError("cycles must be ordered and consecutive, without duplicates or gaps")
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"unit_id": self.unit_id, **observation.model_dump(exclude_none=True)}
            for observation in self.observations
        ]
        return pd.DataFrame(rows)


class RULResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: int
    history_start_cycle: int
    last_cycle: int
    observed_cycles: int
    window_cycles_used: int
    rul_estimate: float
    rul_conservative: float
    safety_base_prediction: float
    safety_margin: float
    decision_value: float
    unit: Literal["cycles"] = "cycles"
    central_model_version: str
    safety_model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    api_version: str
    central_model_version: str
    safety_model_version: str


class DocumentSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: Annotated[str, Field(min_length=3, max_length=500)]
    top_k: Annotated[int, Field(ge=1, le=10)] = 5


class DocumentAskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: Annotated[str, Field(min_length=3, max_length=500)]


class DocumentPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    score: float
    document_id: str
    title: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    source_url: str
    citation: str
    limitations: str


class DocumentSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    retrieval_version: str
    results: list[DocumentPassage]


class DocumentAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    status: Literal["answered", "abstained"]
    answerable: bool
    answer: str
    retrieval_version: str
    generation_version: str
    citations: list[DocumentPassage]


class DocumentStatusResponse(BaseModel):
    status: Literal["ok"]
    retrieval_version: str
    model_name: str
    model_revision: str
    chunk_count: int
    default_top_k: int
    retrieval_method: str


class PlanningEngineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    unit_id: PositiveInt
    conservative_rul: NonNegativeFiniteFloat
    maintenance_duration: PositiveInt
    technicians_required: PositiveInt = 1
    criticality: Literal["low", "medium", "high"]
    preventive_cost: NonNegativeInt | None = None


class MaintenancePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    scenario_id: Annotated[str, Field(min_length=1, max_length=100)] = "api_scenario"
    description: Annotated[str, Field(max_length=500)] = ""
    planning_horizon: Annotated[int, Field(gt=0, le=1000)]
    technicians: Annotated[int, Field(gt=0, le=100)]
    workshop_capacity: Annotated[int, Field(gt=0, le=100)]
    engines: Annotated[list[PlanningEngineRequest], Field(min_length=1, max_length=500)]

    def to_planning_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class MaintenanceAssignmentResponse(BaseModel):
    unit_id: int
    start_cycle: int
    end_cycle: int
    deadline_cycle: int
    maintenance_duration: int
    technicians_required: int
    criticality: Literal["low", "medium", "high"]
    preventive_cost: int
    earliness_cycles: int
    tardiness_cycles: int
    cost: int


class PlanResultResponse(BaseModel):
    method: Literal["heuristic", "cp_sat"]
    status: Literal["optimal", "feasible", "partial", "infeasible", "unknown"]
    objective_cost: int | None
    assignments: list[MaintenanceAssignmentResponse]
    unscheduled_unit_ids: list[int]
    deferred_unit_ids: list[int]
    solve_time_seconds: float
    solver_status: str
    planning_version: str


class MaintenancePlanResponse(BaseModel):
    scenario_id: str
    heuristic: PlanResultResponse
    optimized: PlanResultResponse
    absolute_savings: int | None
    relative_savings: float | None


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


class AgentRuntimeProtocol(Protocol):
    def start(self, payload: MaintenanceAgentRequest) -> MaintenanceAgentRun: ...

    def status(self, incident_id: str) -> MaintenanceAgentRun: ...

    def review(
        self,
        incident_id: str,
        decision: HumanReviewDecision,
    ) -> MaintenanceAgentRun: ...


def _unavailable(request: Request) -> PredictorProtocol:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        message = getattr(request.app.state, "load_error", "Models are not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "models_unavailable", "message": message},
        )
    return predictor


def _documents_unavailable(request: Request) -> DocumentSearchProtocol:
    document_search = getattr(request.app.state, "document_search", None)
    if document_search is None:
        message = getattr(
            request.app.state,
            "document_load_error",
            "Document search is not loaded",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "documents_unavailable", "message": message},
        )
    return document_search


def _answers_unavailable(request: Request) -> AnswerServiceProtocol:
    answer_service = getattr(request.app.state, "answer_service", None)
    if answer_service is None:
        message = getattr(
            request.app.state,
            "answer_load_error",
            "Grounded answer generation is not loaded",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "generation_unavailable", "message": message},
        )
    return answer_service


def _planning_unavailable(request: Request) -> PlanningServiceProtocol:
    planning_service = getattr(request.app.state, "planning_service", None)
    if planning_service is None:
        message = getattr(
            request.app.state,
            "planning_load_error",
            "Maintenance planning is not loaded",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "planning_unavailable", "message": message},
        )
    return planning_service


def _agent_unavailable(request: Request) -> AgentRuntimeProtocol:
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        message = getattr(
            request.app.state,
            "agent_load_error",
            "Supervised maintenance agent is not loaded",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "agent_unavailable", "message": message},
        )
    return runtime


def _citation(result: SearchResult) -> str:
    if result.page_start is None:
        return f"{result.title} — {result.section}"
    if result.page_end is None or result.page_end == result.page_start:
        return f"{result.title}, p. {result.page_start} — {result.section}"
    return f"{result.title}, pp. {result.page_start}-{result.page_end} — {result.section}"


def _passage_response(result: SearchResult) -> DocumentPassage:
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


def create_app(
    *,
    config_path: str | Path = "configs/api.toml",
    predictor: PredictorProtocol | None = None,
    document_search: DocumentSearchProtocol | None = None,
    answer_service: AnswerServiceProtocol | None = None,
    planning_service: PlanningServiceProtocol | None = None,
    agent_runtime: AgentRuntimeProtocol | None = None,
    agent_config_path: str | Path = "configs/agent.toml",
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor
        application.state.load_error = None
        application.state.document_search = document_search
        application.state.document_load_error = None
        application.state.answer_service = answer_service
        application.state.answer_load_error = None
        application.state.planning_service = planning_service
        application.state.planning_load_error = None
        application.state.agent_runtime = agent_runtime
        application.state.agent_load_error = None
        created_agent_runtime: MaintenanceAgentRuntime | None = None
        if predictor is None:
            try:
                config = read_inference_config(config_path)
                application.state.predictor = DualRULPredictor.load(config)
            except (ArtifactLoadError, OSError, ValueError) as exc:
                application.state.load_error = str(exc)
        if document_search is None and predictor is None:
            try:
                application.state.document_search = DocumentSearchService.from_api_config(
                    config_path
                )
            except (CorpusError, OSError, RuntimeError, ValueError) as exc:
                application.state.document_load_error = str(exc)
        elif document_search is None:
            application.state.document_load_error = (
                "Document search was not configured for this application instance"
            )
        if answer_service is None and predictor is None:
            try:
                rag_config = read_document_search_config(config_path)
                generation_settings = read_generation_settings(rag_config)
                loaded_search = application.state.document_search
                if loaded_search is None:
                    raise GenerationError("Document search is unavailable")
                generator = LocalQwenGenerator(
                    generation_settings,
                    local_files_only=True,
                )
                application.state.answer_service = GroundedAnswerService(
                    search=loaded_search,
                    generator=generator,
                    top_k_passages=generation_settings.top_k_passages,
                )
            except (CorpusError, GenerationError, OSError, RuntimeError, ValueError) as exc:
                application.state.answer_load_error = str(exc)
        elif answer_service is None:
            application.state.answer_load_error = (
                "Grounded answer generation was not configured for this application instance"
            )
        if planning_service is None and predictor is None:
            try:
                application.state.planning_service = MaintenancePlanningService.from_api_config(
                    config_path
                )
            except (PlanningError, OSError, ValueError) as exc:
                application.state.planning_load_error = str(exc)
        elif planning_service is None:
            application.state.planning_load_error = (
                "Maintenance planning was not configured for this application instance"
            )
        if agent_runtime is None:
            loaded_predictor = application.state.predictor
            loaded_answers = application.state.answer_service
            loaded_planner = application.state.planning_service
            if all(
                value is not None for value in (loaded_predictor, loaded_answers, loaded_planner)
            ):
                try:
                    required_sensors = tuple(loaded_predictor.model_info()["required_sensors"])
                    created_agent_runtime = MaintenanceAgentRuntime(
                        AgentServices(
                            predictor=loaded_predictor,
                            trends=SensorTrendService(required_sensors),
                            answers=loaded_answers,
                            planner=loaded_planner,
                        ),
                        config_path=agent_config_path,
                    )
                    application.state.agent_runtime = created_agent_runtime
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    application.state.agent_load_error = str(exc)
            else:
                application.state.agent_load_error = (
                    "Prediction, grounded answers, and planning must all be available"
                )
        try:
            yield
        finally:
            if created_agent_runtime is not None:
                created_agent_runtime.close()

    application = FastAPI(
        title="Industrial AI Maintenance API",
        version=__version__,
        description=(
            "Prédit une RUL centrale et une RUL conservatrice. La valeur conservatrice "
            "est celle destinée aux décisions de maintenance. Le service fournit aussi "
            "une recherche documentaire sourcée, un planning de flotte simulé et un "
            "workflow agentique dont la recommandation exige une validation humaine."
        ),
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        loaded = _unavailable(request)
        return HealthResponse(
            status="ok",
            api_version=__version__,
            central_model_version=loaded.central_version,
            safety_model_version=loaded.safety_version,
        )

    @application.get("/models", tags=["models"])
    def models(request: Request) -> dict[str, Any]:
        return _unavailable(request).model_info()

    @application.post("/predict/rul", response_model=RULResponse, tags=["prediction"])
    def predict(payload: RULRequest, request: Request) -> RULResponse:
        loaded = _unavailable(request)
        try:
            result = loaded.predict_frame(payload.to_frame())
        except InferenceInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_history", "message": str(exc)},
            ) from exc
        except ArtifactLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "inference_unavailable", "message": str(exc)},
            ) from exc
        return RULResponse(**asdict(result))

    @application.get(
        "/documents/status",
        response_model=DocumentStatusResponse,
        tags=["documents"],
    )
    def document_status(request: Request) -> DocumentStatusResponse:
        info = _documents_unavailable(request).info()
        return DocumentStatusResponse(status="ok", **info)

    @application.post(
        "/documents/search",
        response_model=DocumentSearchResponse,
        tags=["documents"],
    )
    def document_search_route(
        payload: DocumentSearchRequest,
        request: Request,
    ) -> DocumentSearchResponse:
        loaded = _documents_unavailable(request)
        try:
            results = loaded.search(payload.query, top_k=payload.top_k)
        except (CorpusError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "search_unavailable", "message": str(exc)},
            ) from exc
        passages = [_passage_response(result) for result in results]
        return DocumentSearchResponse(
            query=payload.query,
            retrieval_version=loaded.version,
            results=passages,
        )

    @application.post(
        "/documents/ask",
        response_model=DocumentAnswerResponse,
        tags=["documents"],
    )
    def document_answer_route(
        payload: DocumentAskRequest,
        request: Request,
    ) -> DocumentAnswerResponse:
        loaded = _answers_unavailable(request)
        try:
            result = loaded.ask(payload.question)
        except (CorpusError, GenerationError, OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "generation_failed", "message": str(exc)},
            ) from exc
        return DocumentAnswerResponse(
            question=result.question,
            status="answered" if result.answerable else "abstained",
            answerable=result.answerable,
            answer=result.answer,
            retrieval_version=result.retrieval_version,
            generation_version=result.generation_version,
            citations=[_passage_response(citation) for citation in result.citations],
        )

    @application.post(
        "/maintenance/plan",
        response_model=MaintenancePlanResponse,
        tags=["maintenance"],
    )
    def maintenance_plan_route(
        payload: MaintenancePlanRequest,
        request: Request,
    ) -> MaintenancePlanResponse:
        loaded = _planning_unavailable(request)
        try:
            comparison = loaded.plan(payload.to_planning_payload())
        except PlanningError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_planning_scenario", "message": str(exc)},
            ) from exc
        return MaintenancePlanResponse(**comparison_to_dict(comparison))

    @application.post(
        "/agent/runs",
        response_model=MaintenanceAgentRun,
        tags=["agent"],
    )
    def start_agent_run(
        payload: MaintenanceAgentRequest,
        request: Request,
    ) -> MaintenanceAgentRun:
        runtime = _agent_unavailable(request)
        try:
            return runtime.start(payload)
        except AgentRunConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "agent_run_conflict", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={"code": "invalid_agent_request", "message": str(exc)},
            ) from exc

    @application.get(
        "/agent/runs/{incident_id}",
        response_model=MaintenanceAgentRun,
        tags=["agent"],
    )
    def get_agent_run(incident_id: str, request: Request) -> MaintenanceAgentRun:
        runtime = _agent_unavailable(request)
        try:
            return runtime.status(incident_id)
        except AgentRunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "agent_run_not_found", "message": str(exc)},
            ) from exc

    @application.post(
        "/agent/runs/{incident_id}/review",
        response_model=MaintenanceAgentRun,
        tags=["agent"],
    )
    def review_agent_run(
        incident_id: str,
        payload: HumanReviewDecision,
        request: Request,
    ) -> MaintenanceAgentRun:
        runtime = _agent_unavailable(request)
        try:
            return runtime.review(incident_id, payload)
        except AgentRunNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "agent_run_not_found", "message": str(exc)},
            ) from exc
        except AgentRunConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"code": "agent_run_conflict", "message": str(exc)},
            ) from exc

    return application


app = create_app()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/api.toml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--agent-config", default="configs/agent.toml")
    arguments = parser.parse_args()
    config = read_inference_config(arguments.config)
    application = create_app(
        config_path=arguments.config,
        agent_config_path=arguments.agent_config,
    )
    uvicorn.run(
        application,
        host=arguments.host or config.host,
        port=arguments.port or config.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
