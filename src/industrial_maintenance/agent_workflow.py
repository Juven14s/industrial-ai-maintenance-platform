"""Supervised LangGraph workflow over the existing maintenance domain services."""

from __future__ import annotations

import json
import math
import operator
import unicodedata
from dataclasses import asdict, dataclass, is_dataclass
from typing import Annotated, Any, Protocol, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, RetryPolicy, interrupt
from pydantic import BaseModel, ValidationError

from industrial_maintenance.agent_contracts import (
    AgentAction,
    AgentError,
    HumanReviewDecision,
    MaintenanceAgentRequest,
    MaintenanceAgentResult,
    RouterDecision,
)
from industrial_maintenance.inference import ArtifactLoadError, InferenceInputError
from industrial_maintenance.planning import PlanningError
from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_generation import GenerationError
from industrial_maintenance.sensor_trends import SensorTrendInputError

AGENT_POLICY_VERSION = "maintenance-agent-v1"
ACTION_ORDER: tuple[AgentAction, ...] = (
    "sensor_trends",
    "predict_rul",
    "answer_documents",
    "plan_maintenance",
)


class AgentWorkflowError(RuntimeError):
    """Raised when the graph cannot safely execute or serialize a workflow step."""


class AgentTransientToolError(RuntimeError):
    """Explicitly retryable tool failure, such as a timeout or temporary outage."""


class RouterProtocol(Protocol):
    def route(self, request: MaintenanceAgentRequest) -> RouterDecision: ...


class PredictorProtocol(Protocol):
    def predict_frame(self, frame: pd.DataFrame) -> Any: ...


class TrendServiceProtocol(Protocol):
    def summarize(self, frame: pd.DataFrame) -> Any: ...


class AnswerServiceProtocol(Protocol):
    def ask(self, question: str) -> Any: ...


class PlanningServiceProtocol(Protocol):
    def plan(self, payload: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class AgentServices:
    predictor: PredictorProtocol
    trends: TrendServiceProtocol
    answers: AnswerServiceProtocol
    planner: PlanningServiceProtocol


class MaintenanceAgentState(TypedDict, total=False):
    request: dict[str, Any]
    status: str
    route: dict[str, Any] | None
    trend_summary: dict[str, Any] | None
    rul_prediction: dict[str, Any] | None
    document_answer: dict[str, Any] | None
    planning_payload: dict[str, Any] | None
    planning_result: dict[str, Any] | None
    draft_recommendation: str | None
    human_decision: dict[str, Any] | None
    final_recommendation: str | None
    errors: Annotated[list[dict[str, str]], operator.add]
    trace: Annotated[list[str], operator.add]
    policy_version: str


class KeywordRouter:
    """Auditable fallback router for demonstrations without an external LLM."""

    _TREND_TERMS = ("capteur", "sensor", "tendance", "evolution", "derive", "anomal")
    _RUL_TERMS = (
        "rul",
        "duree de vie",
        "vie restante",
        "cycles restants",
        "pronostic",
    )
    _DOCUMENT_TERMS = (
        "document",
        "manuel",
        "source",
        "procedure",
        "expliquer",
        "pourquoi",
    )
    _PLANNING_TERMS = (
        "planning",
        "planifier",
        "ordonnancer",
        "calendrier",
        "intervention",
    )

    @staticmethod
    def _normalized(text: str) -> str:
        value = unicodedata.normalize("NFKD", text.casefold())
        return "".join(character for character in value if not unicodedata.combining(character))

    def route(self, request: MaintenanceAgentRequest) -> RouterDecision:
        if request.requested_actions is not None:
            actions = list(request.requested_actions)
            rationale = "Actions explicitement demandées par le client."
        else:
            text = self._normalized(" ".join(filter(None, (request.message, request.question))))
            actions: list[AgentAction] = []
            if any(term in text for term in self._TREND_TERMS):
                actions.append("sensor_trends")
            if any(term in text for term in self._RUL_TERMS):
                actions.append("predict_rul")
            if request.question is not None or any(term in text for term in self._DOCUMENT_TERMS):
                actions.append("answer_documents")
            if request.planning is not None or any(term in text for term in self._PLANNING_TERMS):
                actions.append("plan_maintenance")
            if not actions:
                actions = (
                    ["sensor_trends", "predict_rul"]
                    if request.history is not None
                    else ["answer_documents"]
                )
            rationale = "Routage lexical déterministe maintenance-agent-v1."
        return RouterDecision(actions=_ordered_actions(actions), rationale=rationale)


def _ordered_actions(actions: list[AgentAction]) -> list[AgentAction]:
    selected = set(actions)
    if "plan_maintenance" in selected:
        selected.add("predict_rul")
    return [action for action in ACTION_ORDER if action in selected]


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        candidate = value.model_dump(mode="json")
    elif hasattr(value, "to_dict") and callable(value.to_dict):
        candidate = value.to_dict()
    elif is_dataclass(value):
        candidate = asdict(value)
    else:
        candidate = value
    try:
        return json.loads(json.dumps(candidate, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AgentWorkflowError("Tool output is not JSON-serializable") from exc


def _json_object(value: Any, tool_name: str) -> dict[str, Any]:
    serialized = _jsonable(value)
    if not isinstance(serialized, dict):
        raise AgentWorkflowError(f"{tool_name} output must be a JSON object")
    return serialized


def _finite_non_negative(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentWorkflowError(f"RUL {field_name} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise AgentWorkflowError(f"RUL {field_name} must be finite and non-negative")
    return number


def initial_agent_state(request: MaintenanceAgentRequest) -> MaintenanceAgentState:
    return {
        "request": request.model_dump(mode="json", exclude_none=True),
        "status": "running",
        "policy_version": AGENT_POLICY_VERSION,
    }


def _request(state: MaintenanceAgentState) -> MaintenanceAgentRequest:
    try:
        return MaintenanceAgentRequest.model_validate(state["request"])
    except (KeyError, ValidationError) as exc:
        raise AgentWorkflowError("Invalid maintenance agent request in graph state") from exc


def _history_frame(request: MaintenanceAgentRequest) -> pd.DataFrame:
    if request.history is None:
        raise AgentWorkflowError("The selected workflow action requires an engine history")
    return request.history.to_frame()


def _error(node: str, category: str, message: str) -> dict[str, str]:
    return AgentError(node=node, category=category, message=message).model_dump()


def _next_node(state: MaintenanceAgentState) -> str:
    if state.get("status") == "failed":
        return END
    route = RouterDecision.model_validate(state.get("route"))
    field_by_action = {
        "sensor_trends": "trend_summary",
        "predict_rul": "rul_prediction",
        "answer_documents": "document_answer",
        "plan_maintenance": "planning_result",
    }
    for action in route.actions:
        if state.get(field_by_action[action]) is None:
            return action
    return "compose_draft"


def _after_validation(state: MaintenanceAgentState) -> str:
    return END if state.get("status") == "failed" else "route_request"


def _call_with_transient_translation(callback):
    try:
        return callback()
    except AgentTransientToolError:
        raise
    except (ConnectionError, TimeoutError) as exc:
        raise AgentTransientToolError(str(exc)) from exc


def build_maintenance_graph(
    services: AgentServices,
    *,
    router: RouterProtocol | None = None,
    checkpointer=None,
    max_retry_attempts: int = 3,
):
    """Compile the sequential workflow with injected, independently testable services."""

    if max_retry_attempts < 1:
        raise ValueError("max_retry_attempts must be at least 1")
    selected_router = router or KeywordRouter()
    retry_policy = RetryPolicy(
        max_attempts=max_retry_attempts,
        jitter=False,
        retry_on=AgentTransientToolError,
    )

    def reset_run(state: MaintenanceAgentState) -> dict[str, Any]:
        """Clear outputs from any earlier run sharing the same incident thread."""

        return {
            "status": "running",
            "route": None,
            "trend_summary": None,
            "rul_prediction": None,
            "document_answer": None,
            "planning_payload": None,
            "planning_result": None,
            "draft_recommendation": None,
            "human_decision": None,
            "final_recommendation": None,
            "errors": Overwrite(value=[]),
            "trace": Overwrite(value=[]),
            "policy_version": AGENT_POLICY_VERSION,
        }

    def failed_tool(node: str, exc: Exception) -> dict[str, Any]:
        return {
            "status": "failed",
            "errors": [_error(node, "invalid_tool_input_or_output", str(exc))],
            "trace": [node],
        }

    def validate_request(state: MaintenanceAgentState) -> dict[str, Any]:
        try:
            request = _request(state)
        except AgentWorkflowError as exc:
            return {
                "status": "failed",
                "errors": [_error("validate_request", "invalid_request", str(exc))],
                "trace": ["validate_request"],
            }
        return {
            "request": request.model_dump(mode="json", exclude_none=True),
            "status": "running",
            "trace": ["validate_request"],
        }

    def route_request(state: MaintenanceAgentState) -> dict[str, Any]:
        request = _request(state)
        try:
            decision = selected_router.route(request)
            normalized = RouterDecision.model_validate(decision)
            normalized = RouterDecision(
                actions=_ordered_actions(normalized.actions),
                rationale=normalized.rationale,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as exc:
            return {
                "status": "failed",
                "errors": [_error("route_request", "invalid_router_output", str(exc))],
                "trace": ["route_request"],
            }
        missing: list[str] = []
        if (
            any(
                action in normalized.actions
                for action in ("sensor_trends", "predict_rul", "plan_maintenance")
            )
            and request.history is None
        ):
            missing.append("history")
        if "plan_maintenance" in normalized.actions and request.planning is None:
            missing.append("planning")
        if missing:
            return {
                "route": normalized.model_dump(mode="json"),
                "status": "failed",
                "errors": [
                    _error(
                        "route_request",
                        "missing_input",
                        f"Missing required fields: {', '.join(missing)}",
                    )
                ],
                "trace": ["route_request"],
            }
        return {
            "route": normalized.model_dump(mode="json"),
            "trace": ["route_request"],
        }

    def analyze_sensor_trends(state: MaintenanceAgentState) -> dict[str, Any]:
        request = _request(state)
        try:
            result = _call_with_transient_translation(
                lambda: services.trends.summarize(_history_frame(request))
            )
            serialized = _json_object(result, "Sensor trend tool")
            strongest = serialized.get("strongest_changes", [])
            if not isinstance(strongest, list) or any(
                not isinstance(sensor, str) for sensor in strongest
            ):
                raise AgentWorkflowError(
                    "Sensor trend output strongest_changes must be a string list"
                )
        except (AgentWorkflowError, SensorTrendInputError, ValueError) as exc:
            return failed_tool("sensor_trends", exc)
        return {
            "trend_summary": serialized,
            "trace": ["sensor_trends"],
        }

    def predict_rul(state: MaintenanceAgentState) -> dict[str, Any]:
        request = _request(state)
        try:
            result = _call_with_transient_translation(
                lambda: services.predictor.predict_frame(_history_frame(request))
            )
            serialized = _json_object(result, "RUL tool")
            estimate = _finite_non_negative(serialized.get("rul_estimate"), "rul_estimate")
            conservative = _finite_non_negative(
                serialized.get("rul_conservative"), "rul_conservative"
            )
            decision_value = _finite_non_negative(
                serialized.get("decision_value"), "decision_value"
            )
            if decision_value != conservative:
                raise AgentWorkflowError(
                    "RUL decision_value must equal the conservative maintenance estimate"
                )
            serialized["rul_estimate"] = estimate
            serialized["rul_conservative"] = conservative
            serialized["decision_value"] = decision_value
        except (AgentWorkflowError, ArtifactLoadError, InferenceInputError, ValueError) as exc:
            return failed_tool("predict_rul", exc)
        return {"rul_prediction": serialized, "trace": ["predict_rul"]}

    def answer_documents(state: MaintenanceAgentState) -> dict[str, Any]:
        request = _request(state)
        question = request.question or request.message
        try:
            result = _call_with_transient_translation(lambda: services.answers.ask(question))
            serialized = _json_object(result, "Document answer tool")
            answerable = serialized.get("answerable")
            citations = serialized.get("citations")
            answer = serialized.get("answer")
            if (
                not isinstance(answerable, bool)
                or not isinstance(answer, str)
                or not isinstance(citations, list)
            ):
                raise AgentWorkflowError(
                    "Document answer must contain answerable, answer, and a citation list"
                )
            if answerable and not citations:
                raise AgentWorkflowError(
                    "An answerable document response must contain at least one citation"
                )
        except (AgentWorkflowError, CorpusError, GenerationError, ValueError) as exc:
            return failed_tool("answer_documents", exc)
        return {"document_answer": serialized, "trace": ["answer_documents"]}

    def plan_maintenance(state: MaintenanceAgentState) -> dict[str, Any]:
        try:
            request = _request(state)
            if request.history is None or request.planning is None:
                raise AgentWorkflowError("Planning requires history and planning context")
            prediction = state.get("rul_prediction")
            if prediction is None:
                raise AgentWorkflowError("Planning cannot run before conservative RUL prediction")
            current_engine: dict[str, Any] = {
                "unit_id": request.history.unit_id,
                "conservative_rul": float(prediction["decision_value"]),
                "maintenance_duration": request.planning.maintenance_duration,
                "technicians_required": request.planning.technicians_required,
                "criticality": request.planning.criticality,
            }
            if request.planning.preventive_cost is not None:
                current_engine["preventive_cost"] = request.planning.preventive_cost
            engines = [current_engine]
            engines.extend(
                engine.model_dump(mode="json", exclude_none=True)
                for engine in request.planning.other_engines
            )
            payload = {
                "schema_version": 1,
                "scenario_id": request.planning.scenario_id,
                "description": request.planning.description,
                "planning_horizon": request.planning.planning_horizon,
                "technicians": request.planning.technicians,
                "workshop_capacity": request.planning.workshop_capacity,
                "engines": engines,
            }
            result = _call_with_transient_translation(lambda: services.planner.plan(payload))
            serialized_result = _json_object(result, "Maintenance planning tool")
        except (AgentWorkflowError, KeyError, PlanningError, TypeError, ValueError) as exc:
            return failed_tool("plan_maintenance", exc)
        return {
            "planning_payload": _jsonable(payload),
            "planning_result": serialized_result,
            "trace": ["plan_maintenance"],
        }

    def compose_draft(state: MaintenanceAgentState) -> dict[str, Any]:
        request = _request(state)
        parts = [f"Incident {request.incident_id} analysé."]
        if state.get("trend_summary") is not None:
            strongest = state["trend_summary"].get("strongest_changes", [])
            if strongest:
                parts.append("Évolutions descriptives principales : " + ", ".join(strongest) + ".")
            parts.append("Ces tendances ne constituent pas un diagnostic de panne.")
        if state.get("rul_prediction") is not None:
            prediction = state["rul_prediction"]
            parts.append(
                "RUL centrale : "
                f"{float(prediction['rul_estimate']):.2f} cycles ; "
                "RUL conservatrice de décision : "
                f"{float(prediction['decision_value']):.2f} cycles."
            )
        if state.get("document_answer") is not None:
            answer = state["document_answer"]
            parts.append(str(answer.get("answer", "Aucune réponse documentaire disponible.")))
        if state.get("planning_result") is not None:
            parts.append(
                "Un planning simulé a été calculé avec la RUL conservatrice ; "
                "il doit être revu avant toute utilisation opérationnelle."
            )
        parts.append("Les cycles C-MAPSS ne sont pas convertis en jours calendaires.")
        return {
            "draft_recommendation": " ".join(parts),
            "status": "pending_approval",
            "trace": ["compose_draft"],
        }

    def human_review(state: MaintenanceAgentState) -> dict[str, Any]:
        resume_value = interrupt(
            {
                "kind": "maintenance_recommendation_review",
                "incident_id": _request(state).incident_id,
                "draft_recommendation": state["draft_recommendation"],
                "allowed_actions": ["approve", "edit", "reject"],
                "policy_version": AGENT_POLICY_VERSION,
            }
        )
        try:
            decision = HumanReviewDecision.model_validate(resume_value)
        except ValidationError as exc:
            raise AgentWorkflowError("Invalid human review decision") from exc
        if decision.action == "approve":
            status = "approved"
            final = state["draft_recommendation"]
        elif decision.action == "edit":
            status = "edited"
            final = decision.edited_recommendation
        else:
            status = "rejected"
            final = None
        return {
            "human_decision": decision.model_dump(mode="json"),
            "final_recommendation": final,
            "status": status,
            "trace": ["human_review"],
        }

    def finalize(state: MaintenanceAgentState) -> dict[str, Any]:
        return {"trace": ["finalize"]}

    builder = StateGraph(MaintenanceAgentState)
    builder.add_node("reset_run", reset_run)
    builder.add_node("validate_request", validate_request)
    builder.add_node("route_request", route_request)
    builder.add_node("sensor_trends", analyze_sensor_trends, retry_policy=retry_policy)
    builder.add_node("predict_rul", predict_rul, retry_policy=retry_policy)
    builder.add_node("answer_documents", answer_documents, retry_policy=retry_policy)
    builder.add_node("plan_maintenance", plan_maintenance, retry_policy=retry_policy)
    builder.add_node("compose_draft", compose_draft)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "reset_run")
    builder.add_edge("reset_run", "validate_request")
    builder.add_conditional_edges("validate_request", _after_validation)
    builder.add_conditional_edges("route_request", _next_node)
    for action in ACTION_ORDER:
        builder.add_conditional_edges(action, _next_node)
    builder.add_edge("compose_draft", "human_review")
    builder.add_edge("human_review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name=AGENT_POLICY_VERSION)


def result_from_state(state: MaintenanceAgentState) -> MaintenanceAgentResult:
    request = _request(state)
    return MaintenanceAgentResult(
        incident_id=request.incident_id,
        status=state.get("status", "failed"),
        route=state.get("route"),
        trend_summary=state.get("trend_summary"),
        rul_prediction=state.get("rul_prediction"),
        document_answer=state.get("document_answer"),
        planning_result=state.get("planning_result"),
        draft_recommendation=state.get("draft_recommendation"),
        human_decision=state.get("human_decision"),
        final_recommendation=state.get("final_recommendation"),
        errors=state.get("errors", []),
        trace=state.get("trace", []),
    )
