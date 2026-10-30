"""Strict, JSON-serializable contracts for the supervised maintenance workflow."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]

AgentAction = Literal[
    "sensor_trends",
    "predict_rul",
    "answer_documents",
    "plan_maintenance",
]
AgentStatus = Literal[
    "running",
    "failed",
    "pending_approval",
    "approved",
    "edited",
    "rejected",
]
ReviewAction = Literal["approve", "edit", "reject"]


class AgentObservation(BaseModel):
    """One operating cycle with a dynamic, explicit sensor mapping."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    cycle: PositiveInt
    sensors: dict[str, FiniteFloat]

    @field_validator("sensors")
    @classmethod
    def validate_sensor_mapping(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("sensors must contain at least one measurement")
        if any(not name.strip() for name in value):
            raise ValueError("sensor names must be non-empty")
        return value


class AgentEngineHistory(BaseModel):
    """A single ordered engine trajectory suitable for existing domain services."""

    model_config = ConfigDict(extra="forbid", strict=True)

    unit_id: PositiveInt
    observations: Annotated[list[AgentObservation], Field(min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def validate_observations(self) -> AgentEngineHistory:
        cycles = [row.cycle for row in self.observations]
        if any(current != previous + 1 for previous, current in zip(cycles, cycles[1:])):
            raise ValueError("cycles must be ordered and consecutive, without duplicates or gaps")
        expected = set(self.observations[0].sensors)
        if any(set(row.sensors) != expected for row in self.observations[1:]):
            raise ValueError("every observation must contain the same sensor names")
        return self

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "unit_id": self.unit_id,
                    "cycle": observation.cycle,
                    **observation.sensors,
                }
                for observation in self.observations
            ]
        )


class OtherPlanningEngine(BaseModel):
    """A fleet engine whose conservative RUL is already available."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    unit_id: PositiveInt
    conservative_rul: NonNegativeFiniteFloat
    maintenance_duration: PositiveInt
    technicians_required: PositiveInt = 1
    criticality: Literal["low", "medium", "high"]
    preventive_cost: NonNegativeInt | None = None


class AgentPlanningContext(BaseModel):
    """Resources and maintenance metadata; current-engine RUL is deliberately absent."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    scenario_id: Annotated[str, Field(min_length=1, max_length=100)] = "agent_scenario"
    description: Annotated[str, Field(max_length=500)] = ""
    planning_horizon: Annotated[int, Field(gt=0, le=1000)]
    technicians: Annotated[int, Field(gt=0, le=100)]
    workshop_capacity: Annotated[int, Field(gt=0, le=100)]
    maintenance_duration: PositiveInt
    technicians_required: PositiveInt = 1
    criticality: Literal["low", "medium", "high"]
    preventive_cost: NonNegativeInt | None = None
    other_engines: Annotated[list[OtherPlanningEngine], Field(max_length=499)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_other_engines(self) -> AgentPlanningContext:
        unit_ids = [engine.unit_id for engine in self.other_engines]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("other_engines must not contain duplicate unit_id values")
        return self


class MaintenanceAgentRequest(BaseModel):
    """Initial incident submitted to the supervised graph."""

    model_config = ConfigDict(extra="forbid", strict=True)

    incident_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        ),
    ]
    message: Annotated[str, Field(min_length=3, max_length=2000)]
    history: AgentEngineHistory | None = None
    question: Annotated[str, Field(min_length=3, max_length=500)] | None = None
    planning: AgentPlanningContext | None = None
    requested_actions: list[AgentAction] | None = None

    @field_validator("incident_id", "message", "question")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("text fields must be non-empty")
        return stripped

    @field_validator("requested_actions")
    @classmethod
    def deduplicate_actions(cls, value: list[AgentAction] | None) -> list[AgentAction] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("requested_actions must not be empty")
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_planning_units(self) -> MaintenanceAgentRequest:
        if self.history is None or self.planning is None:
            return self
        if any(engine.unit_id == self.history.unit_id for engine in self.planning.other_engines):
            raise ValueError("other_engines must not repeat the current history unit_id")
        return self


class RouterDecision(BaseModel):
    """Structured result produced by either a deterministic or an LLM router."""

    model_config = ConfigDict(extra="forbid", strict=True)

    actions: Annotated[list[AgentAction], Field(min_length=1)]
    rationale: Annotated[str, Field(min_length=1, max_length=500)]

    @field_validator("actions")
    @classmethod
    def unique_actions(cls, value: list[AgentAction]) -> list[AgentAction]:
        return list(dict.fromkeys(value))


class HumanReviewDecision(BaseModel):
    """Human decision used to resume a graph interrupted before finalization."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: ReviewAction
    comment: Annotated[str, Field(max_length=1000)] = ""
    edited_recommendation: Annotated[str, Field(min_length=3, max_length=5000)] | None = None

    @model_validator(mode="after")
    def validate_edit(self) -> HumanReviewDecision:
        if self.action == "edit" and self.edited_recommendation is None:
            raise ValueError("edited_recommendation is required when action is edit")
        if self.action != "edit" and self.edited_recommendation is not None:
            raise ValueError("edited_recommendation is only allowed when action is edit")
        return self


class AgentError(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    node: str
    category: str
    message: str


class MaintenanceAgentResult(BaseModel):
    """Stable, user-facing result extracted from the internal graph state."""

    model_config = ConfigDict(extra="forbid", strict=True)

    incident_id: str
    status: AgentStatus
    route: RouterDecision | None = None
    trend_summary: dict[str, Any] | None = None
    rul_prediction: dict[str, Any] | None = None
    document_answer: dict[str, Any] | None = None
    planning_result: dict[str, Any] | None = None
    draft_recommendation: str | None = None
    human_decision: HumanReviewDecision | None = None
    final_recommendation: str | None = None
    errors: list[AgentError] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class MaintenanceAgentRun(BaseModel):
    """One graph snapshot plus any pending human-review request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    result: MaintenanceAgentResult
    interrupts: list[dict[str, Any]] = Field(default_factory=list)
