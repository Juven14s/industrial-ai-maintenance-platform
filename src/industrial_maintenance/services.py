"""Load one shared set of maintenance domain services for agentic frontends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from industrial_maintenance.agent_workflow import AgentServices
from industrial_maintenance.document_search import (
    DocumentSearchService,
    read_document_search_config,
)
from industrial_maintenance.inference import DualRULPredictor, read_inference_config
from industrial_maintenance.planning import MaintenancePlanningService
from industrial_maintenance.rag_generation import (
    GroundedAnswerService,
    LocalQwenGenerator,
    read_generation_settings,
)
from industrial_maintenance.sensor_trends import SensorTrendService


@dataclass(frozen=True)
class MaintenanceServiceBundle:
    """Heavy services loaded once and shared by orchestration adapters."""

    predictor: DualRULPredictor
    document_search: DocumentSearchService
    answer_service: GroundedAnswerService
    planning_service: MaintenancePlanningService
    sensor_trend_service: SensorTrendService

    def as_agent_services(self) -> AgentServices:
        return AgentServices(
            predictor=self.predictor,
            trends=self.sensor_trend_service,
            answers=self.answer_service,
            planner=self.planning_service,
        )


def load_maintenance_services(
    api_config_path: str | Path = "configs/api.toml",
) -> MaintenanceServiceBundle:
    """Load pinned RUL, RAG, planning, and trend services without global state."""

    inference_config = read_inference_config(api_config_path)
    predictor = DualRULPredictor.load(inference_config)
    document_search = DocumentSearchService.from_api_config(api_config_path)
    rag_config = read_document_search_config(api_config_path)
    generation_settings = read_generation_settings(rag_config)
    generator = LocalQwenGenerator(generation_settings, local_files_only=True)
    answer_service = GroundedAnswerService(
        search=document_search,
        generator=generator,
        top_k_passages=generation_settings.top_k_passages,
    )
    planning_service = MaintenancePlanningService.from_api_config(api_config_path)
    sensor_trend_service = SensorTrendService(predictor.sensor_names)
    return MaintenanceServiceBundle(
        predictor=predictor,
        document_search=document_search,
        answer_service=answer_service,
        planning_service=planning_service,
        sensor_trend_service=sensor_trend_service,
    )


__all__ = ["MaintenanceServiceBundle", "load_maintenance_services"]
