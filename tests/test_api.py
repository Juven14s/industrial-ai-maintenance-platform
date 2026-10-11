"""HTTP contract tests for the FastAPI inference service."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from industrial_maintenance import __version__
from industrial_maintenance.api import create_app
from industrial_maintenance.inference import InferenceInputError, RULPrediction
from industrial_maintenance.planning import MaintenancePlanningService
from industrial_maintenance.rag_generation import ABSTENTION_MESSAGE, GroundedAnswer
from industrial_maintenance.rag_retrieval import SearchResult

REQUIRED_SENSORS = (
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)


def _observation(cycle: int) -> dict[str, int | float]:
    return {
        "cycle": cycle,
        **{
            sensor: float(100 * sensor_index + cycle)
            for sensor_index, sensor in enumerate(REQUIRED_SENSORS, start=1)
        },
    }


def _payload(*cycles: int) -> dict[str, Any]:
    return {
        "unit_id": 7,
        "observations": [_observation(cycle) for cycle in cycles],
    }


class FakePredictor:
    """Small deterministic substitute for the two serialized model bundles."""

    def __init__(self, error: InferenceInputError | None = None) -> None:
        self.error = error
        self.received_frames: list[pd.DataFrame] = []

    @property
    def central_version(self) -> str:
        return "lstm-central123456"

    @property
    def safety_version(self) -> str:
        return "hgb-guard123456"

    def model_info(self) -> dict[str, Any]:
        return {
            "window_length": 30,
            "required_sensors": list(REQUIRED_SENSORS),
            "central": {
                "name": "lstm",
                "version": self.central_version,
                "role": "point_estimate",
            },
            "conservative": {
                "name": "hgb_quantile_005",
                "version": self.safety_version,
                "role": "maintenance_decision",
                "safety_margin_cycles": 4.0,
            },
        }

    def predict_frame(self, frame: pd.DataFrame) -> RULPrediction:
        self.received_frames.append(frame.copy(deep=True))
        if self.error is not None:
            raise self.error
        return RULPrediction(
            unit_id=int(frame["unit_id"].iloc[0]),
            history_start_cycle=int(frame["cycle"].iloc[0]),
            last_cycle=int(frame["cycle"].iloc[-1]),
            observed_cycles=len(frame),
            window_cycles_used=min(len(frame), 30),
            rul_estimate=48.25,
            rul_conservative=27.5,
            safety_base_prediction=31.5,
            safety_margin=4.0,
            decision_value=27.5,
            central_model_version=self.central_version,
            safety_model_version=self.safety_version,
        )


def _document_result() -> SearchResult:
    return SearchResult(
        rank=1,
        score=0.052,
        dense_score=0.84,
        lexical_score=0.21,
        chunk_id="readme-001",
        document_id="cmapss_dataset_readme",
        title="NASA C-MAPSS dataset README",
        section="Experimental Scenario",
        page_start=None,
        page_end=None,
        text="There are three operational settings.",
        source_url="https://example.test/readme",
        limitations="The settings are anonymized.",
    )


class FakeDocumentSearch:
    version = "multilingual-e5-small-test123"

    def __init__(self) -> None:
        self.received_queries: list[tuple[str, int]] = []

    def info(self) -> dict[str, Any]:
        return {
            "retrieval_version": self.version,
            "model_name": "intfloat/multilingual-e5-small",
            "model_revision": "test-revision",
            "chunk_count": 192,
            "default_top_k": 5,
            "retrieval_method": "hybrid_rrf_dense_cosine_tfidf",
        }

    def search(self, query: str, *, top_k: int) -> list[SearchResult]:
        self.received_queries.append((query, top_k))
        return [_document_result()][:top_k]


class FakeAnswerService:
    version = "Qwen3-0.6B-test123"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def ask(self, question: str) -> GroundedAnswer:
        self.questions.append(question)
        if "inconnue" in question:
            return GroundedAnswer(
                question=question,
                answerable=False,
                answer=ABSTENTION_MESSAGE,
                citations=(),
                retrieval_version="multilingual-e5-small-test123",
                generation_version=self.version,
            )
        return GroundedAnswer(
            question=question,
            answerable=True,
            answer="Trois réglages opérationnels influencent les performances.",
            citations=(_document_result(),),
            retrieval_version="multilingual-e5-small-test123",
            generation_version=self.version,
        )


@pytest.fixture
def predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def document_search() -> FakeDocumentSearch:
    return FakeDocumentSearch()


@pytest.fixture
def answer_service() -> FakeAnswerService:
    return FakeAnswerService()


@pytest.fixture
def planning_service() -> MaintenancePlanningService:
    return MaintenancePlanningService("configs/maintenance_planning.toml")


@pytest.fixture
def client(
    predictor: FakePredictor,
    document_search: FakeDocumentSearch,
    answer_service: FakeAnswerService,
    planning_service: MaintenancePlanningService,
):
    with TestClient(
        create_app(
            predictor=predictor,
            document_search=document_search,
            answer_service=answer_service,
            planning_service=planning_service,
        )
    ) as test_client:
        yield test_client


def test_health_reports_loaded_model_versions(client: TestClient, predictor: FakePredictor):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "api_version": __version__,
        "central_model_version": predictor.central_version,
        "safety_model_version": predictor.safety_version,
    }


def test_models_returns_sanitized_predictor_metadata(
    client: TestClient, predictor: FakePredictor
):
    response = client.get("/models")

    assert response.status_code == 200
    assert response.json() == predictor.model_info()


def test_predict_returns_both_rul_values_and_preserves_history(
    client: TestClient, predictor: FakePredictor
):
    response = client.post("/predict/rul", json=_payload(1, 2))

    assert response.status_code == 200
    assert response.json() == {
        "unit_id": 7,
        "history_start_cycle": 1,
        "last_cycle": 2,
        "observed_cycles": 2,
        "window_cycles_used": 2,
        "rul_estimate": 48.25,
        "rul_conservative": 27.5,
        "safety_base_prediction": 31.5,
        "safety_margin": 4.0,
        "decision_value": 27.5,
        "unit": "cycles",
        "central_model_version": predictor.central_version,
        "safety_model_version": predictor.safety_version,
    }
    assert len(predictor.received_frames) == 1
    received = predictor.received_frames[0]
    assert received["unit_id"].tolist() == [7, 7]
    assert received["cycle"].tolist() == [1, 2]
    assert set(received.columns) == {"unit_id", "cycle", *REQUIRED_SENSORS}
    assert "rul" not in received.columns


def test_openapi_documents_the_public_routes_and_closed_observation_schema(client: TestClient):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert document["info"]["title"] == "Industrial AI Maintenance API"
    assert document["info"]["version"] == __version__
    assert set(document["paths"]) >= {
        "/health",
        "/models",
        "/predict/rul",
        "/documents/status",
        "/documents/search",
        "/documents/ask",
        "/maintenance/plan",
    }
    assert "get" in document["paths"]["/health"]
    assert "get" in document["paths"]["/models"]
    assert "post" in document["paths"]["/predict/rul"]
    assert "get" in document["paths"]["/documents/status"]
    assert "post" in document["paths"]["/documents/search"]
    assert "post" in document["paths"]["/documents/ask"]
    assert "post" in document["paths"]["/maintenance/plan"]
    observation_schema = document["components"]["schemas"]["CycleObservation"]
    assert observation_schema["additionalProperties"] is False
    assert set(REQUIRED_SENSORS).issubset(observation_schema["required"])
    assert "rul" not in observation_schema["properties"]


def test_document_status_reports_loaded_index(
    client: TestClient,
    document_search: FakeDocumentSearch,
):
    response = client.get("/documents/status")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", **document_search.info()}


def test_document_search_returns_passages_with_citations(
    client: TestClient,
    document_search: FakeDocumentSearch,
):
    response = client.post(
        "/documents/search",
        json={"query": "Que représentent les réglages ?", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "Que représentent les réglages ?",
        "retrieval_version": document_search.version,
        "results": [
            {
                "rank": 1,
                "score": 0.052,
                "document_id": "cmapss_dataset_readme",
                "title": "NASA C-MAPSS dataset README",
                "section": "Experimental Scenario",
                "page_start": None,
                "page_end": None,
                "text": "There are three operational settings.",
                "source_url": "https://example.test/readme",
                "citation": "NASA C-MAPSS dataset README — Experimental Scenario",
                "limitations": "The settings are anonymized.",
            }
        ],
    }
    assert document_search.received_queries == [("Que représentent les réglages ?", 3)]


def test_document_answer_returns_only_grounded_citations(
    client: TestClient,
    answer_service: FakeAnswerService,
):
    question = "Combien de réglages opérationnels sont présents ?"
    response = client.post("/documents/ask", json={"question": question})

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == question
    assert payload["status"] == "answered"
    assert payload["answerable"] is True
    assert payload["generation_version"] == answer_service.version
    assert payload["citations"][0]["document_id"] == "cmapss_dataset_readme"
    assert payload["citations"][0]["citation"].endswith("Experimental Scenario")
    assert answer_service.questions == [question]


def test_document_answer_exposes_explicit_abstention(
    client: TestClient,
    answer_service: FakeAnswerService,
):
    question = "Quelle information inconnue est demandée ?"
    response = client.post("/documents/ask", json={"question": question})

    assert response.status_code == 200
    assert response.json() == {
        "question": question,
        "status": "abstained",
        "answerable": False,
        "answer": ABSTENTION_MESSAGE,
        "retrieval_version": "multilingual-e5-small-test123",
        "generation_version": answer_service.version,
        "citations": [],
    }


def test_maintenance_plan_compares_heuristic_and_cp_sat(client: TestClient) -> None:
    response = client.post(
        "/maintenance/plan",
        json={
            "scenario_id": "api-test",
            "description": "Small constrained fleet",
            "planning_horizon": 8,
            "technicians": 1,
            "workshop_capacity": 1,
            "engines": [
                {
                    "unit_id": 1,
                    "conservative_rul": 3.0,
                    "maintenance_duration": 2,
                    "technicians_required": 1,
                    "criticality": "high",
                    "preventive_cost": 500,
                },
                {
                    "unit_id": 2,
                    "conservative_rul": 6.0,
                    "maintenance_duration": 2,
                    "technicians_required": 1,
                    "criticality": "medium",
                    "preventive_cost": 400,
                },
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_id"] == "api-test"
    assert payload["heuristic"]["status"] == "feasible"
    assert payload["optimized"]["status"] == "optimal"
    assert payload["optimized"]["planning_version"] == "maintenance-planning-v1"
    assert len(payload["optimized"]["assignments"]) == 2
    assert payload["optimized"]["objective_cost"] <= payload["heuristic"]["objective_cost"]


def test_maintenance_plan_rejects_duplicate_units(client: TestClient) -> None:
    engine = {
        "unit_id": 1,
        "conservative_rul": 3.0,
        "maintenance_duration": 1,
        "criticality": "high",
    }
    response = client.post(
        "/maintenance/plan",
        json={
            "planning_horizon": 5,
            "technicians": 1,
            "workshop_capacity": 1,
            "engines": [engine, engine],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_planning_scenario"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "ab"},
        {"query": "question valide", "top_k": 0},
        {"query": "question valide", "top_k": "5"},
        {"query": "question valide", "unexpected": True},
    ],
)
def test_document_search_rejects_invalid_requests(
    client: TestClient,
    document_search: FakeDocumentSearch,
    payload: dict[str, Any],
):
    response = client.post("/documents/search", json=payload)

    assert response.status_code == 422
    assert document_search.received_queries == []


def test_document_routes_report_unavailable_without_a_retriever(
    predictor: FakePredictor,
):
    with TestClient(create_app(predictor=predictor)) as client:
        status_response = client.get("/documents/status")
        search_response = client.post(
            "/documents/search",
            json={"query": "Qu'est-ce que la RUL ?"},
        )
        answer_response = client.post(
            "/documents/ask",
            json={"question": "Qu'est-ce que la RUL ?"},
        )
        planning_response = client.post(
            "/maintenance/plan",
            json={
                "planning_horizon": 5,
                "technicians": 1,
                "workshop_capacity": 1,
                "engines": [
                    {
                        "unit_id": 1,
                        "conservative_rul": 3.0,
                        "maintenance_duration": 1,
                        "criticality": "high",
                    }
                ],
            },
        )

    assert status_response.status_code == 503
    assert search_response.status_code == 503
    assert answer_response.status_code == 503
    assert planning_response.status_code == 503
    assert status_response.json()["detail"]["code"] == "documents_unavailable"
    assert search_response.json()["detail"]["code"] == "documents_unavailable"
    assert answer_response.json()["detail"]["code"] == "generation_unavailable"
    assert planning_response.json()["detail"]["code"] == "planning_unavailable"


@pytest.mark.parametrize("cycles", [(1, 1), (1, 3), (2, 1)])
def test_predict_rejects_duplicate_gapped_or_unordered_cycles(
    client: TestClient, predictor: FakePredictor, cycles: tuple[int, int]
):
    response = client.post("/predict/rul", json=_payload(*cycles))

    assert response.status_code == 422
    assert "ordered and consecutive" in response.text
    assert predictor.received_frames == []


def test_predict_rejects_a_missing_required_sensor(
    client: TestClient, predictor: FakePredictor
):
    payload = _payload(1)
    del payload["observations"][0]["sensor_11"]

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(item["loc"][-1] == "sensor_11" and item["type"] == "missing" for item in detail)
    assert predictor.received_frames == []


@pytest.mark.parametrize("field", ["rul", "unexpected_measurement"])
def test_predict_rejects_target_or_unknown_fields(
    client: TestClient, predictor: FakePredictor, field: str
):
    payload = _payload(1)
    payload["observations"][0][field] = 12.0

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(
        item["loc"][-1] == field and item["type"] == "extra_forbidden" for item in detail
    )
    assert predictor.received_frames == []


@pytest.mark.parametrize(
    ("location", "value", "expected_type"),
    [
        ("unit_id", 0, "greater_than"),
        ("unit_id", "7", "int_type"),
        ("cycle", 1.0, "int_type"),
        ("sensor_2", "101.0", "float_type"),
        ("sensor_2", True, "float_type"),
    ],
)
def test_predict_rejects_invalid_values_and_types(
    client: TestClient,
    predictor: FakePredictor,
    location: str,
    value: Any,
    expected_type: str,
):
    payload = _payload(1)
    if location == "unit_id":
        payload[location] = value
    else:
        payload["observations"][0][location] = value

    response = client.post("/predict/rul", json=payload)

    assert response.status_code == 422
    assert any(item["type"] == expected_type for item in response.json()["detail"])
    assert predictor.received_frames == []


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/health", None),
        ("get", "/models", None),
        ("post", "/predict/rul", _payload(1)),
    ],
)
def test_routes_report_service_unavailable_when_models_cannot_load(
    tmp_path, method: str, path: str, json_body: dict[str, Any] | None
):
    application = create_app(config_path=tmp_path / "missing-api.toml")
    with TestClient(application) as unavailable_client:
        response = unavailable_client.request(method, path, json=json_body)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "models_unavailable"
    assert detail["message"]


def test_predict_maps_inference_input_errors_to_a_structured_422():
    predictor = FakePredictor(InferenceInputError("short history must start at cycle 1"))
    with TestClient(create_app(predictor=predictor)) as client:
        response = client.post("/predict/rul", json=_payload(10))

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "invalid_history",
            "message": "short history must start at cycle 1",
        }
    }
    assert len(predictor.received_frames) == 1
