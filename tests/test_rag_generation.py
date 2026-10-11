from __future__ import annotations

from pathlib import Path

import pytest

from industrial_maintenance.rag_generation import (
    ABSTENTION_MESSAGE,
    GenerationError,
    GroundedAnswerService,
    parse_generated_decision,
    read_generation_settings,
)
from industrial_maintenance.rag_retrieval import SearchResult


def _result(rank: int, document_id: str) -> SearchResult:
    return SearchResult(
        rank=rank,
        score=0.05,
        dense_score=0.82,
        lexical_score=0.2,
        chunk_id=f"chunk-{rank}",
        document_id=document_id,
        title=f"Document {rank}",
        section="Technical section",
        page_start=rank,
        page_end=rank,
        text=f"Evidence passage {rank}",
        source_url=f"https://example.test/{rank}",
        limitations="Test limitation",
    )


class _FakeSearch:
    version = "retrieval-v1"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class _FakeGenerator:
    version = "generator-v1"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, question: str, passages: list[SearchResult]) -> str:
        self.calls.append((question, [passage.chunk_id for passage in passages]))
        return self.output


def test_generation_settings_are_pinned_and_bounded(tmp_path: Path) -> None:
    config = tmp_path / "rag.toml"
    config.write_text(
        "\n".join(
            [
                "[generation]",
                'model_name = "Qwen/Qwen3-0.6B"',
                'model_revision = "fixed-revision"',
                'device = "cpu"',
                'dtype = "float32"',
                "max_input_tokens = 2048",
                "max_new_tokens = 128",
                "top_k_passages = 5",
            ]
        ),
        encoding="utf-8",
    )

    settings = read_generation_settings(config)

    assert settings.model_revision == "fixed-revision"
    assert settings.top_k_passages == 5


def test_parser_accepts_only_retrieved_citations() -> None:
    decision = parse_generated_decision(
        '{"answerable": true, "answer": "Réponse fondée.", "citations": ["S2", "S2"]}',
        source_count=2,
    )

    assert decision.answerable
    assert decision.answer == "Réponse fondée."
    assert decision.citation_ids == ("S2",)

    with pytest.raises(GenerationError, match="not retrieved"):
        parse_generated_decision(
            '{"answerable": true, "answer": "Réponse", "citations": ["S3"]}',
            source_count=2,
        )


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '{"answerable": true, "answer": "Réponse"}',
        '{"answerable": "yes", "answer": "Réponse", "citations": ["S1"]}',
        '{"answerable": true, "answer": "", "citations": ["S1"]}',
        '{"answerable": true, "answer": "Réponse", "citations": []}',
    ],
)
def test_parser_rejects_unverifiable_generator_outputs(raw: str) -> None:
    with pytest.raises(GenerationError):
        parse_generated_decision(raw, source_count=2)


def test_abstention_discards_text_and_citations() -> None:
    decision = parse_generated_decision(
        '{"answerable": false, "answer": "guess", "citations": ["S1"]}',
        source_count=1,
    )

    assert not decision.answerable
    assert decision.answer == ABSTENTION_MESSAGE
    assert decision.citation_ids == ()


def test_grounded_service_returns_only_cited_retrieved_passages() -> None:
    search = _FakeSearch([_result(1, "first"), _result(2, "second")])
    generator = _FakeGenerator(
        '{"answerable": true, "answer": "La preuve est dans S2.", "citations": ["S2"]}'
    )
    service = GroundedAnswerService(
        search=search,
        generator=generator,
        top_k_passages=5,
    )

    answer = service.ask("Question technique ?")

    assert answer.answerable
    assert answer.answer == "La preuve est dans S2."
    assert [citation.document_id for citation in answer.citations] == ["second"]
    assert answer.retrieval_version == "retrieval-v1"
    assert answer.generation_version == "generator-v1-procedure-guard-v1"
    assert search.calls == [("Question technique ?", 5)]


def test_grounded_service_abstains_without_calling_generator_when_no_passage() -> None:
    search = _FakeSearch([])
    generator = _FakeGenerator("unused")
    service = GroundedAnswerService(
        search=search,
        generator=generator,
        top_k_passages=5,
    )

    answer = service.ask("Question hors corpus ?")

    assert not answer.answerable
    assert answer.answer == ABSTENTION_MESSAGE
    assert answer.citations == ()
    assert generator.calls == []


def test_grounded_service_does_not_infer_a_repair_from_rul_only() -> None:
    search = _FakeSearch([_result(1, "dataset-readme")])
    generator = _FakeGenerator(
        '{"answerable": true, "answer": "Remplacer le composant.", '
        '"citations": ["S1"]}'
    )
    service = GroundedAnswerService(
        search=search,
        generator=generator,
        top_k_passages=5,
    )

    answer = service.ask(
        "Quelle réparation exacte faut-il effectuer si la RUL prédite vaut 20 cycles ?"
    )

    assert not answer.answerable
    assert answer.answer == ABSTENTION_MESSAGE
    assert answer.citations == ()
    assert generator.calls == []
