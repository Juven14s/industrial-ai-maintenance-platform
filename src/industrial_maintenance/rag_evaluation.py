"""Retrieval evaluation with document-and-page relevance judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_retrieval import DenseIndex, EmbedderProtocol, SearchResult


@dataclass(frozen=True)
class ExpectedReference:
    document_id: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class EvaluationQuestion:
    question_id: str
    category: str
    question: str
    answerable: bool
    expected_answer: str
    expected_references: tuple[ExpectedReference, ...]


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                question_id = str(raw["id"])
                references = tuple(
                    ExpectedReference(
                        document_id=str(reference["document_id"]),
                        page_start=reference.get("page_start"),
                        page_end=reference.get("page_end"),
                    )
                    for reference in raw.get("expected_references", [])
                )
                question = EvaluationQuestion(
                    question_id=question_id,
                    category=str(raw["category"]),
                    question=str(raw["question"]),
                    answerable=bool(raw["answerable"]),
                    expected_answer=str(raw["expected_answer"]),
                    expected_references=references,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise CorpusError(f"Invalid evaluation question at line {line_number}") from exc
            if question_id in seen_ids:
                raise CorpusError(f"Duplicate evaluation question id: {question_id}")
            if question.answerable and not question.expected_references:
                raise CorpusError(f"Answerable question {question_id} has no reference")
            if not question.answerable and question.expected_references:
                raise CorpusError(f"Unanswerable question {question_id} has a reference")
            seen_ids.add(question_id)
            questions.append(question)
    if not questions:
        raise CorpusError("The retrieval evaluation set is empty")
    return questions


def _document_matches(result: SearchResult, references: tuple[ExpectedReference, ...]) -> bool:
    return any(result.document_id == reference.document_id for reference in references)


def _reference_matches(result: SearchResult, references: tuple[ExpectedReference, ...]) -> bool:
    for reference in references:
        if result.document_id != reference.document_id:
            continue
        if reference.page_start is None:
            return True
        if result.page_start is None or result.page_end is None:
            continue
        expected_end = reference.page_end or reference.page_start
        if result.page_start <= expected_end and result.page_end >= reference.page_start:
            return True
    return False


def _first_rank(results: list[SearchResult], predicate: Callable[[SearchResult], bool]) -> int | None:
    for result in results:
        if predicate(result):
            return result.rank
    return None


def evaluate_retrieval(
    *,
    index: DenseIndex,
    embedder: EmbedderProtocol,
    questions: list[EvaluationQuestion],
    expand: Callable[[str], str],
    top_k: int,
    max_per_document: int,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    lexical_score_weight: float = 0.0,
    rrf_k: int = 60,
) -> dict[str, Any]:
    if top_k <= 0:
        raise CorpusError("Evaluation top_k must be positive")
    details: list[dict[str, Any]] = []
    page_ranks: list[int | None] = []
    document_ranks: list[int | None] = []

    for question in questions:
        expanded_query = expand(question.question)
        results = index.search(
            expanded_query,
            embedder=embedder,
            top_k=top_k,
            max_per_document=max_per_document,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            lexical_score_weight=lexical_score_weight,
            rrf_k=rrf_k,
        )
        document_rank = None
        reference_rank = None
        if question.answerable:
            document_rank = _first_rank(
                results,
                lambda result: _document_matches(result, question.expected_references),
            )
            reference_rank = _first_rank(
                results,
                lambda result: _reference_matches(result, question.expected_references),
            )
            document_ranks.append(document_rank)
            page_ranks.append(reference_rank)

        details.append(
            {
                "id": question.question_id,
                "category": question.category,
                "question": question.question,
                "expanded_query": expanded_query,
                "answerable": question.answerable,
                "expected_answer": question.expected_answer,
                "expected_references": [reference.__dict__ for reference in question.expected_references],
                "document_rank": document_rank,
                "reference_rank": reference_rank,
                "results": [result.to_dict() for result in results],
            }
        )

    answerable_count = len(page_ranks)
    if answerable_count == 0:
        raise CorpusError("The evaluation set has no answerable questions")

    def hit_rate(ranks: list[int | None], cutoff: int) -> float:
        return sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)

    metrics = {
        "answerable_questions": answerable_count,
        "unanswerable_questions": len(questions) - answerable_count,
        "document_hit_at_1": hit_rate(document_ranks, 1),
        "document_hit_at_3": hit_rate(document_ranks, 3),
        "document_hit_at_5": hit_rate(document_ranks, 5),
        "reference_hit_at_1": hit_rate(page_ranks, 1),
        "reference_hit_at_3": hit_rate(page_ranks, 3),
        "reference_hit_at_5": hit_rate(page_ranks, 5),
        "mrr_at_5": sum(1 / rank for rank in page_ranks if rank is not None and rank <= 5)
        / answerable_count,
    }
    return {
        "schema_version": 1,
        "evaluation_date": date.today().isoformat(),
        "top_k": top_k,
        "max_per_document": max_per_document,
        "retrieval": {
            "method": "reciprocal_rank_fusion",
            "dense_weight": dense_weight,
            "lexical_weight": lexical_weight,
            "lexical_score_weight": lexical_score_weight,
            "rrf_k": rrf_k,
        },
        "metrics": metrics,
        "questions": details,
    }


def write_evaluation(
    evaluation: dict[str, Any],
    *,
    json_path: str | Path,
    report_path: str | Path,
    model_name: str,
    model_revision: str,
) -> None:
    output_json = Path(json_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = evaluation["metrics"]
    rows = []
    for question in evaluation["questions"]:
        if question["answerable"]:
            result = (
                f"rang {question['reference_rank']}"
                if question["reference_rank"] is not None
                else "absent du top-k"
            )
        else:
            result = "abstention à évaluer avec la génération"
        rows.append(
            f"| `{question['id']}` | {question['category']} | "
            f"{'oui' if question['answerable'] else 'non'} | {result} |"
        )

    report = "\n".join(
        [
            "# Évaluation de la recherche documentaire",
            "",
            f"Date : {evaluation['evaluation_date']}  ",
            f"Modèle : `{model_name}`  ",
            f"Révision : `{model_revision}`  ",
            f"Configuration : top-{evaluation['top_k']}, au plus "
            f"{evaluation['max_per_document']} passages par document.",
            "Recherche hybride : similarité sémantique multilingue + TF-IDF, "
            f"fusion RRF (poids {evaluation['retrieval']['dense_weight']:.2f}/"
            f"{evaluation['retrieval']['lexical_weight']:.2f}, "
            f"k={evaluation['retrieval']['rrf_k']}), bonus lexical brut "
            f"{evaluation['retrieval']['lexical_score_weight']:.2f}.",
            "",
            "## Résultats",
            "",
            f"- Questions répondables : {metrics['answerable_questions']}",
            f"- Questions volontairement sans réponse : {metrics['unanswerable_questions']}",
            f"- Hit document@1/@3/@5 : {metrics['document_hit_at_1']:.1%} / "
            f"{metrics['document_hit_at_3']:.1%} / {metrics['document_hit_at_5']:.1%}",
            f"- Hit référence page@1/@3/@5 : {metrics['reference_hit_at_1']:.1%} / "
            f"{metrics['reference_hit_at_3']:.1%} / {metrics['reference_hit_at_5']:.1%}",
            f"- MRR@5 : {metrics['mrr_at_5']:.3f}",
            "",
            "La métrique de référence exige le bon document et, pour les PDF, une page située "
            "dans la plage annotée. Les cinq questions sans réponse servent au futur test "
            "d'abstention et ne sont pas comptées dans les métriques de récupération.",
            "",
            "## Détail par question",
            "",
            "| ID | Catégorie | Répondable | Premier passage pertinent |",
            "|---|---|---:|---|",
            *rows,
            "",
        ]
    )
    output_report = Path(report_path)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
