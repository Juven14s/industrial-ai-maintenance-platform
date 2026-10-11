"""Evaluate grounded generation, citations, abstention and CPU latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tomllib
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from industrial_maintenance.document_search import DocumentSearchService
from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_evaluation import (
    EvaluationQuestion,
    ExpectedReference,
    load_questions,
)
from industrial_maintenance.rag_generation import (
    GenerationError,
    GroundedAnswer,
    GroundedAnswerService,
    LocalQwenGenerator,
    read_generation_settings,
)
from industrial_maintenance.rag_retrieval import SearchResult


def _reference_matches(
    result: SearchResult,
    references: tuple[ExpectedReference, ...],
) -> bool:
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


def _paths(config_path: str | Path) -> tuple[Path, Path, Path]:
    source = Path(config_path).resolve()
    root = source.parent.parent
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        json_path = root / raw["generation_evaluation_output"]
        report_path = root / raw["generation_evaluation_report"]
        checkpoint_path = root / raw["generation_evaluation_checkpoint"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise CorpusError(f"Invalid generation evaluation paths in {source}: {exc}") from exc
    return json_path, report_path, checkpoint_path


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                completed[str(item["id"])] = item
    return completed


def _append_checkpoint(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _synchronize_annotations(
    item: dict[str, Any],
    question: EvaluationQuestion,
) -> dict[str, Any]:
    """Recompute annotation-dependent fields without regenerating an answer."""
    synchronized = dict(item)
    synchronized["category"] = question.category
    synchronized["question"] = question.question
    synchronized["expected_answerable"] = question.answerable
    synchronized["expected_answer"] = question.expected_answer
    synchronized["expected_references"] = [
        asdict(reference) for reference in question.expected_references
    ]
    predicted = synchronized.get("predicted_answerable")
    synchronized["decision_correct"] = predicted == question.answerable
    synchronized["citation_reference_hit"] = False
    if predicted is True and question.answerable:
        citations = [SearchResult(**value) for value in synchronized.get("citations", [])]
        synchronized["citation_reference_hit"] = any(
            _reference_matches(citation, question.expected_references)
            for citation in citations
        )
    return synchronized


def _semantic_similarity(
    search: DocumentSearchService,
    expected: str,
    actual: str,
) -> float:
    expected_embedding = search.embedder.encode_query(expected)
    actual_embedding = search.embedder.encode_query(actual)
    return float(np.dot(expected_embedding, actual_embedding))


def _evaluate_one(
    *,
    question: EvaluationQuestion,
    service: GroundedAnswerService,
    search: DocumentSearchService,
) -> dict[str, Any]:
    started = time.perf_counter()
    answer: GroundedAnswer | None = None
    error: str | None = None
    try:
        answer = service.ask(question.question)
    except (CorpusError, GenerationError, OSError, RuntimeError, ValueError) as exc:
        error = str(exc)
    latency = time.perf_counter() - started

    predicted_answerable = answer.answerable if answer is not None else None
    decision_correct = predicted_answerable == question.answerable
    citation_reference_hit = False
    semantic_similarity = None
    if answer is not None and answer.answerable and question.answerable:
        citation_reference_hit = any(
            _reference_matches(citation, question.expected_references)
            for citation in answer.citations
        )
        semantic_similarity = _semantic_similarity(
            search,
            question.expected_answer,
            answer.answer,
        )

    return {
        "id": question.question_id,
        "category": question.category,
        "question": question.question,
        "expected_answerable": question.answerable,
        "expected_answer": question.expected_answer,
        "expected_references": [asdict(reference) for reference in question.expected_references],
        "predicted_answerable": predicted_answerable,
        "decision_correct": decision_correct,
        "answer": answer.answer if answer is not None else None,
        "citations": (
            [citation.to_dict() for citation in answer.citations]
            if answer is not None
            else []
        ),
        "citation_reference_hit": citation_reference_hit,
        "semantic_similarity": semantic_similarity,
        "latency_seconds": latency,
        "error": error,
    }


def _metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in items if item["expected_answerable"]]
    unanswerable = [item for item in items if not item["expected_answerable"]]
    answered = [item for item in answerable if item["predicted_answerable"] is True]
    valid = [item for item in items if item["error"] is None]
    similarities = [
        float(item["semantic_similarity"])
        for item in answered
        if item["semantic_similarity"] is not None
    ]
    latencies = [float(item["latency_seconds"]) for item in items]
    return {
        "question_count": len(items),
        "valid_output_rate": len(valid) / len(items),
        "answerability_accuracy": sum(item["decision_correct"] for item in items) / len(items),
        "answer_recall": sum(item["predicted_answerable"] is True for item in answerable)
        / len(answerable),
        "abstention_recall": sum(
            item["predicted_answerable"] is False for item in unanswerable
        )
        / len(unanswerable),
        "citation_reference_hit_rate": (
            sum(item["citation_reference_hit"] for item in answered) / len(answered)
            if answered
            else 0.0
        ),
        "mean_answer_similarity": statistics.fmean(similarities) if similarities else None,
        "latency_mean_seconds": statistics.fmean(latencies),
        "latency_median_seconds": statistics.median(latencies),
        "latency_max_seconds": max(latencies),
    }


def _write_outputs(
    *,
    items: list[dict[str, Any]],
    metrics: dict[str, Any],
    json_path: Path,
    report_path: Path,
    retrieval_version: str,
    generation_version: str,
) -> None:
    payload = {
        "schema_version": 1,
        "evaluation_date": date.today().isoformat(),
        "retrieval_version": retrieval_version,
        "generation_version": generation_version,
        "metrics": metrics,
        "questions": items,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = []
    for item in items:
        actual = (
            "erreur"
            if item["predicted_answerable"] is None
            else "réponse" if item["predicted_answerable"] else "abstention"
        )
        rows.append(
            f"| `{item['id']}` | "
            f"{'réponse' if item['expected_answerable'] else 'abstention'} | {actual} | "
            f"{'oui' if item['citation_reference_hit'] else 'non'} | "
            f"{item['latency_seconds']:.1f} s |"
        )

    report = "\n".join(
        [
            "# Évaluation de la génération RAG",
            "",
            f"Date : {payload['evaluation_date']}  ",
            f"Récupérateur : `{retrieval_version}`  ",
            f"Générateur : `{generation_version}`",
            "",
            "## Résultats",
            "",
            f"- Sorties valides : {metrics['valid_output_rate']:.1%}",
            f"- Décisions répondre/s'abstenir correctes : "
            f"{metrics['answerability_accuracy']:.1%}",
            f"- Questions répondables effectivement répondues : {metrics['answer_recall']:.1%}",
            f"- Questions sans réponse correctement refusées : "
            f"{metrics['abstention_recall']:.1%}",
            f"- Référence attendue parmi les citations des réponses : "
            f"{metrics['citation_reference_hit_rate']:.1%}",
            f"- Similarité sémantique moyenne avec la réponse attendue : "
            f"{metrics['mean_answer_similarity']:.3f}",
            f"- Latence CPU moyenne / médiane / maximale : "
            f"{metrics['latency_mean_seconds']:.1f} / "
            f"{metrics['latency_median_seconds']:.1f} / "
            f"{metrics['latency_max_seconds']:.1f} secondes",
            "",
            "La similarité sémantique est un indicateur automatique, pas une preuve de "
            "fidélité factuelle. Le hit de référence vérifie le document et la page annotés. "
            "Le même jeu ayant servi à mettre au point le récupérateur, ces résultats ne sont "
            "pas une estimation indépendante de généralisation.",
            "",
            "Pendant la mise au point, une demande de réparation fondée sur la seule RUL a "
            "révélé un excès de confiance. La politique finale refuse ce type de prescription "
            "lorsqu'aucun passage ne contient de procédure. L'annotation de `damage_02` a aussi "
            "été complétée avec la page 1, dont le résumé énonce explicitement la réponse. Ces "
            "deux corrections expliquent pourquoi le résultat final doit rester présenté comme "
            "une validation de développement.",
            "",
            "## Détail",
            "",
            "| ID | Attendu | Produit | Bonne référence citée | Latence |",
            "|---|---|---|---:|---:|",
            *rows,
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag.toml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--refresh-id",
        action="append",
        default=[],
        help="Re-evaluate this question ID while retaining the other checkpoints",
    )
    arguments = parser.parse_args()

    settings = read_generation_settings(arguments.config)
    search = DocumentSearchService(arguments.config)
    generator = LocalQwenGenerator(settings, local_files_only=True)
    service = GroundedAnswerService(
        search=search,
        generator=generator,
        top_k_passages=settings.top_k_passages,
    )
    json_path, report_path, checkpoint_path = _paths(arguments.config)
    questions = load_questions(search.settings.evaluation_questions)
    if arguments.limit is not None:
        if arguments.limit < 1:
            parser.error("--limit must be positive")
        questions = questions[: arguments.limit]

    completed = _load_checkpoint(checkpoint_path)
    completed = {
        question.question_id: _synchronize_annotations(
            completed[question.question_id], question
        )
        for question in questions
        if question.question_id in completed
    }
    refresh_ids = set(arguments.refresh_id)
    known_ids = {question.question_id for question in questions}
    unknown_ids = refresh_ids - known_ids
    if unknown_ids:
        parser.error(f"unknown --refresh-id: {', '.join(sorted(unknown_ids))}")
    for position, question in enumerate(questions, start=1):
        if question.question_id in completed and question.question_id not in refresh_ids:
            print(f"[{position}/{len(questions)}] {question.question_id}: checkpoint", flush=True)
            continue
        item = _evaluate_one(question=question, service=service, search=search)
        _append_checkpoint(checkpoint_path, item)
        completed[question.question_id] = item
        outcome = (
            "error"
            if item["predicted_answerable"] is None
            else "answer" if item["predicted_answerable"] else "abstain"
        )
        print(
            f"[{position}/{len(questions)}] {question.question_id}: {outcome}, "
            f"{item['latency_seconds']:.1f}s",
            flush=True,
        )

    items = [completed[question.question_id] for question in questions]
    metrics = _metrics(items)
    _write_outputs(
        items=items,
        metrics=metrics,
        json_path=json_path,
        report_path=report_path,
        retrieval_version=search.version,
        generation_version=service.version,
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
