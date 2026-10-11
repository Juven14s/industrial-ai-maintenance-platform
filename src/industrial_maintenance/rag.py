"""Prepare, index and query the local technical-document corpus."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_maintenance.rag_corpus import (
    CorpusError,
    chunk_sections,
    extract_sections,
    load_corpus_manifest,
    sha256_file,
    write_chunks,
)
from industrial_maintenance.rag_evaluation import (
    evaluate_retrieval,
    load_questions,
    write_evaluation,
)
from industrial_maintenance.rag_retrieval import (
    DenseIndex,
    SentenceTransformerEmbedder,
    build_index,
    expand_query,
    load_query_terms,
)


@dataclass(frozen=True)
class RagSettings:
    project_root: Path
    corpus_manifest: Path
    chunks_path: Path
    preparation_summary_path: Path
    index_directory: Path
    query_terms: Path
    evaluation_questions: Path
    evaluation_output: Path
    evaluation_report: Path
    max_words: int
    overlap_words: int
    min_words: int
    model_name: str
    model_revision: str
    query_prefix: str
    passage_prefix: str
    batch_size: int
    device: str
    default_top_k: int
    max_per_document: int
    dense_weight: float
    lexical_weight: float
    lexical_score_weight: float
    rrf_k: int


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def read_rag_settings(path: str | Path) -> RagSettings:
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise CorpusError("Unsupported RAG config schema_version")
    chunking = data["chunking"]
    embeddings = data["embeddings"]
    search = data["search"]
    return RagSettings(
        project_root=project_root,
        corpus_manifest=_resolve(data["corpus_manifest"], project_root),
        chunks_path=_resolve(data["chunks_path"], project_root),
        preparation_summary_path=_resolve(data["preparation_summary_path"], project_root),
        index_directory=_resolve(data["index_directory"], project_root),
        query_terms=_resolve(data["query_terms"], project_root),
        evaluation_questions=_resolve(data["evaluation_questions"], project_root),
        evaluation_output=_resolve(data["evaluation_output"], project_root),
        evaluation_report=_resolve(data["evaluation_report"], project_root),
        max_words=int(chunking["max_words"]),
        overlap_words=int(chunking["overlap_words"]),
        min_words=int(chunking["min_words"]),
        model_name=str(embeddings["model_name"]),
        model_revision=str(embeddings["model_revision"]),
        query_prefix=str(embeddings["query_prefix"]),
        passage_prefix=str(embeddings["passage_prefix"]),
        batch_size=int(embeddings["batch_size"]),
        device=str(embeddings["device"]),
        default_top_k=int(search["default_top_k"]),
        max_per_document=int(search["max_per_document"]),
        dense_weight=float(search["dense_weight"]),
        lexical_weight=float(search["lexical_weight"]),
        lexical_score_weight=float(search["lexical_score_weight"]),
        rrf_k=int(search["rrf_k"]),
    )


def prepare(settings: RagSettings) -> dict[str, Any]:
    documents = load_corpus_manifest(
        settings.corpus_manifest,
        project_root=settings.project_root,
    )
    all_chunks = []
    document_summaries: list[dict[str, Any]] = []
    for document in documents:
        sections = extract_sections(document)
        chunks = chunk_sections(
            sections,
            max_words=settings.max_words,
            overlap_words=settings.overlap_words,
            min_words=settings.min_words,
        )
        all_chunks.extend(chunks)
        document_summaries.append(
            {
                "document_id": document.document_id,
                "source_sha256": document.sha256,
                "included_pages": list(document.included_pages),
                "section_count": len(sections),
                "chunk_count": len(chunks),
                "word_count": sum(chunk.word_count for chunk in chunks),
            }
        )

    chunk_count = write_chunks(settings.chunks_path, all_chunks)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "corpus_manifest": str(settings.corpus_manifest.relative_to(settings.project_root)),
        "corpus_manifest_sha256": sha256_file(settings.corpus_manifest),
        "chunks_path": str(settings.chunks_path.relative_to(settings.project_root)),
        "chunks_sha256": sha256_file(settings.chunks_path),
        "chunking": {
            "max_words": settings.max_words,
            "overlap_words": settings.overlap_words,
            "min_words": settings.min_words,
        },
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "documents": document_summaries,
    }
    settings.preparation_summary_path.parent.mkdir(parents=True, exist_ok=True)
    settings.preparation_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _create_embedder(
    settings: RagSettings,
    *,
    local_files_only: bool,
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        model_name=settings.model_name,
        model_revision=settings.model_revision,
        query_prefix=settings.query_prefix,
        passage_prefix=settings.passage_prefix,
        batch_size=settings.batch_size,
        device=settings.device,
        local_files_only=local_files_only,
    )


def index(settings: RagSettings) -> dict[str, object]:
    if not settings.chunks_path.is_file():
        raise CorpusError("Prepared chunks are missing; run the prepare command first")
    return build_index(
        chunks_path=settings.chunks_path,
        index_directory=settings.index_directory,
        embedder=_create_embedder(settings, local_files_only=False),
    )


def search(settings: RagSettings, *, query: str, top_k: int | None = None) -> list[dict[str, object]]:
    dense_index = DenseIndex.load(settings.index_directory)
    expanded_query = expand_query(query, load_query_terms(settings.query_terms))
    results = dense_index.search(
        expanded_query,
        embedder=_create_embedder(settings, local_files_only=True),
        top_k=top_k or settings.default_top_k,
        max_per_document=settings.max_per_document,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
        lexical_score_weight=settings.lexical_score_weight,
        rrf_k=settings.rrf_k,
    )
    return [result.to_dict() for result in results]


def evaluate(settings: RagSettings) -> dict[str, Any]:
    terms = load_query_terms(settings.query_terms)
    evaluation = evaluate_retrieval(
        index=DenseIndex.load(settings.index_directory),
        embedder=_create_embedder(settings, local_files_only=True),
        questions=load_questions(settings.evaluation_questions),
        expand=lambda query: expand_query(query, terms),
        top_k=settings.default_top_k,
        max_per_document=settings.max_per_document,
        dense_weight=settings.dense_weight,
        lexical_weight=settings.lexical_weight,
        lexical_score_weight=settings.lexical_score_weight,
        rrf_k=settings.rrf_k,
    )
    write_evaluation(
        evaluation,
        json_path=settings.evaluation_output,
        report_path=settings.evaluation_report,
        model_name=settings.model_name,
        model_revision=settings.model_revision,
    )
    return evaluation["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Validate, extract and chunk the corpus")
    commands.add_parser("index", help="Build the normalized dense embedding index")
    commands.add_parser("evaluate", help="Evaluate document-and-page retrieval")
    search_parser = commands.add_parser("search", help="Query the dense index")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=None)
    arguments = parser.parse_args()

    try:
        settings = read_rag_settings(arguments.config)
        if arguments.command == "prepare":
            output = prepare(settings)
        elif arguments.command == "index":
            output = index(settings)
        elif arguments.command == "evaluate":
            output = evaluate(settings)
        else:
            output = search(settings, query=arguments.query, top_k=arguments.top_k)
    except (CorpusError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
