"""Load and query the versioned technical-document retrieval index."""

from __future__ import annotations

import json
import threading
import tomllib
from pathlib import Path
from typing import Any

from industrial_maintenance.rag import _create_embedder, read_rag_settings
from industrial_maintenance.rag_retrieval import (
    DenseIndex,
    SearchResult,
    expand_query,
    load_query_terms,
)


def read_document_search_config(api_config_path: str | Path) -> Path:
    source = Path(api_config_path).resolve()
    project_root = source.parent.parent.resolve()
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
        value = raw["documents"]["rag_config"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid document-search configuration {source}: {exc}") from exc
    if not isinstance(value, str) or not value.strip():
        raise ValueError("documents.rag_config must be a non-empty path")
    path = (project_root / value).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("documents.rag_config must remain inside the project root") from exc
    return path


class DocumentSearchService:
    """Thread-safe document retriever loaded once during API startup."""

    def __init__(self, config_path: str | Path) -> None:
        self.settings = read_rag_settings(config_path)
        self.index = DenseIndex.load(self.settings.index_directory)
        self.embedder = _create_embedder(self.settings, local_files_only=True)
        self.terms = load_query_terms(self.settings.query_terms)
        self._lock = threading.RLock()
        manifest_path = self.settings.index_directory / "index_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._version = (
            f"{manifest['model_name'].split('/')[-1]}-"
            f"{manifest['chunks_sha256'][:12]}-{manifest['embeddings_sha256'][:12]}"
        )

    @classmethod
    def from_api_config(cls, api_config_path: str | Path) -> DocumentSearchService:
        return cls(read_document_search_config(api_config_path))

    @property
    def version(self) -> str:
        return self._version

    def info(self) -> dict[str, Any]:
        return {
            "retrieval_version": self.version,
            "model_name": self.settings.model_name,
            "model_revision": self.settings.model_revision,
            "chunk_count": len(self.index.chunks),
            "default_top_k": self.settings.default_top_k,
            "retrieval_method": "hybrid_rrf_dense_cosine_tfidf",
        }

    def search(self, query: str, *, top_k: int) -> list[SearchResult]:
        expanded_query = expand_query(query, self.terms)
        with self._lock:
            return self.index.search(
                expanded_query,
                embedder=self.embedder,
                top_k=top_k,
                max_per_document=self.settings.max_per_document,
                dense_weight=self.settings.dense_weight,
                lexical_weight=self.settings.lexical_weight,
                lexical_score_weight=self.settings.lexical_score_weight,
                rrf_k=self.settings.rrf_k,
            )
