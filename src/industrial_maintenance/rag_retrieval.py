"""Dense local retrieval over normalized multilingual text embeddings."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from industrial_maintenance.rag_corpus import Chunk, CorpusError, read_chunks, sha256_file


class EmbedderProtocol(Protocol):
    model_name: str
    model_revision: str
    query_prefix: str
    passage_prefix: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers adapter, kept out of extraction-only workflows."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        query_prefix: str,
        passage_prefix: str,
        batch_size: int,
        device: str,
        local_files_only: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise CorpusError(
                "sentence-transformers is required to build or query the dense index; "
                "install the project with the 'rag' extra"
            ) from exc
        self.model_name = model_name
        self.model_revision = model_revision
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name,
            revision=model_revision,
            device=device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [self.passage_prefix + text for text in texts]
        return np.asarray(
            self.model.encode(
                prefixed,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [self.query_prefix + text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embedding[0], dtype=np.float32)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    dense_score: float
    lexical_score: float
    chunk_id: str
    document_id: str
    title: str
    section: str
    page_start: int | None
    page_end: int | None
    text: str
    source_url: str
    limitations: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_query_terms(path: str | Path) -> dict[str, tuple[str, ...]]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise CorpusError("Unsupported query-term schema_version")
    terms = data.get("terms")
    if not isinstance(terms, dict) or not terms:
        raise CorpusError("Query-term configuration is empty")
    return {
        unicodedata.normalize("NFKC", str(source)).casefold(): tuple(
            str(target) for target in targets
        )
        for source, targets in terms.items()
    }


def expand_query(query: str, terms: dict[str, tuple[str, ...]]) -> str:
    normalized = unicodedata.normalize("NFKC", query).casefold()
    additions: list[str] = []
    seen: set[str] = set()
    for source in sorted(terms, key=len, reverse=True):
        if not re.search(rf"(?<!\w){re.escape(source)}", normalized):
            continue
        for target in terms[source]:
            key = target.casefold()
            if key not in seen and key not in normalized:
                additions.append(target)
                seen.add(key)
    if not additions:
        return query
    return f"{query} | Domain terms: {', '.join(additions)}"


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def build_index(
    *,
    chunks_path: str | Path,
    index_directory: str | Path,
    embedder: EmbedderProtocol,
) -> dict[str, object]:
    chunk_path = Path(chunks_path)
    chunks = read_chunks(chunk_path)
    texts = [chunk.embedding_text() for chunk in chunks]
    embeddings = np.asarray(embedder.encode_documents(texts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise CorpusError(
            f"Embedder returned shape {embeddings.shape}; expected ({len(chunks)}, dimension)"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(embeddings)) or not np.allclose(norms, 1.0, atol=1e-3):
        raise CorpusError("Document embeddings must be finite and L2-normalized")

    output_dir = Path(index_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "chunks.jsonl"
    manifest_path = output_dir / "index_manifest.json"
    np.save(embeddings_path, embeddings, allow_pickle=False)
    metadata_path.write_bytes(chunk_path.read_bytes())

    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_name": embedder.model_name,
        "model_revision": embedder.model_revision,
        "query_prefix": embedder.query_prefix,
        "passage_prefix": embedder.passage_prefix,
        "normalized": True,
        "similarity": "hybrid_rrf_dense_cosine_tfidf",
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "chunks_sha256": sha256_file(metadata_path),
        "embeddings_sha256": _array_sha256(embeddings),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class DenseIndex:
    def __init__(self, *, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise CorpusError("Dense index metadata and embeddings are inconsistent")
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        self.lexical_matrix: csr_matrix = self.vectorizer.fit_transform(
            chunk.embedding_text() for chunk in chunks
        ).tocsr()

    @classmethod
    def load(cls, index_directory: str | Path) -> DenseIndex:
        index_dir = Path(index_directory)
        manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        chunks_path = index_dir / "chunks.jsonl"
        chunks = read_chunks(chunks_path)
        embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
        if sha256_file(chunks_path) != manifest["chunks_sha256"]:
            raise CorpusError("Indexed chunk metadata failed its SHA-256 check")
        if _array_sha256(embeddings) != manifest["embeddings_sha256"]:
            raise CorpusError("Embedding matrix failed its SHA-256 check")
        if int(manifest["chunk_count"]) != len(chunks):
            raise CorpusError("Index manifest has an invalid chunk count")
        return cls(chunks=chunks, embeddings=embeddings)

    def search_vector(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int,
        max_per_document: int | None = None,
    ) -> list[SearchResult]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise CorpusError("Query embedding dimension does not match the index")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(query).all() or not np.isclose(norm, 1.0, atol=1e-3):
            raise CorpusError("Query embedding must be finite and L2-normalized")
        if top_k <= 0:
            raise CorpusError("top_k must be positive")
        if max_per_document is not None and max_per_document <= 0:
            raise CorpusError("max_per_document must be positive when provided")

        scores = self.embeddings @ query
        order = np.argsort(-scores, kind="stable")
        results: list[SearchResult] = []
        per_document: dict[str, int] = {}
        for index in order:
            chunk = self.chunks[int(index)]
            current_count = per_document.get(chunk.document_id, 0)
            if max_per_document is not None and current_count >= max_per_document:
                continue
            per_document[chunk.document_id] = current_count + 1
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=float(scores[int(index)]),
                    dense_score=float(scores[int(index)]),
                    lexical_score=0.0,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    limitations=chunk.limitations,
                )
            )
            if len(results) == min(top_k, len(self.chunks)):
                break
        return results

    def search(
        self,
        query: str,
        *,
        embedder: EmbedderProtocol,
        top_k: int,
        max_per_document: int | None = None,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        lexical_score_weight: float = 0.0,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        query_embedding = np.asarray(embedder.encode_query(query), dtype=np.float32).reshape(-1)
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise CorpusError("Query embedding dimension does not match the index")
        norm = float(np.linalg.norm(query_embedding))
        if not np.isfinite(query_embedding).all() or not np.isclose(norm, 1.0, atol=1e-3):
            raise CorpusError("Query embedding must be finite and L2-normalized")
        if top_k <= 0:
            raise CorpusError("top_k must be positive")
        if max_per_document is not None and max_per_document <= 0:
            raise CorpusError("max_per_document must be positive when provided")
        if (
            dense_weight < 0
            or lexical_weight < 0
            or lexical_score_weight < 0
            or dense_weight + lexical_weight + lexical_score_weight == 0
        ):
            raise CorpusError("At least one non-negative retrieval weight must be positive")
        if rrf_k <= 0:
            raise CorpusError("rrf_k must be positive")

        dense_scores = self.embeddings @ query_embedding
        query_vector = self.vectorizer.transform([query])
        lexical_scores = np.asarray((self.lexical_matrix @ query_vector.T).toarray()).reshape(-1)

        dense_order = np.argsort(-dense_scores, kind="stable")
        dense_ranks = np.empty(len(self.chunks), dtype=np.int32)
        dense_ranks[dense_order] = np.arange(1, len(self.chunks) + 1)

        lexical_order = np.argsort(-lexical_scores, kind="stable")
        lexical_ranks = np.empty(len(self.chunks), dtype=np.int32)
        lexical_ranks[lexical_order] = np.arange(1, len(self.chunks) + 1)
        lexical_matches = lexical_scores > 0

        fused_scores = dense_weight / (rrf_k + dense_ranks)
        fused_scores += np.where(
            lexical_matches,
            lexical_weight / (rrf_k + lexical_ranks),
            0.0,
        )
        fused_scores += lexical_score_weight * lexical_scores
        order = np.argsort(-fused_scores, kind="stable")

        results: list[SearchResult] = []
        per_document: dict[str, int] = {}
        for index in order:
            chunk = self.chunks[int(index)]
            current_count = per_document.get(chunk.document_id, 0)
            if max_per_document is not None and current_count >= max_per_document:
                continue
            per_document[chunk.document_id] = current_count + 1
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    score=float(fused_scores[int(index)]),
                    dense_score=float(dense_scores[int(index)]),
                    lexical_score=float(lexical_scores[int(index)]),
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=chunk.title,
                    section=chunk.section,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    source_url=chunk.source_url,
                    limitations=chunk.limitations,
                )
            )
            if len(results) == min(top_k, len(self.chunks)):
                break
        return results
