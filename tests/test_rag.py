from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from industrial_maintenance.rag_corpus import (
    Chunk,
    CorpusError,
    ExtractedSection,
    chunk_sections,
    load_corpus_manifest,
    parse_page_ranges,
    write_chunks,
)
from industrial_maintenance.rag_evaluation import load_questions
from industrial_maintenance.rag_retrieval import (
    DenseIndex,
    build_index,
    expand_query,
    load_query_terms,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_page_ranges_expands_and_validates() -> None:
    assert parse_page_ranges(["2-4", 7, "4-5"], page_count=8) == (2, 3, 4, 5, 7)
    with pytest.raises(CorpusError, match="outside"):
        parse_page_ranges(["3-9"], page_count=8)
    with pytest.raises(CorpusError, match="Invalid"):
        parse_page_ranges(["one-two"], page_count=8)


def test_manifest_verifies_source_integrity(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Data Set: FD001\nUseful technical content.", encoding="utf-8")
    manifest = tmp_path / "corpus.toml"
    manifest.write_text(
        "\n".join(
            [
                "schema_version = 1",
                "[[documents]]",
                'id = "source"',
                'path = "source.txt"',
                'title = "Source"',
                'source_url = "https://example.test/source"',
                'format = "txt"',
                'encoding = "utf-8"',
                "page_count = 0",
                f"size_bytes = {source.stat().st_size}",
                f'sha256 = "{_sha256(source)}"',
                'authority = "primary"',
                'roles = ["dataset_specification"]',
                'scope = "Test scope"',
                'limitations = "Test limitation"',
            ]
        ),
        encoding="utf-8",
    )

    documents = load_corpus_manifest(manifest, project_root=tmp_path)
    assert documents[0].path == source
    assert documents[0].encoding == "utf-8"

    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(CorpusError, match="Size mismatch|SHA-256 mismatch"):
        load_corpus_manifest(manifest, project_root=tmp_path)


def test_chunking_preserves_citation_and_overlap() -> None:
    text = " ".join(f"word{index}" for index in range(25))
    section = ExtractedSection(
        document_id="doc",
        title="Technical document",
        source_url="https://example.test/doc",
        roles=("maintenance",),
        scope="Generic maintenance",
        limitations="Not model specific",
        authority="primary",
        page=12,
        section="Turbine Engine Maintenance",
        text=text,
    )
    first = chunk_sections(
        [section],
        max_words=10,
        overlap_words=2,
        min_words=3,
    )
    second = chunk_sections(
        [section],
        max_words=10,
        overlap_words=2,
        min_words=3,
    )

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) == 3
    assert first[0].page_start == first[0].page_end == 12
    assert first[0].text.split()[-2:] == first[1].text.split()[:2]
    assert "Turbine Engine Maintenance" in first[0].embedding_text()


def test_chunking_keeps_short_structured_sections() -> None:
    section = ExtractedSection(
        document_id="readme",
        title="Dataset README",
        source_url="https://example.test/readme",
        roles=("dataset_specification",),
        scope="Dataset counts",
        limitations="None",
        authority="primary",
        page=None,
        section="Data Set: FD001",
        text="Train trajectories: 100 Test trajectories: 100 Conditions: ONE",
    )
    chunks = chunk_sections(
        [section],
        max_words=220,
        overlap_words=40,
        min_words=30,
    )
    assert len(chunks) == 1
    assert chunks[0].section == "Data Set: FD001"


def test_chunking_merges_short_table_fragments_on_same_page() -> None:
    common = {
        "document_id": "paper",
        "title": "Paper",
        "source_url": "https://example.test/paper",
        "roles": ("sensor_semantics",),
        "scope": "Sensor definitions",
        "limitations": "None",
        "authority": "primary",
        "page": 3,
    }
    sections = [
        ExtractedSection(
            **common,
            section="Table 2. Sensor outputs",
            text=" ".join(f"description{index}" for index in range(35)),
        ),
        ExtractedSection(
            **common,
            section="Nf Physical fan speed rpm",
            text="Nc Physical core speed rpm",
        ),
    ]
    chunks = chunk_sections(
        sections,
        max_words=100,
        overlap_words=20,
        min_words=30,
    )
    assert len(chunks) == 1
    assert "Nf Physical fan speed rpm" in chunks[0].text
    assert "Nc Physical core speed rpm" in chunks[0].text


class _FakeEmbedder:
    model_name = "fake-embedding-model"
    model_revision = "test-revision"
    query_prefix = "query: "
    passage_prefix = "passage: "

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == 3
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.6, 0.0],
            ],
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        assert text == "RUL definition"
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def _chunk(*, chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=document_id,
        source_url=f"https://example.test/{document_id}",
        roles=("dataset_specification",),
        scope="scope",
        limitations="limitations",
        authority="primary",
        page_start=1,
        page_end=1,
        section="section",
        text=text,
        word_count=len(text.split()),
    )


def test_dense_index_build_load_and_search(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            _chunk(chunk_id="a", document_id="readme", text="RUL definition"),
            _chunk(chunk_id="b", document_id="faa", text="engine inspection"),
            _chunk(chunk_id="c", document_id="readme", text="test trajectories"),
        ],
    )
    index_dir = tmp_path / "index"
    manifest = build_index(
        chunks_path=chunks_path,
        index_directory=index_dir,
        embedder=_FakeEmbedder(),
    )

    assert manifest["chunk_count"] == 3
    loaded = DenseIndex.load(index_dir)
    results = loaded.search(
        "RUL definition",
        embedder=_FakeEmbedder(),
        top_k=2,
        max_per_document=1,
    )
    assert [result.document_id for result in results] == ["readme", "faa"]
    assert results[0].chunk_id == "a"
    assert results[0].dense_score == pytest.approx(1.0)
    assert results[0].lexical_score > 0
    assert results[0].score > results[1].score


def test_query_expansion_adds_bilingual_domain_terms(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms.toml"
    terms_path.write_text(
        '\n'.join(
            [
                "schema_version = 1",
                "[terms]",
                '\"panne\" = [\"fault\", \"failure\"]',
                '\"durée de vie restante\" = [\"remaining useful life\", \"RUL\"]',
            ]
        ),
        encoding="utf-8",
    )
    terms = load_query_terms(terms_path)
    expanded = expand_query("Quelle panne limite la durée de vie restante ?", terms)
    assert "fault" in expanded
    assert "failure" in expanded
    assert "remaining useful life" in expanded
    assert "RUL" in expanded


def test_versioned_evaluation_set_has_answerable_and_abstention_cases() -> None:
    questions = load_questions("data/evaluation/rag/questions_v1.jsonl")
    assert len(questions) == 30
    assert sum(question.answerable for question in questions) == 25
    assert sum(not question.answerable for question in questions) == 5
