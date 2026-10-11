"""Validated document extraction and citation-preserving chunking for the RAG corpus."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader


class CorpusError(ValueError):
    """Raised when a corpus source or manifest is invalid."""


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    path: Path
    title: str
    source_url: str
    file_format: str
    sha256: str
    size_bytes: int
    page_count: int
    included_pages: tuple[int, ...]
    roles: tuple[str, ...]
    scope: str
    limitations: str
    authority: str
    encoding: str = "utf-8"


@dataclass(frozen=True)
class ExtractedSection:
    document_id: str
    title: str
    source_url: str
    roles: tuple[str, ...]
    scope: str
    limitations: str
    authority: str
    page: int | None
    section: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source_url: str
    roles: tuple[str, ...]
    scope: str
    limitations: str
    authority: str
    page_start: int | None
    page_end: int | None
    section: str
    text: str
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Chunk:
        normalized = dict(value)
        normalized["roles"] = tuple(normalized.get("roles", ()))
        return cls(**normalized)

    def embedding_text(self) -> str:
        return (
            f"Document: {self.title}\n"
            f"Section: {self.section}\n"
            f"{self.text}"
        )


_PAGE_FOOTER_PATTERNS = (
    re.compile(r"^NASA/TM[—-].*$", re.IGNORECASE),
    re.compile(
        r"^(?:\d+\s+)?ANNUAL CONFERENCE OF THE PROGNOSTICS.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^FAA-H-8083-32B$", re.IGNORECASE),
    re.compile(r"^\d+$"),
    re.compile(r"^\d+-\d+$"),
)

_NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.)\s+[A-Z][^.!?]{1,100}$")
_TABLE_OR_FIGURE = re.compile(r"^(?:TABLE|Table|Figure)\s+\d+(?:\.\d+)?[.\-—:]?.{0,120}$")
_README_HEADING = re.compile(r"^(?:Data Set:\s*FD\d{3}|Experimental Scenario|Reference)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_page_ranges(values: Iterable[str | int], *, page_count: int) -> tuple[int, ...]:
    pages: set[int] = set()
    for value in values:
        if isinstance(value, int):
            start = end = value
        else:
            match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", value)
            if not match:
                raise CorpusError(f"Invalid PDF page range: {value!r}")
            start = int(match.group(1))
            end = int(match.group(2) or start)
        if start < 1 or end < start or end > page_count:
            raise CorpusError(
                f"PDF page range {value!r} is outside the document (1-{page_count})"
            )
        pages.update(range(start, end + 1))
    return tuple(sorted(pages))


def _resolve_source_path(raw_path: str, *, project_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def load_corpus_manifest(path: str | Path, *, project_root: str | Path = ".") -> list[DocumentSpec]:
    manifest_path = Path(path)
    root = Path(project_root).resolve()
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise CorpusError("Unsupported corpus manifest schema_version")

    documents: list[DocumentSpec] = []
    seen_ids: set[str] = set()
    for raw in data.get("documents", []):
        document_id = str(raw["id"])
        if document_id in seen_ids:
            raise CorpusError(f"Duplicate document id: {document_id}")
        seen_ids.add(document_id)

        source_path = _resolve_source_path(str(raw["path"]), project_root=root)
        if not source_path.is_file():
            raise CorpusError(f"Corpus file is missing: {source_path}")
        actual_size = source_path.stat().st_size
        expected_size = int(raw["size_bytes"])
        if actual_size != expected_size:
            raise CorpusError(
                f"Size mismatch for {document_id}: expected {expected_size}, got {actual_size}"
            )
        actual_hash = sha256_file(source_path)
        expected_hash = str(raw["sha256"]).lower()
        if actual_hash != expected_hash:
            raise CorpusError(f"SHA-256 mismatch for {document_id}")

        file_format = str(raw["format"]).lower()
        page_count = int(raw.get("page_count", 0))
        included_values = raw.get("included_pdf_pages")
        excluded_values = {int(value) for value in raw.get("excluded_pdf_pages", [])}
        if file_format == "pdf":
            reader = PdfReader(source_path)
            actual_pages = len(reader.pages)
            if actual_pages != page_count:
                raise CorpusError(
                    f"Page-count mismatch for {document_id}: expected {page_count}, "
                    f"got {actual_pages}"
                )
            if included_values:
                included_pages = parse_page_ranges(included_values, page_count=page_count)
            else:
                included_pages = tuple(
                    page for page in range(1, page_count + 1) if page not in excluded_values
                )
        elif file_format == "txt":
            included_pages = ()
        else:
            raise CorpusError(f"Unsupported corpus format for {document_id}: {file_format}")

        documents.append(
            DocumentSpec(
                document_id=document_id,
                path=source_path,
                title=str(raw["title"]),
                source_url=str(raw["source_url"]),
                file_format=file_format,
                sha256=expected_hash,
                size_bytes=expected_size,
                page_count=page_count,
                included_pages=included_pages,
                roles=tuple(str(value) for value in raw.get("roles", [])),
                scope=str(raw.get("scope", "")),
                limitations=str(raw.get("limitations", "")),
                authority=str(raw.get("authority", "")),
                encoding=str(raw.get("encoding", "utf-8")),
            )
        )
    if not documents:
        raise CorpusError("The corpus manifest contains no documents")
    return documents


def _is_heading(line: str) -> bool:
    if _README_HEADING.fullmatch(line):
        return True
    if _NUMBERED_HEADING.fullmatch(line) or _TABLE_OR_FIGURE.fullmatch(line):
        return True
    letters = [character for character in line if character.isalpha()]
    if 3 <= len(letters) and len(line) <= 100:
        uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
        if uppercase_ratio >= 0.9 and not line.endswith(('.', ';', ',')):
            return True
    return False


def _clean_lines(raw_text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", raw_text.replace("\u00ad", ""))
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=[a-z])", "", text)
    cleaned: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if any(pattern.fullmatch(line) for pattern in _PAGE_FOOTER_PATTERNS):
            continue
        cleaned.append(line)
    return cleaned


def split_into_sections(
    raw_text: str,
    *,
    fallback_section: str,
) -> list[tuple[str, str]]:
    lines = _clean_lines(raw_text)
    sections: list[tuple[str, str]] = []
    section = fallback_section
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        text = re.sub(r"\s+", " ", " ".join(body)).strip()
        if text:
            sections.append((section, text))
        body = []

    for line in lines:
        if _is_heading(line):
            flush()
            section = line
        else:
            body.append(line)
    flush()
    return sections


def extract_sections(document: DocumentSpec) -> list[ExtractedSection]:
    sections: list[ExtractedSection] = []
    if document.file_format == "txt":
        raw_text = document.path.read_text(encoding=document.encoding)
        page_sections = split_into_sections(raw_text, fallback_section=document.title)
        pages_and_sections = ((None, section, text) for section, text in page_sections)
    else:
        reader = PdfReader(document.path)
        extracted: list[tuple[int, str, str]] = []
        for page_number in document.included_pages:
            raw_text = reader.pages[page_number - 1].extract_text() or ""
            if not raw_text.strip():
                raise CorpusError(
                    f"No extractable text for {document.document_id}, PDF page {page_number}"
                )
            fallback = f"PDF page {page_number}"
            for section, text in split_into_sections(raw_text, fallback_section=fallback):
                extracted.append((page_number, section, text))
        pages_and_sections = iter(extracted)

    for page, section, text in pages_and_sections:
        if section.upper() == "CONTENTS":
            continue
        sections.append(
            ExtractedSection(
                document_id=document.document_id,
                title=document.title,
                source_url=document.source_url,
                roles=document.roles,
                scope=document.scope,
                limitations=document.limitations,
                authority=document.authority,
                page=page,
                section=section,
                text=text,
            )
        )
    return sections


def _window_words(
    text: str,
    *,
    max_words: int,
    overlap_words: int,
    min_words: int,
) -> list[str]:
    if max_words <= 0:
        raise CorpusError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise CorpusError("overlap_words must be between 0 and max_words - 1")
    if min_words <= 0 or min_words > max_words:
        raise CorpusError("min_words must be between 1 and max_words")

    words = text.split()
    if len(words) <= max_words:
        # Short sections often contain compact tables or dataset counts. Dropping them would
        # remove high-value facts such as the FD001-FD004 train/test sizes.
        return [text] if words else []
    chunks: list[str] = []
    step = max_words - overlap_words
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        window = words[start:end]
        if len(window) < min_words and chunks:
            break
        chunks.append(" ".join(window))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_sections(
    sections: Iterable[ExtractedSection],
    *,
    max_words: int,
    overlap_words: int,
    min_words: int,
) -> list[Chunk]:
    merged_sections: list[ExtractedSection] = []
    pending: ExtractedSection | None = None
    for extracted in sections:
        is_compact_dataset_table = extracted.section.startswith("Data Set: FD")
        is_short = len(extracted.text.split()) < min_words and not is_compact_dataset_table

        if pending is not None:
            if pending.document_id == extracted.document_id and pending.page == extracted.page:
                extracted = replace(
                    extracted,
                    text=f"{pending.section}. {pending.text} {extracted.text}".strip(),
                )
            pending = None
            is_short = len(extracted.text.split()) < min_words and not is_compact_dataset_table

        if is_short:
            if (
                merged_sections
                and merged_sections[-1].document_id == extracted.document_id
                and merged_sections[-1].page == extracted.page
            ):
                previous = merged_sections[-1]
                merged_sections[-1] = replace(
                    previous,
                    text=f"{previous.text} {extracted.section}. {extracted.text}".strip(),
                )
            else:
                pending = extracted
            continue
        merged_sections.append(extracted)

    # A final isolated fragment is usually a page number, plot label or split caption. It is
    # intentionally discarded when it cannot be attached to another passage from the same page.

    chunks: list[Chunk] = []
    for extracted in merged_sections:
        windows = _window_words(
            extracted.text,
            max_words=max_words,
            overlap_words=overlap_words,
            min_words=min_words,
        )
        for sequence, text in enumerate(windows, start=1):
            identity = "\0".join(
                [
                    extracted.document_id,
                    str(extracted.page or "none"),
                    extracted.section,
                    str(sequence),
                    text,
                ]
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=extracted.document_id,
                    title=extracted.title,
                    source_url=extracted.source_url,
                    roles=extracted.roles,
                    scope=extracted.scope,
                    limitations=extracted.limitations,
                    authority=extracted.authority,
                    page_start=extracted.page,
                    page_end=extracted.page,
                    section=extracted.section,
                    text=text,
                    word_count=len(text.split()),
                )
            )
    if not chunks:
        raise CorpusError("Chunking produced no passages")
    return chunks


def write_chunks(path: str | Path, chunks: Iterable[Chunk]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (TypeError, ValueError, KeyError) as exc:
                raise CorpusError(f"Invalid chunk at line {line_number}") from exc
    if not chunks:
        raise CorpusError("The chunk file is empty")
    return chunks
