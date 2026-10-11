"""Grounded local answer generation over retrieved technical passages."""

from __future__ import annotations

import argparse
import json
import re
import threading
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from industrial_maintenance.rag_corpus import CorpusError
from industrial_maintenance.rag_retrieval import SearchResult

ABSTENTION_MESSAGE = (
    "Je ne trouve pas de preuve suffisante dans la documentation disponible pour répondre "
    "à cette question."
)
GROUNDING_POLICY_VERSION = "procedure-guard-v1"

_PROCEDURE_REQUEST = re.compile(
    r"(?:r[ée]paration|r[ée]parer|remplacer|remplacement|pi[èe]ce|"
    r"intervention)\b",
    flags=re.IGNORECASE,
)
_PROCEDURE_EVIDENCE = re.compile(
    r"(?:repair|r[ée]paration|replace|remplacement|replacement|"
    r"maintenance procedure|corrective action)\b",
    flags=re.IGNORECASE,
)


class GenerationError(RuntimeError):
    """Raised when the local generator cannot produce a valid grounded decision."""


@dataclass(frozen=True)
class GenerationSettings:
    model_name: str
    model_revision: str
    device: str
    dtype: str
    max_input_tokens: int
    max_new_tokens: int
    top_k_passages: int


@dataclass(frozen=True)
class GeneratedDecision:
    answerable: bool
    answer: str
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    answerable: bool
    answer: str
    citations: tuple[SearchResult, ...]
    retrieval_version: str
    generation_version: str


class SearchProtocol(Protocol):
    @property
    def version(self) -> str: ...

    def search(self, query: str, *, top_k: int) -> list[SearchResult]: ...


class GeneratorProtocol(Protocol):
    @property
    def version(self) -> str: ...

    def generate(self, question: str, passages: Sequence[SearchResult]) -> str: ...


def read_generation_settings(path: str | Path) -> GenerationSettings:
    source = Path(path)
    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))["generation"]
        settings = GenerationSettings(
            model_name=str(data["model_name"]),
            model_revision=str(data["model_revision"]),
            device=str(data["device"]),
            dtype=str(data["dtype"]),
            max_input_tokens=int(data["max_input_tokens"]),
            max_new_tokens=int(data["max_new_tokens"]),
            top_k_passages=int(data["top_k_passages"]),
        )
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise CorpusError(f"Invalid generation configuration {source}: {exc}") from exc
    if not settings.model_name.strip() or not settings.model_revision.strip():
        raise CorpusError("Generation model name and revision must be non-empty")
    if settings.max_input_tokens < 256 or settings.max_new_tokens < 32:
        raise CorpusError("Generation token limits are too small")
    if not 1 <= settings.top_k_passages <= 10:
        raise CorpusError("generation.top_k_passages must be between 1 and 10")
    if settings.dtype not in {"auto", "float32", "bfloat16"}:
        raise CorpusError("generation.dtype must be auto, float32 or bfloat16")
    return settings


def _prompt(question: str, passages: Sequence[SearchResult]) -> list[dict[str, str]]:
    sources = []
    for index, passage in enumerate(passages, start=1):
        page = (
            "sans numéro de page"
            if passage.page_start is None
            else f"page PDF {passage.page_start}"
        )
        sources.append(
            f"[S{index}] {passage.title}, {page}, section {passage.section}\n"
            f"Limite de la source : {passage.limitations}\n"
            f"Passage : {passage.text}"
        )
    source_text = "\n\n".join(sources)
    system = (
        "Tu es un assistant de maintenance industrielle strictement documentaire. "
        "Réponds en français uniquement avec les informations explicitement présentes dans "
        "les sources. N'invente aucune correspondance de capteur, procédure, valeur, coût, "
        "date ou diagnostic. Si les sources ne suffisent pas, indique answerable=false. "
        "Retourne un unique objet JSON valide, sans markdown, avec exactement les champs "
        '"answerable" (booléen), "answer" (chaîne) et "citations" (liste de S1, S2...). '
        "Dans citations, écris uniquement les identifiants exacts, par exemple [\"S2\"], "
        "sans extrait ni commentaire. Examine toutes les sources et privilégie celle qui "
        "énonce directement la réponse. Si answerable=true, donne une seule phrase concise "
        "et au moins une citation. "
        "Si answerable=false, laisse answer vide et citations vide."
    )
    user = f"Question : {question}\n\nSources :\n{source_text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class LocalQwenGenerator:
    """Pinned Transformers generator using Qwen in deterministic non-thinking mode."""

    def __init__(
        self,
        settings: GenerationSettings,
        *,
        local_files_only: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise GenerationError(
                "torch and transformers are required for local answer generation"
            ) from exc
        self.settings = settings
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            revision=settings.model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        torch_dtype: Any = (
            "auto" if settings.dtype == "auto" else getattr(torch, settings.dtype)
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            revision=settings.model_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            dtype=torch_dtype,
        )
        self.model.to(settings.device)
        self.model.eval()
        self._lock = threading.RLock()

    @property
    def version(self) -> str:
        return (
            f"{self.settings.model_name.split('/')[-1]}-"
            f"{self.settings.model_revision[:12]}"
        )

    def generate(self, question: str, passages: Sequence[SearchResult]) -> str:
        messages = _prompt(question, passages)
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        ).to(self.settings.device)
        with self._lock, self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_ids = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def parse_generated_decision(raw: str, *, source_count: int) -> GeneratedDecision:
    match = re.search(r"\{.*\}", raw.strip(), flags=re.DOTALL)
    if match is None:
        raise GenerationError("Generator did not return a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GenerationError("Generator returned invalid JSON") from exc
    if set(payload) != {"answerable", "answer", "citations"}:
        raise GenerationError("Generator JSON has an invalid schema")
    answerable = payload["answerable"]
    answer = payload["answer"]
    citation_ids = payload["citations"]
    if not isinstance(answerable, bool) or not isinstance(answer, str):
        raise GenerationError("Generator JSON has invalid field types")
    if not isinstance(citation_ids, list) or any(
        not isinstance(citation_id, str) for citation_id in citation_ids
    ):
        raise GenerationError("Generator citations must be a string list")

    unique_ids = tuple(dict.fromkeys(citation_ids))
    allowed = {f"S{index}" for index in range(1, source_count + 1)}
    if any(citation_id not in allowed for citation_id in unique_ids):
        raise GenerationError("Generator cited a passage that was not retrieved")
    if answerable and (not answer.strip() or not unique_ids):
        raise GenerationError("An answer must contain text and at least one citation")
    if not answerable:
        return GeneratedDecision(
            answerable=False,
            answer=ABSTENTION_MESSAGE,
            citation_ids=(),
        )
    return GeneratedDecision(
        answerable=True,
        answer=answer.strip(),
        citation_ids=unique_ids,
    )


class GroundedAnswerService:
    def __init__(
        self,
        *,
        search: SearchProtocol,
        generator: GeneratorProtocol,
        top_k_passages: int,
    ) -> None:
        self.search_service = search
        self.generator = generator
        self.top_k_passages = top_k_passages

    @property
    def version(self) -> str:
        return f"{self.generator.version}-{GROUNDING_POLICY_VERSION}"

    @staticmethod
    def _requests_unsupported_procedure(
        question: str,
        passages: Sequence[SearchResult],
    ) -> bool:
        """Reject a requested maintenance prescription with no procedural evidence."""
        if _PROCEDURE_REQUEST.search(question) is None:
            return False
        return not any(_PROCEDURE_EVIDENCE.search(passage.text) for passage in passages)

    def ask(self, question: str) -> GroundedAnswer:
        passages = self.search_service.search(question, top_k=self.top_k_passages)
        if not passages:
            return GroundedAnswer(
                question=question,
                answerable=False,
                answer=ABSTENTION_MESSAGE,
                citations=(),
                retrieval_version=self.search_service.version,
                generation_version=self.version,
            )
        if self._requests_unsupported_procedure(question, passages):
            return GroundedAnswer(
                question=question,
                answerable=False,
                answer=ABSTENTION_MESSAGE,
                citations=(),
                retrieval_version=self.search_service.version,
                generation_version=self.version,
            )
        raw = self.generator.generate(question, passages)
        decision = parse_generated_decision(raw, source_count=len(passages))
        citations = tuple(
            passages[int(citation_id[1:]) - 1] for citation_id in decision.citation_ids
        )
        return GroundedAnswer(
            question=question,
            answerable=decision.answerable,
            answer=decision.answer,
            citations=citations,
            retrieval_version=self.search_service.version,
            generation_version=self.version,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rag.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download-model", help="Download and validate the pinned local generator")
    ask_parser = commands.add_parser("ask", help="Retrieve passages and generate a cited answer")
    ask_parser.add_argument("question")
    arguments = parser.parse_args()

    try:
        settings = read_generation_settings(arguments.config)
        generator = LocalQwenGenerator(
            settings,
            local_files_only=arguments.command != "download-model",
        )
        if arguments.command == "download-model":
            output: dict[str, Any] = {"status": "ready", "version": generator.version}
        else:
            from industrial_maintenance.document_search import DocumentSearchService

            service = GroundedAnswerService(
                search=DocumentSearchService(arguments.config),
                generator=generator,
                top_k_passages=settings.top_k_passages,
            )
            output = asdict(service.ask(arguments.question))
    except (CorpusError, GenerationError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
