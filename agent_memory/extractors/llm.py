from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_memory.schemas import MemoryCandidate, MemoryCategory, MemoryItem, ScopeResolution


class MemoryCandidateModel(BaseModel):
    """Structured output model for one candidate memory."""

    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    content: str = Field(min_length=8, description="Concise third-person memory sentence.")
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    evidence_quote: str = Field(
        min_length=3,
        description="Exact short quote from the source user message that supports this memory.",
    )
    entities: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_must_be_specific(cls, value: str) -> str:
        normalized = " ".join(value.casefold().split())
        vague_values = {
            "the user said something important.",
            "the user provided project information.",
            "the user shared a preference.",
            "important information about the user.",
        }
        if normalized in vague_values:
            raise ValueError("memory content is too vague")
        return value.strip()

    @field_validator("entities")
    @classmethod
    def entities_must_be_nonempty_strings(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class MemoryExtractionResult(BaseModel):
    """Structured output model for memory extraction."""

    model_config = ConfigDict(extra="forbid")

    should_save: bool
    reason: str = Field(min_length=1)
    candidates: list[MemoryCandidateModel] = Field(default_factory=list)


class LLMMemoryExtractor:
    """LLM-backed extractor using API-native structured outputs and semantic validation."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        fallback: Any | None = None,
        max_retries: int = 2,
    ):
        self.client = client or self._default_client()
        self.model = model
        self.fallback = fallback
        self.max_retries = max_retries

    def extract(
        self,
        *,
        text: str,
        scope: ScopeResolution,
        related_memories: list[MemoryItem] | None = None,
    ) -> list[MemoryCandidate]:
        try:
            result = self._call_llm(text=text, scope=scope, related_memories=related_memories or [])
            return self._to_candidates(result=result, source_text=text)
        except Exception:
            if self.fallback is None:
                raise
            return self.fallback.extract(
                text=text,
                scope=scope,
                related_memories=related_memories,
            )

    def _call_llm(
        self,
        *,
        text: str,
        scope: ScopeResolution,
        related_memories: list[MemoryItem],
    ) -> MemoryExtractionResult:
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract durable memories for an AI agent memory system. "
                            "Use the provided schema. Save only stable facts, preferences, goals, "
                            "constraints, profile details, and project context. Ignore greetings, "
                            "short acknowledgements, temporary wording, and one-off requests. "
                            "Do not invent facts. Every candidate must include an exact evidence_quote "
                            "copied from the source user message."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(
                            text=text,
                            scope=scope,
                            related_memories=related_memories,
                            last_error=last_error,
                        ),
                    },
                ],
                response_format=MemoryExtractionResult,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                last_error = "Model returned no parsed structured output."
                continue
            semantic_error = self._semantic_error(parsed, source_text=text)
            if semantic_error is None:
                return parsed
            last_error = semantic_error

        raise ValueError(last_error or "LLM memory extraction failed semantic validation.")

    def _build_prompt(
        self,
        *,
        text: str,
        scope: ScopeResolution,
        related_memories: list[MemoryItem],
        last_error: str | None,
    ) -> str:
        related = [
            {
                "id": memory.id,
                "category": memory.category,
                "content": memory.content,
                "scope_type": memory.scope_type,
                "scope_id": memory.scope_id,
            }
            for memory in related_memories[:8]
        ]
        payload: dict[str, Any] = {
            "user_message": text,
            "resolved_scope": {
                "kind": scope.kind,
                "scope_type": scope.scope_type,
                "scope_id": scope.scope_id,
                "reason": scope.reason,
            },
            "related_memories": related,
            "allowed_categories": [
                "profile",
                "preference",
                "project",
                "goal",
                "fact",
                "constraint",
                "relationship",
                "task_state",
            ],
            "business_rules": [
                "Each candidate must be directly supported by evidence_quote.",
                "Do not turn tentative language into a stronger fact.",
                "Do not save information already present in related_memories unless it updates or refines it.",
                "Use should_save=false when the message has no durable memory value.",
            ],
        }
        if last_error:
            payload["previous_validation_error"] = last_error
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _semantic_error(self, result: MemoryExtractionResult, *, source_text: str) -> str | None:
        if not result.should_save and result.candidates:
            return "should_save is false but candidates is not empty."
        for candidate in result.candidates:
            if not self._quote_supported(candidate.evidence_quote, source_text):
                return (
                    "Candidate evidence_quote is not supported by the source message: "
                    f"{candidate.evidence_quote!r}"
                )
            if candidate.category == "project" and not candidate.entities:
                return "Project memories must include at least one entity."
        return None

    def _to_candidates(
        self,
        *,
        result: MemoryExtractionResult,
        source_text: str,
    ) -> list[MemoryCandidate]:
        if not result.should_save:
            return []
        candidates: list[MemoryCandidate] = []
        for candidate in result.candidates:
            if not self._quote_supported(candidate.evidence_quote, source_text):
                continue
            candidates.append(
                MemoryCandidate(
                    category=candidate.category,
                    content=candidate.content,
                    confidence=candidate.confidence,
                    importance=candidate.importance,
                    entities=candidate.entities,
                    metadata={
                        **candidate.metadata,
                        "extractor": "llm",
                        "model": self.model,
                        "evidence_quote": candidate.evidence_quote,
                    },
                )
            )
        return candidates

    @staticmethod
    def _quote_supported(quote: str, source_text: str) -> bool:
        normalized_quote = " ".join(quote.casefold().split())
        normalized_source = " ".join(source_text.casefold().split())
        return normalized_quote in normalized_source

    @staticmethod
    def _default_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai package or pass a compatible client to LLMMemoryExtractor."
            ) from exc
        return OpenAI()


StructuredOutputMode = Literal["openai_pydantic_parse"]
