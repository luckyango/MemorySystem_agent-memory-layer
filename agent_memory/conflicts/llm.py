from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_memory.conflicts.rule_based import RuleBasedConflictResolver
from agent_memory.schemas import MemoryCandidate, MemoryItem, MemoryWriteDecision


class MemoryWriteDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["insert", "ignore", "update"]
    reason: str = Field(min_length=1)
    existing_memory_id: str | None = None
    merged_content: str | None = None

    @model_validator(mode="after")
    def validate_action_fields(self):
        if self.action in {"ignore", "update"} and not self.existing_memory_id:
            raise ValueError("ignore/update requires existing_memory_id")
        if self.action == "update" and not self.merged_content:
            raise ValueError("update requires merged_content")
        return self


class LLMConflictResolver:
    """LLM-backed memory write conflict resolver with rule-based fallback."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        fallback: Any | None = None,
    ):
        self.client = client or self._default_client()
        self.model = model
        self.fallback = fallback or RuleBasedConflictResolver()

    def decide(
        self,
        *,
        candidate: MemoryCandidate,
        related_memories: list[MemoryItem],
    ) -> MemoryWriteDecision:
        try:
            parsed = self._call_llm(candidate=candidate, related_memories=related_memories)
            return self._to_decision(
                parsed=parsed,
                candidate=candidate,
                related_memories=related_memories,
            )
        except Exception:
            return self.fallback.decide(candidate=candidate, related_memories=related_memories)

    def _call_llm(
        self,
        *,
        candidate: MemoryCandidate,
        related_memories: list[MemoryItem],
    ) -> MemoryWriteDecisionModel:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide whether a candidate memory should be inserted, ignored, or used "
                        "to update an existing memory. Ignore duplicates and one-off temporary "
                        "overrides. Update when the candidate refines or replaces an existing "
                        "durable memory. Insert when it is genuinely new."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "candidate": {
                                "category": candidate.category,
                                "content": candidate.content,
                                "confidence": candidate.confidence,
                                "importance": candidate.importance,
                                "entities": candidate.entities,
                                "metadata": candidate.metadata,
                            },
                            "related_memories": [
                                {
                                    "id": memory.id,
                                    "category": memory.category,
                                    "content": memory.content,
                                    "confidence": memory.confidence,
                                    "importance": memory.importance,
                                    "entities": memory.entities,
                                }
                                for memory in related_memories
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            response_format=MemoryWriteDecisionModel,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no conflict decision.")
        return parsed

    def _to_decision(
        self,
        *,
        parsed: MemoryWriteDecisionModel,
        candidate: MemoryCandidate,
        related_memories: list[MemoryItem],
    ) -> MemoryWriteDecision:
        memory_by_id = {memory.id: memory for memory in related_memories}
        existing = memory_by_id.get(parsed.existing_memory_id or "")
        if parsed.action in {"ignore", "update"} and existing is None:
            raise ValueError("LLM referenced an unknown existing_memory_id.")
        return MemoryWriteDecision(
            action=parsed.action,
            candidate=candidate,
            reason=parsed.reason,
            existing_memory=existing,
            merged_content=parsed.merged_content,
        )

    @staticmethod
    def _default_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai package or pass a compatible client to LLMConflictResolver."
            ) from exc
        return OpenAI()
