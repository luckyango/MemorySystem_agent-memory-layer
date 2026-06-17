from __future__ import annotations

from agent_memory.schemas import MemoryCandidate, MemoryItem, MemoryWriteDecision


class RuleBasedConflictResolver:
    """Deterministic baseline for duplicate and simple update detection."""

    def decide(
        self,
        *,
        candidate: MemoryCandidate,
        related_memories: list[MemoryItem],
    ) -> MemoryWriteDecision:
        normalized_candidate = self._normalize(candidate.content)
        same_category = [
            memory for memory in related_memories if memory.category == candidate.category
        ]

        for memory in same_category:
            if self._normalize(memory.content) == normalized_candidate:
                return MemoryWriteDecision(
                    action="ignore",
                    candidate=candidate,
                    existing_memory=memory,
                    reason="Candidate duplicates an existing memory.",
                )

        entity_matches = [
            memory
            for memory in same_category
            if self._has_entity_overlap(candidate.entities, memory.entities)
        ]
        if entity_matches:
            existing = entity_matches[0]
            merged_content = self._merge_content(existing.content, candidate.content)
            return MemoryWriteDecision(
                action="update",
                candidate=candidate,
                existing_memory=existing,
                merged_content=merged_content,
                reason="Candidate shares entities with an existing memory in the same category.",
            )

        return MemoryWriteDecision(
            action="insert",
            candidate=candidate,
            reason="Candidate is new for this scope and category.",
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split()).strip(" .,:;")

    @staticmethod
    def _has_entity_overlap(left: list[str], right: list[str]) -> bool:
        left_set = {item.casefold() for item in left if item.strip()}
        right_set = {item.casefold() for item in right if item.strip()}
        return bool(left_set and right_set and left_set.intersection(right_set))

    @staticmethod
    def _merge_content(existing: str, candidate: str) -> str:
        if candidate.casefold() in existing.casefold():
            return existing
        if existing.casefold() in candidate.casefold():
            return candidate
        return f"{existing} {candidate}"
