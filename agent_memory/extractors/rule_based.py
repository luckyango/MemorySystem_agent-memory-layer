from __future__ import annotations

import re

from agent_memory.schemas import MemoryCandidate, MemoryItem, ScopeResolution


class RuleBasedMemoryExtractor:
    """Extract obvious long-term memory candidates with deterministic rules."""

    TECH_PATTERN = re.compile(
        r"(?:uses?|using|built with|powered by)\s+([A-Za-z0-9_.+-]+)",
        re.I,
    )
    PREFERENCE_PATTERNS = (
        re.compile(r"i prefer\s+(.+?)(?:\.|$)", re.I),
        re.compile(r"i like\s+(.+?)(?:\.|$)", re.I),
    )
    IDENTITY_PATTERNS = (
        re.compile(r"i am\s+(?:a|an)?\s*(.+?)(?:\.|$)", re.I),
    )

    def extract(
        self,
        *,
        text: str,
        scope: ScopeResolution,
        related_memories: list[MemoryItem] | None = None,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        candidates.extend(self._extract_preferences(text))
        candidates.extend(self._extract_identity(text))
        if scope.scope_type == "project":
            candidates.extend(self._extract_project_tech(text))
        return candidates

    def _extract_preferences(self, text: str) -> list[MemoryCandidate]:
        candidates = []
        for pattern in self.PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                preference = self._clean(match.group(1))
                if preference:
                    candidates.append(
                        MemoryCandidate(
                            category="preference",
                            content=f"The user prefers {preference}.",
                            confidence=0.85,
                            importance=0.8,
                            entities=[],
                            metadata={"extractor": "rule_based"},
                        )
                    )
        return candidates

    def _extract_identity(self, text: str) -> list[MemoryCandidate]:
        candidates = []
        for pattern in self.IDENTITY_PATTERNS:
            for match in pattern.finditer(text):
                identity = self._clean(match.group(1))
                if identity:
                    candidates.append(
                        MemoryCandidate(
                            category="profile",
                            content=f"The user is {identity}.",
                            confidence=0.75,
                            importance=0.7,
                            metadata={"extractor": "rule_based"},
                        )
                    )
        return candidates

    def _extract_project_tech(self, text: str) -> list[MemoryCandidate]:
        candidates = []
        for match in self.TECH_PATTERN.finditer(text):
            tech = self._clean(match.group(1))
            if tech:
                candidates.append(
                    MemoryCandidate(
                        category="project",
                        content=f"The project uses {tech}.",
                        confidence=0.8,
                        importance=0.7,
                        entities=[tech],
                        metadata={"extractor": "rule_based", "field": "technical_stack"},
                    )
                )
        return candidates

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip(" .,:;")
