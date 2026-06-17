"""Conflict resolution components for memory writes."""

from agent_memory.conflicts.rule_based import RuleBasedConflictResolver

__all__ = ["LLMConflictResolver", "RuleBasedConflictResolver"]


def __getattr__(name: str):
    if name == "LLMConflictResolver":
        from agent_memory.conflicts.llm import LLMConflictResolver

        return LLMConflictResolver
    raise AttributeError(name)
