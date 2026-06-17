"""Memory extraction components."""

from agent_memory.extractors.rule_based import RuleBasedMemoryExtractor

__all__ = ["LLMMemoryExtractor", "RuleBasedMemoryExtractor"]


def __getattr__(name: str):
    if name == "LLMMemoryExtractor":
        from agent_memory.extractors.llm import LLMMemoryExtractor

        return LLMMemoryExtractor
    raise AttributeError(name)
