"""Scope and project resolution components."""

from agent_memory.resolvers.scope_resolver import RuleBasedScopeResolver

__all__ = ["LLMScopeResolver", "RuleBasedScopeResolver"]


def __getattr__(name: str):
    if name == "LLMScopeResolver":
        from agent_memory.resolvers.llm import LLMScopeResolver

        return LLMScopeResolver
    raise AttributeError(name)
