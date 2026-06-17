"""Memory retrieval components."""

from agent_memory.retrievers.keyword import KeywordRetriever

__all__ = ["ChromaRetriever", "KeywordRetriever"]


def __getattr__(name: str):
    if name == "ChromaRetriever":
        from agent_memory.retrievers.chroma import ChromaRetriever

        return ChromaRetriever
    raise AttributeError(name)
