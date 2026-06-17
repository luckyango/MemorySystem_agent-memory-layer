from __future__ import annotations

from agent_memory.schemas import MemoryContext


class ContextBuilder:
    """Format retrieved memory into a prompt-ready context block."""

    def build(self, context: MemoryContext, *, max_items: int = 8) -> str:
        lines = ["## Retrieved Memory Context", f"Query: {context.query}", ""]

        if context.memories:
            lines.append("### Structured Memories")
            for result in context.memories[:max_items]:
                memory = result.memory
                lines.append(
                    "- "
                    f"[{memory.scope_type}:{memory.scope_id} | {memory.category} | "
                    f"score={result.score:.2f} | confidence={memory.confidence:.2f}] "
                    f"{memory.content}"
                )
            lines.append("")

        if context.recall_messages:
            lines.append("### Raw Recall Messages")
            for result in context.recall_messages[:max_items]:
                message = result.message
                lines.append(
                    "- "
                    f"[{message.role} | score={result.score:.2f} | id={message.id}] "
                    f"{message.content}"
                )
            lines.append("")

        if not context.memories and not context.recall_messages:
            lines.append("No relevant memories found.")

        return "\n".join(lines).rstrip()
