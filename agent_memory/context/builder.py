from __future__ import annotations

from agent_memory.schemas import ContextBudget, MemoryContext


class ContextBuilder:
    """Format retrieved memory into a prompt-ready context block."""

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()

    def build(
        self,
        context: MemoryContext,
        *,
        max_items: int = 8,
        budget: ContextBudget | None = None,
    ) -> str:
        active_budget = budget or self.budget
        lines = ["## Retrieved Memory Context", f"Query: {context.query}", ""]

        session_lines = self._session_lines(context)
        if session_lines:
            lines.extend(self._fit_lines(session_lines, active_budget.session_chars))
            lines.append("")

        if context.memories:
            memory_lines = ["### Structured Memories"]
            for result in context.memories[:max_items]:
                memory = result.memory
                memory_lines.append(
                    "- "
                    f"[{memory.scope_type}:{memory.scope_id} | {memory.category} | "
                    f"score={result.score:.2f} | confidence={memory.confidence:.2f}] "
                    f"{memory.content}"
                )
            lines.extend(self._fit_lines(memory_lines, active_budget.memory_chars))
            lines.append("")

        if context.recall_messages:
            recall_lines = ["### Raw Recall Messages"]
            for result in context.recall_messages[:max_items]:
                message = result.message
                recall_lines.append(
                    "- "
                    f"[{message.role} | score={result.score:.2f} | id={message.id}] "
                    f"{message.content}"
                )
            lines.extend(self._fit_lines(recall_lines, active_budget.recall_chars))
            lines.append("")

        if not context.memories and not context.recall_messages and not session_lines:
            lines.append("No relevant memories found.")

        return self._fit_text("\n".join(lines).rstrip(), active_budget.max_context_chars)

    def _session_lines(self, context: MemoryContext) -> list[str]:
        state = context.session_state
        if state is None and context.active_project is None:
            return []
        lines = ["### Session State"]
        if context.active_project is not None:
            lines.append(f"- Active Project: {context.active_project.name} ({context.active_project.id})")
        elif state and state.active_project_id:
            lines.append(f"- Active Project ID: {state.active_project_id}")
        if state and state.current_task:
            lines.append(f"- Current Task: {state.current_task}")
        if state and state.temporary_constraints:
            lines.append("- Temporary Constraints:")
            for item in state.temporary_constraints:
                lines.append(f"  - {item}")
        return lines

    @staticmethod
    def _fit_lines(lines: list[str], max_chars: int) -> list[str]:
        kept = []
        total = 0
        for line in lines:
            projected = total + len(line) + 1
            if kept and projected > max_chars:
                kept.append("- [truncated]")
                break
            kept.append(line)
            total = projected
        return kept

    @staticmethod
    def _fit_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        suffix = "\n[context truncated]"
        return text[: max(0, max_chars - len(suffix))].rstrip() + suffix
