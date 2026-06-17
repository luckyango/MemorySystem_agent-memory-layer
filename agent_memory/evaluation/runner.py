from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_memory import MemoryLayer


@dataclass(slots=True)
class EvalCase:
    name: str
    inputs: list[str]
    query: str
    expected_memory_substrings: list[str] = field(default_factory=list)
    expected_context_substrings: list[str] = field(default_factory=list)
    expected_memory_count: int | None = None


@dataclass(slots=True)
class EvalResult:
    name: str
    passed: bool
    inputs: list[str]
    query: str
    context: str
    saved_memories: list[str]
    failures: list[str] = field(default_factory=list)


class MemoryEvalRunner:
    """Run deterministic memory behavior fixtures without a live LLM."""

    def run_case(self, case: EvalCase) -> EvalResult:
        with TemporaryDirectory() as tmpdir:
            memory = MemoryLayer(Path(tmpdir) / "memory.sqlite3")
            saved_memories = []
            for item in case.inputs:
                _, saved = memory.process_user_message(
                    user_id="eval_user",
                    session_id="eval_session",
                    content=item,
                )
                saved_memories.extend(memory_item.content for memory_item in saved)

            context = memory.build_context(
                user_id="eval_user",
                query=case.query,
            )
            failures = self._check(case=case, saved_memories=saved_memories, context=context)
            return EvalResult(
                name=case.name,
                passed=not failures,
                inputs=case.inputs,
                query=case.query,
                context=context,
                saved_memories=saved_memories,
                failures=failures,
            )

    def run(self, cases: list[EvalCase]) -> list[EvalResult]:
        return [self.run_case(case) for case in cases]

    @staticmethod
    def _check(case: EvalCase, saved_memories: list[str], context: str) -> list[str]:
        failures = []
        memory_blob = "\n".join(saved_memories)
        for expected in case.expected_memory_substrings:
            if expected not in memory_blob:
                failures.append(f"missing saved memory substring: {expected!r}")
        for expected in case.expected_context_substrings:
            if expected not in context:
                failures.append(f"missing context substring: {expected!r}")
        if case.expected_memory_count is not None and len(saved_memories) != case.expected_memory_count:
            failures.append(
                f"expected {case.expected_memory_count} saved memories, got {len(saved_memories)}"
            )
        return failures
