from __future__ import annotations

from typing import Any

from agent_memory.memory_layer import MemoryLayer
from agent_memory.schemas import Message


class MemoryAgent:
    """Minimal agent loop that uses MemoryLayer for write and read paths."""

    def __init__(
        self,
        *,
        memory: MemoryLayer,
        client: Any,
        model: str = "gpt-4.1-mini",
        system_prompt: str | None = None,
    ):
        self.memory = memory
        self.client = client
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. Use the retrieved memory context when it is relevant. "
            "Do not claim memories that are not present in the context."
        )

    def chat(self, *, user_id: str, session_id: str, content: str) -> str:
        user_message, _ = self.memory.process_user_message(
            user_id=user_id,
            session_id=session_id,
            content=content,
        )
        session_state = self.memory.get_session_state(user_id=user_id, session_id=session_id)
        context_block = self.memory.build_context(
            user_id=user_id,
            query=content,
            session_id=session_id,
            scope_type="project" if session_state and session_state.active_project_id else None,
            scope_id=session_state.active_project_id if session_state else None,
        )
        recent_messages = self.memory.recall_store.list_messages(
            user_id=user_id,
            session_id=session_id,
            limit=8,
        )
        reply = self._call_model(
            context_block=context_block,
            recent_messages=recent_messages,
            user_message=user_message,
        )
        self.memory.record_message(
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=reply,
        )
        return reply

    def _call_model(
        self,
        *,
        context_block: str,
        recent_messages: list[Message],
        user_message: Message,
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": context_block},
        ]
        for message in recent_messages:
            if message.id == user_message.id:
                continue
            if message.role in {"user", "assistant", "system"}:
                messages.append({"role": message.role, "content": message.content})
        messages.append({"role": "user", "content": user_message.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content or ""
