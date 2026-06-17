from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_memory.resolvers.scope_resolver import RuleBasedScopeResolver
from agent_memory.schemas import Project, ScopeResolution
from agent_memory.stores import SQLiteProjectStore


class ScopeResolutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["existing_project", "new_project", "user", "session", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    project_id: str | None = None
    suggested_project_name: str | None = None
    matched_text: str | None = None

    @model_validator(mode="after")
    def validate_fields_for_kind(self):
        if self.kind == "existing_project" and not self.project_id:
            raise ValueError("existing_project requires project_id")
        if self.kind == "new_project" and not self.suggested_project_name:
            raise ValueError("new_project requires suggested_project_name")
        return self


class LLMScopeResolver:
    """LLM-backed scope resolver with rule-based fallback."""

    def __init__(
        self,
        *,
        project_store: SQLiteProjectStore,
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        fallback: Any | None = None,
    ):
        self.project_store = project_store
        self.client = client or self._default_client()
        self.model = model
        self.fallback = fallback or RuleBasedScopeResolver(project_store)

    def resolve(self, *, user_id: str, text: str) -> ScopeResolution:
        try:
            projects = self.project_store.list_projects(user_id=user_id, status="active")
            parsed = self._call_llm(text=text, projects=projects)
            return self._to_scope_resolution(user_id=user_id, parsed=parsed, projects=projects)
        except Exception:
            return self.fallback.resolve(user_id=user_id, text=text)

    def resolve_or_create_project(self, *, user_id: str, text: str) -> ScopeResolution:
        resolution = self.resolve(user_id=user_id, text=text)
        if resolution.kind != "new_project" or not resolution.suggested_project_name:
            return resolution

        project = self.project_store.add_project(
            Project(
                user_id=user_id,
                name=resolution.suggested_project_name,
                aliases=[resolution.suggested_project_name],
            )
        )
        return ScopeResolution(
            kind="existing_project",
            scope_type="project",
            scope_id=project.id,
            project=project,
            confidence=resolution.confidence,
            reason="Created a new project from LLM scope resolution.",
            matched_text=resolution.suggested_project_name,
        )

    def _call_llm(self, *, text: str, projects: list[Project]) -> ScopeResolutionModel:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Resolve the scope of a user message for an AI agent memory system. "
                        "Choose existing_project when the message refers to one listed project, "
                        "new_project when it introduces a new durable project, user for global "
                        "profile/preferences/goals, session for temporary task state, and unknown "
                        "when there is not enough information."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_message": text,
                            "projects": [
                                {
                                    "id": project.id,
                                    "name": project.name,
                                    "aliases": project.aliases,
                                    "description": project.description,
                                }
                                for project in projects
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            response_format=ScopeResolutionModel,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no scope resolution.")
        return parsed

    def _to_scope_resolution(
        self,
        *,
        user_id: str,
        parsed: ScopeResolutionModel,
        projects: list[Project],
    ) -> ScopeResolution:
        project_by_id = {project.id: project for project in projects}
        if parsed.kind == "existing_project":
            project = project_by_id.get(parsed.project_id or "")
            if project is None:
                raise ValueError("LLM referenced an unknown project_id.")
            return ScopeResolution(
                kind="existing_project",
                scope_type="project",
                scope_id=project.id,
                confidence=parsed.confidence,
                reason=parsed.reason,
                project=project,
                matched_text=parsed.matched_text or project.name,
            )
        if parsed.kind == "new_project":
            return ScopeResolution(
                kind="new_project",
                scope_type="project",
                scope_id="",
                confidence=parsed.confidence,
                reason=parsed.reason,
                suggested_project_name=parsed.suggested_project_name,
                matched_text=parsed.matched_text or parsed.suggested_project_name,
            )
        if parsed.kind == "user":
            return ScopeResolution(
                kind="user",
                scope_type="user",
                scope_id="global",
                confidence=parsed.confidence,
                reason=parsed.reason,
            )
        if parsed.kind == "session":
            return ScopeResolution(
                kind="session",
                scope_type="session",
                scope_id="current",
                confidence=parsed.confidence,
                reason=parsed.reason,
            )
        return ScopeResolution(
            kind="unknown",
            scope_type="session",
            scope_id="current",
            confidence=parsed.confidence,
            reason=parsed.reason,
        )

    @staticmethod
    def _default_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai package or pass a compatible client to LLMScopeResolver."
            ) from exc
        return OpenAI()
