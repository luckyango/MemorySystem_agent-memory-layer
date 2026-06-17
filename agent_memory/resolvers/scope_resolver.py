from __future__ import annotations

import re

from agent_memory.schemas import Project, ScopeResolution
from agent_memory.stores import SQLiteProjectStore


class RuleBasedScopeResolver:
    """Resolve whether text belongs to a project, user profile, or session scope.

    This first version intentionally uses transparent rules. It is not meant to be
    linguistically complete; it provides a deterministic baseline before adding LLM
    or embedding-based resolution.
    """

    PROJECT_HINTS = (
        "project",
        "repo",
        "repository",
        "app",
        "system",
        "service",
        "pipeline",
        "项目",
        "仓库",
        "系统",
        "应用",
        "服务",
        "工程",
    )
    NEW_PROJECT_PATTERNS = (
        re.compile(r"(?:new|started|starting|building|working on)\s+(?:a\s+)?(.+?)\s+project", re.I),
        re.compile(r"(?:新|开始|在做|准备做|正在做)(?:一个|个)?(.+?)(?:项目|系统|应用|服务)"),
    )
    USER_SCOPE_HINTS = (
        "i prefer",
        "my preference",
        "call me",
        "我喜欢",
        "我偏好",
        "叫我",
        "我的偏好",
    )

    def __init__(self, project_store: SQLiteProjectStore):
        self.project_store = project_store

    def resolve(self, *, user_id: str, text: str) -> ScopeResolution:
        normalized = self._normalize(text)
        project = self._find_existing_project(user_id, normalized)
        if project is not None:
            return ScopeResolution(
                kind="existing_project",
                scope_type="project",
                scope_id=project.id,
                project=project,
                confidence=0.9,
                reason="Matched an existing project name or alias.",
                matched_text=project.name,
            )

        suggested_name = self._extract_new_project_name(text)
        if suggested_name is not None:
            return ScopeResolution(
                kind="new_project",
                scope_type="project",
                scope_id="",
                confidence=0.7,
                reason="Detected wording that appears to introduce a new project.",
                suggested_project_name=suggested_name,
                matched_text=suggested_name,
            )

        if self._looks_like_user_scope(normalized):
            return ScopeResolution(
                kind="user",
                scope_type="user",
                scope_id="global",
                confidence=0.7,
                reason="Detected a user-level preference or profile statement.",
            )

        return ScopeResolution(
            kind="unknown",
            scope_type="session",
            scope_id="current",
            confidence=0.3,
            reason="No reliable project or user-level scope was detected.",
        )

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
            reason="Created a new project from the detected project mention.",
            matched_text=resolution.suggested_project_name,
        )

    def _find_existing_project(self, user_id: str, normalized_text: str) -> Project | None:
        candidates: list[tuple[int, Project]] = []
        for project in self.project_store.list_projects(user_id, status="active"):
            names = [project.name, *project.aliases]
            for name in names:
                normalized_name = self._normalize(name)
                if normalized_name and normalized_name in normalized_text:
                    candidates.append((len(normalized_name), project))
                    break

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _extract_new_project_name(self, text: str) -> str | None:
        if not self._has_project_hint(text):
            return None

        for pattern in self.NEW_PROJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                return self._clean_project_name(match.group(1))

        return None

    def _looks_like_user_scope(self, normalized_text: str) -> bool:
        return any(hint in normalized_text for hint in self.USER_SCOPE_HINTS)

    def _has_project_hint(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(hint in normalized for hint in self.PROJECT_HINTS)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _clean_project_name(value: str) -> str:
        cleaned = value.strip(" .,:;，。；：")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:80]
