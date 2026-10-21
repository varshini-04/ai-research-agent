"""In-process, per-session history of agent runs."""

from __future__ import annotations

import copy
from typing import Any

DEFAULT_SESSION = "default"


class SessionNotFoundError(Exception):
    """Raised when accessing a session that doesn't exist."""


class ShortTermMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, list[Any]] = {}
        self.create_session(DEFAULT_SESSION)

    def create_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            return False
        self.sessions[session_id] = []
        return True

    def delete_session(self, session_id: str) -> bool:
        if session_id == DEFAULT_SESSION:
            raise ValueError("Cannot delete the default session")
        return self.sessions.pop(session_id, None) is not None

    def _history(self, session_id: str | None) -> list[Any]:
        session_id = session_id or DEFAULT_SESSION
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        return self.sessions[session_id]

    def add(self, obj: Any, session_id: str | None = None) -> None:
        self._history(session_id).append(copy.deepcopy(obj))

    def get_all_objects(self, session_id: str | None = None) -> list[Any]:
        return [copy.deepcopy(o) for o in self._history(session_id)]

    def get_last_object(self, session_id: str | None = None) -> Any | None:
        history = self._history(session_id)
        return copy.deepcopy(history[-1]) if history else None

    def get_all_sessions(self) -> list[str]:
        return list(self.sessions)

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            for sid in self.sessions:
                self.sessions[sid] = []
        else:
            self._history(session_id).clear()
