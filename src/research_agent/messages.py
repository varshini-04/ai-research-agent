"""Chat message models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from research_agent.tooling import ToolCall


class BaseMessage(BaseModel):
    role: str
    content: str | None = ""

    def to_api(self) -> dict:
        """Serialize for the OpenAI chat API (drops None fields and local-only metadata)."""
        return self.model_dump(exclude_none=True, exclude={"token_usage"})


class SystemMessage(BaseMessage):
    role: Literal["system"] = "system"


class UserMessage(BaseMessage):
    role: Literal["user"] = "user"


class ToolMessage(BaseMessage):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str = ""


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AIMessage(BaseMessage):
    role: Literal["assistant"] = "assistant"
    content: str | None = ""
    tool_calls: list[ToolCall] | None = None
    token_usage: TokenUsage | None = None


AnyMessage = SystemMessage | UserMessage | AIMessage | ToolMessage
