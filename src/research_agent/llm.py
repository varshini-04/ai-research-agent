"""Thin wrapper around the OpenAI chat completions API."""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from research_agent.config import get_settings
from research_agent.messages import AIMessage, BaseMessage, TokenUsage, UserMessage
from research_agent.tooling import Tool


class LLM:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        tools: list[Tool] | None = None,
        client: OpenAI | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.chat_model
        self.temperature = temperature
        self.client = client or settings.openai_client()
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def _build_payload(self, messages: list[BaseMessage]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.to_api() for m in messages],
        }
        if self.tools:
            payload["tools"] = [t.to_openai() for t in self.tools.values()]
            payload["tool_choice"] = "auto"
        return payload

    @staticmethod
    def _as_messages(value: str | BaseMessage | list[BaseMessage]) -> list[BaseMessage]:
        if isinstance(value, str):
            return [UserMessage(content=value)]
        if isinstance(value, BaseMessage):
            return [value]
        if isinstance(value, list) and all(isinstance(m, BaseMessage) for m in value):
            return value
        raise ValueError(f"Invalid input type {type(value)}.")

    def invoke(
        self,
        input: str | BaseMessage | list[BaseMessage],
        response_format: type[BaseModel] | None = None,
    ) -> AIMessage:
        payload = self._build_payload(self._as_messages(input))
        if response_format:
            payload["response_format"] = response_format
            response = self.client.chat.completions.parse(**payload)
        else:
            response = self.client.chat.completions.create(**payload)

        message = response.choices[0].message
        usage = response.usage
        return AIMessage(
            content=message.content,
            tool_calls=message.tool_calls or None,
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
            if usage
            else None,
        )
