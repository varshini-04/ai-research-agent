"""Shared fakes so the whole suite runs offline."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai.types.chat import ChatCompletionMessageToolCall


class FakeEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words hashing embedding (no network)."""

    DIM = 128

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for text in input:
            vec = [0.0] * self.DIM
            for word in text.lower().replace("(", " ").replace(")", " ").split():
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name() -> str:
        return "fake-hash-embedding"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> FakeEmbedding:
        return FakeEmbedding()


def tool_call(name: str, args: dict, call_id: str = "call_1") -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall.model_validate(
        {"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}
    )


def completion(content: str | None = None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeOpenAI:
    """Replays scripted completions and records the payloads it receives."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []
        handler = SimpleNamespace(create=self._next, parse=self._next)
        self.chat = SimpleNamespace(completions=handler)

    def _next(self, **payload):
        self.calls.append(payload)
        return self.script.pop(0)


@pytest.fixture
def fake_embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from research_agent import config

    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
