"""Agent tools. ``build_tools`` wires them to a vector store and clients."""

from __future__ import annotations

from openai import OpenAI

from research_agent.config import Settings
from research_agent.tooling import Tool
from research_agent.tools.evaluate_retrieval import make_evaluate_retrieval
from research_agent.tools.retrieve_game import make_retrieve_game
from research_agent.tools.web_search import make_game_web_search
from research_agent.vectorstore import GameStore


def build_tools(settings: Settings, store: GameStore | None = None,
                client: OpenAI | None = None) -> list[Tool]:
    store = store or GameStore.from_settings(settings)
    client = client or settings.openai_client()
    return [
        make_retrieve_game(store),
        make_evaluate_retrieval(client, settings.chat_model),
        make_game_web_search(settings.tavily_api_key),
    ]
