"""Game research agent: RAG over a local knowledge base with LLM-judged web-search fallback."""

from research_agent.agent import Agent, AgentAnswer
from research_agent.config import Settings, get_settings

__all__ = ["Agent", "AgentAnswer", "Settings", "get_settings"]
__version__ = "0.1.0"
