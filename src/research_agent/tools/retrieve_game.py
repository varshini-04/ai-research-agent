from __future__ import annotations

from research_agent.tooling import Tool, tool
from research_agent.vectorstore import GameStore


def make_retrieve_game(store: GameStore, n_results: int = 3) -> Tool:
    @tool
    def retrieve_game(query: str) -> str:
        """Semantic search over the internal video game knowledge base.

        Args:
            query: a question or keywords about a video game (name, platform, genre, year...).
        Returns one game per line with name, platform, release year, genre, publisher, description.
        """
        results = store.search(query, n_results=n_results)
        if not results:
            return "No documents found."
        return "\n".join(r.format() for r in results)

    return retrieve_game
