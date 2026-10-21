from __future__ import annotations

from tavily import TavilyClient

from research_agent.tooling import Tool, tool


def make_game_web_search(api_key: str | None, max_results: int = 3) -> Tool:
    @tool
    def game_web_search(question: str) -> str:
        """Search the web for video game information missing from the knowledge base.

        Args:
            question: a search query about the video game industry.
        Returns titles, URLs and snippets of the top results.
        """
        if not api_key:
            return "Error: web search is unavailable (TAVILY_API_KEY is not set)."
        data = TavilyClient(api_key=api_key).search(query=question, max_results=max_results)
        hits = data.get("results", [])
        if not hits:
            return "No web results found."
        return "\n\n".join(f"{h['title']}\n{h['url']}\n{h['content']}" for h in hits)

    return game_web_search
