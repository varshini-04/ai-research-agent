from typing import Literal

from research_agent.tooling import tool


def test_schema_inference():
    @tool
    def search(
        query: str,
        limit: int = 3,
        mode: Literal["a", "b"] = "a",
        tags: list[str] | None = None,
    ):
        """Search things."""

    spec = search.to_openai()["function"]
    props = spec["parameters"]["properties"]
    assert spec["name"] == "search"
    assert spec["description"] == "Search things."
    assert props["query"] == {"type": "string"}
    assert props["limit"] == {"type": "integer"}
    assert props["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert spec["parameters"]["required"] == ["query"]


def test_tool_is_callable():
    @tool(name="add")
    def _add(a: int, b: int) -> int:
        return a + b

    assert _add.name == "add"
    assert _add(1, 2) == 3
