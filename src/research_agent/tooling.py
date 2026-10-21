"""Turn plain Python functions into OpenAI function-calling tools."""

from __future__ import annotations

import datetime
import inspect
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, Union, get_args, get_origin, get_type_hints

from openai.types.chat import ChatCompletionMessageToolCall

ToolCall: TypeAlias = ChatCompletionMessageToolCall

_PRIMITIVES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    datetime.date: "string",
    datetime.datetime: "string",
}


def _json_schema_type(typ: Any) -> dict:
    origin = get_origin(typ)

    if origin is Literal:
        return {"type": "string", "enum": list(get_args(typ))}

    if origin is Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        non_none = [a for a in get_args(typ) if a is not type(None)]
        return _json_schema_type(non_none[0]) if len(non_none) == 1 else {"type": "string"}

    if origin is list:
        args = get_args(typ)
        return {"type": "array", "items": _json_schema_type(args[0] if args else str)}

    if origin is dict:
        args = get_args(typ)
        return {
            "type": "object",
            "additionalProperties": _json_schema_type(args[1] if len(args) > 1 else str),
        }

    return {"type": _PRIMITIVES.get(typ, "string")}


class Tool:
    """A callable plus the JSON schema the model needs to call it."""

    def __init__(self, func: Callable, name: str | None = None, description: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        self.parameters = [
            {
                "name": key,
                "schema": _json_schema_type(hints.get(key, str)),
                "required": param.default is inspect.Parameter.empty,
            }
            for key, param in signature.parameters.items()
        ]

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p["name"]: p["schema"] for p in self.parameters},
                    "required": [p["name"] for p in self.parameters if p["required"]],
                    "additionalProperties": False,
                },
            },
        }

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<Tool name={self.name} params={[p['name'] for p in self.parameters]}>"


def tool(func: Callable | None = None, *, name: str | None = None, description: str | None = None):
    """Decorator: ``@tool`` or ``@tool(name="foo")``."""

    def wrap(f: Callable) -> Tool:
        return Tool(f, name=name, description=description)

    return wrap(func) if func else wrap
