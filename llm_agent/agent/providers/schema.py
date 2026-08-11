"""
schema.py — Tool schema conversion between provider dialects.

`tool_schemas.TOOLS` stays the single source of truth, expressed in
google-genai's `FunctionDeclaration` types. Providers that speak plain JSON
Schema (Ollama, and the OpenAI-compatible world generally) get a converted
view produced here, so tool definitions never have to be maintained twice.

The only real difference between the two dialects is that genai serialises
`type` as an upper-case enum (`OBJECT`, `STRING`) while JSON Schema wants
lower-case (`object`, `string`). Everything else — nested properties,
`required`, `enum`, array `items` — round-trips unchanged.
"""

from __future__ import annotations

from typing import Any


def _normalize(node: Any) -> Any:
    """Recursively lower-case genai `type` enums into JSON Schema types."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "type":
                # genai dumps either a Type enum or its string name.
                raw = getattr(value, "value", value)
                out[key] = raw.lower() if isinstance(raw, str) else raw
            else:
                out[key] = _normalize(value)
        return out
    if isinstance(node, list):
        return [_normalize(item) for item in node]
    return node


def function_declarations(tools: list) -> list:
    """Flatten genai `Tool` wrappers into their `FunctionDeclaration` list."""
    return [fd for tool in tools for fd in (tool.function_declarations or [])]


def to_json_schema_tools(tools: list) -> list[dict]:
    """
    Convert genai `Tool` objects into OpenAI/Ollama-style tool definitions.

    Returns a list of::

        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": {...}}}

    The result is plain JSON-serialisable data, ready to POST to Ollama.
    """
    converted: list[dict] = []
    for fd in function_declarations(tools):
        dumped = fd.model_dump(exclude_none=True)
        parameters = _normalize(
            dumped.get("parameters") or {"type": "object", "properties": {}}
        )
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": dumped["name"],
                    "description": dumped.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return converted
