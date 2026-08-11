"""
base.py — Provider-neutral message format and the LLMProvider interface.

The agentic loop in `loop.py` speaks only the vocabulary defined here; each
provider translates to and from its own wire format. Messages are plain
JSON-serialisable dicts rather than vendor objects, which keeps
`st.session_state.history` serialisable and makes the loop testable without
a live model.

Message shapes
--------------
    {"role": "user",      "content": "<text>"}
    {"role": "assistant", "content": "<text>", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "name": "<tool>",    "content": <result>}

`tool_calls` is omitted when the assistant returned prose only. Tool results
keep their native Python structure; providers sanitise at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single function call requested by the model."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    # Opaque provider round-trip token. Gemini 3.x attaches a `thought_signature`
    # to every functionCall part and *rejects* history that echoes the call back
    # without it (400 INVALID_ARGUMENT). Stored base64 so history stays
    # JSON-serialisable; providers that don't use it simply ignore it.
    signature: str | None = None
    # Correlation id. When a provider populates it, the matching tool result must
    # carry it back so parallel calls pair up correctly.
    id: str | None = None

    def to_dict(self) -> dict:
        out = {"name": self.name, "args": self.args}
        if self.signature:
            out["signature"] = self.signature
        if self.id:
            out["id"] = self.id
        return out

    @staticmethod
    def from_dict(data: dict) -> "ToolCall":
        return ToolCall(
            name=data["name"],
            args=data.get("args") or {},
            signature=data.get("signature"),
            id=data.get("id"),
        )


@dataclass
class ModelResponse:
    """One model reply, normalised across providers."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-native response, kept for logging/debugging
    # Signature belonging to the text part, when the provider emits one.
    signature: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# Message constructors — used by loop.py so the dict shapes live in one place
# ---------------------------------------------------------------------------


def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_message(response: ModelResponse) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": response.text}
    if response.tool_calls:
        msg["tool_calls"] = [tc.to_dict() for tc in response.tool_calls]
    if response.signature:
        msg["signature"] = response.signature
    return msg


def tool_message(name: str, result: Any, call_id: str | None = None) -> dict:
    msg: dict[str, Any] = {"role": "tool", "name": name, "content": result}
    if call_id:
        msg["id"] = call_id
    return msg


def iter_tool_calls(message: dict) -> list[ToolCall]:
    """Read tool calls back out of a stored assistant message."""
    return [ToolCall.from_dict(tc) for tc in message.get("tool_calls") or []]


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """
    Minimal contract the agentic loop depends on.

    Implementations own their own transport, retry policy, and error
    classification — `loop.py` only sees `ModelResponse` or a `RuntimeError`
    carrying a user-facing message.
    """

    name: str
    model: str

    def chat(
        self,
        messages: list[dict],
        tools: list,
        system: str,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ModelResponse:
        """Send one request and return the normalised reply."""
        ...

    def preflight(self) -> None:
        """
        Validate that this provider can serve requests at all.

        Raises RuntimeError with an actionable message if not. Called before
        the first request of a turn; cheap providers may make this a no-op.
        """
        ...
