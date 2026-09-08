"""
config.py — Central configuration for the AIDT LLM agent.

All values are overridable via environment variables.
GEMINI_API_KEY is required and raises EnvironmentError at import time if missing.
"""

import logging
import os
import pathlib
from typing import NamedTuple

import httpx
from dotenv import load_dotenv

# The .env sits beside the app, and is found whatever the working directory
# is. Plain `load_dotenv()` searches upward from the cwd, which works when
# Streamlit is launched from `llm_agent/` and silently finds nothing when a
# batch run starts at the repository root — the agent then reports a missing
# API key on a machine that has one.
_ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE if _ENV_FILE.exists() else None)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
BASE_URL: str = os.environ.get("DT_BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
# "google" = Google Generative AI (needs GEMINI_API_KEY);
# "openai" = the OpenAI API, or anything speaking its chat-completions dialect
#            (needs OPENAI_API_KEY);
# "ollama" = a local Ollama server (no key, no data leaves the machine).
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "google").strip().lower()

# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
# Default matches llm_agent/.env.example — keep the two in sync.
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

_api_key = os.environ.get("GEMINI_API_KEY", "")
# Not raising here — the Streamlit app handles the missing-key setup flow.
GEMINI_API_KEY: str = _api_key

# How hard a thinking-capable Gemini model reasons before answering, lowest to
# highest. Gemini's ladder is its own — four rungs where OpenAI has five, and
# "minimal" where OpenAI says "none" — so it is spelled out here rather than
# forced onto a shared scale that would match neither API.
GEMINI_THINKING_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high")

# Empty means "send nothing", which leaves the model on its own dynamic budget.
# That is the default because it is what this provider has always done: the
# call has only ever set include_thoughts=False, which suppresses *returning*
# the thought trace and does not stop the model thinking. Pinning a level here
# would quietly change latency, cost and answer quality for every existing
# user, so the parity knob is opt-in rather than a new default.
GEMINI_THINKING_LEVEL: str = os.environ.get(
    "GEMINI_THINKING_LEVEL", ""
).strip().lower()

# ---------------------------------------------------------------------------
# OpenAI (and OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------
# Default matches llm_agent/.env.example — keep the two in sync. Any model set
# here must support tool calling: CONDUCTOR drives the grid entirely through
# tools, so a model without it can only chat.
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")

# How hard a reasoning model thinks before answering, lowest to highest.
# Ordered, because the settings UI renders them as a slider-like choice.
OPENAI_REASONING_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")

# "medium" rather than "none", matching what the other two backends already do:
# the Gemini path leaves the thinking budget at its default and Ollama defaults
# OLLAMA_THINK to true. A turn here plans several tool calls and then reasons
# over grid results, so some deliberation earns its cost.
#
# Sent only when non-empty, and dropped automatically if the model rejects it —
# non-reasoning models and some OpenAI-compatible servers do not accept the
# parameter at all.
OPENAI_REASONING_EFFORT: str = os.environ.get(
    "OPENAI_REASONING_EFFORT", "medium"
).strip().lower()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# Which OpenAI endpoint to drive.
#
#   auto      — /v1/responses when a reasoning level is asked for and the base
#               URL is OpenAI's own; /v1/chat/completions otherwise.
#   chat      — always /v1/chat/completions.
#   responses — always /v1/responses.
#
# The split exists because the two endpoints can do different things. Newer
# reasoning models refuse function tools combined with any reasoning effort on
# chat/completions — and CONDUCTOR sends tools on every turn — so reasoning is
# only reachable through /v1/responses. Meanwhile most OpenAI-*compatible*
# servers (vLLM, OpenRouter, Groq) implement only chat/completions, which is
# why a custom base URL stays on it unless told otherwise.
OPENAI_APIS: tuple[str, ...] = ("auto", "chat", "responses")
OPENAI_API: str = os.environ.get("OPENAI_API", "auto").strip().lower()

# Overridable so the same provider serves anything speaking the OpenAI
# chat-completions dialect — Azure OpenAI, OpenRouter, Groq, a local vLLM or
# llama.cpp server. Only the base changes; the wire format is identical.
OPENAI_BASE_URL: str = os.environ.get(
    "OPENAI_BASE_URL", "https://api.openai.com/v1"
).rstrip("/")

# Hosted, so far quicker than Ollama, but this agent's turns carry ~25k tokens
# of prompt and can fan out over several tool calls. Matches the Gemini path's
# DT_MODEL_REQUEST_TIMEOUT_MS rather than Ollama's much longer local budget.
OPENAI_TIMEOUT_S: float = float(os.environ.get("OPENAI_TIMEOUT_S", "180"))

# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------
# Default matches llm_agent/.env.example — keep the two in sync. Any model set
# here must support tool calling, as with every other backend: CONDUCTOR drives
# the grid entirely through tools.
#
# Haiku by default — the cheapest Claude model, matching the cost posture of
# the other hosted backends here (the Gemini path defaults to a Flash model).
# Set `claude-sonnet-5` or `claude-opus-5` for a more capable tier.
#
# Note that Haiku 4.5 predates adaptive thinking: it takes the older
# `budget_tokens` shape and rejects `output_config` outright. The provider
# picks the right shape from the model id, so `ANTHROPIC_EFFORT` below works
# either way — on Haiku it maps onto a token budget rather than an effort
# level. Its context window is also 200K rather than 1M, which is ample for a
# ~25k-token turn but worth knowing.
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# How hard the model thinks before answering, lowest to highest. Ordered,
# because the settings UI renders them as a slider-like choice. Empty means
# "send nothing" — the API default, which is `high`. On a 4.5-generation model
# these map onto thinking-token budgets instead; see providers/claude.py.
ANTHROPIC_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# `medium` for the same reason the OpenAI path picks it: a turn here plans
# several tool calls and then reasons over grid results, so some deliberation
# earns its cost, and the top of the range earns it only on hard problems.
ANTHROPIC_EFFORT: str = os.environ.get("ANTHROPIC_EFFORT", "medium").strip().lower()

# Whether the model thinks before answering. On current models this is adaptive
# thinking — the model decides when and how much, with `effort` above setting
# the depth; on a 4.5-generation model it enables a token budget instead.
# Set to "0"/"false" to turn it off, which the current models accept only at
# effort `high` or below.
ANTHROPIC_THINKING: bool = os.environ.get(
    "ANTHROPIC_THINKING", "1"
).strip().lower() not in ("0", "false", "no")

# Overridable for gateways and proxies that speak the Anthropic wire format.
ANTHROPIC_BASE_URL: str = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
).rstrip("/")

# Matches the OpenAI budget: hosted, but this agent's turns carry ~25k tokens
# of prompt and schemas, and a thinking model can spend a while on top.
ANTHROPIC_TIMEOUT_S: float = float(os.environ.get("ANTHROPIC_TIMEOUT_S", "180"))

# Output cap. The skill's guidance is not to lowball this — hitting the cap
# truncates mid-thought and costs a retry. Answers here are operator summaries
# rather than documents, so 16k is generous without inviting a timeout on the
# non-streaming path.
ANTHROPIC_MAX_TOKENS: int = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))

# ---------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "gemma4:12b-mlx")

# Ollama defaults num_ctx to ~4096 REGARDLESS of what the model supports, and
# silently truncates the *front* of an over-long prompt — which drops the system
# prompt first, with no error. This agent's system prompt plus 20 tool schemas
# measures ~25k tokens before any conversation, so a generous window is required.
#
# 32768 rather than 65536: measured on an M3 Pro, quadrupling the window made no
# difference to prompt-processing speed (139 tok/s either way) while reserving
# several GB more KV cache. Pay for headroom you use, not headroom you don't.
OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))

# How long Ollama keeps the model — and crucially its prompt cache — resident.
#
# This is the single biggest lever on perceived speed. Processing the ~25k-token
# prefix costs ~180s cold, but Ollama reuses the cached prefix on subsequent
# calls, dropping that to ~0.2s. The default 5m unload throws the cache away
# between questions and makes every one of them pay the cold cost again.
OLLAMA_KEEP_ALIVE: str = os.environ.get("OLLAMA_KEEP_ALIVE", "60m")

# Whether thinking-capable models (Gemma 4 among them) may reason before answering.
#
# On by default, which is what the Gemini path actually does: it sets
# include_thoughts=False, and that only suppresses *returning* the trace — the
# model still thinks, since thinking_budget is left at its default. Disabling
# reasoning outright is a quality decision, not a parity one, so it is opt-in.
#
# Measured cost on a 12B local model: ~12s per call. Turn it off if you would
# rather have speed than reasoning quality.
OLLAMA_THINK: bool = os.environ.get("OLLAMA_THINK", "true").strip().lower() == "true"

# Tokens held back from the window for the model's own reply. A fixed reserve,
# not a fraction: replies run a few hundred tokens regardless of window size, so
# scaling this with num_ctx would waste budget that history needs.
OLLAMA_OUTPUT_RESERVE: int = int(os.environ.get("OLLAMA_OUTPUT_RESERVE", "4096"))

# Local generation is far slower than a hosted API, especially on the first call
# when the model is still being loaded into memory.
OLLAMA_TIMEOUT_S: float = float(os.environ.get("OLLAMA_TIMEOUT_S", "600"))

# ---------------------------------------------------------------------------
# HTTP / agent loop
# ---------------------------------------------------------------------------
HTTP_TIMEOUT: float = float(os.environ.get("DT_HTTP_TIMEOUT", "120.0"))
MAX_AGENT_TURNS: int = int(os.environ.get("DT_MAX_AGENT_TURNS", "8"))

# Timeout for the per-turn grid-constants fetch. Previously 3s — far tighter
# than everything else here — which meant a briefly busy backend silently
# swapped the agent onto the wrong network. Missing this fetch is a correctness
# problem, not a latency one, so it gets a realistic budget.
GRID_CONSTANTS_TIMEOUT: float = float(
    os.environ.get("DT_GRID_CONSTANTS_TIMEOUT", "15.0")
)
MODEL_REQUEST_TIMEOUT_MS: int = int(
    os.environ.get("DT_MODEL_REQUEST_TIMEOUT_MS", "180000")
)
MODEL_RETRY_ATTEMPTS: int = int(os.environ.get("DT_MODEL_RETRY_ATTEMPTS", "10"))
MODEL_OVERLOADED_RETRY_DELAY_S: int = int(
    os.environ.get("DT_MODEL_OVERLOADED_RETRY_DELAY_S", "90")
)

# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------
# Whether the agent is shown its own provenance findings before its answer is
# delivered. `off` is the baseline every earlier result was produced under and
# stays the default; `warn` records what a reflection loop would have fired on
# without changing any answer; `inform` hands the findings back to the agent
# and lets it decide. See `reflection.py` for why informing and correcting are
# deliberately different things.
REFLECTION_MODE: str = os.environ.get("CONDUCTOR_REFLECTION", "off").strip().lower()

# ---------------------------------------------------------------------------
# Session logging
# ---------------------------------------------------------------------------
# Every turn is written as one JSON line holding the prompt, the resolved tool
# calls, the full tool results and the answer. That record is the only durable
# evidence of what the agent actually did, and the offline graders in
# `evaluation/` read nothing else.
#
# It used to be a single file truncated at process start, so each run destroyed
# the one before it and nothing could be graded after the fact. Runs now get
# their own file and `last_session.jsonl` is kept as a pointer to the newest,
# so existing habits still work.
SESSION_LOG_DIR: str = os.environ.get(
    "CONDUCTOR_SESSION_LOG_DIR",
    str(pathlib.Path(__file__).parent.parent / "session_logs"),
)

# An explicit file overrides the per-run name, so a batch sweep can collect
# every turn it runs in one place.
SESSION_LOG_PATH: str = os.environ.get("CONDUCTOR_SESSION_LOG", "")

# ---------------------------------------------------------------------------
# Default grid constants — fallback only.
# Callers should use get_grid_constants() for the live, network-aware values.
# ---------------------------------------------------------------------------
DEFAULT_GRID_CONSTANTS: dict = {
    "name": "PGLib IEEE 14-Bus System",
    "n_lines": 15,
    "n_trafos": 5,
    "n_substations": 14,
    "substation_names": [f"Bus_{i}" for i in range(1, 15)],
    "vm_lower": 0.94,
    "vm_upper": 1.06,
    "max_loading_pct": 100.0,
    "load_scaling_min": 0.0,
    "load_scaling_max": 4.0,
    "slack_max_mw_default": 999.0,
    "slack_max_mw_min": 0.0,
    "slack_max_mw_max": 999.0,
    "slack_name": "External Grid",
    "grid_type": "transmission",
    "description": "Synthetic operating data generated from the bundled PGLib IEEE 14-bus MATPOWER case.",
    "voltage_level_kv": 0.0,
}


class GridConstants(NamedTuple):
    """Grid constants plus whether they actually came from the backend."""

    values: dict
    live: bool
    error: str | None = None

    @property
    def name(self) -> str:
        return self.values.get("name", "Unknown Grid")


# Status of the most recent fetch, so the UI can warn without re-querying.
_last_fetch: GridConstants | None = None


def fetch_grid_constants() -> GridConstants:
    """
    Fetch live grid constants from the backend `/api/grid_constants`.

    Returns the values *and* whether they are real. This distinction matters:
    the fallback describes a completely different network (the bundled IEEE
    14-bus case), so silently substituting it makes the agent describe a grid
    that is not loaded — wrong bus names, wrong voltage limits, wrong topology —
    with no indication anything went wrong. Callers must decide what to do about
    a degraded fetch rather than being handed plausible-looking wrong data.
    """
    global _last_fetch

    try:
        resp = httpx.get(
            f"{BASE_URL}/api/grid_constants", timeout=GRID_CONSTANTS_TIMEOUT
        )
        if resp.status_code == 200:
            _last_fetch = GridConstants(values=resp.json(), live=True)
            return _last_fetch
        error = f"backend returned HTTP {resp.status_code}"
        logger.warning(
            "GET /api/grid_constants returned %d — falling back to %s.",
            resp.status_code, DEFAULT_GRID_CONSTANTS["name"],
        )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        logger.warning("Could not reach backend for grid constants: %s", exc)

    _last_fetch = GridConstants(
        values=DEFAULT_GRID_CONSTANTS, live=False, error=error
    )
    return _last_fetch


def get_grid_constants() -> dict:
    """Live grid constants, falling back to defaults. Prefer
    `fetch_grid_constants()` when you need to know which you got."""
    return fetch_grid_constants().values


class GridLimits(NamedTuple):
    vm_lower: float
    vm_upper: float
    max_loading: float


def grid_limits() -> GridLimits:
    """
    Voltage and loading limits for the network currently loaded.

    Read at call time, not bound at import: these are the fallbacks charts use
    when a tool result carries no thresholds of its own, and freezing them at
    import pinned every chart to the bundled example network's limits regardless
    of which grid was loaded.

    Uses the most recent backend fetch — which the agent loop refreshes every
    turn — so callers pay no HTTP request.

    Lives here rather than in `renderers` so it can be used (and tested) without
    pulling in plotly, which is a UI-only dependency absent from CI.
    """
    status = last_grid_constants_status()
    values = status.values if status is not None else DEFAULT_GRID_CONSTANTS
    return GridLimits(
        vm_lower=values.get("vm_lower", DEFAULT_GRID_CONSTANTS["vm_lower"]),
        vm_upper=values.get("vm_upper", DEFAULT_GRID_CONSTANTS["vm_upper"]),
        max_loading=values.get(
            "max_loading_pct", DEFAULT_GRID_CONSTANTS["max_loading_pct"]
        ),
    )


def network_fingerprint(values: dict) -> str:
    """
    A stable identity for the loaded network.

    Built from the properties that cannot differ between two networks anyone
    would consider the same: name, size, and bus names. Deliberately excludes
    operating limits, which an operator may legitimately retune on the same
    grid without it becoming a different one.
    """
    parts = [
        str(values.get("name", "")),
        str(values.get("n_substations", "")),
        str(values.get("n_lines", "")),
        str(values.get("n_trafos", "")),
        ",".join(values.get("substation_names", []) or []),
    ]
    return "|".join(parts)


def is_network_change(previous: str | None, status: GridConstants) -> bool:
    """
    Whether the backend is now serving a different network than `previous`.

    Returns False for a degraded fetch even though the fallback constants
    describe a different grid: a momentary backend outage must never be
    mistaken for a network swap, because the caller's response is to discard
    the user's conversation.

    Also False when there is no previous fingerprint — the first observation
    establishes the baseline rather than signalling a change.
    """
    if not status.live or previous is None:
        return False
    return previous != network_fingerprint(status.values)


def last_grid_constants_status() -> GridConstants | None:
    """The most recent fetch result, or None if none has happened yet."""
    return _last_fetch
