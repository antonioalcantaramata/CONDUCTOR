"""
config.py — Central configuration for the AIDT LLM agent.

All values are overridable via environment variables.
GEMINI_API_KEY is required and raises EnvironmentError at import time if missing.
"""

import logging
import os
from typing import NamedTuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
BASE_URL: str = os.environ.get("DT_BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
# "google" = Google Generative AI (needs GEMINI_API_KEY);
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
