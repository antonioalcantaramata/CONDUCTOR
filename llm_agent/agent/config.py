"""
config.py — Central configuration for the AIDT LLM agent.

All values are overridable via environment variables.
GEMINI_API_KEY is required and raises EnvironmentError at import time if missing.
"""

import logging
import os

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


def get_grid_constants() -> dict:
    """Fetch live grid constants from the backend /api/grid_constants endpoint.

    Called on every agent turn so the system prompt automatically reflects
    any network uploaded at runtime.  Falls back to DEFAULT_GRID_CONSTANTS if
    the backend is unreachable (e.g. during unit tests or early startup).
    """
    try:
        resp = httpx.get(f"{BASE_URL}/api/grid_constants", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(
            "GET /api/grid_constants returned %d — using fallback constants.",
            resp.status_code,
        )
    except Exception as exc:
        logger.warning("Could not reach backend for grid constants: %s", exc)
    return DEFAULT_GRID_CONSTANTS
