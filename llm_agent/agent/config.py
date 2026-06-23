"""
config.py — Central configuration for the AIDT LLM agent.

All values are overridable via environment variables.
GEMINI_API_KEY is required and raises EnvironmentError at import time if missing.
"""

import json
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
# Gemini
# ---------------------------------------------------------------------------
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

_api_key = os.environ.get("GEMINI_API_KEY", "")
# Not raising here — the Streamlit app handles the missing-key setup flow.
GEMINI_API_KEY: str = _api_key

# ---------------------------------------------------------------------------
# HTTP / agent loop
# ---------------------------------------------------------------------------
HTTP_TIMEOUT: float = float(os.environ.get("DT_HTTP_TIMEOUT", "120.0"))
MAX_AGENT_TURNS: int = int(os.environ.get("DT_MAX_AGENT_TURNS", "20"))
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
