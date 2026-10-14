"""
gemini.py — newer Gemini models than the catalog lists, found live.

Google's `GET /v1beta/models` (free to call with the API key; it generates
nothing) lists every model the key can reach with its input token limit and
supported methods, but nothing about free-tier quotas. So a newer model —
`gemini-3.9-flash-lite` once the catalog lists 3.5 — is offered as a
candidate, never switched to: its free-tier limits are unknown, and it needs a
passing tool-calling test before it can be chosen.

Families are Flash-Lite, Flash and Pro by version; speech, image, live, audio,
embedding and alias models (`gemini-flash-latest`) are not candidates.
"""

from __future__ import annotations

import re

import httpx

from . import cache

MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_FAMILY = re.compile(r"^gemini-(?P<ver>\d+(?:\.\d+)?)-(?P<kind>flash-lite|flash|pro)(?P<rest>(?:-preview)?(?:-[0-9-]+)?)$")


def family(model_id: str) -> tuple[str, tuple[int, ...], bool] | None:
    """(kind, version, preview) — e.g. "gemini-3-flash-preview" →
    ("flash", (3,), True); None for anything that is not a plain text model."""
    m = _FAMILY.match(model_id or "")
    if not m:
        return None
    return m["kind"], tuple(int(p) for p in m["ver"].split(".")), "preview" in (m["rest"] or "")


def fetch(api_key: str, timeout: float = 20.0) -> list[dict]:
    """Every model the key can reach: id, input_token_limit, methods."""
    models, token = [], None
    while True:
        params = {"pageSize": 1000, **({"pageToken": token} if token else {})}
        resp = httpx.get(MODELS_URL, params=params, headers={"x-goog-api-key": api_key}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json() or {}
        for m in data.get("models") or []:
            models.append({"id": str(m.get("name", "")).removeprefix("models/"),
                           "input_token_limit": m.get("inputTokenLimit"),
                           "methods": list(m.get("supportedGenerationMethods") or [])})
        token = data.get("nextPageToken")
        if not token:
            return models


def newer_candidates(catalog_models: list[dict], listed: list[dict]) -> list[dict]:
    """Listed text models newer than every catalog model of the same kind."""
    best: dict[str, tuple[int, ...]] = {}
    for m in catalog_models:
        parsed = family(m.get("id", ""))
        if parsed:
            kind, version, _ = parsed
            best[kind] = max(best.get(kind, version), version)
    known = {m.get("id") for m in catalog_models}
    out = []
    for m in sorted(listed, key=lambda x: x["id"]):
        parsed = family(m["id"])
        if m["id"] in known or not parsed or "generateContent" not in m.get("methods", []):
            continue
        kind, version, preview = parsed
        if kind in best and version > best[kind]:
            out.append({"id": m["id"], "label": f"{m['id']} (newer{', preview' if preview else ''}, unconfirmed)",
                        "context_window": m.get("input_token_limit"), "tools": True,
                        "smoke_test_required": True, "candidate": True,
                        "note": "Newer than the catalog's models. Google's list does not give free-tier limits; "
                                "run Test tool calling to try it, and check its limits in AI Studio."})
    return out


def discover(api_key: str, catalog_models: list[dict], *, force: bool = False) -> dict:
    """{"candidates", "fetched_at", "error"}; never raises. Needs a key."""
    if not api_key:
        return {"candidates": [], "fetched_at": None, "error": None}
    hit = None if force else cache.get("gemini_models")
    error = None
    if hit is None:
        try:
            hit = cache.put("gemini_models", {"models": fetch(api_key)})
        except Exception as exc:  # noqa: BLE001
            error = f"Could not read Google's model list: {exc}"
            hit = cache.get_any("gemini_models")
    listed = (hit or {}).get("models") or []
    return {"candidates": newer_candidates(catalog_models, listed), "fetched_at": (hit or {}).get("fetched_at"),
            "error": error}
