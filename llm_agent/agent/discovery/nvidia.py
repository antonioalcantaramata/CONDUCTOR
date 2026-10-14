"""
nvidia.py — NVIDIA Build's free endpoints: a curated allowlist, checked live.

`GET https://integrate.api.nvidia.com/v1/models` is public but lists only
`id`, `object`, `created` and `owned_by` — nothing about pricing, free-endpoint
status, tool support or context length. So the catalog keeps an allowlist
(with each model card's context length and tool support), and at launch it is
validated against the live list: only allowlisted models NVIDIA currently
lists are offered. A model that disappears, or starts failing, is treated as
gone — free endpoints get deprecated, and the website is never scraped.

Models marked `smoke_test_required` in the catalog are offered only after a
tool-calling test has passed for them (`cache.smoke_result`).

New versions are spotted, not assumed: an ID in the live list from the same
family as an allowlisted model (same owner, name and suffix) with a higher
version — `z-ai/glm-5.4` next to `z-ai/glm-5.3` — is offered as a candidate.
Its free-endpoint status and context length are unknown (the list does not say),
so it too needs a passing tool-calling test before it can be chosen.
"""

from __future__ import annotations

import re

import httpx

from . import cache

_FAMILY = re.compile(r"^(?P<owner>[^/]+)/(?P<stem>.*?)(?P<ver>\d+(?:\.\d+)*)(?P<suffix>(?:-[a-z][a-z0-9]*)*)$")

MODELS_URL = "https://integrate.api.nvidia.com/v1/models"


def fetch_ids(timeout: float = 20.0) -> list[str]:
    resp = httpx.get(MODELS_URL, timeout=timeout)
    resp.raise_for_status()
    return sorted(str(m["id"]) for m in (resp.json() or {}).get("data") or [] if m.get("id"))


def validate(allowlist: list[dict], live_ids: list[str] | set[str] | None) -> dict:
    """{"available": allowlisted entries NVIDIA lists (catalog order),
    "missing": allowlisted IDs it does not, "validated": whether a live list
    was available}. Without a live list, the allowlist is returned as is,
    unvalidated."""
    if live_ids is None:
        return {"available": list(allowlist), "missing": [], "validated": False}
    live = set(live_ids)
    available = [m for m in allowlist if m.get("id") in live]
    missing = [m["id"] for m in allowlist if m.get("id") not in live]
    return {"available": available, "missing": missing, "validated": True}


def family(model_id: str) -> tuple[tuple, tuple[int, ...]] | None:
    """((owner, name stem, suffix), version) — e.g. "z-ai/glm-5.3-flash" →
    (("z-ai", "glm-", "-flash"), (5, 3)); None when there is no version."""
    m = _FAMILY.match(model_id or "")
    if not m:
        return None
    return (m["owner"], m["stem"], m["suffix"]), tuple(int(p) for p in m["ver"].split("."))


def newer_candidates(allowlist: list[dict], live_ids) -> list[dict]:
    """Live IDs newer than every allowlisted model of their family, as
    catalog-like entries that need a tool-calling test."""
    best: dict[tuple, tuple[int, ...]] = {}
    for m in allowlist:
        parsed = family(m.get("id", ""))
        if parsed:
            key, version = parsed
            best[key] = max(best.get(key, version), version)
    known = {m.get("id") for m in allowlist}
    out = []
    for model_id in sorted(live_ids or []):
        parsed = family(model_id)
        if model_id in known or not parsed:
            continue
        key, version = parsed
        if key in best and version > best[key]:
            out.append({"id": model_id, "label": f"{model_id} (newer, unconfirmed)", "context_window": None,
                        "tools": True, "smoke_test_required": True, "candidate": True,
                        "note": "Newer than the allowlisted version. NVIDIA's list does not say whether its "
                                "endpoint is free or how much context it has; run Test tool calling to try it."})
    return out


def needs_test(entry: dict) -> bool:
    """A model offered only after a passing tool-calling test."""
    if not entry.get("smoke_test_required"):
        return False
    result = cache.smoke_result("nvidia", entry["id"])
    return not (result and result.get("ok"))


def discover(allowlist: list[dict], *, force: bool = False) -> dict:
    """The validated allowlist, using the cached live list when fresh.
    Never raises: a failed fetch uses the last cached list, or none."""
    hit = None if force else cache.get("nvidia")
    error = None
    if hit is None:
        try:
            hit = cache.put("nvidia", {"ids": fetch_ids()})
        except Exception as exc:  # noqa: BLE001
            error = f"Could not reach NVIDIA's model list: {exc}"
            hit = cache.get_any("nvidia")
    ids = (hit or {}).get("ids")
    result = validate(allowlist, ids)
    return {**result, "candidates": newer_candidates(allowlist, ids) if ids else [],
            "fetched_at": (hit or {}).get("fetched_at"), "error": error}


def next_model(available: list[dict], failed: set[str]) -> str | None:
    """The next allowlisted model to try after one is gone."""
    for m in available:
        if m["id"] not in failed and not needs_test(m):
            return m["id"]
    return None
