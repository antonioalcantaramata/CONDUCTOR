"""
cache.py — a small local cache for discovery results and tool-calling tests.

One JSON file (`discovery.json` in the cache folder) with a section per
source, each stamped with when it was fetched. Lists are kept for
`DISCOVERY_CACHE_TTL_H` hours (default 24); "Refresh list" ignores the age.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from ..usage import cache_dir

DEFAULT_TTL_H = 24.0


def ttl_seconds() -> float:
    try:
        return float(os.environ.get("DISCOVERY_CACHE_TTL_H") or DEFAULT_TTL_H) * 3600
    except ValueError:
        return DEFAULT_TTL_H * 3600


def _path():
    return cache_dir() / "discovery.json"


def read_all() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_all(data: dict) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".discovery", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def get(section: str, *, now: float | None = None, max_age: float | None = None) -> dict | None:
    """The cached section when younger than the TTL (or `max_age` seconds);
    None when absent or stale."""
    entry = read_all().get(section)
    if not isinstance(entry, dict) or "fetched_at" not in entry:
        return None
    age = (time.time() if now is None else now) - float(entry["fetched_at"])
    return entry if age <= (ttl_seconds() if max_age is None else max_age) else None


def get_any(section: str) -> dict | None:
    """The cached section whatever its age (a fallback when a fetch fails)."""
    entry = read_all().get(section)
    return entry if isinstance(entry, dict) else None


def put(section: str, payload: dict, *, now: float | None = None) -> dict:
    data = read_all()
    entry = {**payload, "fetched_at": time.time() if now is None else now}
    data[section] = entry
    write_all(data)
    return entry


def smoke_key(provider: str, model: str) -> str:
    return f"{provider}|{model}"


def smoke_result(provider: str, model: str, *, now: float | None = None) -> dict | None:
    """A tool-calling test result for this provider and model, if one was run
    within the cache window."""
    entry = (read_all().get("smoke") or {}).get(smoke_key(provider, model))
    if not isinstance(entry, dict):
        return None
    if (time.time() if now is None else now) - float(entry.get("at") or 0) > ttl_seconds():
        return None
    return entry


def put_smoke(provider: str, model: str, result: dict, *, now: float | None = None) -> None:
    data = read_all()
    data.setdefault("smoke", {})[smoke_key(provider, model)] = {**result, "at": time.time() if now is None else now}
    write_all(data)
