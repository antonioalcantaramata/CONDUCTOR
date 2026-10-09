"""
Every analysis endpoint reads its network through one accessor (sandbox
phase 0, `sandbox_and_recommender.md`).

`_net_for` / `_db_for` are where a request's `network` — the base, or later a
proposal — is resolved. These tests keep it the only way in: a new endpoint
that reads `app_data["net"]` directly would study the base even when asked
about a proposal, silently.
"""

import pathlib
import re
import types

import pytest
from fastapi import HTTPException

import main_backend as mb

SOURCE = pathlib.Path(mb.__file__).read_text(encoding="utf-8")

# Functions that load, replace or describe the base itself rather than study a
# network: they may touch app_data directly.
LOADERS = {
    "_restore_timeseries_from_sentinel", "_lifespan_generic", "_lifespan_from_uploaded",
    "health", "network_topology", "get_grid_constants_endpoint", "upload_timeseries",
    "upload_network", "upload_advanced_admittance", "reset_active_network",
    "_net_for", "_db_for", "_build_and_store_ybus",
}


def _direct_reads() -> list[str]:
    found, current = [], None
    for line in SOURCE.split("\n"):
        m = re.match(r"(async )?def (\w+)\(", line)
        if m:
            current = m.group(2)
        if current in LOADERS or re.match(r'\s*app_data\["(net|db_\w+)"\]\s*=', line):
            continue
        if re.search(r'app_data\["(net|db_full)"\]', line):
            found.append(f"{current}: {line.strip()}")
    return found


def test_no_analysis_reads_the_network_directly():
    assert _direct_reads() == []


def test_the_base_is_the_default():
    for request in (None, types.SimpleNamespace(), types.SimpleNamespace(network=None),
                    types.SimpleNamespace(network="base"), types.SimpleNamespace(network=" ")):
        assert mb._network_name(request) == mb.BASE_NETWORK


def test_an_unknown_network_is_refused_not_answered_from_the_base():
    with pytest.raises(HTTPException) as exc:
        mb._net_for(types.SimpleNamespace(network="dc_bus9"))
    assert exc.value.status_code == 400 and "dc_bus9" in exc.value.detail


def test_no_network_loaded_is_a_503(monkeypatch):
    monkeypatch.setitem(mb.app_data, "net", None)
    with pytest.raises(HTTPException) as exc:
        mb._net_for()
    assert exc.value.status_code == 503


def test_databases_by_kind(monkeypatch):
    monkeypatch.setitem(mb.app_data, "db_full", {"x": 1})
    assert mb._db_for("full") == {"x": 1}
    with pytest.raises(KeyError):
        mb._db_for("n1_line")  # outages need no database of their own


def test_endpoints_still_answer_through_it(backend_client):
    r = backend_client.post("/api/grid/rsa", json={})
    assert r.status_code == 200 and "secure" in r.json()
