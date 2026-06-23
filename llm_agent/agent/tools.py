"""
tools.py — HTTP wrapper tool functions for the digital twin agent.

Every function:
- Accepts **kwargs for forward-compatibility with new backend fields.
- Passes kwargs through to the request body via {**required_fields, **kwargs}.
- Appends (function_name, result) to _last_tool_results before returning.
- Returns a plain dict (JSON-serializable) on success.
- Returns {"error": str, "endpoint": str, "status_code": int | None} on failure.
"""

from __future__ import annotations

import difflib
import re
import httpx
from .config import BASE_URL, HTTP_TIMEOUT

# ---------------------------------------------------------------------------
# Shared state — cleared by loop.py at the start of every agent turn.
# Streamlit reads this list to decide which charts to render.
# ---------------------------------------------------------------------------
_last_tool_results: list[tuple[str, dict]] = []
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _unsupported_kwargs_result(tool_name: str, kwargs: dict, supported_extra: set[str]) -> dict | None:
    """Return a structured error when a tool receives unsupported extra args."""
    unsupported = sorted(set(kwargs) - supported_extra)
    if not unsupported:
        return None

    supported_text = ", ".join(sorted(supported_extra)) if supported_extra else "none"
    return _record(
        tool_name,
        {
            "error": (
                f"Unsupported arguments for {tool_name}: {unsupported}. "
                f"Supported extra arguments: {supported_text}."
            ),
            "unsupported_arguments": unsupported,
            "supported_extra_arguments": sorted(supported_extra),
        },
    )


def _post(endpoint: str, body: dict) -> dict:
    """Execute a POST request and return parsed JSON or an error dict."""
    url = f"{BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.post(url, json=body)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = None
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                detail = payload.get("detail")
        except Exception:  # noqa: BLE001
            detail = None
        result = {
            "error": str(exc),
            "endpoint": url,
            "status_code": exc.response.status_code,
        }
        if detail is not None:
            result["detail"] = detail
        return result
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "endpoint": url, "status_code": None}


def _get(endpoint: str) -> dict:
    """Execute a GET request and return parsed JSON or an error dict."""
    url = f"{BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        return {
            "error": str(exc),
            "endpoint": url,
            "status_code": exc.response.status_code,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "endpoint": url, "status_code": None}


def _record(name: str, result: dict) -> dict:
    """Append result to _last_tool_results and return it."""
    _last_tool_results.append((name, result))
    return result


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _expand_day_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if _DATE_ONLY_RE.fullmatch(text):
        return f"{text} 00:00:00"
    return text


def _split_historical_target(target: str) -> tuple[str, str | None]:
    raw = str(target or "all").strip()
    if ":" not in raw:
        return "all", None
    kind, ident = raw.split(":", 1)
    kind = kind.strip().lower()
    ident = ident.strip()
    if kind in {"bus", "line", "trafo"} and ident:
        return kind, ident
    return "all", None


def _fetch_target_candidates(kind: str) -> list[str]:
    snapshot = _post(
        "/api/grid/rsa",
        {
            "load_scaling_factor": 1.0,
            "vm_upper_pu": 1.05,
            "vm_lower_pu": 0.95,
            "max_line_loading_pct": 90.0,
            "max_trafo_loading_pct": 90.0,
        },
    )
    if "error" in snapshot:
        return []

    if kind == "bus":
        raw_names = [str(row.get("bus_name", "")).strip() for row in snapshot.get("all_voltages", [])]
    elif kind == "line":
        raw_names = [str(row.get("line_name", "")).strip() for row in snapshot.get("all_line_loading", [])]
    elif kind == "trafo":
        raw_names = [str(row.get("trafo_name", "")).strip() for row in snapshot.get("all_trafo_loading", [])]
    else:
        return []

    deduped: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        if not name or name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _best_target_candidate(identifier: str, candidates: list[str]) -> tuple[str | None, float]:
    ident = str(identifier or "").strip()
    if not ident or not candidates:
        return None, 0.0

    ident_norm = _normalize_text(ident)
    for cand in candidates:
        if cand.strip().lower() == ident.lower():
            return cand, 1.0

    contains = [cand for cand in candidates if ident_norm and ident_norm in _normalize_text(cand)]
    if len(contains) == 1:
        return contains[0], 0.95

    best = None
    best_score = 0.0
    for cand in candidates:
        score = difflib.SequenceMatcher(None, ident_norm, _normalize_text(cand)).ratio()
        if score > best_score:
            best_score = score
            best = cand
    return best, best_score


# ---------------------------------------------------------------------------
# 2.1 — Current timestamp
# ---------------------------------------------------------------------------

def get_current_timestamp() -> dict:
    """
    GET /api/time/current

    Returns the current simulation timestamp.
    """
    result = _get("/api/time/current")
    return _record("get_current_timestamp", result)


# ---------------------------------------------------------------------------
# 2.2 — Advance timestamp
# ---------------------------------------------------------------------------

def advance_timestamp(steps: int = 1, target_timestamp: str | None = None, **kwargs) -> dict:
    """
    POST /api/time/advance

    If target_timestamp is given, jumps the simulation clock directly to that
    timestamp in a single API call (prefix matching supported on the server).
    Otherwise advances by `steps` 15-minute ticks sequentially.
    """
    if target_timestamp is not None:
        result = _post("/api/time/advance", {"target_timestamp": target_timestamp})
        if "error" in result:
            return _record("advance_timestamp", result)
        ts = result.get("new_timestamp", result.get("current_timestamp", ""))
        return _record("advance_timestamp", {"current_timestamp": ts, "timestamps_traversed": [ts]})

    timestamps_traversed: list[str] = []
    current_timestamp: str = ""

    for _ in range(steps):
        body: dict = {**kwargs}
        result = _post("/api/time/advance", body)
        if "error" in result:
            error_result = {**result, "timestamps_traversed": timestamps_traversed}
            return _record("advance_timestamp", error_result)
        ts = result.get("current_timestamp", result.get("new_timestamp", result.get("timestamp", "")))
        timestamps_traversed.append(ts)
        current_timestamp = ts

    final = {
        "timestamps_traversed": timestamps_traversed,
        "current_timestamp": current_timestamp,
    }
    return _record("advance_timestamp", final)


# ---------------------------------------------------------------------------
# 2.3 — Run real-time security assessment
# ---------------------------------------------------------------------------

def run_rsa(
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/grid/rsa

    vm_upper_pu / vm_lower_pu: voltage violation thresholds in p.u.
    max_line_loading_pct / max_trafo_loading_pct: thermal overload thresholds.

    Known future kwargs:
    # TODO: tap_override (dict) — transformer tap positions override
    # TODO: slack_max_mw (float) — external-grid capacity cap (currently unused by RSA)
    # TODO: custom_load_profile (dict) — substation name → MW override
    """
    body = {
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    result = _post("/api/grid/rsa", body)
    return _record("run_rsa", result)


# ---------------------------------------------------------------------------
# 2.4 — Simulate single contingency
# ---------------------------------------------------------------------------

def simulate_contingency(
    element_type: str,
    element_index: int,
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/contingency/simulate

    element_type: "line" or "trafo"
    element_index: 0-22 for lines, 0-15 for trafos
    vm_upper_pu / vm_lower_pu: voltage violation thresholds in p.u.
    max_line_loading_pct / max_trafo_loading_pct: thermal overload thresholds.

    Known future kwargs:
    # TODO: simultaneous_outages (list) — list of additional elements for N-2
    # TODO: return_all_voltages (bool) — include full voltage profile in response
    """
    body = {
        "element_type": element_type,
        "element_index": element_index,
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    result = _post("/api/contingency/simulate", body)
    return _record("simulate_contingency", result)


# ---------------------------------------------------------------------------
# 2.5 — Simulate all contingencies (full N-1 sweep)
# ---------------------------------------------------------------------------

def simulate_all_contingencies(
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/contingency/simulate_all

    Executes a full N-1 sweep on the backend — do NOT loop simulate_contingency.
    vm_upper_pu / vm_lower_pu: voltage violation thresholds in p.u.
    max_line_loading_pct / max_trafo_loading_pct: thermal overload thresholds.

    Known future kwargs:
    # TODO: element_types (list) — filter to "line" only or "trafo" only
    # TODO: return_rankings (bool) — request severity-ranked summary from backend
    """
    unsupported = _unsupported_kwargs_result(
        "simulate_all_contingencies",
        kwargs,
        supported_extra={"sgen_scaling_factor", "data_source", "timestamp"},
    )
    if unsupported is not None:
        return unsupported

    body = {
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    result = _post("/api/contingency/simulate_all", body)
    return _record("simulate_all_contingencies", result)


# ---------------------------------------------------------------------------
# 2.6 — Optimize post-contingency dispatch
# ---------------------------------------------------------------------------

def optimize_contingency(
    element_type: str,
    element_index: int,
    load_scaling_factor: float = 1.0,
    slack_max_mw: float | None = None,
    opf_vm_upper: float = 1.05,
    opf_vm_lower: float = 0.95,
    opf_lambda_p: float = 0.01,
    opf_lambda_q: float = 0.001,
    pg_max_overrides: dict | None = None,
    pg_min_overrides: dict | None = None,
    fixed_setpoints: dict | None = None,
    opf_min_power_factor: float = 0.95,
    opf_current_safety_margin: float = 0.9,
    **kwargs,
) -> dict:
    """
    POST /api/contingency/optimize

    slack_max_mw: External-grid capacity cap in MW. Omit to use the loaded profile limit.
    opf_vm_upper / opf_vm_lower: voltage bounds enforced inside the AC OPF.
    opf_lambda_p / opf_lambda_q: objective weights on P and Q deviation.
    pg_max_overrides / pg_min_overrides: per-substation capacity overrides (MW).
    fixed_setpoints: pin specific generators/slack at exact MW values.
    opf_min_power_factor: minimum generator power factor (default 0.95).
    opf_current_safety_margin: branch current limit safety factor (default 0.9).

    Known future kwargs:
    # TODO: disabled_generators (list) — substation names forced offline
    # TODO: flexibility_bounds (dict) — per-substation {min, max} override
    """
    body = {
        "element_type": element_type,
        "element_index": element_index,
        "load_scaling_factor": load_scaling_factor,
        "slack_max_mw": slack_max_mw,
        "opf_vm_upper": opf_vm_upper,
        "opf_vm_lower": opf_vm_lower,
        "opf_lambda_p": opf_lambda_p,
        "opf_lambda_q": opf_lambda_q,
        "pg_max_overrides": pg_max_overrides or {},
        "pg_min_overrides": pg_min_overrides or {},
        "fixed_setpoints": fixed_setpoints or {},
        "opf_min_power_factor": opf_min_power_factor,
        "opf_current_safety_margin": opf_current_safety_margin,
        **kwargs,
    }
    result = _post("/api/contingency/optimize", body)
    return _record("optimize_contingency", result)


# ---------------------------------------------------------------------------
# 2.7 — Optimize flexibility (normal operation)
# ---------------------------------------------------------------------------

def optimize_flexibility(
    load_scaling_factor: float = 1.0,
    disabled_generators: list[str] | None = None,
    slack_max_mw: float | None = None,
    slack_q_max_mvar: float | None = None,
    opf_vm_upper: float = 1.05,
    opf_vm_lower: float = 0.95,
    opf_lambda_p: float = 0.01,
    opf_lambda_q: float = 0.001,
    pg_max_overrides: dict | None = None,
    pg_min_overrides: dict | None = None,
    fixed_setpoints: dict | None = None,
    opf_min_power_factor: float = 0.95,
    opf_current_safety_margin: float = 0.9,
    **kwargs,
) -> dict:
    """
    POST /api/flexibility/optimize

    disabled_generators: list of substation names whose generators are forced offline.
    slack_max_mw: External-grid capacity cap in MW. Omit to use the loaded profile limit.
    slack_q_max_mvar: separate reactive power cap on the external-grid interface (Mvar). None = same as slack_max_mw.
    opf_vm_upper / opf_vm_lower: voltage bounds enforced inside the AC OPF.
    opf_lambda_p / opf_lambda_q: objective weights on P and Q deviation.
    pg_max_overrides / pg_min_overrides: per-substation capacity overrides (MW).
    fixed_setpoints: pin specific generators/slack at exact MW values.
    opf_min_power_factor: minimum generator power factor (default 0.95).
    opf_current_safety_margin: branch current limit safety factor (default 0.9).

    Known future kwargs:
    # TODO: target_import_mw (float) — desired external-grid import setpoint
    # TODO: flexibility_bounds (dict) — per-substation {min, max} override
    """
    body = {
        "load_scaling_factor": load_scaling_factor,
        "disabled_generators": disabled_generators or [],
        "slack_max_mw": slack_max_mw,
        "slack_q_max_mvar": slack_q_max_mvar,
        "opf_vm_upper": opf_vm_upper,
        "opf_vm_lower": opf_vm_lower,
        "opf_lambda_p": opf_lambda_p,
        "opf_lambda_q": opf_lambda_q,
        "pg_max_overrides": pg_max_overrides or {},
        "pg_min_overrides": pg_min_overrides or {},
        "fixed_setpoints": fixed_setpoints or {},
        "opf_min_power_factor": opf_min_power_factor,
        "opf_current_safety_margin": opf_current_safety_margin,
        **kwargs,
    }
    result = _post("/api/flexibility/optimize", body)
    return _record("optimize_flexibility", result)


# ---------------------------------------------------------------------------
# 2.8 — Evaluate KPIs
# ---------------------------------------------------------------------------

def evaluate_kpis(
    load_scaling_factor: float = 1.0,
    slack_max_mw: float | None = None,
    slack_q_max_mvar: float | None = None,
    opf_vm_upper: float = 1.05,
    opf_vm_lower: float = 0.95,
    opf_lambda_p: float = 0.01,
    opf_lambda_q: float = 0.001,
    pg_max_overrides: dict | None = None,
    pg_min_overrides: dict | None = None,
    fixed_setpoints: dict | None = None,
    opf_min_power_factor: float = 0.95,
    opf_current_safety_margin: float = 0.9,
    **kwargs,
) -> dict:
    """
    POST /api/kpi/evaluate

    slack_max_mw drives the constrained-scenario solve (external-grid derating).
    slack_q_max_mvar: separate reactive power cap on the external-grid interface (Mvar). None = same as slack_max_mw.
    opf_vm_upper / opf_vm_lower: voltage bounds enforced inside the AC OPF.
    opf_lambda_p / opf_lambda_q: objective weights on P and Q deviation.
    pg_max_overrides / pg_min_overrides: per-substation capacity overrides (MW).
    fixed_setpoints: pin specific generators/slack at exact MW values.
    opf_min_power_factor: minimum generator power factor (default 0.95).
    opf_current_safety_margin: branch current limit safety factor (default 0.9).

    Known future kwargs:
    # TODO: disabled_generators (list) — substation names forced offline
    """
    body = {
        "load_scaling_factor": load_scaling_factor,
        "slack_max_mw": slack_max_mw,
        "slack_q_max_mvar": slack_q_max_mvar,
        "opf_vm_upper": opf_vm_upper,
        "opf_vm_lower": opf_vm_lower,
        "opf_lambda_p": opf_lambda_p,
        "opf_lambda_q": opf_lambda_q,
        "pg_max_overrides": pg_max_overrides or {},
        "pg_min_overrides": pg_min_overrides or {},
        "fixed_setpoints": fixed_setpoints or {},
        "opf_min_power_factor": opf_min_power_factor,
        "opf_current_safety_margin": opf_current_safety_margin,
        **kwargs,
    }
    result = _post("/api/kpi/evaluate", body)
    return _record("evaluate_kpis", result)


# ---------------------------------------------------------------------------
# 2.9 — Forecast KPIs (24 h ahead, ~120 s)
# ---------------------------------------------------------------------------

def forecast_kpis(
    load_scaling_factor: float = 1.0,
    slack_max_mw: float | None = None,
    slack_q_max_mvar: float | None = None,
    opf_vm_upper: float = 1.05,
    opf_vm_lower: float = 0.95,
    opf_lambda_p: float = 0.01,
    opf_lambda_q: float = 0.001,
    pg_max_overrides: dict | None = None,
    pg_min_overrides: dict | None = None,
    opf_min_power_factor: float = 0.95,
    opf_current_safety_margin: float = 0.9,
    **kwargs,
) -> dict:
    """
    POST /api/kpi/forecast

    Returns 96 × 15-min intervals (24 hours) of KPI-1 forecasts.
    This is a slow call (~120 s). Only invoke when user explicitly asks for a
    forecast or "next 24 hours".
    slack_q_max_mvar: separate reactive power cap on cable (Mvar). None = same as slack_max_mw.
    opf_vm_upper / opf_vm_lower: voltage bounds enforced inside the AC OPF.
    opf_lambda_p / opf_lambda_q: objective weights on P and Q deviation.
    pg_max_overrides / pg_min_overrides: per-substation capacity overrides (MW).
    opf_min_power_factor: minimum generator power factor (default 0.95).
    opf_current_safety_margin: branch current limit safety factor (default 0.9).

    Known future kwargs: (none defined yet)
    """
    body = {
        "load_scaling_factor": load_scaling_factor,
        "slack_max_mw": slack_max_mw,
        "slack_q_max_mvar": slack_q_max_mvar,
        "opf_vm_upper": opf_vm_upper,
        "opf_vm_lower": opf_vm_lower,
        "opf_lambda_p": opf_lambda_p,
        "opf_lambda_q": opf_lambda_q,
        "pg_max_overrides": pg_max_overrides or {},
        "pg_min_overrides": pg_min_overrides or {},
        "opf_min_power_factor": opf_min_power_factor,
        "opf_current_safety_margin": opf_current_safety_margin,
        **kwargs,
    }
    result = _post("/api/kpi/forecast", body)
    return _record("forecast_kpis", result)


# ---------------------------------------------------------------------------
# 2.10 — Scan RSA over multiple timesteps (composite tool)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2.x — Current network conditions snapshot
# ---------------------------------------------------------------------------

def get_element_timeseries(
    element_type: str,
    element_name: str,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    n_steps: int | None = None,
    step_size: int = 1,
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/grid/element_timeseries

    Focus tool: scan a time range and return the evolution of one specific
    bus, line, or transformer element over time. Does NOT mutate the clock.

    element_type : "bus" | "line" | "trafo"
    element_name : partial name match (e.g. "Åkirkeby" matches "Åkirkeby 10.5 kV")
    start_timestamp / end_timestamp : ISO prefix of the time window endpoints
    n_steps : cap on number of ticks to scan
    step_size : sample every Nth tick (default 1, use 4 for hourly resolution)
    """
    body: dict = {
        "element_type": element_type,
        "element_name": element_name,
        "step_size": step_size,
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    if start_timestamp is not None:
        body["start_timestamp"] = start_timestamp
    if end_timestamp is not None:
        body["end_timestamp"] = end_timestamp
    if n_steps is not None:
        body["n_steps"] = n_steps
    result = _post("/api/grid/element_timeseries", body)
    return _record("get_element_timeseries", result)


# ---------------------------------------------------------------------------
# 2.x — Current network conditions snapshot
# ---------------------------------------------------------------------------

def get_current_conditions(**kwargs) -> dict:
    """
    POST /api/network/snapshot

    Returns the raw network state at a timestamp:
    per-generator active/reactive power and installed Pmax,
    per-load consumption, external grid import, and network totals.
    Use this whenever the user asks about current generation levels,
    maximum generator capacity, load, or external-grid import.

    Honors data_source ('measurements'|'forecasts') and an optional timestamp
    via kwargs; defaults to the live measurement clock.
    """
    result = _post("/api/network/snapshot", {**kwargs})
    return _record("get_current_conditions", result)


# ---------------------------------------------------------------------------
# 2.x — Compare two optimization results (pure-client, no backend call)
# ---------------------------------------------------------------------------

_OPT_TOOL_NAMES = frozenset({
    "optimize_flexibility", "optimize_contingency", "evaluate_kpis",
})
_RSA_TOOL_NAMES = frozenset({"run_rsa"})
_ALL_COMPARABLE_NAMES = _OPT_TOOL_NAMES | _RSA_TOOL_NAMES


def _normalize_to_comparable(result: dict) -> dict:
    """Adapt an RSA result to the same shape expected by compare_results.

    OPF results are returned unchanged.  RSA results have their voltage
    and dispatch fields remapped so the diff logic can treat both uniformly:
      * ``all_voltages``  → ``bus_voltages_post_opf``
      * ``gen_dispatch``  → ``activated_resources``  (Pg_base == Pg_new,
        so the delta against an OPF result reflects what the optimiser changed)
    """
    if "bus_voltages_post_opf" in result or "activated_resources" in result:
        return result  # already OPF-shaped
    out = dict(result)
    out["_source"] = "rsa"  # tag so dispatch diff can detect RSA origin
    if "all_voltages" in result:
        out["bus_voltages_post_opf"] = result["all_voltages"]
    if "gen_dispatch" in result:
        out["activated_resources"] = [
            {
                "element": g["name"],
                "Pg_base": g["Pg_mw"], "Pg_new": g["Pg_mw"],
                "Qg_base": g["Qg_mvar"], "Qg_new": g["Qg_mvar"],
                "Pg_up": 0.0, "Pg_down": 0.0, "Qg_up": 0.0, "Qg_down": 0.0,
            }
            for g in result["gen_dispatch"]
        ]
    return out


def compare_results(
    label_a: str = "Baseline",
    label_b: str = "Scenario",
    result_a: dict | None = None,
    result_b: dict | None = None,
) -> dict:
    """
    Pure-client diff of two tool results (optimize_flexibility,
    optimize_contingency, or evaluate_kpis). No HTTP call is made.

    If result_a / result_b are not supplied, the function automatically
    uses the last two optimization results recorded in _last_tool_results
    for the current agent turn. This is the preferred usage — the LLM
    should call compare_results(label_a="...", label_b="...") immediately
    after running two optimization tools without re-passing the objects.

    Computes per-generator ΔPg_new and ΔQg_new, per-bus ΔVm (when
    bus_voltages_post_opf is present in both results), and ΔKPI (when
    metrics dicts are present in both results).
    """
    # Auto-populate from the current turn's tool history if not supplied.
    if result_a is None or result_b is None:
        comparable = [
            r for name, r in _last_tool_results if name in _ALL_COMPARABLE_NAMES
        ]
        if len(comparable) < 2:
            return _record("compare_results", {
                "error": "Need at least two comparable results in the current turn. "
                         "Run run_rsa() then optimize_flexibility() (or run two optimizers) "
                         "before calling compare_results."
            })
        result_a = comparable[-2]
        result_b = comparable[-1]
    result_a = _normalize_to_comparable(result_a)
    result_b = _normalize_to_comparable(result_b)
    diff: dict = {
        "label_a": label_a,
        "label_b": label_b,
        "dispatch_diff": [],
        "voltage_diff": [],
        "kpi_diff": {},
    }

    # --- Dispatch diff ---
    # When label_a is raw/RSA state and label_b is OPF, RSA gen_dispatch uses
    # different names ("0", "Slack") than OPF activated_resources ("Gen_bus1",
    # "ExtGrid_0"), and RSA omits condenser reactive power entirely.  In this
    # case use the OPF's own Pg_base/Qg_base as the "before" values — they are
    # the pre-optimisation power flow and match the RSA state after the tap fix.
    # For OPF-vs-OPF comparisons, fall back to the standard name-based merge.
    def _dispatch_map(res: dict) -> dict:
        rows = res.get("activated_resources") or res.get("dispatch_results") or []
        return {r.get("element") or r.get("name"): r for r in rows}

    rows_b = result_b.get("activated_resources") or result_b.get("dispatch_results") or []
    a_is_rsa = result_a.get("_source") == "rsa" or not result_a.get("activated_resources")

    if rows_b and a_is_rsa:
        for r in sorted(rows_b, key=lambda x: x.get("element") or x.get("name") or ""):
            name = r.get("element") or r.get("name")
            pg_a = float(r.get("Pg_base") or 0.0)
            pg_b = float(r.get("Pg_new") or 0.0)
            qg_a = float(r.get("Qg_base") or 0.0)
            qg_b = float(r.get("Qg_new") or 0.0)
            diff["dispatch_diff"].append({
                "name": name,
                "Pg_new_a": round(pg_a, 4),
                "Pg_new_b": round(pg_b, 4),
                "delta_Pg": round(pg_b - pg_a, 4),
                "Qg_new_a": round(qg_a, 4),
                "Qg_new_b": round(qg_b, 4),
                "delta_Qg": round(qg_b - qg_a, 4),
            })
    else:
        map_a = _dispatch_map(result_a)
        map_b = _dispatch_map(result_b)
        for name in sorted(set(map_a) | set(map_b)):
            ra = map_a.get(name, {})
            rb = map_b.get(name, {})
            pg_a = float(ra.get("Pg_new") or ra.get("Pg_base") or 0.0)
            pg_b = float(rb.get("Pg_new") or rb.get("Pg_base") or 0.0)
            qg_a = float(ra.get("Qg_new") or ra.get("Qg_base") or 0.0)
            qg_b = float(rb.get("Qg_new") or rb.get("Qg_base") or 0.0)
            diff["dispatch_diff"].append({
                "name": name,
                "Pg_new_a": round(pg_a, 4),
                "Pg_new_b": round(pg_b, 4),
                "delta_Pg": round(pg_b - pg_a, 4),
                "Qg_new_a": round(qg_a, 4),
                "Qg_new_b": round(qg_b, 4),
                "delta_Qg": round(qg_b - qg_a, 4),
            })

    # --- Voltage diff ---
    # Prefer OPF's own bus_voltages_base (pre-optimisation state) over RSA
    # voltages when result_b is an OPF result, so voltage comparison is
    # self-consistent with the dispatch comparison.
    def _voltage_map(res: dict, use_base: bool = False) -> dict:
        key = "bus_voltages_base" if use_base else "bus_voltages_post_opf"
        return {v["bus_name"]: v["vm_pu"] for v in res.get(key, [])}

    use_base_for_a = a_is_rsa and bool(result_b.get("bus_voltages_base"))
    vm_a = _voltage_map(result_b, use_base=True) if use_base_for_a else _voltage_map(result_a)
    vm_b = _voltage_map(result_b)
    for bus in sorted(set(vm_a) | set(vm_b)):
        va = vm_a.get(bus)
        vb = vm_b.get(bus)
        if va is not None and vb is not None:
            diff["voltage_diff"].append({
                "bus_name": bus,
                "vm_pu_a": round(float(va), 4),
                "vm_pu_b": round(float(vb), 4),
                "delta_vm": round(float(vb) - float(va), 4),
            })

    # --- KPI diff ---
    def _kpi_map(res: dict) -> dict:
        return {k: v for k, v in res.get("metrics", {}).items() if isinstance(v, (int, float))}

    kpi_a = _kpi_map(result_a)
    kpi_b = _kpi_map(result_b)
    for key in sorted(set(kpi_a) | set(kpi_b)):
        ka = kpi_a.get(key)
        kb = kpi_b.get(key)
        if ka is not None and kb is not None:
            diff["kpi_diff"][key] = {
                "value_a": round(float(ka), 2),
                "value_b": round(float(kb), 2),
                "delta": round(float(kb) - float(ka), 2),
            }

    return _record("compare_results", diff)


def scan_rsa_over_time(
    n_steps: int = 4,
    load_scaling_factor: float = 1.0,
    **kwargs,
) -> dict:
    """
    Composite tool: advance + run_rsa repeated n_steps times.

    Collects a time-series of key grid security metrics without adding
    intermediate results to _last_tool_results (only the final summary is added).

    kwargs are passed through to each run_rsa call.
    """
    unsupported = _unsupported_kwargs_result(
        "scan_rsa_over_time",
        kwargs,
        supported_extra={
            "sgen_scaling_factor",
            "vm_upper_pu",
            "vm_lower_pu",
            "max_line_loading_pct",
            "max_trafo_loading_pct",
        },
    )
    if unsupported is not None:
        return unsupported

    timestamps: list[str] = []
    violation_counts: list[int] = []
    min_voltage: list[float] = []
    max_voltage: list[float] = []
    max_line_loading: list[float] = []
    max_trafo_loading: list[float] = []

    for _ in range(n_steps):
        # Advance clock one tick
        adv = _post("/api/time/advance", {})
        if "error" in adv:
            error_result = {**adv, "partial_timestamps": timestamps}
            return _record("scan_rsa_over_time", error_result)
        ts = adv.get("new_timestamp", adv.get("current_timestamp", adv.get("timestamp", "")))
        timestamps.append(ts)

        # Run RSA
        rsa_body = {"load_scaling_factor": load_scaling_factor, **kwargs}
        rsa = _post("/api/grid/rsa", rsa_body)
        if "error" in rsa:
            error_result = {**rsa, "partial_timestamps": timestamps}
            return _record("scan_rsa_over_time", error_result)

        violation_counts.append(rsa.get("total_violations", 0))

        voltages = [v.get("vm_pu", 1.0) for v in rsa.get("all_voltages", [])]
        min_voltage.append(min(voltages) if voltages else 1.0)
        max_voltage.append(max(voltages) if voltages else 1.0)

        line_loadings = [
            l.get("loading_percent", 0.0) for l in rsa.get("all_line_loading", [])
        ]
        max_line_loading.append(max(line_loadings) if line_loadings else 0.0)

        trafo_loadings = [
            t.get("loading_percent", 0.0) for t in rsa.get("all_trafo_loading", [])
        ]
        max_trafo_loading.append(max(trafo_loadings) if trafo_loadings else 0.0)

    result = {
        "timestamps": timestamps,
        "violation_counts": violation_counts,
        "min_voltage": min_voltage,
        "max_voltage": max_voltage,
        "max_line_loading": max_line_loading,
        "max_trafo_loading": max_trafo_loading,
        "any_violations": any(v > 0 for v in violation_counts),
    }
    return _record("scan_rsa_over_time", result)


def find_worst_case_timestamp(
    metric: str = "violations",
    n_steps: int | None = None,
    step_size: int = 1,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/worst_case

    Scans timestamps starting from the current simulation position **without**
    mutating the clock (current_index is preserved on the server).

    Runs pandapower (no OPF) at each tick and returns:
    - worst_timestamp / worst_value for the requested metric
    - worst_per_metric: best timestamp for every supported metric
    - series: full time-series lists for charting

    metric options:
      "violations"   — most simultaneous security violations (voltage + thermal)
    "slack_import" — highest external-grid import (MW)
      "max_voltage"  — highest bus voltage (p.u.)
      "min_voltage"  — lowest bus voltage (p.u.) — worst = most negative
      "max_loading"  — highest line loading (%)
    """
    body: dict = {"metric": metric, "step_size": step_size, **kwargs}
    if n_steps is not None:
        body["n_steps"] = n_steps
    result = _post("/api/rsa/worst_case", body)
    return _record("find_worst_case_timestamp", result)


def scan_scenarios(
    sgen_scales: list | None = None,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    n_steps: int | None = None,
    step_size: int = 1,
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/scan_scenarios

    Run RSA across a time range for multiple renewable output scaling factors
    simultaneously (parallel on the server). Does NOT advance the clock.

    sgen_scales : list of floats, e.g. [0.7, 1.0, 1.3] for P10/P50/P90.
                  Each factor multiplies all sgen p_mw and q_mvar columns.
                  Default: [0.7, 1.0, 1.3]
    """
    body: dict = {
        "step_size": step_size,
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    if sgen_scales is not None:
        body["sgen_scales"] = sgen_scales
    if start_timestamp is not None:
        body["start_timestamp"] = start_timestamp
    if end_timestamp is not None:
        body["end_timestamp"] = end_timestamp
    if n_steps is not None:
        body["n_steps"] = n_steps
    result = _post("/api/rsa/scan_scenarios", body)
    return _record("scan_scenarios", result)


# ---------------------------------------------------------------------------
# Probabilistic RSA (Monte Carlo)
# ---------------------------------------------------------------------------

def run_probabilistic_rsa(
    n_samples: int = 200,
    sgen_sigma: float = 0.0,
    load_sigma: float = 0.05,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/probabilistic

    Run a Monte Carlo probabilistic security assessment at the current timestamp.
        Draws n_samples Latin Hypercube samples for generation and load uncertainty,
        runs a power flow for each, and returns:
    - per-bus/line/trafo violation probability
    - voltage P5/P50/P95 envelopes per bus
    - distribution of total violation count
    - overall probability that any violation occurs

        Guidance on uncertainty meaning:
        - By default, `sgen_sigma=0.0` (no generator uncertainty) and only load
            uncertainty is active unless the user explicitly requests generator uncertainty.
        - `sgen_sigma` treats current sgen `p_mw` as a forecast estimate of renewable
            availability, samples forecast error around it, clips by physical capacity,
            and applies setpoint ceiling logic (available resource, not commanded setpoint,
            is uncertain).
        - `load_sigma` remains multiplicative demand uncertainty around current load.
    """
    body: dict = {
        "n_samples": n_samples,
        "sgen_sigma": sgen_sigma,
        "load_sigma": load_sigma,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    result = _post("/api/rsa/probabilistic", body)
    return _record("run_probabilistic_rsa", result)


def optimize_robust_flexibility(
    robust_method: str = "heuristic",
    risk_target: float = 0.05,
    sgen_sigma: float = 0.0,
    load_sigma: float = 0.05,
    confidence: float | None = None,
    n_samples: int = 200,
    validation_samples: int | None = None,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    slack_max_mw: float | None = None,
    load_scaling_factor: float = 1.0,
    target_p_any: float | None = None,
    max_iter: int = 3,
    min_improvement: float = 0.005,
    alpha: float | None = None,
    beta: float = 1e-3,
    n_scenarios: int | None = None,
    scenario_k_cap: int = 120,
    allowed_violation_fraction: float = 0.0,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/flexibility/robust

    Robust OPF via constraint tightening (back-off approach).

    Orchestrates:
    1. Probabilistic RSA → per-bus voltage P-confidence envelope.
     2. Compute symmetric per-bus back-offs:
         - upper: Δu_b = max(0, V_b,pct - V_b,base)
         - lower: Δl_b = max(0, V_b,base - V_b,1-pct)
     3. Solve OPF with tightened per-bus bounds:
         - vm_upper_b = vm_upper_pu - Δu_b
         - vm_lower_b = vm_lower_pu + Δl_b
     4. Iterate tighten-and-solve until target_p_any is reached or improvement stalls.
     5. Certify post-OPF risk on an independent validation sample set.

    Returns:
    - activated_resources (same as optimize_flexibility)
    - bus_voltages_post_opf
    - back_off_upper_per_bus / back_off_lower_per_bus
    - tightened_upper_bounds / tightened_lower_bounds
    - p_any_violation_before / p_any_violation_after
    - p_any_violation_after_validation
    - robust_loop_iterations / robust_loop_stop_reason
    - confidence, sgen_sigma, n_samples, validation_samples used
    """
    body: dict = {
        "risk_target": risk_target,
        "sgen_sigma": sgen_sigma,
        "load_sigma": load_sigma,
        "robust_method": robust_method,
        "n_samples": n_samples,
        "validation_samples": validation_samples,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "slack_max_mw": slack_max_mw,
        "load_scaling_factor": load_scaling_factor,
        "target_p_any": target_p_any,
        "max_iter": max_iter,
        "min_improvement": min_improvement,
        "beta": beta,
        "n_scenarios": n_scenarios,
        "scenario_k_cap": scenario_k_cap,
        "allowed_violation_fraction": allowed_violation_fraction,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
    }
    if confidence is not None:
        body["confidence"] = confidence
    if alpha is not None:
        body["alpha"] = alpha
    if target_p_any is not None:
        body["target_p_any"] = target_p_any
    result = _post("/api/flexibility/robust", body)
    return _record("optimize_robust_flexibility", result)


def compute_flexibility_envelope(
    gen_name: str,
    p_min_mw: float | None = None,
    p_max_mw: float | None = None,
    q_min_mvar: float | None = None,
    q_max_mvar: float | None = None,
    resolution: int = 10,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 100.0,
    max_trafo_loading_pct: float = 100.0,
    load_scaling_factor: float = 1.0,
    reference_state: str = "scada",
    dispatch_overrides: dict | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/flexibility/envelope

    Sweeps a single generator over a rectangular (P, Q) grid. The reference
    operating point can be selected with reference_state:
    - scada (default): current measurement-based state
    - post_opf: latest cached optimization dispatch from app_data[last_dispatch_result]
    - custom: apply dispatch_overrides before the sweep

    Key result fields:
    - envelope: list of {p_mw, q_mvar, feasible, max_vm_pu, max_loading_pct,
                          binding_constraint}
    - base_point: {p_mw, q_mvar} — current SCADA setpoint of the generator
    - safe_q_range_at_base_p: [q_min, q_max] — safe reactive range at
      the generator's current active output
    - n_feasible / n_total: count summary
    """
    body: dict = {
        "gen_name": gen_name,
        "resolution": resolution,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        "load_scaling_factor": load_scaling_factor,
        "reference_state": reference_state,
    }
    if p_min_mw is not None:
        body["p_min_mw"] = p_min_mw
    if p_max_mw is not None:
        body["p_max_mw"] = p_max_mw
    if q_min_mvar is not None:
        body["q_min_mvar"] = q_min_mvar
    if q_max_mvar is not None:
        body["q_max_mvar"] = q_max_mvar
    if dispatch_overrides is not None:
        body["dispatch_overrides"] = dispatch_overrides
    body.update(kwargs)
    result = _post("/api/flexibility/envelope", body)
    return _record("compute_flexibility_envelope", result)


def compute_hosting_capacity(
    bus: int | str,
    q_mode: str = "all",
    power_factor: float = 0.95,
    pf_sign: str = "absorbing",
    mode: str = "deterministic",
    risk_threshold: float = 0.05,
    n_samples: int = 200,
    added_gen_sigma: float = 0.1,
    load_sigma: float = 0.0,
    uncertainty_scope: str = "added_generation_only",
    timestamp: str | None = None,
    p_max_search_mw: float | None = None,
    tolerance_mw: float = 0.1,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    **kwargs,
) -> dict:
    """
    POST /api/flexibility/hosting_capacity

    Single-bus hosting-capacity scan.
    Deterministic mode checks one operating point; probabilistic mode accepts
    the largest candidate whose sampled violation probability stays below
    risk_threshold.
    """
    body: dict = {
        "bus": bus,
        "q_mode": q_mode,
        "power_factor": power_factor,
        "pf_sign": pf_sign,
        "mode": mode,
        "risk_threshold": risk_threshold,
        "n_samples": n_samples,
        "added_gen_sigma": added_gen_sigma,
        "load_sigma": load_sigma,
        "uncertainty_scope": uncertainty_scope,
        "tolerance_mw": tolerance_mw,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
    }
    if timestamp is not None:
        body["timestamp"] = timestamp
    if p_max_search_mw is not None:
        body["p_max_search_mw"] = p_max_search_mw
    body.update(kwargs)
    result = _post("/api/flexibility/hosting_capacity", body)
    return _record("compute_hosting_capacity", result)


def compute_historical_risk(
    target: str = "all",
    window_start: str | None = None,
    window_end: str | None = None,
    condition: str | None = None,
    n_bins: int = 4,
    near_miss_band: float = 0.01,
    worst_n: int = 5,
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float = 1.05,
    vm_lower_pu: float = 0.95,
    max_line_loading_pct: float = 90.0,
    max_trafo_loading_pct: float = 90.0,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/historical_risk

    Empirical historical violation / near-miss statistics over a real SCADA
    window. First implementation slice returns exceedance frequency/count,
    worst episodes, and a duration curve over the signed limit-margin metric.
    Conditional binning parameters are accepted but currently deferred.
    """
    resolved_window_start = _expand_day_prefix(window_start)
    resolved_window_end = _expand_day_prefix(window_end)

    body: dict = {
        "target": target,
        "condition": condition,
        "n_bins": n_bins,
        "near_miss_band": near_miss_band,
        "worst_n": worst_n,
        "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        "parallel": parallel,
    }
    if resolved_window_start is not None:
        body["window_start"] = resolved_window_start
    if resolved_window_end is not None:
        body["window_end"] = resolved_window_end
    if max_workers is not None:
        body["max_workers"] = max_workers
    body.update(kwargs)
    result = _post("/api/rsa/historical_risk", body)

    if "error" in result and result.get("status_code") == 400:
        # Retry once if backend reports ambiguous date prefixes.
        detail = str(result.get("detail") or result.get("error", "")).lower()
        if "ambiguous" in detail:
            retried = False
            for key in ("window_start", "window_end"):
                value = body.get(key)
                expanded = _expand_day_prefix(value)
                if expanded is not None and expanded != value:
                    body[key] = expanded
                    retried = True
            if retried:
                result = _post("/api/rsa/historical_risk", body)

    if "error" in result:
        kind, identifier = _split_historical_target(str(body.get("target", target)))
        detail = str(result.get("detail") or result.get("error", "")).lower()
        not_found = "was not found in rsa output" in detail or "target must be" in detail
        if kind != "all" and identifier and result.get("status_code") in {400, 404} and not_found:
            candidates = _fetch_target_candidates(kind)
            ident_norm = _normalize_text(identifier)
            contains = [
                cand for cand in candidates
                if ident_norm and ident_norm in _normalize_text(cand)
            ]

            if len(contains) > 1:
                prioritized = contains + [cand for cand in candidates if cand not in contains]
                result = {
                    **result,
                    "target_suggestions": prioritized[:12],
                    "hint": (
                        "Target name is ambiguous and matches multiple elements. "
                        "Choose one of target_suggestions and retry with the exact label."
                    ),
                }
                return _record("compute_historical_risk", result)

            best, score = _best_target_candidate(identifier, candidates)
            if best and score >= 0.82 and best != identifier:
                body["target"] = f"{kind}:{best}"
                retry = _post("/api/rsa/historical_risk", body)
                if "error" not in retry:
                    retry["target_autocorrected_from"] = f"{kind}:{identifier}"
                    retry["target_autocorrected_to"] = body["target"]
                    return _record("compute_historical_risk", retry)
                result = retry
            if candidates:
                result = {
                    **result,
                    "target_suggestions": candidates[:12],
                    "hint": (
                        "Target name did not match exact RSA labels. "
                        "Retry with one of target_suggestions, or allow autocorrect by using a closer name."
                    ),
                }

    return _record("compute_historical_risk", result)
