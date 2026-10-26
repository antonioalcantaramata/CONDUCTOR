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

import contextlib
import contextvars
import difflib
import json
import re
import httpx
from .config import BASE_URL, HTTP_TIMEOUT
from .investigation_record import REQUESTS_FILE as TOOL_REQUESTS_FILE

# ---------------------------------------------------------------------------
# Shared state — cleared by loop.py at the start of every agent turn.
# Streamlit reads this list to decide which charts to render.
# ---------------------------------------------------------------------------
_last_tool_results: list[tuple[str, dict]] = []
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _unsupported_kwargs_result(tool_name: str, kwargs: dict, supported_extra: set[str]) -> dict | None:
    """Return a structured error when a tool receives unsupported extra args.

    The wording carries a deliberate instruction, and the reason is the
    system-prompt ablation. Without the prompt's refusal rule, an unsupported
    request — generator unavailability combined with N-1 screening — was not
    declined: the model improvised a workaround and reported a result as if the
    combined analysis were valid. With the rule it declined, but the rule is
    prose and the model complied only most of the time.

    `refusal_required` and the sentence beside it move the boundary into the
    payload, where compliance does not depend on how much of the system prompt
    the model is currently weighting. This is the same move `validate_call`
    makes for impossible parameters, one level up: the study a caller asked for
    does not exist, and no combination of the studies that do exist answers it.
    """
    unsupported = sorted(set(kwargs) - supported_extra)
    if not unsupported:
        return None

    supported_text = ", ".join(sorted(supported_extra)) if supported_extra else "none"
    return _record(
        tool_name,
        {
            "error": (
                f"Unsupported arguments for {tool_name}: {unsupported}. "
                f"Supported extra arguments: {supported_text}. "
                "This analysis is not available in the requested combination. "
                "Tell the user it is unsupported and say what is available "
                "instead. Do NOT approximate it with a different study, and do "
                "NOT report a result as though the requested one had run."
            ),
            "unsupported_arguments": unsupported,
            "supported_extra_arguments": sorted(supported_extra),
            "refusal_required": True,
        },
    )


# Security limits are sent only when someone chose them. The backend fills
# every limit it is not sent with the loaded network's own, and honours every
# one it is sent — so a default written here would be indistinguishable from
# an operator asking for exactly that value.
_LIMIT_KEYS = frozenset({
    "vm_upper_pu", "vm_lower_pu", "max_line_loading_pct", "max_trafo_loading_pct",
    "opf_vm_upper", "opf_vm_lower",
})


# Studies that run a full study at each of several moments (~10 s each): waited
# for well past the default request timeout.
LONG_STUDY_TIMEOUT_S = 900.0
_SLOW_ENDPOINTS = frozenset({"/api/flexibility/hold", "/api/flexibility/plan",
                             "/api/flexibility/security_over_time", "/api/kpi/over_time"})


def _timeout_for(endpoint: str) -> float:
    return LONG_STUDY_TIMEOUT_S if endpoint in _SLOW_ENDPOINTS else HTTP_TIMEOUT


# Where a long study reports its progress while it runs: set by the agent
# loop around each tool call (it shows in the app's live status), else None.
progress_hook = None
_PROGRESS_POLL_S = 1.0


def _say_progress(text: str) -> None:
    hook = progress_hook
    if hook is not None:
        try:
            hook(text)
        except Exception:  # noqa: BLE001 — progress is a courtesy, never a failure
            pass


def _post_following(endpoint: str, body: dict, what: str) -> dict:
    """POST a long study and, while it runs, read its progress (GET
    /api/progress/{id}) and pass it on: "3 of 9 moments done (19:00)"."""
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    job = uuid.uuid4().hex
    with ThreadPoolExecutor(1) as pool:
        # In this thread's context: `unrecorded` must hold in the worker too.
        future = pool.submit(contextvars.copy_context().run, _post, endpoint, {**body, "progress_id": job})
        said = None
        while True:
            try:
                return future.result(timeout=_PROGRESS_POLL_S)
            except TimeoutError:
                pass
            try:
                with httpx.Client(timeout=5.0) as client:
                    seen = client.get(f"{BASE_URL}/api/progress/{job}").json()
            except Exception:  # noqa: BLE001 — the study goes on; only its progress is unknown
                continue
            if not seen.get("known") or not seen.get("total"):
                continue
            text = f"{seen.get('done', 0)} of {seen['total']} {what} done"
            if seen.get("workers", 1) > 1:
                text += f" · {seen['workers']} in parallel"
            if seen.get("done"):
                elapsed = float(seen.get("elapsed_s") or 0)
                left = elapsed / seen["done"] * (seen["total"] - seen["done"])
                text += f" · about {max(1, round(left))} s left" if seen["done"] < seen["total"] else ""
            if text != said:
                said = text
                _say_progress(text)


# Endpoints that keep what they compute (an option set, "the last dispatch"):
# told not to while a check re-runs a study (`unrecorded`).
_KEEPS_RESULTS = ("/api/flexibility/optimize", "/api/contingency/optimize")


def _post(endpoint: str, body: dict) -> dict:
    """Execute a POST request and return parsed JSON or an error dict."""
    body = {k: v for k, v in body.items() if not (k in _LIMIT_KEYS and v is None)}
    if not _RECORDING.get() and endpoint in _KEEPS_RESULTS:
        body["remember"] = False
    url = f"{BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=_timeout_for(endpoint)) as client:
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


# Enough to describe a substation without ever returning a network.
_MAX_NEIGHBOURS = 12


# False while a check re-runs studies (`unrecorded`).
_RECORDING: contextvars.ContextVar = contextvars.ContextVar("recording", default=True)


@contextlib.contextmanager
def unrecorded():
    """Tool calls inside leave no trace: their results are not added to
    `_last_tool_results` (the answer's charts), and the backend keeps nothing
    they compute (no option set, no "last dispatch"). For a check that re-runs
    a study under another reading (robustness.py)."""
    token = _RECORDING.set(False)
    try:
        yield
    finally:
        _RECORDING.reset(token)


def _record(name: str, result: dict) -> dict:
    """Append result to _last_tool_results and return it (not inside `unrecorded`)."""
    if _RECORDING.get():
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
    reached_end = False

    for _ in range(steps):
        body: dict = {**kwargs}
        result = _post("/api/time/advance", body)
        if "error" in result:
            error_result = {**result, "timestamps_traversed": timestamps_traversed}
            return _record("advance_timestamp", error_result)
        if result.get("status") == "finished":
            # The clock is already on the last tick; it stays there. Reading a
            # timestamp out of this reply recorded "" as the current time.
            reached_end = True
            break
        ts = result.get("current_timestamp", result.get("new_timestamp", result.get("timestamp", "")))
        timestamps_traversed.append(ts)
        current_timestamp = ts

    if reached_end and not current_timestamp:
        now = _get("/api/time/current")
        current_timestamp = now.get("current_timestamp", now.get("timestamp", ""))
    final = {
        "timestamps_traversed": timestamps_traversed,
        "current_timestamp": current_timestamp,
        "reached_end_of_data": reached_end,
    }
    return _record("advance_timestamp", final)


# ---------------------------------------------------------------------------
# 2.3 — Run real-time security assessment
# ---------------------------------------------------------------------------

def run_rsa(
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
        supported_extra={"sgen_scaling_factor", "data_source", "timestamp", "network",
                         "load_power_factor", "option"},
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
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
    vm_upper_pu / vm_lower_pu: voltage bounds enforced inside the AC OPF.
    opf_lambda_p / opf_lambda_q: objective weights on P and Q deviation.
    pg_max_overrides / pg_min_overrides: per-substation capacity overrides (MW).
    fixed_setpoints: pin specific generators/slack at exact MW values.
    opf_min_power_factor: minimum generator power factor (default 0.95).
    opf_current_safety_margin: branch current limit safety factor (default 0.9).

    Known future kwargs:
    # TODO: disabled_generators (list) — substation names forced offline
    # TODO: flexibility_bounds (dict) — per-substation {min, max} override
    """
    if slack_max_mw is not None:
        # The contingency endpoint has no external-grid cap. The value used to
        # be sent and silently dropped by the backend, so "outage plus a cable
        # derated to 20 MW" returned an uncapped dispatch presented as capped.
        return _unsupported_kwargs_result(
            "optimize_contingency", {"slack_max_mw": slack_max_mw}, supported_extra=set())
    body = {
        "element_type": element_type,
        "element_index": element_index,
        "load_scaling_factor": load_scaling_factor,
        # The engine spells these `opf_vm_*`; the tool layer does not, so one
        # ceiling has one name everywhere the model can see it. Translated
        # here rather than propagated: an alias the guards have to know about
        # is a guard that can be forgotten.
        "opf_vm_upper": vm_upper_pu,
        "opf_vm_lower": vm_lower_pu,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
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
    vm_upper_pu / vm_lower_pu: voltage bounds enforced inside the AC OPF.
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
        # The engine spells these `opf_vm_*`; the tool layer does not, so one
        # ceiling has one name everywhere the model can see it. Translated
        # here rather than propagated: an alias the guards have to know about
        # is a guard that can be forgotten.
        "opf_vm_upper": vm_upper_pu,
        "opf_vm_lower": vm_lower_pu,
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
# 2.7b — Flexibility over time: hold an option, plan a window, security cost
# ---------------------------------------------------------------------------

_OVER_TIME_ENDPOINTS = {"hold": "/api/flexibility/hold", "plan": "/api/flexibility/plan",
                        "security": "/api/flexibility/security_over_time"}


def flexibility_over_time(study: str = "", option: str | None = None, start: str | None = None,
                          end: str | None = None, hours: float | None = None, moments: int | None = None,
                          n1: str | None = None, screen_n1: bool | None = None, objective: str | None = None,
                          steadiness: float | None = None, ramp_pct_per_h: float | None = None,
                          **kwargs) -> dict:
    """
    POST /api/flexibility/hold | /plan | /security_over_time

    study="hold": a corrective option (`option`, its ref) held from its moment
    over the next `hours` — when it stops keeping the limits (`screen_n1`: the
    N-1 screen at each moment too). study="plan": one corrective plan over a
    window (`start`, `end` or `hours`, `moments`), instructions kept from one
    moment to the next (`steadiness`, `ramp_pct_per_h`), `objective`
    least_disturbance or cheapest. study="security": the cost of N-1 security
    (`n1` preventive / corrective / both) at each moment of a window.
    """
    # Normalised once: "Hold" reached the hold endpoint without its option.
    requested = study
    study = str(study or "").strip().lower()
    endpoint = _OVER_TIME_ENDPOINTS.get(study)
    if endpoint is None:
        return {"error": f"study must be one of {sorted(_OVER_TIME_ENDPOINTS)}; got {requested!r}."}
    unsupported = _unsupported_kwargs_result(
        "flexibility_over_time", kwargs,
        supported_extra={"data_source", "network", "load_power_factor", "load_scaling_factor", "hold_units",
                         "branch_limits", "vm_upper_pu", "vm_lower_pu", "max_line_loading_pct",
                         "max_trafo_loading_pct"})
    if unsupported is not None:
        return unsupported
    if study == "hold" and not option:
        return {"error": "study='hold' needs option: the ref of a corrective option (e.g. 'S3:cheapest')."}
    body = {k: v for k, v in {
        "option": option if study == "hold" else None, "start": start, "end": end, "hours": hours,
        "moments": moments, "n1": (bool(screen_n1) if study == "hold" else n1 if study == "security" else None),
        "objective": objective if study == "plan" else None,
        "steadiness": steadiness if study == "plan" else None,
        "ramp_pct_per_h": ramp_pct_per_h if study == "plan" else None, **kwargs}.items() if v is not None}
    if study == "security":
        # The slow one: its progress is followed and shown while it runs.
        result = _post_following(endpoint, body, "moments")
    else:
        result = _post(endpoint, body)                # a long timeout: _SLOW_ENDPOINTS
    return _record("flexibility_over_time", result)

# ---------------------------------------------------------------------------
# 2.8 — Key performance indicators: at a moment, and over a window
# ---------------------------------------------------------------------------

# What both KPI tools pass on besides the operating point (timestamp, data
# source, network, load power factor — forwarded by the loop).
_KPI_PASSED = {"option", "n1", "hold_units", "branch_limits", "load_scaling_factor", "slack_max_mw",
               "slack_q_max_mvar", "pg_max_overrides", "pg_min_overrides", "fixed_setpoints",
               "opf_current_safety_margin", "opf_lambda_p", "opf_lambda_q", "opf_min_power_factor",
               "max_line_loading_pct", "max_trafo_loading_pct"}
_KPI_POINT = {"timestamp", "data_source", "network", "load_power_factor"}


def _kpi_body(vm_upper_pu, vm_lower_pu, kwargs: dict) -> dict:
    """The request body: one ceiling, one name — a voltage limit given here
    judges each state's power flow and bounds the options' OPF alike."""
    return {k: v for k, v in {"vm_upper_pu": vm_upper_pu, "vm_lower_pu": vm_lower_pu,
                              "opf_vm_upper": vm_upper_pu, "opf_vm_lower": vm_lower_pu, **kwargs}.items()
            if v is not None}


def evaluate_kpis(vm_upper_pu: float | None = None, vm_lower_pu: float | None = None, **kwargs) -> dict:
    """
    POST /api/kpi/evaluate

    The key performance indicators at one moment: the schedule against a
    corrective action — the cheapest at that moment, or `option` (a ref,
    held from its own moment when applied later) — each checked by AC power
    flow. KPI-1/2/3, N-0 and N-1 margins, the flexibility the units can
    deliver within every limit, the flexibility required, and costs (`n1`:
    the cost of N-1 security above the cheapest too).
    """
    unsupported = _unsupported_kwargs_result("evaluate_kpis", kwargs, supported_extra=_KPI_PASSED | _KPI_POINT)
    if unsupported is not None:
        return unsupported
    result = _post("/api/kpi/evaluate", _kpi_body(vm_upper_pu, vm_lower_pu, kwargs))
    return _record("evaluate_kpis", result)


def kpis_over_time(start: str | None = None, end: str | None = None, hours: float | None = None,
                   moments: int | None = None, vm_upper_pu: float | None = None,
                   vm_lower_pu: float | None = None, **kwargs) -> dict:
    """
    POST /api/kpi/over_time

    The indicators of evaluate_kpis at each moment of a window (`start`,
    `end` or `hours`, at most `moments`), each with its own corrective action
    or `option` held: ranges, worst moments, the actions' cost and the MWh
    moved against the MWh required. The moments run in parallel; the progress
    is followed and shown while they do.
    """
    unsupported = _unsupported_kwargs_result("kpis_over_time", kwargs,
                                             supported_extra=_KPI_PASSED | _KPI_POINT)
    if unsupported is not None:
        return unsupported
    body = _kpi_body(vm_upper_pu, vm_lower_pu, {"start": start, "end": end, "hours": hours,
                                                "moments": moments, **kwargs})
    result = _post_following("/api/kpi/over_time", body, "moments")
    return _record("kpis_over_time", result)


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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/grid/element_timeseries

    Focus tool: scan a time range and return the evolution of one specific
    bus, line, or transformer element over time. Does NOT mutate the clock.

    element_type : "bus" | "line" | "trafo"
    element_name : partial name match (e.g. "Alpha" matches "Alpha 10.5 kV")
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

def locate_network_element(element: str, network: str | None = None, **kwargs) -> dict:
    """
    GET /api/network/resolve (which element), then /api/network/topology (what it is joined to).

    Answers "what is X, and what is it joined to" — and returns the index each
    neighbouring branch has, which is what the contingency and optimisation
    tools need to act on it.

    Exists because the model had no way to turn a name into an index. Asked
    which substations would lose supply if the line between Golf and
    Hotel were out, it guessed `element_index=11` — the Foxtrot–Charlie
    line — ran a perfectly good contingency on it, and reported the result as
    the answer. Every figure was real and the element was wrong.

    The full topology never reaches the model: it is thousands of tokens on a
    small network and hundreds of thousands on a large one. Only the matched
    element and its immediate neighbours are returned.
    """
    wanted = str(element or "").strip()
    if not wanted:
        return {"error": "locate_network_element needs the name of a bus, line or transformer."}
    # Which element: the backend's one lookup, the same every study uses — its
    # label with or without the tag, the tag, its ends, a kind word, an index
    # (observed live: "line Bus_8 -> Bus_9" found nothing here and worked elsewhere).
    on = network if network and network != "base" else None
    from urllib.parse import urlencode
    found = _get("/api/network/resolve?" + urlencode({"name": wanted, **({"network": on} if on else {})}))
    if "error" in found or not {"element_type", "index"} <= set(found):
        return {"error": found.get("detail") or found.get("error") or f"The name lookup gave no element: {found}"}

    # On a proposal the topology includes the buses and lines its edits added.
    topology = _get("/api/network/topology" + (f"?network={on}" if on else ""))
    if "error" in topology:
        return topology
    buses = {int(b["index"]): b for b in topology.get("buses") or []}
    branches = topology.get("branches") or []
    index = int(found["index"])

    if found["element_type"] == "bus":
        bus = buses.get(index, {"name": found["name"], "index": index})
        neighbours = []
        for branch in branches:
            ends = (int(branch["from_bus"]), int(branch["to_bus"]))
            if index not in ends:
                continue
            other = ends[1] if ends[0] == index else ends[0]
            neighbours.append({
                "name": buses.get(other, {}).get("name", f"Bus_{other}"),
                "bus_index": other,
                "via": branch["name"],
                "branch_kind": branch["kind"],
                # The name the contingency and optimisation tools expect.
                "element_type": branch["kind"],
                "element_index": int(branch["index"]),
                "in_service": branch.get("in_service", True),
            })
        return _record("locate_network_element", {
            "element": bus["name"],
            "kind": "bus",
            "bus_index": index,
            "vn_kv": bus.get("vn_kv"),
            # Observed live: a 1 kV placeholder read as the bus's real voltage, and a
            # 20 kV substation at it refused as a mismatch.
            **({"vn_kv_note": topology["voltage_note"]} if topology.get("voltage_note") else {}),
            "in_service": bus.get("in_service", True),
            "connected_to": neighbours[:_MAX_NEIGHBOURS],
            "n_connections": len(neighbours),
            "truncated": len(neighbours) > _MAX_NEIGHBOURS,
            **({"network": topology["network"]} if "network" in topology else {}),
        })

    branch = next((b for b in branches if b["kind"] == found["element_type"] and int(b["index"]) == index), None)
    if branch is None:
        return {"error": f"{found['name']} is not in the topology of this network."}
    return _record("locate_network_element", {
        "element": branch["name"],
        "kind": branch["kind"],
        "element_type": branch["kind"],
        "element_index": int(branch["index"]),
        "between": [buses.get(int(branch["from_bus"]), {}).get("name"),
                    buses.get(int(branch["to_bus"]), {}).get("name")],
        "in_service": branch.get("in_service", True),
        **({"network": topology["network"]} if "network" in topology else {}),
    })


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


def _first_number(*values) -> float:
    """The first value that is a number — 0.0 included — else 0.0.

    `a or b` treated a real 0.0 as missing: a unit curtailed from 4 MW to 0 MW
    was compared as 4 → 4 (its base), and with the base falling through to
    another field the reported change even flipped sign.
    """
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _unusable_for_comparison(result: dict) -> str | None:
    """Why a result cannot be one side of a comparison, or None if it can."""
    if not isinstance(result, dict):
        return "it is not a result"
    if "error" in result:
        return "it is an error"
    if result.get("feasible") is False:
        return "it is infeasible"
    status = str(result.get("status", "")).lower()
    if status in {"infeasible", "failed", "error", "not_converged"}:
        return f"its status is {status!r}"
    return None


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
        # Infeasible or failed results are not states to compare against: an
        # infeasible OPF lists no resources, and diffing it against an RSA
        # reported every generator dropping to 0 MW.
        comparable = [
            r for name, r in _last_tool_results
            if name in _ALL_COMPARABLE_NAMES and _unusable_for_comparison(r) is None
        ]
        if len(comparable) < 2:
            return _record("compare_results", {
                "error": "Need at least two comparable results in the current turn. "
                         "Run run_rsa() then optimize_flexibility() (or run two optimizers) "
                         "before calling compare_results."
            })
        result_a = comparable[-2]
        result_b = comparable[-1]
    for label, res in ((label_a, result_a), (label_b, result_b)):
        reason = _unusable_for_comparison(res)
        if reason is not None:
            return _record("compare_results", {
                "error": f"Cannot compare {label!r}: {reason}. Compare only results "
                         "that completed and are feasible."})
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
            pg_a = _first_number(r.get("Pg_base"))
            pg_b = _first_number(r.get("Pg_new"))
            qg_a = _first_number(r.get("Qg_base"))
            qg_b = _first_number(r.get("Qg_new"))
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
            # A unit missing from one side did not move there (contingency
            # results list only the units that did), so it sits at its base —
            # taken from the other side, not assumed to be 0 MW.
            pg_a = _first_number(ra.get("Pg_new"), ra.get("Pg_base"), rb.get("Pg_base"))
            pg_b = _first_number(rb.get("Pg_new"), rb.get("Pg_base"), ra.get("Pg_base"))
            qg_a = _first_number(ra.get("Qg_new"), ra.get("Qg_base"), rb.get("Qg_base"))
            qg_b = _first_number(rb.get("Qg_new"), rb.get("Qg_base"), ra.get("Qg_base"))
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
            "network",
            "load_power_factor",
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
    min_voltage: list[float | None] = []
    max_voltage: list[float | None] = []
    max_line_loading: list[float] = []
    max_trafo_loading: list[float] = []
    thresholds_used = None
    reached_end = False

    def _peak(rows: list, default: float) -> float:
        # Out-of-service elements report no loading (None), not 0 %.
        values = [r.get("loading_percent") for r in rows]
        values = [v for v in values if isinstance(v, (int, float))]
        return max(values) if values else default

    for _ in range(n_steps):
        # Advance clock one tick
        adv = _post("/api/time/advance", {})
        if "error" in adv:
            error_result = {**adv, "partial_timestamps": timestamps}
            return _record("scan_rsa_over_time", error_result)
        if adv.get("status") == "finished":
            # Past the last tick the backend no longer advances. Carrying on
            # re-assessed the final tick under an empty timestamp, n times.
            reached_end = True
            break
        ts = adv.get("new_timestamp", adv.get("current_timestamp", adv.get("timestamp", "")))
        timestamps.append(ts)

        # Run RSA
        rsa_body = {"load_scaling_factor": load_scaling_factor, **kwargs}
        rsa = _post("/api/grid/rsa", rsa_body)
        if "error" in rsa:
            error_result = {**rsa, "partial_timestamps": timestamps}
            return _record("scan_rsa_over_time", error_result)

        violation_counts.append(rsa.get("total_violations", 0))
        thresholds_used = thresholds_used or rsa.get("thresholds_used")

        # No voltage back is unknown, not 1.0 p.u.
        voltages = [v.get("vm_pu") for v in rsa.get("all_voltages", [])]
        voltages = [v for v in voltages if isinstance(v, (int, float))]
        min_voltage.append(min(voltages) if voltages else None)
        max_voltage.append(max(voltages) if voltages else None)

        max_line_loading.append(_peak(rsa.get("all_line_loading", []), 0.0))
        max_trafo_loading.append(_peak(rsa.get("all_trafo_loading", []), 0.0))

    result = {
        "timestamps": timestamps,
        "reached_end_of_data": reached_end,
        "thresholds_used": thresholds_used,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/probabilistic

    Run a Monte Carlo probabilistic security assessment at one moment: the
    current timestamp, or `timestamp` (forwarded in kwargs) when given.
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
    risk_target: float | None = None,
    sgen_sigma: float = 0.0,
    load_sigma: float = 0.05,
    confidence: float | None = None,
    n_samples: int = 200,
    validation_samples: int | None = None,
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
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
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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

    `risk_target`, `confidence`, `alpha` and `target_p_any` are sent only when
    given: the backend reads a sent `risk_target` as chosen and refuses a
    `confidence` or `alpha` that disagrees with it, and with none of them it
    applies its own default (0.05).
    """
    body: dict = {
        "sgen_sigma": sgen_sigma,
        "load_sigma": load_sigma,
        "robust_method": robust_method,
        "n_samples": n_samples,
        "validation_samples": validation_samples,
        "vm_upper_pu": vm_upper_pu,
        "vm_lower_pu": vm_lower_pu,
        "slack_max_mw": slack_max_mw,
        "load_scaling_factor": load_scaling_factor,
        "max_iter": max_iter,
        "min_improvement": min_improvement,
        "beta": beta,
        "n_scenarios": n_scenarios,
        "scenario_k_cap": scenario_k_cap,
        "allowed_violation_fraction": allowed_violation_fraction,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
    }
    if risk_target is not None:
        body["risk_target"] = risk_target
    if confidence is not None:
        body["confidence"] = confidence
    if alpha is not None:
        body["alpha"] = alpha
    if target_p_any is not None:
        body["target_p_any"] = target_p_any
    # Everything else the caller passed, `timestamp` and `data_source` above
    # all. Without this the endpoint — which accepts both, via
    # `_TimeseriesRequest` — silently runs at the simulation clock while the
    # rest of the answer describes the requested operating point.
    body.update(kwargs)
    result = _post("/api/flexibility/robust", body)
    return _record("optimize_robust_flexibility", result)


def compute_flexibility_envelope(
    gen_name: str,
    p_min_mw: float | None = None,
    p_max_mw: float | None = None,
    q_min_mvar: float | None = None,
    q_max_mvar: float | None = None,
    resolution: int = 20,
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
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
    # 0 (or less) is no search bound at all: some models fill an optional
    # number with 0, which the backend refuses — the backend's own choice it is.
    if p_max_search_mw is not None and p_max_search_mw > 0:
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
    near_miss_band: float = 0.005,
    worst_n: int = 5,
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/rsa/historical_risk

    Empirical historical violation / near-miss statistics over a window:
    exceedance frequency/count, worst episodes, a duration curve (the signed
    limit margin of one element, or the violation count system-wide) and,
    with `condition`, statistics per bin.
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


def compute_violation_attribution(
    timestamp: str | None = None,
    data_source: str = "measurements",
    load_scaling_factor: float = 1.0,
    vm_lower_pu: float | None = None,
    vm_upper_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    top_n_sources: int = 20,
    top_n_drivers: int = 5,
    **kwargs,
) -> dict:
    """Rank the injections driving each violated element at one operating point."""
    unsupported = _unsupported_kwargs_result(
        "compute_violation_attribution", kwargs,
        {"sgen_scaling_factor", "delta_mw", "delta_mvar", "network", "load_power_factor"},
    )
    if unsupported is not None:
        return unsupported

    body = {
        "data_source": data_source,
        "load_scaling_factor": load_scaling_factor,
        "top_n_sources": top_n_sources,
        "top_n_drivers": top_n_drivers,
        **kwargs,
    }
    if timestamp is not None:
        body["timestamp"] = _expand_day_prefix(timestamp)
    for key, value in (
        ("vm_lower_pu", vm_lower_pu),
        ("vm_upper_pu", vm_upper_pu),
        ("max_line_loading_pct", max_line_loading_pct),
        ("max_trafo_loading_pct", max_trafo_loading_pct),
    ):
        if value is not None:
            body[key] = value

    return _record("compute_violation_attribution", _post("/api/rsa/attribution", body))


def investigate(
    element: str | None = None,
    element_type: str | None = None,
    reference_timestamp: str | None = None,
    reference_data_source: str | None = None,
    reference_network: str | None = None,
    hypotheses: list | None = None,
    classical: list | None = None,
    recurring: bool | None = None,
    check_references: bool | None = None,
    reference_mode: str | None = None,
    outage: str | None = None,
    reference_outage: str | None = None,
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
    reference_start_timestamp: str | None = None,
    reference_end_timestamp: str | None = None,
    timestamp: str | None = None,
    load_scaling_factor: float = 1.0,
    vm_lower_pu: float | None = None,
    vm_upper_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """Why one element's voltage or loading is what it is: its change since a
    reference state, divided among the changes that caused it."""
    unsupported = _unsupported_kwargs_result(
        "investigate", kwargs, {"data_source", "network", "load_power_factor"})
    if unsupported is not None:
        return unsupported
    body = {"load_scaling_factor": load_scaling_factor, **kwargs}
    for key, value in (
        ("element", element), ("element_type", element_type),
        ("timestamp", _expand_day_prefix(timestamp)),
        ("reference_timestamp", _expand_day_prefix(reference_timestamp)),
        ("reference_data_source", reference_data_source), ("reference_network", reference_network),
        ("reference_mode", reference_mode), ("outage", outage), ("reference_outage", reference_outage),
        ("start_timestamp", start_timestamp), ("end_timestamp", end_timestamp),
        ("reference_start_timestamp", reference_start_timestamp),
        ("reference_end_timestamp", reference_end_timestamp),
        ("vm_lower_pu", vm_lower_pu), ("vm_upper_pu", vm_upper_pu),
        ("max_line_loading_pct", max_line_loading_pct), ("max_trafo_loading_pct", max_trafo_loading_pct),
    ):
        # "" is how a model that fills every field leaves one out.
        if value is not None and not (isinstance(value, str) and not value.strip()):
            body[key] = value
    if hypotheses:
        body["hypotheses"] = hypotheses
    if classical:
        body["classical"] = classical
    if recurring:
        body["recurring"] = True
    if check_references is not None:
        body["check_references"] = bool(check_references)
    return _record("investigate", _post("/api/investigate", body))


def test_prevention(
    element: str | None = None,
    element_type: str | None = None,
    actions: list | None = None,
    combine: bool = False,
    warning_margin: float | None = None,
    outage: str | None = None,
    timestamp: str | None = None,
    load_scaling_factor: float = 1.0,
    vm_lower_pu: float | None = None,
    vm_upper_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """Could an episode outside the limits have been avoided, and how early was it seen."""
    unsupported = _unsupported_kwargs_result(
        "test_prevention", kwargs, {"data_source", "network", "load_power_factor"})
    if unsupported is not None:
        return unsupported
    body = {"load_scaling_factor": load_scaling_factor, "combine": bool(combine), **kwargs}
    for key, value in (
        ("element", element), ("element_type", element_type), ("outage", outage),
        ("timestamp", _expand_day_prefix(timestamp)), ("warning_margin", warning_margin),
        ("vm_lower_pu", vm_lower_pu), ("vm_upper_pu", vm_upper_pu),
        ("max_line_loading_pct", max_line_loading_pct), ("max_trafo_loading_pct", max_trafo_loading_pct),
    ):
        if value is not None and not (isinstance(value, str) and not value.strip()):
            body[key] = value
    # A model that fills every field sends zeros for the changes it leaves out.
    cleaned = []
    for a in actions or []:
        if isinstance(a, dict) and str(a.get("unit") or "").strip():
            cleaned.append({k: v for k, v in a.items() if k == "unit" or v not in (None, "", 0, 0.0)})
    if cleaned:
        body["actions"] = cleaned
    return _record("test_prevention", _post("/api/investigate/prevention", body))


def look_ahead(
    start_timestamp: str | None = None,
    hours: float = 24.0,
    elements: list | None = None,
    uncertainty_pct: float = 5.0,
    max_episodes: int | None = None,
    data_source: str = "forecasts",
    load_scaling_factor: float = 1.0,
    vm_lower_pu: float | None = None,
    vm_upper_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """What the forecast says is coming, its cause, when to decide, and what would prevent it."""
    unsupported = _unsupported_kwargs_result("look_ahead", kwargs, {"network", "load_power_factor"})
    if unsupported is not None:
        return unsupported
    body = {"data_source": (data_source or "").strip() or "forecasts", "hours": hours or 24.0,
            "uncertainty_pct": 5.0 if uncertainty_pct in (None, "") else uncertainty_pct,
            "load_scaling_factor": load_scaling_factor, **kwargs}
    for key, value in (
        ("start_timestamp", _expand_day_prefix(start_timestamp)), ("max_episodes", max_episodes or None),
        ("vm_lower_pu", vm_lower_pu), ("vm_upper_pu", vm_upper_pu),
        ("max_line_loading_pct", max_line_loading_pct), ("max_trafo_loading_pct", max_trafo_loading_pct),
    ):
        if value is not None and not (isinstance(value, str) and not value.strip()):
            body[key] = value
    named = [str(e).strip() for e in elements or [] if str(e).strip()]
    if named:
        body["elements"] = named
    return _record("look_ahead", _post("/api/investigate/look_ahead", body))




def _tools_called() -> set[str]:
    """Tools that returned a result (not an error) so far in this question."""
    return {name for name, result in _last_tool_results
            if isinstance(result, dict) and not result.get("error") and name != "record_investigation"}


def _save_tool_requests(symptom: str, rows: list[dict]) -> None:
    """Append the tools an investigation needed and did not have, for the
    developers (`python -m evaluation.tool_requests`). Best effort: a full disk
    must not cost the operator the answer."""
    import datetime
    from .usage import cache_dir

    wanted = [r for r in rows if r["verdict"] == "untested" and r.get("missing_tool")]
    if not wanted or not _RECORDING.get():
        return
    try:
        path = cache_dir() / TOOL_REQUESTS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(path, "a", encoding="utf-8") as fh:
            for r in wanted:
                fh.write(json.dumps({"time": stamp, "symptom": symptom, "hypothesis": r["hypothesis"],
                                     "missing_tool": r["missing_tool"],
                                     "missing_kind": r.get("missing_kind")}) + "\n")
    except OSError:
        pass


def record_investigation(symptom: str = "", hypotheses: list | None = None, **kwargs) -> dict:
    """The agent's record of an investigation: each hypothesis it considered,
    its verdict, and the tools whose results in this question decide it.

    The verdicts are the agent's, but they cannot float free of the studies
    (`investigation_record.validate`): a supported or refuted hypothesis must
    cite tools that returned a result in this question and agree with the
    investigation's own verdict for that factor; an untested one must describe
    a tool CONDUCTOR lacks. Anything else is refused with the reasons, for the
    agent to correct. The tools named as missing are kept for the developers.
    """
    unsupported = _unsupported_kwargs_result("record_investigation", kwargs, set())
    if unsupported is not None:
        return unsupported
    from . import investigation_record as ir

    rows, problems = ir.validate(hypotheses, _tools_called(), ir.study_factors(_last_tool_results))
    if problems:
        return _record("record_investigation", {
            "error": "The investigation record was not accepted: " + " ".join(problems), "problems": problems})
    _save_tool_requests(symptom, rows)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ir.VERDICTS}
    return _record("record_investigation", {
        "status": "recorded", "symptom": symptom, "hypotheses": rows, "counts": counts,
        "tool_suggestions": [r["missing_tool"] for r in rows
                             if r["verdict"] == "untested" and r.get("missing_kind") == ir.ANALYSIS],
        "data_suggestions": [r["missing_tool"] for r in rows
                             if r["verdict"] == "untested" and r.get("missing_kind") == ir.DATA],
        "note": ("In the answer: the supported hypotheses as findings, the refuted ones as ruled out, and the "
                 "untested ones as not tested — with the tool that would be needed — never as findings."),
    })


# ---------------------------------------------------------------------------
# Study assumptions
# ---------------------------------------------------------------------------

def get_data_quality(**kwargs) -> dict:
    """
    GET /api/data/quality

    The quality report of the loaded measurements and forecasts: which columns
    they carry (measured reactive power), coverage, gaps and how they were
    filled, implausible values. Reads the data; studies nothing.
    """
    unsupported = _unsupported_kwargs_result("get_data_quality", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    return _record("get_data_quality", _get("/api/data/quality"))


def set_load_power_factor(value: float | None = None, reset: bool = False, **kwargs) -> dict:
    """
    POST /api/assumptions

    Change the session's load power factor — the default every later study
    uses to give measured loads their reactive power — or, with reset=True,
    go back to 0.95. For one study only, pass `load_power_factor` to that
    study instead.
    """
    unsupported = _unsupported_kwargs_result("set_load_power_factor", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    if reset == (value is not None):
        return {"error": "Give either value (a power factor between 0.5 and 1.0) or reset=true."}
    return _record("set_load_power_factor",
                   _post("/api/assumptions", {"load_power_factor": None if reset else value}))


# The settings /api/generators takes (backend/generators.py FIELDS), by the
# slot the model gives each one's value in.
GENERATOR_CONTROLS = ("voltage", "power_factor", "fixed_q")
_GENERATOR_SLOTS = {"control": "control", "vm_pu": "value", "pf": "value", "q_mvar": "value",
                    "q_min_mvar": "value", "q_max_mvar": "value", "absorbing": "on", "dispatchable": "on"}
GENERATOR_SETTINGS = tuple(_GENERATOR_SLOTS)


def _generator_changes(items: list) -> list[dict] | str:
    """One item per setting (unit, setting, and its value in the slot that
    setting uses) → one change per unit, as /api/generators takes them.

    Only the named setting's slot is read, so a model that fills in every
    slot changes nothing else: given one object per unit with every setting,
    such a model would also have set the reactive limits to 0 and the unit
    not dispatchable."""
    by_unit: dict[str, dict] = {}
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not item.get("unit") or not item.get("setting"):
            return f"Change {n}: give unit, setting (one) and its value."
        unit, setting = str(item["unit"]).strip(), item["setting"]
        slot = _GENERATOR_SLOTS.get(setting)
        if slot is None:
            return f"Change {n}: setting is one of {', '.join(GENERATOR_SETTINGS)}."
        value = item.get(slot)
        if slot == "control" and value not in GENERATOR_CONTROLS:
            return f"Change {n} ({unit}): give control as one of {', '.join(GENERATOR_CONTROLS)}."
        if slot == "value" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return f"Change {n} ({unit}, {setting}): give the number in value."
        if slot == "on" and not isinstance(value, bool):
            return f"Change {n} ({unit}, {setting}): give true or false in on."
        change = by_unit.setdefault(unit.casefold(), {"unit": unit})
        if setting in change:
            return f"Change {n}: {unit} {setting} is given twice."
        change[setting] = value
    return list(by_unit.values())


def set_generator_model(changes: list | None = None, reset: bool = False, **kwargs) -> dict:
    """
    GET / POST /api/generators

    How each generating unit is modelled — voltage control or a fixed power
    factor, its set-point and reactive limits, whether the OPF may dispatch it.
    With no arguments, returns every unit and its model; with `changes` (one
    item per setting), applies them (kept for this network, visible in the
    System tab); reset=True takes every unit back to how it was loaded.
    """
    unsupported = _unsupported_kwargs_result("set_generator_model", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    if not changes and not reset:
        return _record("set_generator_model", _get("/api/generators"))
    if changes and reset:
        return {"error": "Give either changes or reset=true, not both."}
    grouped = _generator_changes(list(changes or []))
    if isinstance(grouped, str):
        return {"error": grouped}
    return _record("set_generator_model", _post("/api/generators", {"changes": grouped, "reset": bool(reset)}))


# The prices /api/costs takes (backend/costs.py FIELDS).
COST_FIELDS = ("cost_eur_per_mwh", "cost_eur_per_mw2h", "cost_eur_per_h", "up_eur_per_mwh",
               "down_eur_per_mwh", "q_eur_per_mvarh", "q_band_tan_phi", "q_penalty_eur_per_mvarh",
               "dr_share_pct", "dr_eur_per_mwh", "relief_eur_per_mwh")


def _cost_changes(items: list) -> list[dict] | str:
    """One item per price (unit, field, value) → one change per unit, as
    /api/costs takes them; a message when an item is incomplete.

    The model is given one price per item, not a unit with every price: a
    model that fills in every property it is shown turned "reactive price to
    0" into all eight prices at 0."""
    by_unit: dict[str, dict] = {}
    for n, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not item.get("unit") or not item.get("field"):
            return f"Change {n}: give unit, field (one price) and value, or to_default=true."
        unit, field, value = str(item["unit"]).strip(), item["field"], item.get("value")
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not item.get("to_default") and not number:
            return f"Change {n} ({unit}, {field}): give a value, or to_default=true."
        # A value of 0 beside to_default=true is a filled-in blank; any other
        # value says two things at once.
        if item.get("to_default") and number and value != 0:
            return f"Change {n} ({unit}, {field}): give either a value or to_default=true, not both."
        change = by_unit.setdefault(unit.casefold(), {"unit": unit})
        if field in change:
            return f"Change {n}: {unit} {field} is given twice."
        change[field] = None if item.get("to_default") else value
    return list(by_unit.values())


def set_unit_costs(changes: list | None = None, reset: bool = False, **kwargs) -> dict:
    """
    GET / POST /api/costs

    Each unit's production cost (c0 + c1·P + c2·P²), redispatch costs up/down
    and reactive price, and the external grid's reactive band, from the case,
    the synthetic rule or the operator. With no arguments, returns them; with
    `changes` (one item per price), applies them; reset=True restores every
    default.
    """
    unsupported = _unsupported_kwargs_result("set_unit_costs", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    if not changes and not reset:
        return _record("set_unit_costs", _get("/api/costs"))
    if changes and reset:
        return {"error": "Give either changes or reset=true, not both."}
    grouped = _cost_changes(list(changes or []))
    if isinstance(grouped, str):
        return {"error": grouped}
    return _record("set_unit_costs", _post("/api/costs", {"changes": grouped, "reset": bool(reset)}))


def rename_elements(changes: list | None = None, reset: bool = False, **kwargs) -> dict:
    """
    GET / POST /api/names

    Rename buses, lines, transformers, loads or units of the loaded network.
    With no arguments, returns every renamed element; reset=True restores
    every name as loaded.
    """
    unsupported = _unsupported_kwargs_result("rename_elements", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    if not changes and not reset:
        result = _get("/api/names")
        if isinstance(result, dict) and "elements" in result:
            result = {k: v for k, v in result.items() if k != "elements"}   # the renamed ones suffice
        return _record("rename_elements", result)
    if changes and reset:
        return {"error": "Give either changes or reset=true, not both."}
    result = _post("/api/names", {"changes": list(changes or []), "reset": bool(reset)})
    if isinstance(result, dict):
        result.pop("elements", None)
    return _record("rename_elements", result)


# ---------------------------------------------------------------------------
# Network proposals (sandbox)
# ---------------------------------------------------------------------------

def edit_proposal(proposal: str, edits: list | None = None, replace: bool = False, **kwargs) -> dict:
    """
    POST /api/proposals/edit

    Create a proposal (the base network plus edits) or add edits to it. The
    base is never modified; nothing is stored when an edit is invalid.
    """
    unsupported = _unsupported_kwargs_result("edit_proposal", kwargs, supported_extra=set())
    if unsupported is not None:
        return unsupported
    body = {"proposal": proposal, "edits": list(edits or []), "replace": bool(replace)}
    return _record("edit_proposal", _post("/api/proposals/edit", body))


def compare_networks(
    proposal: str = "",
    study: str = "rsa",
    load_scaling_factor: float = 1.0,
    vm_upper_pu: float | None = None,
    vm_lower_pu: float | None = None,
    max_line_loading_pct: float | None = None,
    max_trafo_loading_pct: float | None = None,
    **kwargs,
) -> dict:
    """
    POST /api/proposals/compare

    The same study on the base and on a proposal at one operating point, and
    the difference.
    """
    unsupported = _unsupported_kwargs_result(
        "compare_networks", kwargs,
        supported_extra={"data_source", "timestamp", "sgen_scaling_factor", "against", "proposals",
                         "start_timestamp", "end_timestamp", "step_size", "load_power_factor"})
    if unsupported is not None:
        return unsupported
    body = {
        "proposal": proposal, "study": study, "load_scaling_factor": load_scaling_factor,
        "vm_upper_pu": vm_upper_pu, "vm_lower_pu": vm_lower_pu,
        "max_line_loading_pct": max_line_loading_pct,
        "max_trafo_loading_pct": max_trafo_loading_pct,
        **kwargs,
    }
    return _record("compare_networks", _post("/api/proposals/compare", body))
