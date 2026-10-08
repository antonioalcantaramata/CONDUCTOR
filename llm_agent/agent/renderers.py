"""
renderers.py — Plotly chart builders for each digital twin tool result type.

Every renderer:
- Accepts the raw tool result dict.
- Returns a go.Figure (or list of go.Figure for render_rsa).
- Handles empty / missing data gracefully without raising exceptions.
- Uses CHART_THEME for consistent styling.

Exports:
  RENDERER_MAP: dict[str, Callable]  — maps tool name → renderer function
"""

from __future__ import annotations

from typing import Callable
import math
import re

import plotly.graph_objects as go

from .config import grid_limits
from .network_map import MARKER_PX, NetworkMap

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

import plotly.io as pio


class _C:
    """One meaning per colour, for every chart.

    Before this, ~35 literal colours carried overlapping meanings — red marked
    violations but also "max voltage", "before" bars and decreases; orange was a
    violation in one chart and a decrease in the next. Hues follow a
    colour-blind-aware scheme; violations also get a dashed line or an `x`
    marker where it matters, so meaning never rests on hue alone.
    """
    VIOLATION = "#D64545"   # a violation, a limit line — and nothing else
    OK = "#2E9E6B"          # secure, feasible, within limits
    WARN = "#E8A317"        # near a limit, the worst moment, attention
    PRIMARY = "#1D6FE8"     # the main data series (the brand blue)
    TEAL = "#0E9F9A"        # a second data series; an increase
    PURPLE = "#8E5CD9"      # a third data series; a decrease
    NEUTRAL = "#9AA3B2"     # no data, uncontrollable, reference
    INK = "#2E3140"         # text, zero lines
    GRID = "#ECEFF5"


pio.templates["conductor"] = go.layout.Template(
    layout={
        "font": {"family": "Source Sans 3, Source Sans Pro, Inter, system-ui, sans-serif",
                 "size": 12, "color": _C.INK},
        "title": {"font": {"size": 15, "color": _C.INK}, "x": 0.0, "xanchor": "left",
                  "xref": "paper"},
        "colorway": [_C.PRIMARY, _C.TEAL, _C.PURPLE, _C.WARN, _C.NEUTRAL, _C.OK],
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "xaxis": {"gridcolor": _C.GRID, "linecolor": "#D5DAE3", "zerolinecolor": "#C9CFDA",
                  "ticks": "outside", "tickcolor": "#D5DAE3", "title": {"font": {"size": 12}}},
        "yaxis": {"gridcolor": _C.GRID, "linecolor": "#D5DAE3", "zerolinecolor": "#C9CFDA",
                  "ticks": "outside", "tickcolor": "#D5DAE3", "title": {"font": {"size": 12}}},
        "legend": {"font": {"size": 11}, "bgcolor": "rgba(255,255,255,0.8)"},
        "hoverlabel": {"bgcolor": "white", "bordercolor": "#D5DAE3",
                       "font": {"size": 12, "color": _C.INK}},
        "bargap": 0.25,
    }
)
pio.templates.default = "conductor"


CHART_THEME: dict = {
    "template": "conductor",
    "margin": {"l": 40, "r": 20, "t": 50, "b": 60},
}

def _coerce_float(value, fallback: float) -> float:
    """Return value as float when possible, else fallback.

    Some backend payload fields may be None (e.g., non-finite vm_pu sanitized
    to null). Renderers must remain resilient to those values.
    """
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _float_or_none(value) -> float | None:
    """Return a finite float or None when the value is missing/invalid."""
    try:
        if value is None:
            return None
        out = float(value)
        return out if out == out and out not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


# Colour for an element with no value — out of service, or not solved. Never
# the "healthy" colour: an element with no reading has not been shown to be fine.
_NO_DATA_COLOR = "#bbbbbb"


def _over(value, limit) -> bool:
    """`value > limit`, False when either is missing.

    The backend reports NaN results (an out-of-service transformer's loading,
    for one) as None, and a bare comparison raised TypeError — which, before
    the chart guard in app.py, took the whole conversation view down.
    """
    v, lim = _float_or_none(value), _float_or_none(limit)
    return v is not None and lim is not None and v > lim


def _voltage_range(series: list, lower, upper, pad: float = 0.02) -> list[float]:
    """A voltage axis that shows every value and both limits.

    The fixed [0.88, 1.12] it replaces dropped a 0.85 p.u. bus off the chart
    without notice — hiding exactly the severe undervoltage the operator is
    looking for.
    """
    values = [v for v in (_float_or_none(x) for seq in series for x in (seq or [])) if v is not None]
    limits = [v for v in (_float_or_none(lower), _float_or_none(upper)) if v is not None]
    pool = values + limits or [0.9, 1.1]
    return [min(min(pool) - pad, 0.88), max(max(pool) + pad, 1.12)]


def _empty_figure(title: str, color: str = "#333") -> go.Figure:
    """Return a blank figure with a descriptive centred title.

    Tagged in `layout.meta` as a status rather than a chart, so the app can
    show it as a one-line message instead of a 450 px empty canvas.
    """
    tone = {_C.VIOLATION: "error", _C.OK: "ok"}.get(color, "info")
    fig = go.Figure()
    fig.update_layout(
        meta={"conductor_status": title, "tone": tone},
        title={"text": title, "font": {"color": color}},
        **CHART_THEME,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": title,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": color},
            }
        ],
    )
    return fig


def _safe_labels(names: list, prefix: str = "Item") -> list[str]:
    """Replace empty/None labels with index-based fallbacks to prevent Plotly
    from stacking multiple items at the same x-position."""
    return [
        str(n).strip() if (n is not None and str(n).strip()) else f"{prefix}_{i}"
        for i, n in enumerate(names)
    ]


def _unique_labels(labels: list[str]) -> list[str]:
    """Ensure labels are unique for categorical bar axes.

    Plotly aggregates bars that share the same category label. When element
    names collide (for example after renaming), append a stable index suffix
    so each row keeps its own bar.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for label in labels:
        count = seen.get(label, 0) + 1
        seen[label] = count
        out.append(label if count == 1 else f"{label} [{count}]")
    return out


def _compact_branch_label(label: str, max_len: int = 28) -> str:
    """Build a compact axis label for a branch while keeping stable identity.

    Expected long format: "CODE | From -> To [L12]" — the shape every backend
    endpoint now uses for both lines ([L…]) and transformers ([T…]).
    Compact format: "CODE [L12]", or "From -> To [L12]" where the branch
    carries no name of its own, since the endpoints are then all that
    distinguishes it. Falls back to truncation when the pattern is absent.
    """
    raw = str(label).strip()
    if not raw:
        return raw

    # Keep the stable element index when present.
    idx_match = re.search(r"\[[LT]\d+\]$", raw)
    idx_suffix = f" {idx_match.group(0)}" if idx_match else ""

    head, sep, _ = raw.partition("|")
    # An unnamed branch has no code before the "|", only endpoints. Dropping
    # them would leave every bar labelled with nothing but its index.
    code = re.sub(r"\s*\[[LT]\d+\]$", "", (head if sep else raw)).strip()
    compact = f"{code}{idx_suffix}".strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "…"


# ---------------------------------------------------------------------------
# 4.1 — RSA (3 figures)
# ---------------------------------------------------------------------------


def render_rsa(result: dict) -> list[go.Figure]:
    """
    Return three Plotly figures for an RSA result:
      [0] Bus voltage scatter
      [1] Line loading bar
      [2] Transformer loading bar
    """
    timestamp = result.get("timestamp", result.get("current_timestamp", ""))
    title_suffix = f" — {timestamp}" if timestamp else ""

    # Read actual thresholds used by the backend (fall back to constants if absent)
    thresholds = result.get("thresholds_used", {})
    vm_lower = _coerce_float(thresholds.get("vm_lower_pu", grid_limits().vm_lower), grid_limits().vm_lower)
    vm_upper = _coerce_float(thresholds.get("vm_upper_pu", grid_limits().vm_upper), grid_limits().vm_upper)
    max_loading = _coerce_float(thresholds.get("max_line_loading_pct", grid_limits().max_loading), grid_limits().max_loading)

    # ------------------------------------------------------------------
    # Figure 1: Bus voltages
    # ------------------------------------------------------------------
    all_voltages = result.get("all_voltages", [])
    if not all_voltages:
        fig_v = _empty_figure(f"No data — run the analysis first{title_suffix}")
    else:
        valid_points = []
        for row in all_voltages:
            vm = _float_or_none(row.get("vm_pu"))
            if vm is None:
                continue
            valid_points.append((row.get("bus_name", ""), vm))

        if not valid_points:
            fig_v = _empty_figure(f"No in-service bus voltages available{title_suffix}")
        else:
            bus_names = _safe_labels([name for name, _ in valid_points], "Bus")
            vm_values = [vm for _, vm in valid_points]
            colors = [
                _C.VIOLATION if (v < vm_lower or v > vm_upper) else _C.PRIMARY
                for v in vm_values
            ]
            fig_v = go.Figure()
            fig_v.add_trace(
                go.Scatter(
                    x=bus_names,
                    y=vm_values,
                    mode="markers",
                    marker={"color": colors, "size": 9},
                    name="Voltage (p.u.)",
                )
            )
            # Reference lines
            for limit, label in [(vm_upper, f"V_max {vm_upper:.2f}"), (vm_lower, f"V_min {vm_lower:.2f}")]:
                fig_v.add_hline(
                    y=limit,
                    line_dash="dash",
                    line_color=_C.VIOLATION,
                    annotation_text=label,
                    annotation_position="top right" if limit == vm_upper else "bottom right",
                    annotation_font_color=_C.VIOLATION,
                )
            _v_min = min(vm_values)
            _v_max = max(vm_values)
            _y_lo = min(_v_min - 0.02, vm_lower - 0.02)
            _y_hi = max(_v_max + 0.02, vm_upper + 0.02)
            fig_v.update_layout(
                title=f"Bus Voltage Profile (p.u.){title_suffix}",
                xaxis={"tickangle": 45, "title": "Bus"},
                yaxis={"title": "Voltage (p.u.)", "range": [round(_y_lo, 3), round(_y_hi, 3)]},
                **CHART_THEME,
            )

    # ------------------------------------------------------------------
    # Figure 2: Line loading
    # ------------------------------------------------------------------
    all_line_loading = result.get("all_line_loading", [])
    if not all_line_loading:
        fig_l = _empty_figure("No data — run the analysis first")
    else:
        line_names_raw = _safe_labels([l.get("line_name", "") for l in all_line_loading], "Line")
        line_names_axis = [_compact_branch_label(n) for n in line_names_raw]
        line_names = _unique_labels(line_names_axis)
        line_vals = [_float_or_none(l.get("loading_percent")) for l in all_line_loading]
        line_colors = [_NO_DATA_COLOR if v is None else (_C.VIOLATION if _over(v, max_loading) else _C.PRIMARY)
                       for v in line_vals]

        fig_l = go.Figure(
            go.Bar(
                x=line_names,
                y=line_vals,
                marker_color=line_colors,
                name="Loading (%)",
                customdata=line_names_raw,
                hovertemplate="%{customdata}<br>Loading: %{y:.2f}%<extra></extra>",
            )
        )
        fig_l.add_hline(
            y=max_loading,
            line_dash="dash",
            line_color=_C.VIOLATION,
            annotation_text=f"{max_loading:.0f}% limit",
            annotation_position="top right",
            annotation_font_color=_C.VIOLATION,
        )
        fig_l.update_layout(
            title="Line Loading (%)",
            xaxis={"tickangle": 45, "title": "Line"},
            yaxis={"title": "Loading (%)"},
            **CHART_THEME,
        )

    # ------------------------------------------------------------------
    # Figure 3: Transformer loading
    # ------------------------------------------------------------------
    all_trafo_loading = result.get("all_trafo_loading", [])
    if not all_trafo_loading:
        fig_t = _empty_figure("No data — run the analysis first")
    else:
        trafo_names_raw = _safe_labels([t.get("trafo_name", "") for t in all_trafo_loading], "Trafo")
        trafo_names_axis = [_compact_branch_label(n) for n in trafo_names_raw]
        trafo_names = _unique_labels(trafo_names_axis)
        trafo_vals = [_float_or_none(t.get("loading_percent")) for t in all_trafo_loading]
        trafo_colors = [
            _NO_DATA_COLOR if v is None else (_C.VIOLATION if _over(v, max_loading) else _C.PRIMARY)
            for v in trafo_vals
        ]

        fig_t = go.Figure(
            go.Bar(
                x=trafo_names,
                y=trafo_vals,
                marker_color=trafo_colors,
                name="Loading (%)",
                customdata=trafo_names_raw,
                hovertemplate="%{customdata}<br>Loading: %{y:.2f}%<extra></extra>",
            )
        )
        fig_t.add_hline(
            y=max_loading,
            line_dash="dash",
            line_color=_C.VIOLATION,
            annotation_text=f"{max_loading:.0f}% limit",
            annotation_position="top right",
            annotation_font_color=_C.VIOLATION,
        )
        fig_t.update_layout(
            title="Transformer Loading (%)",
            xaxis={"tickangle": 45, "title": "Transformer"},
            yaxis={"title": "Loading (%)"},
            **CHART_THEME,
        )

    return [fig_v, fig_l, fig_t]


# ---------------------------------------------------------------------------
# 4.2 — Contingency violations
# ---------------------------------------------------------------------------

_VIOLATION_COLORS = {
    "bus_vm_pu": _C.WARN,
    "line_loading": _C.VIOLATION,
    "trafo_loading": _C.VIOLATION,
}


_PROBLEM_LABELS = {"line_loading": "Line overload", "trafo_loading": "Transformer overload"}


def render_contingency_violations(result: dict):
    """N-1 results: which outage causes what, worst first.

    The previous chart drew one bar per violated element, every bar of length
    1, and left out the outage that caused each violation — the question an
    N-1 screen exists to answer. A sorted table answers it directly. Outages
    with no converged power flow are stated first: they are the most severe
    result an N-1 screen can return, not an absence of violations.
    """
    is_full_sweep = "total_outages_tested" in result
    thresholds = result.get("thresholds_used") or {}
    non_converged = result.get("non_converged_outages") or []

    if result.get("converged") is False:
        return _empty_figure(
            "The power flow did not converge after this outage — the system has no "
            "stable operating point without this element. Treat it as insecure.",
            color=_C.VIOLATION)

    figs = []
    if non_converged:
        listed = ", ".join(non_converged[:6]) + (" …" if len(non_converged) > 6 else "")
        figs.append(_empty_figure(
            f"{len(non_converged)} outage(s) with no converged power flow — no stable "
            f"operating point, treat as insecure: {listed}", color=_C.VIOLATION))

    violations = result.get("violations") or []
    if not violations:
        if result.get("system_secure") or result.get("system_n1_secure"):
            return _empty_figure("✅ No violations detected", color=_C.OK)
        return figs or _empty_figure("No violation data available")

    vlo = _float_or_none(thresholds.get("vm_lower_pu"))
    vhi = _float_or_none(thresholds.get("vm_upper_pu"))
    rows = []
    for v in violations:
        kind = v.get("violation_type", "")
        value = _float_or_none(v.get("value"))
        if kind == "bus_vm_pu":
            # Which side of the band, judged by the midpoint: values arrive
            # rounded to 3 decimals, so a 0.9396 p.u. violation of a 0.94
            # floor reads as exactly 0.940 and a `< floor` test calls it an
            # overvoltage.
            mid = (vlo + vhi) / 2 if vlo is not None and vhi is not None else 1.0
            low = value is not None and value < mid
            problem = "Undervoltage" if low else "Overvoltage"
            limit = vlo if low else vhi
            # Distance past the limit, in % of nominal, so voltage and loading
            # violations sort on one scale.
            if value is not None and limit is not None:
                severity = max((limit - value) if low else (value - limit), 0.0) * 100
            else:
                severity = 0.0
            shown_value = f"{value:.3f} p.u." if value is not None else "—"
            shown_limit = f"{limit:.3f} p.u." if limit is not None else "—"
        else:
            problem = _PROBLEM_LABELS.get(kind, kind or "Violation")
            key = "max_trafo_loading_pct" if kind == "trafo_loading" else "max_line_loading_pct"
            limit = _float_or_none(thresholds.get(key))
            severity = value - limit if value is not None and limit is not None else 0.0
            shown_value = f"{value:.1f} %" if value is not None else "—"
            shown_limit = f"{limit:.0f} %" if limit is not None else "—"
        rows.append((severity, str(v.get("outage_cause") or "—"),
                     str(v.get("element_name") or f"Element_{v.get('element_index', '?')}"),
                     problem, shown_value, shown_limit))
    rows.sort(key=lambda r: r[0], reverse=True)
    # Every row: the app shows this figure as a scrollable, copyable table
    # (app.py `_table_of`), so there is no height to cap it for.
    shown = rows

    columns = list(zip(*[r[1:] for r in shown]))
    headers = ["Outage", "Violated element", "Problem", "Value", "Limit"]
    if not is_full_sweep:
        columns, headers = columns[1:], headers[1:]
    fig = go.Figure(go.Table(
        header={"values": [f"<b>{h}</b>" for h in headers], "fill_color": "#eef1f7",
                "align": "left", "font": {"size": 12}},
        cells={"values": columns, "align": "left", "height": 26, "font": {"size": 12},
               "fill_color": [["#fdecea" if r[3] != "—" else "white" for r in shown]]},
    ))
    if is_full_sweep:
        title = (f"N-1 screen — {result.get('total_outages_causing_violations', '?')} of "
                 f"{result.get('total_outages_tested', '?')} outages cause violations")
    else:
        title = "Violations after the outage"
    note = f" (worst {len(shown)} of {len(rows)} shown)" if len(rows) > len(shown) else ""
    fig.update_layout(
        title=f"{title}<br><sup>{len(rows)} violation(s), worst first{note}</sup>",
        height=min(700, 110 + 27 * len(shown)),
        **{**CHART_THEME, "margin": {"l": 10, "r": 10, "t": 70, "b": 10}},
    )
    figs.append(fig)
    return figs if len(figs) > 1 else fig


# ---------------------------------------------------------------------------
# 4.3 — Generator dispatch
# ---------------------------------------------------------------------------


def _render_post_opf_voltages(result: dict) -> go.Figure | None:
    """Scatter chart of post-OPF bus voltages with upper/lower limit lines."""
    bus_voltages = result.get("bus_voltages_post_opf")
    if not bus_voltages:
        return None

    bus_names = _safe_labels([bv.get("bus_name") or str(bv.get("bus", "")) for bv in bus_voltages], "Bus")
    vm_pu = [bv["vm_pu"] for bv in bus_voltages]
    vm_lower = result.get("opf_vm_lower_used", grid_limits().vm_lower)
    vm_upper = result.get("opf_vm_upper_used", grid_limits().vm_upper)

    # Dots are red when outside the OPF bounds (should not happen, but surface if so)
    colors = [
        _C.VIOLATION if (v < vm_lower - 1e-4 or v > vm_upper + 1e-4) else _C.PRIMARY
        for v in vm_pu
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bus_names,
            y=vm_pu,
            mode="markers",
            marker={"color": colors, "size": 9},
            name="Vm post-OPF (p.u.)",
        )
    )
    fig.add_hline(
        y=vm_upper, line_dash="dash", line_color=_C.VIOLATION,
        annotation_text=f"Upper {vm_upper} p.u.", annotation_position="top right",
    )
    fig.add_hline(
        y=vm_lower, line_dash="dash", line_color=_C.VIOLATION,
        annotation_text=f"Lower {vm_lower} p.u.", annotation_position="bottom right",
    )
    y_pad = 0.02
    y_lo = min(vm_lower - y_pad, min(vm_pu) - y_pad)
    y_hi = max(vm_upper + y_pad, max(vm_pu) + y_pad)
    timestamp = result.get("timestamp", "")
    fig.update_layout(
        title="Post-OPF Bus Voltages (p.u.)" + (f" — {timestamp}" if timestamp else ""),
        xaxis={"title": "Bus", "tickangle": 45},
        yaxis={"title": "Voltage (p.u.)", "range": [y_lo, y_hi]},
        **CHART_THEME,
    )
    return fig


def render_dispatch(result: dict) -> list[go.Figure]:
    """
    Grouped bar chart for optimize_flexibility / optimize_contingency results.

    Returns a list of figures:
      - Always contains the dispatch bar chart as the first element.
      - Appends a bus-voltage scatter chart when ``bus_voltages_post_opf``
        is present in the result (Issue 1).

    The dispatch chart shows two bars per element normally, or three bars
    when ``dispatch_pre_disable`` is present (Issue 3):
      1. Light grey — P before generators were disabled.
      2. Grey       — P_base (operating point OPF departs from).
      3. Green/orange — P_new (OPF result).
    """
    # Any failed solve, not just the one spelled "infeasible": an iteration
    # limit or a capped solve that cannot meet its cap used to fall through to
    # "No dispatch data — run the optimizer first".
    if result.get("status") == "infeasible" or result.get("feasible") is False:
        reason = result.get("message") or "violations cannot be resolved by local flexibility"
        if str(reason).strip().rstrip(".").lower() == "infeasible":
            reason = "violations cannot be resolved by local flexibility"
        return [_empty_figure(f"❌ Optimizer did not find a usable dispatch — {reason}", color=_C.VIOLATION)]

    resources = result.get("activated_resources", [])
    dispatch_pre_disable = result.get("dispatch_pre_disable")  # dict or None

    if not resources and not dispatch_pre_disable:
        robust_status = str(result.get("status", "")).lower()
        stop_reason = str(result.get("robust_loop_stop_reason", "")).lower()
        if robust_status == "already_secure" or stop_reason == "target_reached":
            return [_empty_figure("No redispatch required — the current operating point already meets the robust target")]
        return [_empty_figure("No dispatch data — run the optimizer first")]

    # Build a unified element ordering: pre-disable keys first (preserves disabled gens),
    # then any additional elements from the OPF result.
    resource_map = {r.get("element", str(i)): r for i, r in enumerate(resources)}
    if dispatch_pre_disable:
        all_elements = list(
            dict.fromkeys(
                list(dispatch_pre_disable.keys())
                + [r.get("element", "") for r in resources]
            )
        )
    else:
        all_elements = [r.get("element", "") for r in resources]
    all_elements = _safe_labels(all_elements, "Gen")

    pg_base = [resource_map[e]["Pg_base"] if e in resource_map else None for e in all_elements]
    pg_new = [resource_map[e]["Pg_new"] if e in resource_map else None for e in all_elements]

    pg_new_colors = [
        (_C.OK if (n >= b) else _C.WARN) if (n is not None and b is not None) else _C.NEUTRAL
        for n, b in zip(pg_new, pg_base)
    ]

    pg_up = [resource_map[e].get("Pg_up", 0.0) if e in resource_map else 0.0 for e in all_elements]
    pg_down = [-abs(resource_map[e].get("Pg_down", 0.0)) if e in resource_map else 0.0 for e in all_elements]

    qg_new  = [resource_map[e].get("Qg_new")  if e in resource_map else None for e in all_elements]
    has_q   = any(v is not None for v in qg_new)

    fig = go.Figure()

    # --- Optional pre-disable bar (Issue 3) ---
    if dispatch_pre_disable:
        pg_pre = [dispatch_pre_disable.get(e) for e in all_elements]
        fig.add_trace(
            go.Bar(
                name="P pre-disable (MW)",
                x=all_elements,
                y=pg_pre,
                marker_color=_C.NEUTRAL,
                opacity=0.9,
            )
        )

    fig.add_trace(
        go.Bar(
            name="P_base (MW)",
            x=all_elements,
            y=pg_base,
            marker_color=_C.NEUTRAL,
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Bar(
            name="P_new (MW)",
            x=all_elements,
            y=pg_new,
            marker_color=pg_new_colors,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Pg_up headroom",
            x=all_elements,
            y=pg_up,
            marker_color="lightgreen",
            opacity=0.4,
            visible="legendonly",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Pg_down headroom",
            x=all_elements,
            y=pg_down,
            marker_color="lightsalmon",
            opacity=0.4,
            visible="legendonly",
        )
    )

    # --- Reactive power traces (hidden by default, revealed via toggle button) ---
    if has_q:
        qg_new_colors_q = [
            "teal" if (v is not None and v >= 0) else _C.WARN
            for v in qg_new
        ]
        fig.add_trace(
            go.Bar(
                name="Q_new (MVAR)",
                x=all_elements,
                y=qg_new,
                marker_color=qg_new_colors_q,
                visible=False,
            )
        )

    status_label = result.get("status", "")
    title = "Generator Dispatch — Base vs Optimized (MW)"
    if dispatch_pre_disable:
        disabled_list = [e for e in dispatch_pre_disable if e not in resource_map]
        if disabled_list:
            title += f"<br><sup>Disabled: {', '.join(disabled_list)}</sup>"
    if status_label:
        title += f"<br><sup>Status: {status_label}</sup>"

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis={"title": "Substation / Generator", "tickangle": 45},
        yaxis={"title": "Power (MW)"},
        **CHART_THEME,
    )

    # Toggle buttons — only shown when Q data is available
    if has_q:
        # P traces: (P_pre +) P_base, P_new, Pg_up, Pg_down  → 4 or 5 traces
        # Q traces: Q_new only                                → 1 trace
        n_p = 5 if dispatch_pre_disable else 4
        p_vis: list = (
            [True, True, True, "legendonly", "legendonly"]
            if dispatch_pre_disable
            else [True, True, "legendonly", "legendonly"]
        ) + [False]
        q_vis: list = [False] * n_p + [True]
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.0,
                xanchor="left",
                y=-0.22,
                yanchor="top",
                showactive=True,
                buttons=[
                    dict(
                        label="Active Power (P)",
                        method="update",
                        args=[{"visible": p_vis}, {"yaxis.title.text": "Power (MW)"}],
                    ),
                    dict(
                        label="Reactive Power (Q)",
                        method="update",
                        args=[{"visible": q_vis}, {"yaxis.title.text": "Power (MVAR)"}],
                    ),
                ],
            )],
            margin={"l": 40, "r": 20, "t": 50, "b": 120},
        )

    row1: list[go.Figure] = [fig]
    voltage_fig = _render_post_opf_voltages(result)
    if voltage_fig is not None:
        row1.append(voltage_fig)

    figs = row1  # kept for reference; actual return is row-structured below
    delta_row: list[go.Figure] = []

    # --- Delta plots ----------------------------------------------------------
    # ΔP per generator
    delta_p_elems = [e for e in all_elements if e in resource_map]
    delta_p = [
        round(float(resource_map[e].get("Pg_new", 0) or 0) - float(resource_map[e].get("Pg_base", 0) or 0), 6)
        for e in delta_p_elems
    ]
    if any(abs(v) > 1e-6 for v in delta_p):
        dp_colors = [_C.OK if v >= 0 else _C.WARN for v in delta_p]
        fig_dp = go.Figure(go.Bar(
            x=delta_p_elems, y=delta_p,
            marker_color=dp_colors,
            text=[f"{v:+.4f}" if abs(v) > 1e-5 else "" for v in delta_p],
            textposition="outside",
        ))
        fig_dp.add_hline(y=0, line_color=_C.INK, line_width=0.8)
        fig_dp.update_layout(
            title="ΔP — Active Power Change (MW)",
            xaxis={"title": "Generator", "tickangle": 45},
            yaxis={"title": "ΔP (MW)"},
            **CHART_THEME,
        )
        delta_row.append(fig_dp)

    # ΔQ per generator
    delta_q_elems = [
        e for e in all_elements
        if e in resource_map
        and resource_map[e].get("Qg_new") is not None
        and resource_map[e].get("Qg_base") is not None
    ]
    delta_q = [
        round(float(resource_map[e].get("Qg_new", 0) or 0) - float(resource_map[e].get("Qg_base", 0) or 0), 6)
        for e in delta_q_elems
    ]
    if any(abs(v) > 1e-6 for v in delta_q):
        dq_colors = [_C.OK if v >= 0 else _C.VIOLATION for v in delta_q]
        fig_dq = go.Figure(go.Bar(
            x=delta_q_elems, y=delta_q,
            marker_color=dq_colors,
            text=[f"{v:+.4f}" if abs(v) > 1e-5 else "" for v in delta_q],
            textposition="outside",
        ))
        fig_dq.add_hline(y=0, line_color=_C.INK, line_width=0.8)
        fig_dq.update_layout(
            title="ΔQ — Reactive Power Change (MVAr)",
            xaxis={"title": "Generator", "tickangle": 45},
            yaxis={"title": "ΔQ (MVAr)"},
            **CHART_THEME,
        )
        delta_row.append(fig_dq)

    # ΔV per bus (post-OPF voltage minus pre-OPF base voltage from result)
    bus_voltages = result.get("bus_voltages_post_opf", [])
    base_voltages = result.get("bus_voltages_base", [])   # present if backend adds it
    if not base_voltages and bus_voltages:
        # Fallback: approximate base from resources Qg_base context isn't available,
        # so only render ΔV when an explicit base is provided.
        base_voltages = []
    if bus_voltages and base_voltages:
        base_map = {bv.get("bus_name", str(bv.get("bus", ""))): float(bv["vm_pu"]) for bv in base_voltages}
        dv_names, dv_vals, dv_colors = [], [], []
        for bv in bus_voltages:
            bname = bv.get("bus_name", str(bv.get("bus", "")))
            if bname in base_map:
                dv = round(float(bv["vm_pu"]) - base_map[bname], 6)
                dv_names.append(bname)
                dv_vals.append(dv)
                dv_colors.append(_C.VIOLATION if dv > 0 else _C.PRIMARY)
        if dv_names and any(abs(v) > 1e-5 for v in dv_vals):
            fig_dv = go.Figure(go.Bar(
                x=dv_names, y=dv_vals,
                marker_color=dv_colors,
                text=[f"{v:+.5f}" if abs(v) > 1e-5 else "" for v in dv_vals],
                textposition="outside",
            ))
            fig_dv.add_hline(y=0, line_color=_C.INK, line_width=0.8)
            fig_dv.update_layout(
                title="ΔV — Bus Voltage Change (p.u.)",
                xaxis={"title": "Bus", "tickangle": 45},
                yaxis={"title": "ΔV (p.u.)"},
                **CHART_THEME,
            )
            delta_row.append(fig_dv)

    # Return two rows: [dispatch + voltages] and [ΔP, ΔQ, ΔV].
    # app.py detects list[list[Figure]] and renders each row separately.
    if delta_row:
        return [row1, delta_row]
    return row1


# ---------------------------------------------------------------------------
# 4.4 — KPI gauges
# ---------------------------------------------------------------------------


def _make_gauge(
    value: float,
    title: str,
    row: int,
    col: int,
    color_thresholds: list[tuple[float, str]],
) -> go.Indicator:
    """Build a single gauge indicator."""
    # Determine color based on thresholds (list of (threshold, color) ascending)
    bar_color = color_thresholds[0][1]
    for threshold, color in color_thresholds:
        if value >= threshold:
            bar_color = color

    return go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, 50], "color": "#fee8e8"},
                {"range": [50, 90], "color": "#fef9e8"},
                {"range": [90, 100], "color": "#e8fee8"},
            ],
        },
        domain={
            "row": row,
            "column": col,
        },
    )


def render_kpis(result: dict) -> go.Figure:
    """
    Three (or six) gauge indicators in a single figure.
    """
    metrics = result.get("metrics", {})
    constrained = result.get("constrained_metrics", {})
    timestamp = result.get("timestamp", result.get("current_timestamp", ""))

    if not metrics:
        return _empty_figure("No KPI data — run evaluate_kpis first")
    status = str(result.get("status", "success")).lower()
    if status not in ("success", "optimal"):
        # A failed evaluation used to be drawn as three red 0 % gauges — read
        # as measured values. Say it failed, with the backend's reason.
        reason = result.get("message") or result.get("detail") or status
        return _empty_figure(f"KPI evaluation did not complete: {reason}", color=_C.VIOLATION)

    kpi1 = _float_or_none(metrics.get("kpi_1_target_demand_flex_pct"))
    kpi2 = _float_or_none(metrics.get("kpi_2_flex_utilization_pct"))
    kpi3 = _float_or_none(metrics.get("kpi_3_prevented_violation_ratio_pct"))

    has_constrained = bool(constrained) and constrained.get("status") == "optimal"
    n_rows = 2 if has_constrained else 1

    specs = [[{"type": "indicator"}] * 3 for _ in range(n_rows)]
    subplot_titles = ["KPI-3: Violations prevented", "KPI-2: Flexibility used", "KPI-1: Demand flexibility available"]
    if has_constrained:
        slack = constrained.get("slack_max_mw", "?")
        subplot_titles += [
            f"KPI-3 (constrained {slack} MW)",
            f"KPI-2 (constrained {slack} MW)",
            f"KPI-1 (constrained {slack} MW)",
        ]

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows,
        cols=3,
        specs=specs,
        subplot_titles=subplot_titles,
    )

    # Row 1: unconstrained
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=kpi3,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": _NO_DATA_COLOR if kpi3 is None else
                    (_C.OK if kpi3 == 100 else (_C.WARN if kpi3 > 50 else _C.VIOLATION))
                },
            },
            number={"suffix": "%"},
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=kpi2,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": _NO_DATA_COLOR if kpi2 is None else
                    (_C.OK if kpi2 > 90 else (_C.WARN if kpi2 > 50 else _C.VIOLATION))
                },
            },
            number={"suffix": "%"},
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=kpi1,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": _C.PRIMARY},
            },
            number={"suffix": "%"},
        ),
        row=1, col=3,
    )

    # Row 2: constrained scenario (optional)
    if has_constrained:
        ck1 = _float_or_none(constrained.get("kpi_1_target_demand_flex_pct"))
        ck2 = _float_or_none(constrained.get("kpi_2_flex_utilization_pct"))
        ck3 = _float_or_none(constrained.get("kpi_3_prevented_violation_ratio_pct"))

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ck3,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": _NO_DATA_COLOR if ck3 is None else
                        (_C.OK if ck3 == 100 else (_C.WARN if ck3 > 50 else _C.VIOLATION))
                    },
                },
                number={"suffix": "%"},
            ),
            row=2, col=1,
        )
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ck2,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": _NO_DATA_COLOR if ck2 is None else
                        (_C.OK if ck2 > 90 else (_C.WARN if ck2 > 50 else _C.VIOLATION))
                    },
                },
                number={"suffix": "%"},
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ck1,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": _C.PRIMARY},
                },
                number={"suffix": "%"},
            ),
            row=2, col=3,
        )

    title_text = "System KPIs"
    if timestamp:
        title_text += f" — {timestamp}"

    fig.update_layout(
        title=title_text,
        height=350 * n_rows,
        **CHART_THEME,
    )
    if constrained and not has_constrained:
        # The constrained scenario ran and was not optimal; it used to vanish.
        slack = constrained.get("slack_max_mw", "?")
        why = constrained.get("message") or constrained.get("status") or "not solved"
        return [fig, _empty_figure(
            f"Constrained scenario (±{slack} MW external grid): {why}", color=_C.VIOLATION)]
    return fig


# ---------------------------------------------------------------------------
# 4.5 — KPI-1 forecast
# ---------------------------------------------------------------------------


def render_kpi_forecast(result: dict) -> go.Figure:
    """24-hour KPI-1 forecast line chart."""
    forecast = result.get("forecast", [])
    if not forecast:
        return _empty_figure("No forecast data — run forecast_kpis first")

    timestamps = [f.get("timestamp", str(i)) for i, f in enumerate(forecast)]
    kpi1_vals = [f.get("kpi_1_target_demand_flex_pct", 0.0) for f in forecast]

    fig = go.Figure(
        go.Scatter(
            x=timestamps,
            y=kpi1_vals,
            mode="lines+markers",
            line={"color": _C.PRIMARY},
            marker={"size": 5},
            name="KPI-1 (%)",
        )
    )
    fig.update_layout(
        title="24-Hour KPI-1 Forecast — Target Demand Flexibility (%)",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={"title": "KPI-1 (%)", "range": [0, 100]},
        **CHART_THEME,
    )
    return fig


# ---------------------------------------------------------------------------
# 4.6 — Time-series (scan_rsa_over_time)
# ---------------------------------------------------------------------------


def render_time_series(result: dict) -> list[go.Figure]:
    """Voltage/violation chart plus a separate thermal-loading chart over time."""
    timestamps = result.get("timestamps", [])
    if not timestamps:
        return [_empty_figure("No time-series data — run scan_rsa_over_time first")]

    min_v = result.get("min_voltage", [])
    max_v = result.get("max_voltage", [])
    violations = result.get("violation_counts", [])
    max_line_loading = result.get("max_line_loading", [])
    max_trafo_loading = result.get("max_trafo_loading", [])

    thresholds = result.get("thresholds_used", {})
    vm_lower = thresholds.get("vm_lower_pu", grid_limits().vm_lower)
    vm_upper = thresholds.get("vm_upper_pu", grid_limits().vm_upper)
    max_loading = thresholds.get("max_line_loading_pct", grid_limits().max_loading)
    max_trafo_loading_pct = thresholds.get("max_trafo_loading_pct", grid_limits().max_loading)

    voltage_fig = go.Figure()

    # Violation count bars on secondary y-axis (drawn first so lines appear on top)
    voltage_fig.add_trace(
        go.Bar(
            x=timestamps,
            y=violations,
            name="Violation Count",
            marker_color="rgba(220, 80, 80, 0.25)",
            yaxis="y2",
        )
    )

    # Voltage envelope on primary y-axis
    if min_v:
        voltage_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=min_v,
                mode="lines+markers",
                line={"color": _C.PRIMARY, "dash": "dot", "width": 2},
                marker={"size": 5},
                name="Min Voltage (p.u.)",
            )
        )
    if max_v:
        voltage_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=max_v,
                mode="lines+markers",
                line={"color": _C.PURPLE, "dash": "dot", "width": 2},
                marker={"size": 5},
                name="Max Voltage (p.u.)",
            )
        )

    # Reference lines via shapes (reliable with overlaying dual-axis)
    known = [v for v in (_float_or_none(x) for x in violations) if v is not None]
    max_violations = max(known) if known else 1
    voltage_fig.update_layout(
        title="Grid Security Evolution Over Time",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={
            "title": "Voltage (p.u.)",
            "range": _voltage_range([min_v, max_v], vm_lower, vm_upper),
        },
        yaxis2={
            "title": "Violation Count", "tickformat": "d",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "rangemode": "nonnegative",
            "range": [0, max(max_violations * 4, 4)],  # keep bars visually short
        },
        shapes=[
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": vm_upper, "y1": vm_upper,
                "line": {"dash": "dash", "color": _C.VIOLATION, "width": 1},
            },
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": vm_lower, "y1": vm_lower,
                "line": {"dash": "dash", "color": _C.VIOLATION, "width": 1},
            },
        ],
        annotations=[
            {
                "x": 1.01, "y": vm_upper, "xref": "paper", "yref": "y",
                "text": f"V_max {vm_upper:.2f}", "showarrow": False,
                "xanchor": "left", "font": {"color": _C.VIOLATION},
            },
            {
                "x": 1.01, "y": vm_lower, "xref": "paper", "yref": "y",
                "text": f"V_min {vm_lower:.2f}", "showarrow": False,
                "xanchor": "left", "font": {"color": _C.VIOLATION},
            },
        ],
        barmode="overlay",
        **CHART_THEME,
    )

    thermal_fig = go.Figure()
    if max_line_loading:
        thermal_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=max_line_loading,
                mode="lines+markers",
                line={"color": _C.WARN, "width": 2},
                marker={"size": 5},
                name="Max Line Loading (%)",
            )
        )
    if max_trafo_loading:
        thermal_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=max_trafo_loading,
                mode="lines+markers",
                line={"color": _C.TEAL, "dash": "dot", "width": 2},
                marker={"size": 5},
                name="Max Trafo Loading (%)",
            )
        )

    peak_loading = max(max_line_loading) if max_line_loading else 0.0
    peak_trafo = max(max_trafo_loading) if max_trafo_loading else 0.0
    thermal_upper = max(max(peak_loading, peak_trafo, max_loading, max_trafo_loading_pct) * 1.15, 110)
    thermal_fig.update_layout(
        title="Thermal Loading Evolution Over Time",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={
            "title": "Loading (%)",
            "range": [0, thermal_upper],
        },
        shapes=[
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": max_loading, "y1": max_loading,
                "line": {"dash": "dash", "color": _C.WARN, "width": 1},
            },
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": max_trafo_loading_pct, "y1": max_trafo_loading_pct,
                "line": {"dash": "dash", "color": _C.TEAL, "width": 1},
            },
        ],
        annotations=[
            {
                "x": 1.01, "y": max_loading, "xref": "paper", "yref": "y",
                "text": f"Line limit {max_loading:.0f}%", "showarrow": False,
                "xanchor": "left", "font": {"color": _C.WARN},
            },
            {
                "x": 1.01, "y": max_trafo_loading_pct, "xref": "paper", "yref": "y",
                "text": f"Trafo limit {max_trafo_loading_pct:.0f}%", "showarrow": False,
                "xanchor": "left", "font": {"color": _C.TEAL},
            },
        ],
        **CHART_THEME,
    )

    return [voltage_fig, thermal_fig]


# ---------------------------------------------------------------------------
# 4.x — Current conditions snapshot
# ---------------------------------------------------------------------------

def render_conditions(result: dict) -> list[go.Figure]:
    """
    Two side-by-side bar charts for a get_current_conditions snapshot:
      [0] Generator active power (Pg, MW) vs installed maximum (Pg_max)
      [1] Load consumption per bus (MW)
    """
    timestamp = result.get("timestamp", "")
    title_suffix = f" — {timestamp}" if timestamp else ""

    # --- Figure 1: Generator dispatch vs Pmax ---
    generators = result.get("generators", [])
    if not generators:
        fig_gen = _empty_figure("No generator data")
    else:
        names    = [g["name"]       for g in generators]
        pg_mw    = [_float_or_none(g.get("Pg_mw")) for g in generators]
        pg_max   = [g.get("Pg_max_mw") for g in generators]
        # Colour bars: green when producing, grey when idle (≈0) or unknown
        colors = [_C.OK if _over(p, 0.01) else _C.NEUTRAL for p in pg_mw]

        fig_gen = go.Figure()
        fig_gen.add_trace(go.Bar(
            name="Pg (MW)",
            x=names, y=pg_mw,
            marker_color=colors,
        ))
        # Pmax as markers if available
        if any(v is not None and not (isinstance(v, float) and v != v) for v in pg_max):
            fig_gen.add_trace(go.Scatter(
                name="Pg_max (MW)",
                x=names, y=pg_max,
                mode="markers",
                marker={"symbol": "line-ew", "size": 12, "color": _C.VIOLATION,
                        "line": {"width": 2, "color": _C.VIOLATION}},
            ))

        # Add ext grid as a separate bar at the end
        ext = result.get("ext_grid", {})
        if ext:
            ext_label = ext.get("name") or "Slack"
            fig_gen.add_trace(go.Bar(
                name=f"{ext_label} (MW)",
                x=[ext_label], y=[ext.get("P_import_mw", 0)],
                marker_color=_C.PRIMARY,
            ))

        totals = result.get("totals", {})
        subtitle = (
            f"Total gen: {totals.get('total_generation_mw', '?')} MW | "
            f"Import: {totals.get('net_import_mw', '?')} MW | "
            f"Load: {totals.get('total_load_mw', '?')} MW"
        )
        fig_gen.update_layout(
            title=f"Generation & Installed Capacity (MW){title_suffix}<br><sup>{subtitle}</sup>",
            barmode="overlay",
            xaxis={"title": "Substation / Generator", "tickangle": 45},
            yaxis={"title": "Power (MW)"},
            **CHART_THEME,
        )

    # --- Figure 2: Load per bus ---
    loads = result.get("loads", [])
    if not loads:
        fig_load = _empty_figure("No load data")
    else:
        bus_names = [l["bus"]   for l in loads]
        p_mw      = [l["P_mw"] for l in loads]
        fig_load = go.Figure(go.Bar(
            x=bus_names, y=p_mw,
            marker_color=_C.PRIMARY,
            name="Load (MW)",
        ))
        fig_load.update_layout(
            title=f"Load per Bus (MW){title_suffix}",
            xaxis={"title": "Bus", "tickangle": 45},
            yaxis={"title": "Load (MW)"},
            **CHART_THEME,
        )

    return [fig_gen, fig_load]


# ---------------------------------------------------------------------------
# 4.7 — Result diff
# ---------------------------------------------------------------------------


def render_diff(result: dict) -> list[go.Figure]:
    """
    Return 2–3 Plotly figures for a compare_results diff:
      [0] ΔPg bar chart (always)
      [1] ΔQg bar chart (always — reactive power is often the key corrective action)
      [2] ΔVm bar chart (when bus_voltages_post_opf was present in both results)
    """
    label_a = result.get("label_a", "A")
    label_b = result.get("label_b", "B")
    title_suffix = f"{label_b} − {label_a}"

    figs: list[go.Figure] = []

    # --- ΔPg bar chart ---
    dispatch_diff = result.get("dispatch_diff", [])
    if not dispatch_diff:
        figs.append(_empty_figure("No dispatch diff data"))
    else:
        names = _safe_labels([d["name"] for d in dispatch_diff], "Gen")
        deltas_pg = [d["delta_Pg"] for d in dispatch_diff]
        colors = [
            _C.PURPLE if d < -0.01 else (_C.TEAL if d > 0.01 else _C.NEUTRAL)
            for d in deltas_pg
        ]
        fig_pg = go.Figure(
            go.Bar(
                x=names,
                y=deltas_pg,
                marker_color=colors,
                name="ΔPg (MW)",
                text=[f"{d:+.2f}" for d in deltas_pg],
                textposition="outside",
            )
        )
        fig_pg.add_hline(y=0, line_color=_C.NEUTRAL, line_width=1)
        fig_pg.update_layout(
            title=f"Active Power Dispatch Diff (MW) — {title_suffix}",
            xaxis={"title": "Generator", "tickangle": 45},
            yaxis={"title": "ΔPg (MW)"},
            **CHART_THEME,
        )
        figs.append(fig_pg)

    # --- ΔQg bar chart ---
    if dispatch_diff:
        deltas_qg = [d["delta_Qg"] for d in dispatch_diff]
        qg_colors = [
            _C.PURPLE if d < -0.001 else (_C.TEAL if d > 0.001 else _C.NEUTRAL)
            for d in deltas_qg
        ]
        fig_qg = go.Figure(
            go.Bar(
                x=names,
                y=deltas_qg,
                marker_color=qg_colors,
                name="ΔQg (MVAr)",
                text=[f"{d:+.4f}" for d in deltas_qg],
                textposition="outside",
            )
        )
        fig_qg.add_hline(y=0, line_color=_C.NEUTRAL, line_width=1)
        fig_qg.update_layout(
            title=f"Reactive Power Dispatch Diff (MVAr) — {title_suffix}",
            xaxis={"title": "Generator", "tickangle": 45},
            yaxis={"title": "ΔQg (MVAr)"},
            **CHART_THEME,
        )
        figs.append(fig_qg)

    # --- ΔVm bar chart (optional) ---
    voltage_diff = result.get("voltage_diff", [])
    if voltage_diff:
        bus_names = _safe_labels([v["bus_name"] for v in voltage_diff], "Bus")
        deltas_vm = [v["delta_vm"] for v in voltage_diff]
        vm_colors = [
            _C.PURPLE if abs(d) > 0.005 else _C.PRIMARY for d in deltas_vm
        ]
        fig_vm = go.Figure(
            go.Bar(
                x=bus_names,
                y=deltas_vm,
                marker_color=vm_colors,
                name="ΔVm (p.u.)",
                text=[f"{d:+.4f}" for d in deltas_vm],
                textposition="outside",
            )
        )
        fig_vm.add_hline(y=0, line_color=_C.NEUTRAL, line_width=1)
        fig_vm.update_layout(
            title=f"Bus Voltage Diff (p.u.) — {title_suffix}",
            xaxis={"title": "Bus", "tickangle": 45},
            yaxis={"title": "ΔVm (p.u.)"},
            **CHART_THEME,
        )
        figs.append(fig_vm)

    return figs if figs else [_empty_figure("No diff data available")]


def render_worst_case(result: dict) -> list:
    """Time-series of grid stress metrics with the worst point highlighted."""
    series = result.get("series", {})
    timestamps = series.get("timestamps", [])
    if not timestamps:
        return [_empty_figure("No scan data — run find_worst_case_timestamp first")]

    worst_ts = result.get("worst_timestamp")
    metric = result.get("metric", "violations")
    n_scanned = result.get("n_scanned", len(timestamps))

    violations = series.get("violation_counts", [])
    min_v = series.get("min_voltage", [])
    max_v = series.get("max_voltage", [])
    slack = series.get("slack_import_mw", [])

    thresholds = result.get("thresholds_used", {})
    vm_lower = thresholds.get("vm_lower_pu", grid_limits().vm_lower)
    vm_upper = thresholds.get("vm_upper_pu", grid_limits().vm_upper)

    known = [v for v in (_float_or_none(x) for x in violations) if v is not None]
    max_violations = max(known) if known else 1

    fig = go.Figure()

    # Violation count bars on secondary axis
    fig.add_trace(go.Bar(
        x=timestamps, y=violations,
        name="Violation Count",
        marker_color="rgba(220, 80, 80, 0.25)",
        yaxis="y2",
    ))

    # Voltage envelope on primary axis
    if min_v:
        fig.add_trace(go.Scatter(
            x=timestamps, y=min_v,
            mode="lines+markers",
            line={"color": _C.PRIMARY, "dash": "dot", "width": 2},
            marker={"size": 4},
            name="Min Voltage (p.u.)",
        ))
    if max_v:
        fig.add_trace(go.Scatter(
            x=timestamps, y=max_v,
            mode="lines+markers",
            line={"color": _C.PURPLE, "dash": "dot", "width": 2},
            marker={"size": 4},
            name="Max Voltage (p.u.)",
        ))

    # External-grid import as secondary trace (not on voltage axis — skip unless second chart)
    # Highlight the worst point with a star marker on the corresponding series
    if worst_ts and worst_ts in timestamps:
        wi = timestamps.index(worst_ts)
        highlight_y: float | None = None
        highlight_name = ""
        if metric == "violations" and violations:
            highlight_y = violations[wi]
            highlight_name = f"Worst: {violations[wi]} violations"
        elif metric == "min_voltage" and min_v:
            highlight_y = min_v[wi]
            highlight_name = f"Worst: {min_v[wi]:.4f} p.u."
        elif metric == "max_voltage" and max_v:
            highlight_y = max_v[wi]
            highlight_name = f"Worst: {max_v[wi]:.4f} p.u."

        if highlight_y is not None:
            fig.add_trace(go.Scatter(
                x=[worst_ts], y=[highlight_y],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": _C.WARN,
                        "line": {"color": _C.WARN, "width": 1.5}},
                text=[highlight_name],
                textposition="top center",
                textfont={"color": _C.WARN, "size": 11},
                name="Worst point",
                yaxis="y2" if metric == "violations" else "y",
            ))

    # Build shapes and annotations lists (add_vline fails on string x-axis values)
    shapes = [
        {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
         "yref": "y", "y0": vm_upper, "y1": vm_upper,
         "line": {"dash": "dash", "color": _C.VIOLATION, "width": 1}},
        {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
         "yref": "y", "y0": vm_lower, "y1": vm_lower,
         "line": {"dash": "dash", "color": _C.VIOLATION, "width": 1}},
    ]
    annotations = [
        {"x": 1.01, "y": vm_upper, "xref": "paper", "yref": "y",
         "text": f"V_max {vm_upper:.2f}", "showarrow": False,
         "xanchor": "left", "font": {"color": _C.VIOLATION}},
        {"x": 1.01, "y": vm_lower, "xref": "paper", "yref": "y",
         "text": f"V_min {vm_lower:.2f}", "showarrow": False,
         "xanchor": "left", "font": {"color": _C.VIOLATION}},
    ]
    if worst_ts:
        shapes.append(
            {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
             "yref": "paper", "y0": 0, "y1": 1,
             "line": {"dash": "dash", "color": _C.WARN, "width": 1.5}}
        )
        annotations.append(
            {"x": worst_ts, "y": 1, "xref": "x", "yref": "paper",
             "text": f"Worst ({metric})", "showarrow": False,
             "xanchor": "left", "font": {"color": _C.WARN, "size": 10}}
        )

    fig.update_layout(
        title=f"Worst-case Scan — {n_scanned} timestamps ({metric})",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={"title": "Voltage (p.u.)", "range": _voltage_range([min_v, max_v], vm_lower, vm_upper)},
        yaxis2={
            "title": "Violation Count", "tickformat": "d",
            "overlaying": "y", "side": "right",
            "showgrid": False, "rangemode": "nonnegative",
            "range": [0, max(max_violations * 4, 4)],
        },
        shapes=shapes,
        annotations=annotations,
        barmode="overlay",
        **CHART_THEME,
    )

    # Second figure: external-grid import over time
    figs = [fig]
    if slack:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=timestamps, y=slack,
            mode="lines+markers",
            line={"color": _C.PRIMARY, "width": 2},
            marker={"size": 4},
            name="External Grid import (MW)",
        ))
        if worst_ts and worst_ts in timestamps and metric == "slack_import":
            wi = timestamps.index(worst_ts)
            fig2.add_trace(go.Scatter(
                x=[worst_ts], y=[slack[wi]],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": _C.WARN,
                        "line": {"color": _C.WARN, "width": 1.5}},
                text=[f"Worst: {slack[wi]:.2f} MW"],
                textposition="top center",
                textfont={"color": _C.WARN, "size": 11},
                name="Worst point",
            ))
        fig2_shapes = []
        if worst_ts and worst_ts in timestamps and metric == "slack_import":
            fig2_shapes.append(
                {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
                 "yref": "paper", "y0": 0, "y1": 1,
                 "line": {"dash": "dash", "color": _C.WARN, "width": 1.5}}
            )
        fig2.update_layout(
            title="External Grid Import (MW) — Scan",
            xaxis={"title": "Timestamp", "tickangle": 45},
            yaxis={"title": "Import (MW)"},
            shapes=fig2_shapes,
            **CHART_THEME,
        )
        figs.append(fig2)

    # Third figure: max line loading over time, with the thermal limit line.
    max_line = series.get("max_line_loading", [])
    if max_line:
        max_load_pct = thresholds.get("max_line_loading_pct", 100.0)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=timestamps, y=max_line,
            mode="lines+markers",
            line={"color": _C.TEAL, "width": 2},
            marker={"size": 4},
            name="Max line loading (%)",
        ))
        if worst_ts and worst_ts in timestamps and metric == "max_loading":
            wi = timestamps.index(worst_ts)
            fig3.add_trace(go.Scatter(
                x=[worst_ts], y=[max_line[wi]],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": _C.WARN,
                        "line": {"color": _C.WARN, "width": 1.5}},
                text=[f"Worst: {max_line[wi]:.0f}%"],
                textposition="top center",
                textfont={"color": _C.WARN, "size": 11},
                name="Worst point",
            ))
        fig3_shapes = [
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
             "yref": "y", "y0": max_load_pct, "y1": max_load_pct,
             "line": {"dash": "dash", "color": _C.VIOLATION, "width": 1}},
        ]
        fig3_annotations = [
            {"x": 1.01, "y": max_load_pct, "xref": "paper", "yref": "y",
             "text": f"Limit {max_load_pct:.0f}%", "showarrow": False,
             "xanchor": "left", "font": {"color": _C.VIOLATION}},
        ]
        if worst_ts and worst_ts in timestamps and metric == "max_loading":
            fig3_shapes.append(
                {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
                 "yref": "paper", "y0": 0, "y1": 1,
                 "line": {"dash": "dash", "color": _C.WARN, "width": 1.5}}
            )
        fig3.update_layout(
            title="Max Line Loading (%) — Scan",
            xaxis={"title": "Timestamp", "tickangle": 45},
            yaxis={"title": "Loading (%)", "rangemode": "nonnegative"},
            shapes=fig3_shapes,
            annotations=fig3_annotations,
            **CHART_THEME,
        )
        figs.append(fig3)

    return figs


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

# Colour palette for scenario lines (up to 6 scenarios)
_SCENARIO_COLOURS = [_C.PRIMARY, _C.PURPLE, _C.TEAL, "darkorchid", _C.WARN, "teal"]


def render_scenarios(result: dict) -> list[go.Figure]:
    """Multi-scenario RSA comparison charts.

    Returns up to 3 figures:
      [0] Violation count over time — one line per scenario
    [1] External-grid import (MW) over time — one line per scenario
      [2] Summary bar chart: total violations per scenario
    """
    if "error" in result:
        return [_empty_figure(f"Error: {result['error']}", color=_C.VIOLATION)]

    scenarios = result.get("scenarios", [])
    if not scenarios:
        return [_empty_figure("No scenario data — run scan_scenarios first")]

    thresholds = result.get("thresholds", {})
    figs: list[go.Figure] = []

    # ---- Figure 1: violations over time ------------------------------------
    fig_viol = go.Figure()
    for i, sc in enumerate(scenarios):
        colour = _SCENARIO_COLOURS[i % len(_SCENARIO_COLOURS)]
        fig_viol.add_trace(go.Scatter(
            x=sc.get("timestamps", []),
            y=sc.get("series", {}).get("violations", []),
            mode="lines",
            line={"color": colour, "width": 2},
            name=sc.get("label", f"×{sc.get('sgen_scale', '?')}"),
        ))
    fig_viol.update_layout(
        title="Security Violations Over Time — Renewable Scenario Comparison",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={"title": "Violation count"},
        **CHART_THEME,
    )
    figs.append(fig_viol)

    # ---- Figure 2: Sweden import over time ---------------------------------
    fig_slack = go.Figure()
    for i, sc in enumerate(scenarios):
        colour = _SCENARIO_COLOURS[i % len(_SCENARIO_COLOURS)]
        fig_slack.add_trace(go.Scatter(
            x=sc.get("timestamps", []),
            y=sc.get("series", {}).get("slack_import_mw", []),
            mode="lines",
            line={"color": colour, "width": 2},
            name=sc.get("label", f"×{sc.get('sgen_scale', '?')}"),
        ))
    fig_slack.update_layout(
        title="External Grid Import (MW) — Renewable Scenario Comparison",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={"title": "Import (MW)"},
        **CHART_THEME,
    )
    figs.append(fig_slack)

    # ---- Figure 3: summary bar chart ---------------------------------------
    labels = [sc.get("label", f"×{sc.get('sgen_scale', '?')}") for sc in scenarios]
    totals = [sc.get("summary", {}).get("total_violations", 0) for sc in scenarios]
    colours = [_SCENARIO_COLOURS[i % len(_SCENARIO_COLOURS)] for i in range(len(scenarios))]

    fig_bar = go.Figure(go.Bar(
        x=labels,
        y=totals,
        marker_color=colours,
        text=[str(v) for v in totals],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Total Violations per Scenario",
        xaxis={"title": "Scenario"},
        yaxis={"title": "Total violations"},
        **CHART_THEME,
    )
    figs.append(fig_bar)

    return figs


def render_element_timeseries(result: dict) -> list[go.Figure]:
    """
    Time-series line charts for a focused bus/line/trafo element.

    Returns 1–2 figures:
      [0] Primary metric over time (vm_pu for bus, loading_percent for line/trafo)
      [1] Active power flow over time (p_mw injection for bus, p_from_mw / p_hv_mw)
    Violation timestamps are highlighted with larger red markers.
    """
    if "error" in result:
        return [_empty_figure(f"Error: {result['error']}", color=_C.VIOLATION)]

    etype = result.get("element_type", "unknown")
    ename = result.get("element_name", "?")
    timestamps = result.get("timestamps", [])
    series = result.get("series", {})
    thresholds = result.get("thresholds", {})
    n_scanned = result.get("n_scanned", len(timestamps))

    if not timestamps or not series:
        return [_empty_figure(f"No data for {etype} '{ename}'")]

    title_base = f"'{ename}' — {n_scanned} timestamps"
    figs: list[go.Figure] = []

    if etype == "bus":
        vm = series.get("vm_pu", [])
        vm_upper = thresholds.get("vm_upper_pu", grid_limits().vm_upper)
        vm_lower = thresholds.get("vm_lower_pu", grid_limits().vm_lower)
        violation_mask = [_over(v, vm_upper) or _over(vm_lower, v) for v in vm]

        fig = go.Figure()
        fig.add_hrect(
            y0=vm_lower, y1=vm_upper,
            fillcolor="rgba(0,200,0,0.07)", line_width=0,
            annotation_text="Normal band", annotation_position="top left",
            annotation_font_size=10,
        )
        fig.add_trace(go.Scatter(
            x=timestamps, y=vm,
            mode="lines+markers",
            line={"color": _C.PRIMARY, "width": 2},
            marker={
                "color": [_C.VIOLATION if v else _C.PRIMARY for v in violation_mask],
                "size": [9 if v else 4 for v in violation_mask],
            },
            name="vm_pu",
        ))
        fig.add_hline(y=vm_upper, line_dash="dash", line_color=_C.VIOLATION,
                      annotation_text=f"V_max {vm_upper:.2f}",
                      annotation_position="top right")
        fig.add_hline(y=vm_lower, line_dash="dash", line_color=_C.VIOLATION,
                      annotation_text=f"V_min {vm_lower:.2f}",
                      annotation_position="bottom right")
        y_pad = 0.015
        y_lo = min(vm_lower - y_pad, min(vm) - y_pad) if vm else vm_lower - y_pad
        y_hi = max(vm_upper + y_pad, max(vm) + y_pad) if vm else vm_upper + y_pad
        n_viol = sum(violation_mask)
        title = f"Bus Voltage Over Time — {title_base}"
        if n_viol:
            title += f"<br><sup>⚠ {n_viol} violation tick(s)</sup>"
        fig.update_layout(
            title=title,
            xaxis={"title": "Timestamp", "tickangle": 45},
            yaxis={"title": "Voltage (p.u.)", "range": [y_lo, y_hi]},
            **CHART_THEME,
        )
        figs.append(fig)

        p_mw = series.get("p_mw", [])
        if p_mw:
            fig2 = go.Figure(go.Scatter(
                x=timestamps, y=p_mw, mode="lines",
                line={"color": _C.WARN, "width": 1.5},
                name="P injection (MW)",
            ))
            fig2.update_layout(
                title=f"Bus Active Power Injection (MW) — {title_base}",
                xaxis={"title": "Timestamp", "tickangle": 45},
                yaxis={"title": "P (MW)"},
                **CHART_THEME,
            )
            figs.append(fig2)

    else:  # line or trafo
        loading = series.get("loading_percent", [])
        threshold_pct = (
            thresholds.get("max_line_loading_pct", grid_limits().max_loading)
            if etype == "line"
            else thresholds.get("max_trafo_loading_pct", grid_limits().max_loading)
        )
        violation_mask = [_over(v, threshold_pct) for v in loading]
        n_viol = sum(violation_mask)
        elem_label = "Line" if etype == "line" else "Transformer"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=loading,
            mode="lines+markers",
            line={"color": _C.PRIMARY, "width": 2},
            marker={
                "color": [_C.VIOLATION if v else _C.PRIMARY for v in violation_mask],
                "size": [9 if v else 4 for v in violation_mask],
            },
            fill="tozeroy",
            fillcolor="rgba(70,130,180,0.12)",
            name="Loading (%)",
        ))
        fig.add_hline(
            y=threshold_pct, line_dash="dash", line_color=_C.VIOLATION,
            annotation_text=f"{threshold_pct:.0f}% limit",
            annotation_position="top right",
        )
        y_hi = max(threshold_pct * 1.15, max(loading) * 1.05) if loading else threshold_pct * 1.15
        title = f"{elem_label} Loading (%) Over Time — {title_base}"
        if n_viol:
            title += f"<br><sup>⚠ {n_viol} overload tick(s)</sup>"
        fig.update_layout(
            title=title,
            xaxis={"title": "Timestamp", "tickangle": 45},
            yaxis={"title": "Loading (%)", "range": [0, y_hi]},
            **CHART_THEME,
        )
        figs.append(fig)

        p_key = "p_from_mw" if etype == "line" else "p_hv_mw"
        p_label = "P_from (MW)" if etype == "line" else "P_hv (MW)"
        p_vals = series.get(p_key, [])
        if p_vals:
            fig2 = go.Figure(go.Scatter(
                x=timestamps, y=p_vals, mode="lines",
                line={"color": _C.WARN, "width": 1.5},
                name=p_label,
            ))
            fig2.update_layout(
                title=f"{elem_label} Active Power Flow (MW) — {title_base}",
                xaxis={"title": "Timestamp", "tickangle": 45},
                yaxis={"title": "MW"},
                **CHART_THEME,
            )
            figs.append(fig2)

    return figs


# ---------------------------------------------------------------------------
# 4.14 — Probabilistic RSA (3 figures)
# ---------------------------------------------------------------------------

def render_probabilistic_rsa(result: dict) -> list[go.Figure]:
    """Render Monte Carlo probabilistic RSA results.

    Returns up to 3 figures:
      [0] Bar chart — per-element violation probability (sorted descending)
      [1] P5/P50/P95 voltage range plot — top 15 buses by P95
      [2] Histogram — distribution of total violation count across samples
    """
    if "error" in result:
        return [_empty_figure(f"Error: {result['error']}", color=_C.VIOLATION)]

    ts = result.get("timestamp", "")
    n_samples = result.get("n_samples", "?")
    n_converged = result.get("n_converged", "?")
    p_any = result.get("p_any_violation", 0.0)
    exp_viol = result.get("expected_violations", 0.0)
    sgen_sigma = result.get("samples_summary", {}).get("sgen_sigma", 0.15)
    thresholds = result.get("thresholds", {})
    vm_upper = thresholds.get("vm_upper_pu", grid_limits().vm_upper)
    vm_lower = thresholds.get("vm_lower_pu", grid_limits().vm_lower)

    subtitle = (
        f"{n_converged}/{n_samples} samples | "
        f"sgen σ={sgen_sigma:.0%} | "
        f"P(any violation)={p_any:.1%} | "
        f"E[violations]={exp_viol:.2f}"
    )
    figs: list[go.Figure] = []

    # ---- Figure 1: per-element violation probability bar chart -------------
    bus_probs: dict = result.get("bus_violation_probability", {})
    line_probs: dict = result.get("line_violation_probability", {})
    trafo_probs: dict = result.get("trafo_violation_probability", {})

    all_elements = (
        [(name, prob, "Bus")   for name, prob in bus_probs.items()]
        + [(name, prob, "Line")  for name, prob in line_probs.items()]
        + [(name, prob, "Trafo") for name, prob in trafo_probs.items()]
    )
    all_elements.sort(key=lambda x: x[1], reverse=True)

    if all_elements:
        names   = _safe_labels([e[0] for e in all_elements], "Elem")
        probs   = [e[1] for e in all_elements]
        etypes  = [e[2] for e in all_elements]
        colour_map = {"Bus": _C.PRIMARY, "Line": _C.WARN, "Trafo": _C.TEAL}
        colours = [colour_map.get(t, _C.NEUTRAL) for t in etypes]
        fig_bar = go.Figure(go.Bar(
            x=names, y=probs,
            marker_color=colours,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        ))
        fig_bar.add_hline(y=0.05, line_dash="dot", line_color=_C.WARN,
                          annotation_text="5%", annotation_position="top right")
        fig_bar.add_hline(y=0.20, line_dash="dot", line_color=_C.VIOLATION,
                          annotation_text="20%", annotation_position="top right")
        fig_bar.update_layout(
            title=f"Violation Probability per Element — {ts}<br><sup>{subtitle}</sup>",
            xaxis={"title": "Element", "tickangle": 45},
            yaxis={
                "title": "Violation probability",
                "tickformat": ".0%",
                "range": [0, min(1.1, max(probs) * 1.35)],
            },
            **CHART_THEME,
        )
    else:
        fig_bar = _empty_figure(
            f"No violations in any sample — grid fully secure ({subtitle})",
            color=_C.OK,
        )
    figs.append(fig_bar)

    # ---- Figure 2: voltage P5/P50/P95 per bus ------------------------------
    voltage_pct: dict = result.get("voltage_percentiles", {})
    if voltage_pct:
        sorted_buses = sorted(voltage_pct.items(), key=lambda x: x[1]["p95"], reverse=True)[:15]
        bnames   = _safe_labels([b[0] for b in sorted_buses], "Bus")
        p5_vals  = [b[1]["p5"]  for b in sorted_buses]
        p50_vals = [b[1]["p50"] for b in sorted_buses]
        p95_vals = [b[1]["p95"] for b in sorted_buses]

        fig_box = go.Figure()
        fig_box.add_hrect(
            y0=vm_lower, y1=vm_upper,
            fillcolor="rgba(0,200,0,0.07)", line_width=0,
            annotation_text="Normal band", annotation_position="top left",
            annotation_font_size=10,
        )
        for i, bname in enumerate(bnames):
            fig_box.add_trace(go.Scatter(
                x=[bname, bname], y=[p5_vals[i], p95_vals[i]],
                mode="lines",
                line={"color": "lightsteelblue", "width": 6},
                showlegend=(i == 0),
                name="P5–P95 range",
            ))
        fig_box.add_trace(go.Scatter(
            x=bnames, y=p50_vals, mode="markers",
            marker={"color": _C.PRIMARY, "size": 8},
            name="P50 (median)",
        ))
        fig_box.add_trace(go.Scatter(
            x=bnames, y=p95_vals, mode="markers",
            marker={"color": _C.PURPLE, "size": 6, "symbol": "triangle-up"},
            name="P95",
        ))
        fig_box.add_hline(y=vm_upper, line_dash="dash", line_color=_C.VIOLATION,
                          annotation_text=f"{vm_upper} p.u. limit")
        fig_box.add_hline(y=vm_lower, line_dash="dash", line_color=_C.VIOLATION,
                          annotation_text=f"{vm_lower} p.u. limit")
        fig_box.update_layout(
            title=f"Voltage P5 / P50 / P95 Envelope — Top 15 Buses by P95 — {ts}",
            xaxis={"title": "Bus", "tickangle": 45},
            yaxis={"title": "Voltage (p.u.)"},
            **CHART_THEME,
        )
        figs.append(fig_box)

    # ---- Figure 3: total violation count histogram -------------------------
    hist_data: dict = result.get("violation_count_histogram", {})
    if hist_data:
        counts = sorted(int(k) for k in hist_data.keys())
        freqs  = [hist_data[str(k)] for k in counts]
        colours = [_C.PURPLE if c > 0 else _C.PRIMARY for c in counts]
        fig_hist = go.Figure(go.Bar(
            x=[str(c) for c in counts], y=freqs,
            marker_color=colours,
            text=[str(f) for f in freqs],
            textposition="outside",
        ))
        fig_hist.update_layout(
            title=(
                f"Total Violations per Sample — {ts}<br>"
                f"<sup>Blue = secure, Red = ≥1 violation</sup>"
            ),
            xaxis={"title": "Violations in sample"},
            yaxis={"title": "Samples"},
            **CHART_THEME,
        )
        figs.append(fig_hist)

    return figs if figs else [_empty_figure("No probabilistic RSA data")]


# ---------------------------------------------------------------------------
# 4.15 — Robust Flexibility (3 figures)
# ---------------------------------------------------------------------------

def render_robust_flexibility(result: dict) -> list:
    """
    Renders the result of optimize_robust_flexibility.

    Row 1  — Generator Dispatch + Post-OPF Bus Voltages (reused from render_dispatch).
    Row 2  — ΔP + ΔQ + ΔV delta plots (reused from render_dispatch).
    Row 3  — Robust-specific: Back-off per bus | Voltages vs tightened bounds | Risk reduction.
    """
    # Rows 1 & 2 — reuse render_dispatch (works because robust result is a superset)
    dispatch_output = render_dispatch(result)
    if dispatch_output and isinstance(dispatch_output[0], list):
        rows: list = list(dispatch_output)      # already multi-row
    else:
        rows = [dispatch_output]                # single row, wrap it

    # ── Robust row ─────────────────────────────────────────────────────────
    robust_row: list[go.Figure] = []

    # ── Figure 1: Back-off per bus (upper + lower) ────────────────────────
    back_off_up: dict = result.get("back_off_upper_per_bus", result.get("back_off_per_bus", {}))
    back_off_low: dict = result.get("back_off_lower_per_bus", {})
    tightened_up: dict = result.get("tightened_upper_bounds", result.get("tightened_bounds", {}))
    tightened_low: dict = result.get("tightened_lower_bounds", {})

    if back_off_up or back_off_low:
        buses = sorted(set(back_off_up.keys()) | set(back_off_low.keys()))
        buses_sorted = sorted(
            buses,
            key=lambda b: max(float(back_off_up.get(b, 0.0)), float(back_off_low.get(b, 0.0))),
            reverse=True,
        )
        deltas_up = [float(back_off_up.get(b, 0.0)) for b in buses_sorted]
        deltas_low = [float(back_off_low.get(b, 0.0)) for b in buses_sorted]
        tight_up_sorted = [tightened_up.get(b, None) for b in buses_sorted]
        tight_low_sorted = [tightened_low.get(b, None) for b in buses_sorted]

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=buses_sorted,
            y=deltas_up,
            marker_color=_C.VIOLATION,
            name="Upper back-off Δu",
            customdata=list(zip(deltas_up, tight_up_sorted)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Upper back-off Δu: %{customdata[0]:.5f} p.u.<br>"
                "Tightened upper: %{customdata[1]:.4f} p.u.<extra></extra>"
            ),
        ))
        fig1.add_trace(go.Bar(
            x=buses_sorted,
            y=deltas_low,
            marker_color=_C.PRIMARY,
            name="Lower back-off Δl",
            customdata=list(zip(deltas_low, tight_low_sorted)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Lower back-off Δl: %{customdata[0]:.5f} p.u.<br>"
                "Tightened lower: %{customdata[1]:.4f} p.u.<extra></extra>"
            ),
        ))
        fig1.update_layout(
            title="Back-off Per Bus (Upper and Lower Tightening)",
            xaxis_title="Bus",
            yaxis_title="Back-off (p.u.)",
            xaxis_tickangle=-45,
            barmode="group",
            height=380,
        )
        robust_row.append(fig1)

    # ── Figure 2: Post-OPF voltages vs tightened bounds ────────────────────
    bus_voltages: list[dict] = result.get("bus_voltages_post_opf", [])
    if bus_voltages and (tightened_up or tightened_low):
        bus_names = [str(bv.get("bus_name", bv.get("bus", i))) for i, bv in enumerate(bus_voltages)]
        vm_vals = [float(bv.get("vm_pu", 0.0)) for bv in bus_voltages]
        tight_ubs = [float(tightened_up.get(b, result.get("opf_vm_upper_used", 1.05))) for b in bus_names]
        tight_lbs = [float(tightened_low.get(b, result.get("opf_vm_lower_used", 0.95))) for b in bus_names]
        vm_upper_global = float(result.get("opf_vm_upper_used", 1.05))
        vm_lower_global = float(result.get("opf_vm_lower_used", 0.95))

        # Colour: red if outside global bounds, amber if in tightening margin, green otherwise.
        point_colors = []
        for v, ub, lb in zip(vm_vals, tight_ubs, tight_lbs):
            if v > vm_upper_global or v < vm_lower_global:
                point_colors.append(_C.VIOLATION)   # red: constraint violated
            elif v > ub or v < lb:
                point_colors.append(_C.WARN)   # amber: in the back-off margin
            else:
                point_colors.append(_C.OK)   # green: within tightened bound

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=bus_names, y=vm_vals,
            mode="markers",
            marker=dict(color=point_colors, size=8, symbol="circle"),
            name="Post-OPF voltage (p.u.)",
            hovertemplate="<b>%{x}</b><br>vm_pu: %{y:.4f}<extra></extra>",
        ))
        # Tightened upper bounds as step line
        fig2.add_trace(go.Scatter(
            x=bus_names, y=tight_ubs,
            mode="lines",
            line=dict(color=_C.WARN, dash="dot", width=1.5),
            name="Tightened upper bound",
            hovertemplate="<b>%{x}</b><br>Tightened UB: %{y:.4f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=bus_names, y=tight_lbs,
            mode="lines",
            line=dict(color=_C.PRIMARY, dash="dot", width=1.5),
            name="Tightened lower bound",
            hovertemplate="<b>%{x}</b><br>Tightened LB: %{y:.4f}<extra></extra>",
        ))
        # Global limits
        fig2.add_hline(y=vm_upper_global, line_dash="dash", line_color=_C.VIOLATION,
                       annotation_text=f"Original UB {vm_upper_global:.3f}", annotation_position="top right")
        fig2.add_hline(y=vm_lower_global, line_dash="dash", line_color=_C.PRIMARY,
                       annotation_text=f"LB {vm_lower_global:.3f}", annotation_position="bottom right")
        fig2.update_layout(
            title="Post-OPF Bus Voltages vs Tightened Bounds",
            xaxis_title="Bus",
            yaxis_title="Voltage (p.u.)",
            xaxis_tickangle=-45,
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        robust_row.append(fig2)

    # ── Figure 3: Risk reduction summary ──────────────────────────────────
    p_before = result.get("p_any_violation_before")
    p_after = result.get("p_any_violation_after")
    p_after_validation = result.get("p_any_violation_after_validation")
    confidence = result.get("confidence", 0.95)
    sgen_sigma = result.get("sgen_sigma", 0.15)

    if p_before is not None:
        labels = ["Before Robust OPF"]
        values = [float(p_before) * 100]
        bar_cols = [_C.VIOLATION]
        text = [f"{values[0]:.1f}%"]
        if p_after_validation is not None:
            labels.append("After Robust OPF (Certified)")
            values.append(float(p_after_validation) * 100)
            bar_cols.append(_C.PRIMARY)
            text.append(f"{values[-1]:.1f}%")
        elif p_after is not None:
            labels.append("After Robust OPF")
            values.append(float(p_after) * 100)
            bar_cols.append(_C.OK)
            text.append(f"{values[-1]:.1f}%")
        else:
            labels.append("After Robust OPF")
            values.append(0.0)
            bar_cols.append(_C.NEUTRAL)
            text.append("N/A")

        fig3 = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=bar_cols,
            text=text,
            textposition="outside",
        ))
        fig3.add_hline(y=100 * (1 - confidence), line_dash="dash", line_color=_C.WARN,
                       annotation_text=f"Reference: {100*(1-confidence):.0f}% (not guaranteed system-wide)",
                       annotation_position="top right")
        fig3.update_layout(
            title=(
                f"Risk Reduction — P(any violation) at {confidence*100:.0f}% confidence "
                f"| σ_sgen={sgen_sigma*100:.0f}%"
            ),
            yaxis_title="P(any violation) [%]",
            yaxis=dict(range=[0, max(max(values) * 1.3, 5)]),
            height=340,
        )
        robust_row.append(fig3)

    if robust_row:
        rows.append(robust_row)

    # ── Robust diagnostics row: upper/lower split + curtailment ───────────
    diagnostics_row: list[go.Figure] = []

    p_over_before = result.get("p_any_bus_overvoltage_before")
    p_under_before = result.get("p_any_bus_undervoltage_before")
    p_over_after = result.get("p_any_bus_overvoltage_after")
    p_under_after = result.get("p_any_bus_undervoltage_after")

    if p_over_before is not None or p_under_before is not None:
        split_labels = ["Overvoltage", "Undervoltage"]
        split_before = [
            100.0 * float(p_over_before or 0.0),
            100.0 * float(p_under_before or 0.0),
        ]
        split_after = [
            100.0 * float(p_over_after or 0.0),
            100.0 * float(p_under_after or 0.0),
        ]

        fig_split = go.Figure()
        fig_split.add_trace(go.Bar(x=split_labels, y=split_before, name="Before", marker_color=_C.VIOLATION))
        fig_split.add_trace(go.Bar(x=split_labels, y=split_after, name="After", marker_color=_C.OK))
        fig_split.update_layout(
            title="Risk Split by Voltage Side",
            xaxis_title="Violation side",
            yaxis_title="P(any bus-side violation) [%]",
            barmode="group",
            height=330,
        )
        diagnostics_row.append(fig_split)

    loop_history = result.get("robust_loop_iterations", []) or []
    if loop_history:
        loop_iters = [int(step.get("iteration", i + 1)) for i, step in enumerate(loop_history)]
        loop_p_any = [100.0 * float(step.get("p_any_calibration", 0.0)) for step in loop_history]
        loop_curt = [float(step.get("expected_curtailment_mw_calibration", 0.0)) for step in loop_history]
        stop_reason = str(result.get("robust_loop_stop_reason", "unknown"))

        fig_loop = go.Figure()
        fig_loop.add_trace(go.Scatter(
            x=loop_iters,
            y=loop_p_any,
            mode="lines+markers",
            name="P(any) calibration [%]",
            marker=dict(color=_C.VIOLATION),
            line=dict(color=_C.VIOLATION),
            yaxis="y1",
        ))
        fig_loop.add_trace(go.Scatter(
            x=loop_iters,
            y=loop_curt,
            mode="lines+markers",
            name="Expected curtailment [MW]",
            marker=dict(color=_C.TEAL),
            line=dict(color=_C.TEAL, dash="dot"),
            yaxis="y2",
        ))
        fig_loop.update_layout(
            title=f"Robust Loop Convergence (stop: {stop_reason})",
            xaxis_title="Iteration",
            yaxis=dict(title="P(any) calibration [%]"),
            yaxis2=dict(
                title="Curtailment [MW]",
                overlaying="y",
                side="right",
            ),
            height=330,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        diagnostics_row.append(fig_loop)

    curtail_before = result.get("expected_curtailment_mw_before")
    curtail_after = result.get("expected_curtailment_mw_after")
    curtail_post_det = result.get("deterministic_setpoint_curtailment_mw_post_opf")
    if curtail_before is not None or curtail_after is not None or curtail_post_det is not None:
        labels = ["Before (MC avg)", "After (MC avg)", "Post-OPF deterministic"]
        vals = [
            float(curtail_before or 0.0),
            float(curtail_after or 0.0),
            float(curtail_post_det or 0.0),
        ]
        fig_curt = go.Figure(go.Bar(
            x=labels,
            y=vals,
            marker_color=[_C.PURPLE, _C.TEAL, _C.NEUTRAL],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ))
        fig_curt.update_layout(
            title="Renewable Curtailment Visibility",
            xaxis_title="Stage",
            yaxis_title="Curtailment [MW]",
            height=330,
        )
        diagnostics_row.append(fig_curt)

    if diagnostics_row:
        rows.append(diagnostics_row)

    # ── Risk detail row: per-bus and per-line probabilities ───────────────
    risk_detail_row: list[go.Figure] = []
    bus_before: dict = result.get("bus_violation_probability_before", {}) or {}
    bus_after: dict = result.get("bus_violation_probability_after", {}) or {}
    line_before: dict = result.get("line_violation_probability_before", {}) or {}
    line_after: dict = result.get("line_violation_probability_after", {}) or {}
    trafo_before: dict = result.get("trafo_violation_probability_before", {}) or {}
    trafo_after: dict = result.get("trafo_violation_probability_after", {}) or {}

    if bus_before or bus_after:
        buses = sorted(set(bus_before.keys()) | set(bus_after.keys()))
        buses = sorted(
            buses,
            key=lambda b: max(float(bus_before.get(b, 0.0)), float(bus_after.get(b, 0.0))),
            reverse=True,
        )[:12]
        before_vals = [100.0 * float(bus_before.get(b, 0.0)) for b in buses]
        after_vals = [100.0 * float(bus_after.get(b, 0.0)) for b in buses]

        fig_bus = go.Figure()
        fig_bus.add_trace(go.Bar(x=buses, y=before_vals, name="Before", marker_color=_C.VIOLATION))
        fig_bus.add_trace(go.Bar(x=buses, y=after_vals, name="After", marker_color=_C.OK))
        fig_bus.update_layout(
            title="Bus Violation Risk (Top 12)",
            xaxis_title="Bus",
            yaxis_title="Violation probability [%]",
            xaxis_tickangle=-45,
            barmode="group",
            height=360,
        )
        risk_detail_row.append(fig_bus)

    if line_before or line_after:
        lines = sorted(set(line_before.keys()) | set(line_after.keys()))
        lines = sorted(
            lines,
            key=lambda l: max(float(line_before.get(l, 0.0)), float(line_after.get(l, 0.0))),
            reverse=True,
        )[:12]
        before_vals = [100.0 * float(line_before.get(l, 0.0)) for l in lines]
        after_vals = [100.0 * float(line_after.get(l, 0.0)) for l in lines]

        fig_line = go.Figure()
        fig_line.add_trace(go.Bar(x=lines, y=before_vals, name="Before", marker_color=_C.VIOLATION))
        fig_line.add_trace(go.Bar(x=lines, y=after_vals, name="After", marker_color=_C.OK))
        fig_line.update_layout(
            title="Line Violation Risk (Top 12)",
            xaxis_title="Line",
            yaxis_title="Violation probability [%]",
            xaxis_tickangle=-45,
            barmode="group",
            height=360,
        )
        risk_detail_row.append(fig_line)

    if trafo_before or trafo_after:
        trafos = sorted(set(trafo_before.keys()) | set(trafo_after.keys()))
        trafos = sorted(
            trafos,
            key=lambda t: max(float(trafo_before.get(t, 0.0)), float(trafo_after.get(t, 0.0))),
            reverse=True,
        )[:12]
        before_vals = [100.0 * float(trafo_before.get(t, 0.0)) for t in trafos]
        after_vals = [100.0 * float(trafo_after.get(t, 0.0)) for t in trafos]

        fig_trafo = go.Figure()
        fig_trafo.add_trace(go.Bar(x=trafos, y=before_vals, name="Before", marker_color=_C.VIOLATION))
        fig_trafo.add_trace(go.Bar(x=trafos, y=after_vals, name="After", marker_color=_C.OK))
        fig_trafo.update_layout(
            title="Transformer Violation Risk (Top 12)",
            xaxis_title="Transformer",
            yaxis_title="Violation probability [%]",
            xaxis_tickangle=-45,
            barmode="group",
            height=360,
        )
        risk_detail_row.append(fig_trafo)

    if risk_detail_row:
        rows.append(risk_detail_row)

    return rows if rows else [_empty_figure("No robust flexibility data")]


# ---------------------------------------------------------------------------
# 4.x — PQ Flexibility Envelope
# ---------------------------------------------------------------------------

def render_flexibility_envelope(result: dict) -> list:
    """Two-figure PQ feasibility map for compute_flexibility_envelope.

    Figure 1 — Feasibility heatmap (green = feasible, red = infeasible,
                grey = no convergence). Capability curve arc overlaid.
    Figure 2 — Max bus voltage heatmap (RdYlGn_r colour scale) so the
                operator can see how close each (P,Q) point is to the
                voltage ceiling.

    Both figures mark the current base operating point with a star.
    Returned as [[fig1, fig2]] so app.py renders them side by side.
    """
    import math

    envelope = result.get("envelope", [])
    if not envelope:
        return [_empty_figure("No envelope data — run compute_flexibility_envelope first")]

    gen_name        = result.get("gen_name", "")
    base_point      = result.get("base_point", {})
    vm_upper        = result.get("vm_upper_pu", 1.05)
    vm_lower        = result.get("vm_lower_pu", 0.95)
    ts              = result.get("timestamp", "")
    safe_q          = result.get("safe_q_range_at_base_p")
    pf_cap          = result.get("capability_curve_pf", 0.9)
    q_cap_at_base_p = result.get(
        "q_cap_at_base_p",
        base_point.get("p_mw", 0.0) * math.tan(math.acos(pf_cap)) if base_point else None,
    )

    # Build sorted unique axes
    p_vals = sorted(set(round(r["p_mw"],   4) for r in envelope))
    q_vals = sorted(set(round(r["q_mvar"], 4) for r in envelope))
    p_idx  = {p: i for i, p in enumerate(p_vals)}
    q_idx  = {q: i for i, q in enumerate(q_vals)}

    # z matrices: feasibility (0/1/-1), max_vm_pu, and min_vm_pu
    n_p, n_q = len(p_vals), len(q_vals)
    z_feas   = [[None] * n_p for _ in range(n_q)]
    z_vm     = [[None] * n_p for _ in range(n_q)]
    z_min_vm = [[None] * n_p for _ in range(n_q)]

    for r in envelope:
        pi = p_idx[round(r["p_mw"],   4)]
        qi = q_idx[round(r["q_mvar"], 4)]
        if not r.get("converged", True) and not r.get("feasible", False):
            z_feas[qi][pi] = -1          # grey: non-converged
        else:
            z_feas[qi][pi] = 1 if r.get("feasible") else 0
        z_vm[qi][pi]     = r.get("max_vm_pu")
        z_min_vm[qi][pi] = r.get("min_vm_pu")

    # ── Overlay curves ────────────────────────────────────────────────────
    # 1. MVA capability arc  (hardware limit): Q = sqrt(S_rated² − P²)
    #    This is a true circular arc — the inverter cannot operate outside it.
    # 2. PF=0.9 lines (grid-code limit): Q = P × tan(acos(0.9))
    #    Straight lines from origin; stricter at low P than the arc.
    tan_phi  = math.tan(math.acos(pf_cap))
    pg_max   = result.get("p_range", [0.0, max(p_vals)])[1]
    s_rated  = pg_max / pf_cap                  # S_rated = Pmax / 0.9
    cap_p    = [p for p in p_vals]

    # Capability arc (circular)
    arc_q_pos = [math.sqrt(max(0.0, s_rated**2 - p**2)) for p in cap_p]
    arc_q_neg = [-q for q in arc_q_pos]

    # PF=0.9 lines (linear)
    pf_q_pos  = [p * tan_phi for p in cap_p]
    pf_q_neg  = [-q for q in pf_q_pos]

    # Mark grid points outside PF=0.9 grid-code constraint as infeasible (red)
    for qi, q in enumerate(q_vals):
        for pi, p in enumerate(p_vals):
            if z_feas[qi][pi] != -1 and abs(q) > p * tan_phi + 1e-6:
                z_feas[qi][pi] = 0

    def _add_overlays(fig, star_color="white"):
        # MVA arc — solid grey, labelled; both pos+neg share a legendgroup so
        # clicking the legend entry toggles both lines simultaneously
        fig.add_trace(go.Scatter(
            x=cap_p, y=arc_q_pos, mode="lines",
            line=dict(color="rgba(80,80,80,0.8)", dash="solid", width=2),
            name="MVA capability arc", legendgroup="arc",
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=cap_p, y=arc_q_neg, mode="lines",
            line=dict(color="rgba(80,80,80,0.8)", dash="solid", width=2),
            legendgroup="arc", showlegend=False, hoverinfo="skip",
        ))
        # PF=0.9 lines — dashed, thinner
        fig.add_trace(go.Scatter(
            x=cap_p, y=pf_q_pos, mode="lines",
            line=dict(color="rgba(80,80,80,0.6)", dash="dash", width=1.2),
            name=f"PF={pf_cap} grid-code limit", legendgroup="pf",
            showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=cap_p, y=pf_q_neg, mode="lines",
            line=dict(color="rgba(80,80,80,0.6)", dash="dash", width=1.2),
            legendgroup="pf", showlegend=False, hoverinfo="skip",
        ))
        # Base operating point star
        if base_point:
            fig.add_trace(go.Scatter(
                x=[base_point.get("p_mw")], y=[base_point.get("q_mvar")],
                mode="markers",
                marker=dict(symbol="star", size=14, color=star_color,
                            line=dict(color=_C.INK, width=1.5)),
                name="Base operating point",
            ))
    colorscale_feas = [
        [0.0,  "#aec7e8"],   # -1 → grey (no convergence)
        [0.49, "#aec7e8"],
        [0.50, _C.VIOLATION],   #  0 → red (infeasible)
        [0.74, _C.VIOLATION],
        [0.75, _C.OK],   #  1 → green (feasible)
        [1.0,  _C.OK],
    ]

    fig1 = go.Figure()
    fig1.add_trace(go.Heatmap(
        x=p_vals, y=q_vals, z=z_feas,
        colorscale=colorscale_feas,
        showscale=False,
        zmin=-1, zmax=1,
        hovertemplate=(
            "P=%{x:.4f} MW, Q=%{y:.4f} MVAr<br>"
            "Feasible: %{z}<extra></extra>"
        ),
        name="Feasibility",
    ))
    _add_overlays(fig1, star_color="white")
    # Safe Q range bar — clipped to PF=0.9 grid-code limit at base P
    if safe_q and base_point:
        bar_y0 = safe_q[0]
        bar_y1 = safe_q[1]
        if q_cap_at_base_p is not None:
            bar_y0 = max(bar_y0, -q_cap_at_base_p)
            bar_y1 = min(bar_y1,  q_cap_at_base_p)
        fig1.add_shape(
            type="line",
            x0=base_point["p_mw"], x1=base_point["p_mw"],
            y0=bar_y0, y1=bar_y1,
            line=dict(color="white", width=3, dash="solid"),
        )

    title1 = f"PQ Feasibility Map — {gen_name}"
    if ts:
        title1 += f"<br><sup>{ts}</sup>"
    if safe_q:
        title1 += f"<br><sup>Safe Q at P={base_point.get('p_mw', '?'):.3f} MW: [{safe_q[0]:.3f}, {safe_q[1]:.3f}] MVAr</sup>"
    fig1.update_layout(
        title=title1,
        xaxis={"title": "P (MW)"},
        yaxis={"title": "Q (MVAr)"},
        **CHART_THEME,
    )

    # ── Figure 2: Max voltage heatmap ──────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Heatmap(
        x=p_vals, y=q_vals, z=z_vm,
        colorscale="RdYlGn_r",
        colorbar=dict(title="max Vm (p.u.)", thickness=12, x=1.0),
        zmin=1.0, zmax=vm_upper + 0.01,
        hovertemplate=(
            "P=%{x:.4f} MW, Q=%{y:.4f} MVAr<br>"
            "max Vm=%{z:.4f} p.u.<extra></extra>"
        ),
        name="Max bus voltage",
    ))
    _add_overlays(fig2, star_color=_C.INK)
    fig2.update_layout(
        title=f"System Max Bus Voltage — {gen_name} PQ Sweep" + (f"<br><sup>{ts}</sup>" if ts else ""),
        xaxis={"title": "P (MW)"},
        yaxis={"title": "Q (MVAr)"},
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1),
        **CHART_THEME,
    )

    # ── Figure 3: Min voltage heatmap ───────────────────────────────────────
    # RdYlGn (not reversed): green at 1.0 (nominal), red at vm_lower (undervoltage)
    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(
        x=p_vals, y=q_vals, z=z_min_vm,
        colorscale="RdYlGn",
        colorbar=dict(title="min Vm (p.u.)", thickness=12, x=1.0),
        zmin=vm_lower - 0.01, zmax=1.0,
        hovertemplate=(
            "P=%{x:.4f} MW, Q=%{y:.4f} MVAr<br>"
            "min Vm=%{z:.4f} p.u.<extra></extra>"
        ),
        name="Min bus voltage",
    ))
    _add_overlays(fig3, star_color=_C.INK)
    fig3.update_layout(
        title=f"System Min Bus Voltage — {gen_name} PQ Sweep" + (f"<br><sup>{ts}</sup>" if ts else ""),
        xaxis={"title": "P (MW)"},
        yaxis={"title": "Q (MVAr)"},
        legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.75)", bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1),
        **CHART_THEME,
    )

    return [[fig1, fig2, fig3]]


def render_historical_risk(result: dict) -> list[go.Figure]:
    """Render empirical historical risk results.

    First slice: duration curve plus a compact summary/episodes panel.
    """
    if result.get("error"):
        return [_empty_figure(str(result["error"]), color=_C.VIOLATION)]

    duration_curve = result.get("duration_curve", []) or []
    if not duration_curve:
        return [_empty_figure("No historical risk data available")]

    target_label = result.get("target_label", result.get("target", "Target"))
    window_start = result.get("window_start", "")
    window_end = result.get("window_end", "")
    title_suffix = f"<br><sup>{window_start} → {window_end}</sup>" if window_start and window_end else ""

    x_vals = [100.0 * float(row.get("rank_fraction", 0.0)) for row in duration_curve]
    y_vals = [float(row.get("severity", 0.0)) for row in duration_curve]
    hover_vals = [float(row.get("value", 0.0)) for row in duration_curve]
    hover_ts = [str(row.get("timestamp", "")) for row in duration_curve]

    metric_name = "Margin to limit" if result.get("used_margin_fallback") else "Violation severity"
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        line=dict(color=_C.PRIMARY, width=2),
        customdata=list(zip(hover_vals, hover_ts)),
        hovertemplate=(
            "Window fraction: %{x:.1f}%<br>"
            "Severity: %{y:.4f}<br>"
            "Raw value: %{customdata[0]:.4f}<br>"
            "Timestamp: %{customdata[1]}<extra></extra>"
        ),
        name=metric_name,
    ))
    fig_curve.add_hline(
        y=float(result.get("duration_curve_limit", 0.0)),
        line_dash="dash",
        line_color=_C.VIOLATION,
        annotation_text="Limit boundary",
        annotation_position="top right",
    )
    fig_curve.update_layout(
        title=f"Historical Risk Duration Curve — {target_label}{title_suffix}",
        xaxis={"title": "Window fraction [%]"},
        yaxis={"title": metric_name},
        **CHART_THEME,
    )

    exceedance_pct = 100.0 * float(result.get("exceedance_frequency", 0.0) or 0.0)
    near_miss_pct = 100.0 * float(result.get("near_miss_frequency", 0.0) or 0.0)
    fig_summary = go.Figure()
    fig_summary.add_trace(go.Bar(
        x=["Exceedance", "Near miss"],
        y=[exceedance_pct, near_miss_pct],
        marker_color=[_C.VIOLATION, _C.WARN],
        text=[f"{exceedance_pct:.1f}%", f"{near_miss_pct:.1f}%"],
        textposition="outside",
        name="Frequency",
    ))

    worst_episodes = result.get("worst_episodes", []) or []
    if worst_episodes:
        episodes_for_plot = sorted(
            worst_episodes,
            key=lambda ep: str(ep.get("start", "")),
        )
        ep_labels = [str(ep.get("start", ""))[:16] for ep in episodes_for_plot]
        ep_vals = [int(ep.get("duration_steps", 0)) * 15 for ep in episodes_for_plot]
        fig_summary.add_trace(go.Scatter(
            x=ep_labels,
            y=ep_vals,
            mode="markers+lines",
            marker=dict(color=_C.PURPLE, size=9),
            line=dict(color=_C.PURPLE, dash="dot"),
            yaxis="y2",
            name="Worst episodes [min]",
            customdata=[float(ep.get("peak_severity", 0.0)) for ep in episodes_for_plot],
            hovertemplate="Episode %{x}<br>Duration: %{y} min<br>Peak severity: %{customdata:.4f}<extra></extra>",
        ))

    fig_summary.update_layout(
        title="Historical Risk Summary",
        xaxis={"title": "Metric / worst episode"},
        yaxis={"title": "Frequency [%]"},
        yaxis2={"title": "Episode duration [min]", "overlaying": "y", "side": "right"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **CHART_THEME,
    )

    conditional_bins = result.get("conditional_bins") or []
    if conditional_bins:
        bin_labels = [str(row.get("label", "")) for row in conditional_bins]
        exc_vals = [100.0 * float(row.get("exceedance_frequency", 0.0) or 0.0) for row in conditional_bins]
        near_vals = [100.0 * float(row.get("near_miss_frequency", 0.0) or 0.0) for row in conditional_bins]
        n_steps = [int(row.get("n_timesteps", 0) or 0) for row in conditional_bins]
        max_samples = max(n_steps) if n_steps else 1
        sample_axis_max = max(1.0, float(max_samples) * 1.1)

        fig_cond = go.Figure()
        fig_cond.add_trace(go.Bar(
            x=bin_labels,
            y=exc_vals,
            marker_color=_C.VIOLATION,
            name="Exceedance [%]",
            text=[f"{v:.1f}%" for v in exc_vals],
            textposition="outside",
            offsetgroup="exceedance",
        ))
        fig_cond.add_trace(go.Bar(
            x=bin_labels,
            y=near_vals,
            marker_color=_C.WARN,
            name="Near miss [%]",
            text=[f"{v:.1f}%" for v in near_vals],
            textposition="outside",
            offsetgroup="near_miss",
        ))
        fig_cond.add_trace(go.Scatter(
            x=bin_labels,
            y=n_steps,
            mode="lines+markers",
            line=dict(color=_C.PRIMARY, dash="dot"),
            marker=dict(size=8),
            name="Samples",
            yaxis="y2",
            hovertemplate="Bin %{x}<br>Samples: %{y}<extra></extra>",
        ))
        fig_cond.update_layout(
            title=f"Conditional Risk — {result.get('condition_used', 'condition')}",
            xaxis={"title": "Condition bin"},
            yaxis={"title": "Risk frequency [%]"},
            yaxis2={
                "title": "Sample count",
                "overlaying": "y",
                "side": "right",
                "range": [0, sample_axis_max],
                "rangemode": "tozero",
            },
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **CHART_THEME,
        )
        return [fig_curve, fig_summary, fig_cond]

    return [fig_curve, fig_summary]


def render_hosting_capacity(result: dict) -> list[go.Figure]:
    """Render hosting capacity with deterministic/probabilistic-specific views."""
    if result.get("error"):
        return [_empty_figure(str(result["error"]), color=_C.VIOLATION)]

    mode_name = str(result.get("mode", "deterministic")).strip().lower()
    scan_scope = str(result.get("scan_scope") or "single_bus").strip().lower()
    if mode_name == "deterministic" and scan_scope == "all_buses":
        bus_results = result.get("bus_results") or []
        if not bus_results:
            return [_empty_figure("No all-bus hosting data available")]

        rows_flat: list[dict] = []
        for bus_entry in bus_results:
            bus_name = str(bus_entry.get("bus", ""))
            bus_idx = bus_entry.get("bus_index", None)
            for mode_row in bus_entry.get("results") or []:
                rows_flat.append(
                    {
                        "bus": bus_name,
                        "bus_index": bus_idx,
                        "q_mode": str(mode_row.get("q_mode", "")),
                        "hosting_capacity_mw": float(mode_row.get("hosting_capacity_mw", 0.0) or 0.0),
                        "binding": str((mode_row.get("binding_constraint") or {}).get("type", "none")),
                    }
                )

        if not rows_flat:
            return [_empty_figure("No all-bus hosting data available")]

        mode_order = ["unity", "fixed_pf", "reactive_proxy"]
        available_modes = [m for m in mode_order if any(r["q_mode"] == m for r in rows_flat)]
        if not available_modes:
            available_modes = sorted({str(r["q_mode"]) for r in rows_flat})

        rows_by_mode = {
            m: {r["bus"]: r for r in rows_flat if r["q_mode"] == m}
            for m in available_modes
        }

        default_sort_mode = "fixed_pf" if "fixed_pf" in available_modes else available_modes[0]
        ordered_buses = sorted(
            rows_by_mode[default_sort_mode].keys(),
            key=lambda b: rows_by_mode[default_sort_mode][b]["hosting_capacity_mw"],
            reverse=True,
        )
        top_buses = ordered_buses[: min(20, len(ordered_buses))]

        fig_rank = go.Figure()
        mode_colors = {
            "unity": _C.TEAL,
            "fixed_pf": _C.OK,
            "reactive_proxy": _C.VIOLATION,
        }
        for m in available_modes:
            mode_map = rows_by_mode[m]
            x_vals = [float(mode_map[b]["hosting_capacity_mw"]) if b in mode_map else 0.0 for b in top_buses]
            bindings = [str(mode_map[b]["binding"]) if b in mode_map else "none" for b in top_buses]
            fig_rank.add_trace(go.Bar(
                x=x_vals,
                y=top_buses,
                orientation="h",
                marker_color=mode_colors.get(m, _C.OK),
                customdata=bindings,
                hovertemplate=(
                    "Mode: " + m + "<br>Bus: %{y}<br>Hosting capacity: %{x:.3f} MW"
                    "<br>Binding: %{customdata}<extra></extra>"
                ),
                name=m,
            ))

        ts = str(result.get("timestamp", ""))
        subtitle = f"<br><sup>{ts} | legend toggles q-handling modes</sup>" if ts else "<br><sup>legend toggles q-handling modes</sup>"
        fig_rank.update_layout(
            title=f"Deterministic Hosting Capacity Ranking — All Buses{subtitle}",
            xaxis={"title": "Hosting capacity [MW]"},
            yaxis={"title": "Bus", "autorange": "reversed"},
            barmode="group",
            legend={"title": {"text": "Q-handling mode"}},
            **CHART_THEME,
        )

        per_mode_avg = []
        for m in mode_order:
            vals = [r["hosting_capacity_mw"] for r in rows_flat if r["q_mode"] == m]
            if vals:
                per_mode_avg.append({"q_mode": m, "avg": float(sum(vals) / len(vals))})

        fig_mode = go.Figure()
        if per_mode_avg:
            fig_mode.add_trace(go.Bar(
                x=[r["q_mode"] for r in per_mode_avg],
                y=[r["avg"] for r in per_mode_avg],
                marker_color=[_C.TEAL, _C.OK, _C.VIOLATION][: len(per_mode_avg)],
                text=[f"{r['avg']:.2f} MW" for r in per_mode_avg],
                textposition="outside",
                name="Average hosting",
            ))
        fig_mode.update_layout(
            title="Average Hosting Capacity by Reactive Strategy (All Buses)",
            xaxis={"title": "Reactive strategy"},
            yaxis={"title": "Average hosting capacity [MW]"},
            **CHART_THEME,
        )

        return [fig_rank, fig_mode]

    rows = result.get("results") or []
    if not rows:
        return [_empty_figure("No hosting capacity data available")]

    modes = [str(r.get("q_mode", "")) for r in rows]
    capacities = [float(r.get("hosting_capacity_mw", 0.0) or 0.0) for r in rows]
    bindings = [
        str((r.get("binding_constraint") or {}).get("type", "none"))
        for r in rows
    ]
    risk_vals = [float(r.get("p_any_violation_at_best", 0.0) or 0.0) for r in rows]
    boundary_risk_vals = [
        (r.get("binding_constraint") or {}).get("sample_probability", None)
        for r in rows
    ]
    n_conv = [int(r.get("n_converged", 0) or 0) for r in rows]
    n_samples = [int(r.get("n_samples", 0) or 0) for r in rows]
    uncertainty_scope = str(result.get("uncertainty_scope") or "")
    risk_threshold = result.get("risk_threshold")
    mode_color_map = {
        "unity": _C.TEAL,
        "fixed_pf": _C.OK,
        "reactive_proxy": _C.VIOLATION,
        "voltage_control": _C.VIOLATION,
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=modes,
        y=capacities,
        marker_color=[mode_color_map.get(str(m).strip().lower(), _C.OK) for m in modes],
        text=[f"{v:.2f} MW" for v in capacities],
        textposition="outside",
        customdata=bindings,
        hovertemplate=(
            "Mode: %{x}<br>Hosting capacity: %{y:.3f} MW"
            "<br>Binding: %{customdata}<extra></extra>"
        ),
        name="Hosting capacity",
    ))

    bus_label = str(result.get("bus", "bus"))
    ts = str(result.get("timestamp", ""))
    subtitle_bits = [ts] if ts else []
    if mode_name == "probabilistic":
        if risk_threshold is not None:
            subtitle_bits.append(f"risk ≤ {float(risk_threshold):.1%}")
        if uncertainty_scope:
            subtitle_bits.append(uncertainty_scope.replace("_", " "))
        if result.get("synthetic_uncertainty"):
            subtitle_bits.append("synthetic uncertainty")
    subtitle = f"<br><sup>{' | '.join(subtitle_bits)}</sup>" if subtitle_bits else ""
    fig.update_layout(
        title=f"{'Probabilistic' if mode_name == 'probabilistic' else 'Deterministic'} Hosting Capacity — {bus_label}{subtitle}",
        xaxis={"title": "Reactive strategy"},
        yaxis={"title": "Hosting capacity [MW]"},
        **CHART_THEME,
    )

    for i, mode in enumerate(modes):
        binding_text = str(bindings[i]).strip().lower()
        if binding_text in {"", "none"}:
            continue
        fig.add_annotation(
            x=mode,
            y=capacities[i],
            text=f"{bindings[i]}",
            yshift=18,
            showarrow=False,
            font={"size": 11, "color": "#555"},
        )

    if mode_name != "probabilistic":
        return [fig]

    mode_color_map = {
        "unity": _C.TEAL,
        "fixed_pf": _C.OK,
        "reactive_proxy": _C.VIOLATION,
    }
    risk_mode_colors = [mode_color_map.get(str(m).strip().lower(), "#495057") for m in modes]

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=modes,
        y=risk_vals,
        marker={
            "color": risk_mode_colors,
            "pattern": {"shape": "/", "fgcolor": _C.INK, "size": 7, "solidity": 0.22},
            "line": {"color": _C.INK, "width": 1.0},
        },
        opacity=0.55,
        text=[f"{v:.1%}" for v in risk_vals],
        textposition="outside",
        customdata=list(zip(bindings, n_conv, n_samples)),
        hovertemplate=(
            "Mode: %{x}<br>P(any violation) at accepted best: %{y:.2%}"
            "<br>Binding: %{customdata[0]}"
            "<br>Converged samples: %{customdata[1]}/%{customdata[2]}<extra></extra>"
        ),
        name="Accepted best risk",
    ))
    fig_risk.add_trace(go.Bar(
        x=modes,
        y=boundary_risk_vals,
        marker={
            "color": risk_mode_colors,
            "pattern": {"shape": "x", "fgcolor": _C.INK, "size": 7, "solidity": 0.38},
            "line": {"color": _C.INK, "width": 1.0},
        },
        opacity=0.95,
        text=[
            (f"{float(v):.1%}" if v is not None else "n/a")
            for v in boundary_risk_vals
        ],
        textposition="outside",
        customdata=bindings,
        hovertemplate=(
            "Mode: %{x}<br>P(any violation) at first infeasible boundary: %{y:.2%}"
            "<br>Binding: %{customdata}<extra></extra>"
        ),
        name="First infeasible risk",
    ))
    if risk_threshold is not None:
        fig_risk.add_hline(
            y=float(risk_threshold),
            line_dash="dot",
            line_color=_C.VIOLATION,
            annotation_text=f"risk threshold {float(risk_threshold):.1%}",
            annotation_position="top left",
        )
    fig_risk.update_layout(
        title=f"Boundary Risk Check — {bus_label}{subtitle}",
        xaxis={"title": "Reactive strategy"},
        yaxis={"title": "Violation probability"},
        barmode="group",
        **CHART_THEME,
    )

    return [fig, fig_risk]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def render_violation_attribution(result: dict) -> list[go.Figure]:
    """One ranked-driver chart per violated element.

    Sensitivities are signed: a negative bar means increasing that source pushes
    the quantity *down*, which relieves an overvoltage and worsens an
    undervoltage. Controllable sources are distinguished from loads because only
    the former are an action the operator can take.
    """
    figures: list[go.Figure] = []
    all_violations = result.get("violations") or []
    for violation in all_violations[:4]:
        drivers = violation.get("drivers") or []
        if not drivers:
            continue

        is_voltage = violation.get("quantity") == "vm_pu"
        unit = "p.u. / MVAr" if is_voltage else "% / MW"
        key = "d_per_mvar" if is_voltage else "d_per_mw"
        axis = "MVAr" if is_voltage else "MW"

        # Largest effect at the top: plotted bottom-up, so ascending by size.
        rows = [(abs(d[key]), str(d.get("source", "?")), d[key], bool(d.get("controllable")))
                for d in drivers if _float_or_none(d.get(key)) is not None]
        rows.sort()
        names = [r[1] for r in rows]
        values = [r[2] for r in rows]
        colors = [_C.PRIMARY if r[3] else _C.NEUTRAL for r in rows]

        if not names:
            continue

        fig = go.Figure(go.Bar(
            x=values, y=names, orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:+.5f} " + unit + "<extra></extra>",
        ))
        value, limit = _float_or_none(violation.get("value")), _float_or_none(violation.get("limit"))
        if value is not None and limit is not None:
            # The raw value printed with 16 digits ("0.9365962009235865").
            shown = (f"{value:.3f} p.u. vs limit {limit:.3f}" if is_voltage
                     else f"{value:.1f} % vs limit {limit:.0f} %")
        else:
            shown = "value unavailable"
        fig.update_layout(
            **CHART_THEME,
            title=f"What drives {violation.get('element', '?')} ({shown})",
            xaxis={"title": f"Change in {'voltage (p.u.)' if is_voltage else 'loading (%)'} "
                            f"per {axis} injected  •  blue = controllable, grey = load",
                   "exponentformat": "power"},
            yaxis_title="",
            height=max(220, 40 * len(names) + 120),
        )
        figures.append(fig)
    if len(all_violations) > 4 and figures:
        # Said, not silently cut: the list is ordered, the rest exist.
        figures.append(_empty_figure(
            f"Showing the drivers of 4 of {len(all_violations)} violations; "
            "the answer and the audit panel cover all of them."))
    return figures


RENDERER_MAP: dict[str, Callable] = {
    "run_rsa": render_rsa,                              # returns list of 3 figs
    "simulate_contingency": render_contingency_violations,
    "simulate_all_contingencies": render_contingency_violations,
    "optimize_flexibility": render_dispatch,
    "optimize_contingency": render_dispatch,
    "evaluate_kpis": render_kpis,
    "forecast_kpis": render_kpi_forecast,
    "scan_rsa_over_time": render_time_series,
    "get_current_conditions": render_conditions,
    "compare_results": render_diff,
    "compute_violation_attribution": render_violation_attribution,
    "find_worst_case_timestamp": render_worst_case,
    "scan_scenarios": render_scenarios,
    "get_element_timeseries": render_element_timeseries,
    "run_probabilistic_rsa": render_probabilistic_rsa,
    "optimize_robust_flexibility": render_robust_flexibility,
    "compute_flexibility_envelope": render_flexibility_envelope,
    "compute_hosting_capacity": render_hosting_capacity,
    "compute_historical_risk": render_historical_risk,
}


# ---------------------------------------------------------------------------
# Network map
#
# The system as a picture. Positions come from `network_map`, which has to
# know how large things are drawn — so marker sizes live there and are only
# read here.
#
# Design positions, all deliberate:
#   * Voltage is the bus fill; loading is the branch width and colour. The two
#     quantities read first, and they do not compete for one channel.
#   * A bus with no measured voltage is hollow, never a shade. A near-nominal
#     voltage sits at the pale middle of a diverging scale, so colouring the
#     unmeasured ones made "healthy" and "no data" identical.
#   * Violations get a ring rather than a colour, so how far out a bus is
#     stays readable.
#   * Generation and demand are separate marks, sized by the square root of
#     magnitude so one large unit does not erase the small ones.
# ---------------------------------------------------------------------------

_VOLTAGE_SCALE = [
    [0.00, "#2166ac"], [0.35, "#92c5de"], [0.50, "#eef0f2"],
    [0.65, "#f4a582"], [1.00, _C.VIOLATION],
]
_LOADING_BANDS = ((60.0, "#4d9221", "≤ 60 % loaded"),
                  (90.0, _C.WARN, "60–90 %"),
                  (float("inf"), _C.VIOLATION, "> 90 %"))
_UNMEASURED_EDGE = "#b8b8b8"
# Above the network's own loading limit — a violation, not just "heavily
# loaded". 91 % and 160 % used to share the "> 90 %" band.
_OVER_LIMIT_EDGE = "#67001f"
_OUT_OF_SERVICE = "#d4d4d4"
_EXTERNAL = "#3f3f46"
_GENERATION = "#1b7837"
_DEMAND = "#762a83"

_MAX_LABELS = 28
_FONT_PX = 10
_PLOT_PX = 660

# Where a label sits relative to its marker, in units of
# (half-marker, half-width, half-height).
_PLACEMENTS = {
    "middle right": (1.0, 1.0, 0.0, 0.0), "middle left": (-1.0, -1.0, 0.0, 0.0),
    "top center": (0.0, 0.0, 1.0, 1.0), "bottom center": (0.0, 0.0, -1.0, -1.0),
    "top right": (0.7, 1.0, 0.7, 1.0), "top left": (-0.7, -1.0, 0.7, 1.0),
    "bottom right": (0.7, 1.0, -0.7, -1.0), "bottom left": (-0.7, -1.0, -0.7, -1.0),
}
_BY_ANGLE = ("middle right", "top right", "top center", "top left",
             "middle left", "bottom left", "bottom center", "bottom right")


def _band_label(edge) -> str:
    """Which legend group a branch belongs to."""
    kind = "transformer" if edge.kind == "trafo" else "line"
    if not edge.in_service:
        return f"{kind}, out of service"
    if edge.loading_pct is None:
        return f"{kind}, not measured"
    for threshold, _, label in _LOADING_BANDS:
        if edge.loading_pct < threshold:
            return f"{kind} {label}"
    return f"{kind} {_LOADING_BANDS[-1][2]}"


def _loading_colour(loading):
    if loading is None:
        return _UNMEASURED_EDGE
    for threshold, colour, _ in _LOADING_BANDS:
        if loading < threshold:
            return colour
    return _LOADING_BANDS[-1][1]


def _blend(a: str, b: str, t: float) -> str:
    ar, ag, ab = (int(a[i:i + 2], 16) for i in (1, 3, 5))
    br, bg, bb = (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (round(ar + (br - ar) * t), round(ag + (bg - ag) * t),
                              round(ab + (bb - ab) * t))


def _voltage_colour(vm_pu: float, lower: float, upper: float) -> str:
    span = max(upper - 1.0, 1.0 - lower, 1e-6)
    fraction = min(1.0, max(0.0, 0.5 + (vm_pu - 1.0) / (2 * span)))
    for (p0, c0), (p1, c1) in zip(_VOLTAGE_SCALE, _VOLTAGE_SCALE[1:]):
        if fraction <= p1:
            return _blend(c0, c1, 0.0 if p1 == p0 else (fraction - p0) / (p1 - p0))
    return _VOLTAGE_SCALE[-1][1]


def _label_box(x, y, placement, half, width, height):
    fx, wx, fy, hy = _PLACEMENTS[placement]
    cx = x + fx * half + wx * width / 2
    cy = y + fy * half + hy * height / 2
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def _hits(box, others) -> bool:
    return any(box[0] < o[2] and o[0] < box[2] and box[1] < o[3] and o[1] < box[3]
               for o in others)


def _place_labels(xs, ys, names, sizes, ranked, obstacles, per_px):
    """Keep the names that fit, trying every position around each marker.

    A budget alone was not enough: a distribution bus sits close to the
    substation it hangs off, so both names landed in the same few pixels.
    Markers count as obstacles too — otherwise a label covers the very bus it
    is naming, or the triangle showing what that bus injects.
    """
    if not xs:
        return names, "top center", 0
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    taken, kept = list(obstacles), {}

    for _, _, slot in sorted(ranked)[:_MAX_LABELS]:
        text = names[slot]
        if not text:
            continue
        width = len(text) * 0.55 * _FONT_PX * per_px
        height = 1.25 * _FONT_PX * per_px
        half = (sizes[slot] / 2 + 3) * per_px
        angle = math.degrees(math.atan2(ys[slot] - cy, xs[slot] - cx))
        start = int(round(angle / 45.0)) % 8
        for step in (0, 1, -1, 2, -2, 3, -3, 4):
            placement = _BY_ANGLE[(start + step) % 8]
            box = _label_box(xs[slot], ys[slot], placement, half, width, height)
            if not _hits(box, taken):
                taken.append(box)
                kept[slot] = placement
                break

    return ([name if slot in kept else "" for slot, name in enumerate(names)],
            [kept.get(slot, "top center") for slot in range(len(names))],
            sum(1 for _, _, slot in ranked if names[slot] and slot not in kept))


def render_network_map(network: NetworkMap, positions: dict, *,
                       vm_lower: float = 0.95, vm_upper: float = 1.05,
                       highlight: set | None = None,
                       subtitle: str = "",
                       max_loading: float | None = None) -> go.Figure:
    """The loaded network, coloured by whatever state has been joined onto it."""
    highlight = highlight or set()
    measured = any(n.vm_pu is not None for n in network.nodes.values())
    drawn = [n for n in network.nodes.values() if n.index in positions]
    if not drawn:
        return _empty_figure("No network to draw")

    xs = [positions[n.index][0] for n in drawn]
    ys = [positions[n.index][1] for n in drawn]
    span = max(max(ys) - min(ys), max(xs) - min(xs)) or 1.0
    per_px = span / _PLOT_PX
    power_offset = 0.022 * span

    traces: list[go.Scatter] = []
    hover_x, hover_y, hover_text = [], [], []
    shown_groups: set[str] = set()

    for edge in network.edges:
        if edge.from_bus not in positions or edge.to_bus not in positions:
            continue
        x0, y0 = positions[edge.from_bus]
        x1, y1 = positions[edge.to_bus]
        if edge.kind == "ext_grid":
            colour, width, detail = _EXTERNAL, 3.0, "external connection"
        else:
            colour = _OUT_OF_SERVICE if not edge.in_service else _loading_colour(edge.loading_pct)
            width = 1.6 if edge.loading_pct is None else 1.6 + 3.4 * min(edge.loading_pct, 120.0) / 120.0
            detail = ("not measured" if edge.loading_pct is None
                      else f"{edge.loading_pct:.1f}% loaded")
            # Over the limit, or named among the violations: drawn as one. The
            # highlight used to reach buses only, so an overloaded line looked
            # like any other heavily loaded one.
            over = (edge.in_service and (
                edge.name in highlight
                or (max_loading is not None and edge.loading_pct is not None
                    and edge.loading_pct > max_loading)))
            if over:
                colour, width = _OVER_LIMIT_EDGE, max(width, 5.5)
                detail += f" — over the {max_loading:g} % limit" if max_loading else " — violation"

        # Every legend entry is a real trace in a group, so clicking one hides
        # the branches it describes. They used to be empty placeholder traces:
        # the key looked interactive, and clicking it did nothing at all.
        if edge.kind == "ext_grid":
            group = "external"
        elif over:
            group = f"{'transformer' if edge.kind == 'trafo' else 'line'} over its limit"
        else:
            group = _band_label(edge)
        first = group not in shown_groups
        shown_groups.add(group)

        traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
            legendgroup=group, showlegend=first, name=group,
            line=dict(color=colour, width=width,
                      dash="dot" if edge.kind == "trafo" else "solid"),
        ))
        # A two-point line answers the cursor only at its endpoints, so
        # hovering the middle of a branch did nothing at all.
        label = (f"<b>{edge.name}</b><br>{edge.kind} · {detail}"
                 + ("" if edge.in_service else "<br>out of service"))
        for fraction in (0.2, 0.35, 0.5, 0.65, 0.8):
            hover_x.append(x0 + (x1 - x0) * fraction)
            hover_y.append(y0 + (y1 - y0) * fraction)
            hover_text.append(label)

    if hover_x:
        traces.append(go.Scatter(
            x=hover_x, y=hover_y, mode="markers", showlegend=False,
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            hoverinfo="text", hovertext=hover_text,
        ))

    fills, outlines, widths, sizes, hovers, names, symbols, ranked, obstacles = (
        [], [], [], [], [], [], [], [], [])
    states: list[str] = []
    for slot, node in enumerate(drawn):
        flagged = node.name in highlight
        if node.kind == "external":
            symbols.append("square")
            fills.append(_EXTERNAL)
            states.append("external grid (slack)")
        elif not node.in_service:
            symbols.append("circle")
            fills.append(_OUT_OF_SERVICE)
            states.append("out of service")
        elif node.vm_pu is None:
            # Hollow, not pale: a near-nominal voltage sits at the white middle
            # of the scale, so colouring the unmeasured made "healthy" and "no
            # data" look the same.
            symbols.append("circle-open")
            fills.append(_UNMEASURED_EDGE)
            states.append("no measurement")
        else:
            symbols.append("circle")
            fills.append(_voltage_colour(node.vm_pu, vm_lower, vm_upper))
            states.append("violating" if flagged else "measured")

        outlines.append("#111111" if flagged else "#5b5b5b")
        widths.append(3.0 if flagged else 1.0)
        size = MARKER_PX.get(node.kind, 10) * network.marker_scale
        sizes.append(size + 3 if flagged else size)

        if node.kind == "external":
            flow = ("" if not node.gen_mw else
                    f"<br>{'exporting' if node.gen_mw < 0 else 'importing'} "
                    f"{abs(node.gen_mw):.2f} MW")
            hovers.append(f"<b>{node.name}</b><br>external grid (slack)"
                          f"<br>{node.vn_kv:g} kV{flow}")
        else:
            voltage = "not measured" if node.vm_pu is None else f"{node.vm_pu:.4f} pu"
            hovers.append(
                f"<b>{node.name}</b><br>{node.kind} · {node.vn_kv:g} kV · bus {node.index}"
                f"<br>V = {voltage}"
                + (f"<br>load {node.load_mw:.2f} MW" if node.load_mw else "")
                + (f"<br>gen {node.gen_mw:.2f} MW" if node.gen_mw else "")
                + ("<br><b>violating</b>" if flagged else "")
                + ("" if node.in_service else "<br>out of service"))

        names.append(node.name)
        ranked.append((
            0 if node.kind in ("slack", "external") else
            1 if flagged else 2 if node.kind == "substation" else
            3 if node.kind == "distribution" else 4,
            -(abs(node.load_mw) + abs(node.gen_mw)), slot,
        ))
        half = (sizes[slot] / 2 + 3) * per_px
        up = power_offset + 9 * per_px if node.gen_mw else 0.0
        down = power_offset + 9 * per_px if node.load_mw else 0.0
        obstacles.append((xs[slot] - half, ys[slot] - half - down,
                          xs[slot] + half, ys[slot] + half + up))

    labels, placements, dropped = _place_labels(
        xs, ys, names, sizes, ranked, obstacles, per_px)

    # Buses are split by state rather than drawn as one trace with per-point
    # styling. Plotly's legend toggles whole traces, so a single trace would
    # give a key that cannot filter anything — and a legend that mixes items
    # which respond with items which do not is worse than one that does
    # neither.
    for state in ("violating", "measured", "no measurement",
                  "out of service", "external grid (slack)"):
        members = [slot for slot, node in enumerate(drawn) if states[slot] == state]
        if not members:
            continue
        traces.append(go.Scatter(
            x=[xs[i] for i in members], y=[ys[i] for i in members],
            mode="markers+text", legendgroup=state, showlegend=True, name=state,
            marker=dict(size=[sizes[i] for i in members],
                        color=[fills[i] for i in members],
                        symbol=[symbols[i] for i in members],
                        line=dict(color=[outlines[i] for i in members],
                                  width=[widths[i] for i in members])),
            text=[labels[i] for i in members],
            textposition=[placements[i] for i in members],
            textfont=dict(size=_FONT_PX, color="#333"),
            hoverinfo="text", hovertext=[hovers[i] for i in members],
        ))

    traces.extend(_power_traces(drawn, positions, power_offset))
    if measured:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=0.1, color=[vm_lower, vm_upper], colorscale=_VOLTAGE_SCALE,
                        cmin=vm_lower, cmax=vm_upper, showscale=True,
                        # Anchored to the bottom: the legend grows downward
                        # from the top and a centred bar ran straight into it.
                        colorbar=dict(title=dict(text="V (pu)", side="right"),
                                      thickness=12, len=0.38, x=1.02,
                                      y=0.0, yanchor="bottom",
                                      tickvals=[vm_lower, 1.0, vm_upper]))))

    title = f"<b>{network.name}</b>"
    if subtitle:
        title += f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        hovermode="closest", plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=74, b=44), height=720,
        legend=dict(orientation="v", x=1.0, y=1.0, xanchor="left",
                    bgcolor="rgba(255,255,255,0.8)", borderwidth=0,
                    font=dict(size=10)),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        # Equal aspect: stretching one axis would misrepresent the shape.
        yaxis=dict(visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        annotations=[dict(
            # Said on the figure: a force-directed layout looks like a map and
            # is not one.
            text="schematic connectivity, not geography"
                 + (f" · {dropped} name(s) hidden, hover for them" if dropped else ""),
            xref="paper", yref="paper", x=0.0, y=-0.045, showarrow=False,
            font=dict(size=10, color="#888"), xanchor="left")],
    )
    return fig


def _power_traces(drawn, positions, offset):
    peak = max((max(abs(n.load_mw), abs(n.gen_mw)) for n in drawn), default=0.0)
    if peak <= 0:
        return []
    traces = []
    for attr, symbol, colour, label in (("gen_mw", "triangle-up", _GENERATION, "generation"),
                                        ("load_mw", "triangle-down", _DEMAND, "demand")):
        xs, ys, sizes, hovers = [], [], [], []
        for node in drawn:
            value = getattr(node, attr)
            if not value or node.kind == "external":
                continue
            x, y = positions[node.index]
            xs.append(x)
            ys.append(y + (offset if attr == "gen_mw" else -offset))
            sizes.append(6 + 12 * (abs(value) / peak) ** 0.5)
            hovers.append(f"<b>{node.name}</b><br>{label} {abs(value):.3f} MW")
        if xs:
            traces.append(go.Scatter(
                x=xs, y=ys, mode="markers", legendgroup=label,
                showlegend=True, name=label,
                marker=dict(size=sizes, color=colour, symbol=symbol, line=dict(width=0)),
                hoverinfo="text", hovertext=hovers))
    return traces
