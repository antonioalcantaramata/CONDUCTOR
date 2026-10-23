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
    # Two more categorical hues, for series that need five (the flexibility
    # options). With PRIMARY, TEAL and PURPLE they pass the palette validator
    # (lightness band, chroma, colour-blind separation, contrast on white).
    MAGENTA = "#D9578F"
    OLIVE = "#7B8A12"
    # A sixth, for the sixth strategy: passes against every one of the five
    # (normal vision; against OLIVE in the 6–8 colour-blind band, so the
    # legend and the labels carry the name too).
    SIENNA = "#A0522D"


# One colour per flexibility strategy, whatever else an answer shows: the
# colour follows the option, never its rank.
OPTION_COLOURS = {"least_disturbance": _C.PRIMARY, "cheapest": _C.TEAL, "fewest_units": _C.PURPLE,
                  "without_main_unit": _C.MAGENTA, "least_import": _C.OLIVE, "most_headroom": _C.SIENNA,
                  # A near-optimal set holds the cheapest and its alternatives only,
                  # an N-1 set the least-disturbance, the cheapest and the secure
                  # ones: the validated hues are reused, each once per chart.
                  "near_optimal_1": _C.PURPLE, "near_optimal_2": _C.MAGENTA, "near_optimal_3": _C.OLIVE,
                  "near_optimal_4": _C.SIENNA, "near_optimal_5": _C.PRIMARY,
                  "n1_preventive": _C.PURPLE, "n1_corrective": _C.MAGENTA,
                  # A seventh strategy: no seventh hue passes against the six, so
                  # it shares most headroom's and carries a hatch (composite encoding).
                  "demand_response": _C.SIENNA,
                  # The last resort is its own set: least load reduction, cheapest with it.
                  "last_resort_least": _C.PRIMARY, "last_resort_cheapest": _C.TEAL}
OPTION_PATTERNS = {"demand_response": "/"}
# A horizontal legend whose entries keep their whole label.
_WIDE_LEGEND = {"orientation": "h", "y": -0.22, "x": 0, "entrywidth": 190, "entrywidthmode": "pixels"}
# Units in a chart of their own (a plan): the categorical order, then folded to neutral.
UNIT_COLOURS = (_C.PRIMARY, _C.TEAL, _C.PURPLE, _C.MAGENTA, _C.OLIVE, _C.SIENNA)
_PRICE_SOURCES = {"case": "the case file", "synthetic": "the synthetic rule", "operator": "your changes"}


pio.templates["conductor"] = go.layout.Template(
    layout={
        "font": {"family": "Source Sans 3, Source Sans Pro, Inter, system-ui, sans-serif",
                 "size": 12, "color": _C.INK},
        "title": {"font": {"size": 15, "color": _C.INK}, "x": 0.0, "xanchor": "left",
                  "xref": "paper"},
        "colorway": [_C.PRIMARY, _C.TEAL, _C.PURPLE, _C.WARN, _C.NEUTRAL, _C.OK],
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        # `automargin`: long category labels ("Bus_3 -> Bus_8 [T1]") widen the
        # margin instead of being cut off in a narrow column.
        "xaxis": {"gridcolor": _C.GRID, "linecolor": "#D5DAE3", "zerolinecolor": "#C9CFDA",
                  "ticks": "outside", "tickcolor": "#D5DAE3", "title": {"font": {"size": 12}},
                  "automargin": True},
        "yaxis": {"gridcolor": _C.GRID, "linecolor": "#D5DAE3", "zerolinecolor": "#C9CFDA",
                  "ticks": "outside", "tickcolor": "#D5DAE3", "title": {"font": {"size": 12}},
                  "automargin": True},
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
    if not all_trafo_loading and result.get("all_line_loading"):
        # A network without transformers (the IEEE 30 demo): nothing to draw, and
        # "run the analysis first" read as if the assessment had failed.
        return [fig_v, fig_l]
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


def _screened(result: dict) -> str:
    """What an N-1 screen ran on: the schedule, or a corrective option applied."""
    option = result.get("option") or {}
    if not option:
        return "N-1 screen"
    applied = f"with option {option.get('ref')} ({option.get('label')}) applied"
    return (f"N-1-1 screen after the outage of {option['after_outage']}, {applied}" if option.get("after_outage")
            else f"N-1 screen {applied}")


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
            secure = "✅ No violations detected"
            return _empty_figure(f"{secure} — {_screened(result)}" if result.get("option") else secure, color=_C.OK)
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
        title = (f"{_screened(result)} — {result.get('total_outages_causing_violations', '?')} of "
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


def _render_options(result: dict) -> list[go.Figure]:
    """optimize_flexibility(options=true): the options side by side — a
    copyable table of the same figures for every option, then the change each
    asks of every unit (active power; reactive power when any option moves it).
    No option is highlighted: the operator chooses."""
    when = str(result.get("timestamp") or "")[:16]
    if result.get("status") == "no_action_needed":
        return [_empty_figure(f"No corrective action needed at {when}: the schedule meets every limit", _C.OK)]
    options = [o for o in result.get("options") or [] if isinstance(o, dict)]
    if not options:
        return [_empty_figure("No corrective options were found")]

    def num(v, fmt):
        return format(v, fmt) if isinstance(v, (int, float)) and not isinstance(v, bool) else "—"

    headers = ["Option", "What it does", "Units called", "MW moved", "Mvar moved", "Import MW",
               "Cost €/h", "Above cheapest €/h", "Lowest V p.u.", "Highest loading %", "Power flow"]

    def checked(o):
        """The independent AC power flow with the option's dispatch applied."""
        check = o.get("power_flow_check") or {}
        if not check:
            return "—"
        if check.get("confirmed"):
            return "✓ confirmed"
        first = (check.get("violations") or ["not confirmed"])[0]
        return "✗ " + (first if len(first) <= 40 else first[:39] + "…")
    def name(o):
        also = " and ".join(o.get("also") or [])
        return o.get("label", o.get("id")) + (f" (also {also})" if also else "")

    rows = [[name(o), o.get("how", ""), str(o.get("n_units_called", "—")),
             num(o.get("moved_mw"), ".2f"), num(o.get("moved_mvar"), ".2f"), num(o.get("import_mw"), ".1f"),
             num(o.get("cost_eur_per_h"), ".2f"), num(o.get("above_cheapest_eur_per_h"), "+.2f"),
             num(o.get("min_vm_pu"), ".3f"), num(o.get("max_loading_pct"), ".1f"), checked(o)] for o in options]
    widths = [1.2, 2.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 0.9, 1.0, 1.1]
    if any(o.get("load_reduction_mw") for o in options):
        # Load reduced (demand response and relief): beside the units called.
        headers.insert(3, "Load reduced MW")
        widths.insert(3, 0.8)
        for row, o in zip(rows, options):
            row.insert(3, num(o.get("load_reduction_mw") or 0.0, ".2f"))
    sources = [_PRICE_SOURCES.get(k, k) for k in (result.get("costs") or {}).get("prices_from") or {}]
    kept = list(result.get("held_units") or {})
    after = f" after the outage of {result['outage']['element']}" if (result.get("outage") or {}).get("element") else ""
    held = (f" with {', '.join(kept)} at {'its' if len(kept) == 1 else 'their'} schedule" if kept else "")
    caps = result.get("branch_limits") or {}
    if caps:
        held += (" and " if held else " with ") + ", ".join(f"{_tag(n)} at most {p:g} %" for n, p in caps.items())
    study = result.get("n1_study") or {}
    table = go.Figure(go.Table(
        columnwidth=widths,
        header={"values": [f"<b>{h}</b>" for h in headers], "fill_color": "#eef1f7", "align": "left",
                "font": {"size": 12}},
        cells={"values": [list(col) for col in zip(*rows)], "align": "left", "height": 26, "font": {"size": 12}},
    ))
    table.update_layout(
        title=((f"Near-optimal alternatives{after} at {when}<br><sup>Each meets every limit{held} and costs at most "
                f"{result.get('budget_eur_per_h', 0):.2f} €/h ({result.get('near_optimal_pct', 0):g} % above the "
                f"cheapest) · prices from {', '.join(sources) or 'the cost table'}</sup>")
               if result.get("set") == "near_optimal" else
               (f"Options for N-1 security{after} at {when}<br><sup>{len(study.get('outages_studied') or [])} critical "
                f"outage(s) studied ({' and '.join(study.get('modes') or [])}) · each meets every limit{held} with "
                f"everything in service · prices from {', '.join(sources) or 'the cost table'}</sup>")
               if result.get("set") == "n1" else
               (f"Last resort{after} at {when}: options that also reduce load<br><sup>Redispatching the units alone "
                f"cannot clear the violations · demand response first, load relief only beyond it · each meets "
                f"every limit{held} · prices from {', '.join(sources) or 'the cost table'}</sup>")
               if result.get("set") == "last_resort" else
               (f"Corrective options{after} at {when}<br><sup>Each meets every limit{held} · cost of the action, "
                f"prices from {', '.join(sources) or 'the cost table'}</sup>")),
        height=_table_height(headers, rows, widths), **{**CHART_THEME, "margin": {"l": 10, "r": 10, "t": 70, "b": 10}})
    figures = [table]
    if result.get("set") == "n1":
        figures += _n1_tables(options, study, when)
    figures += _load_reduction_tables(options, when)
    if result.get("switching"):
        figures.append(_switching_table(result["switching"], when))

    for key, unit, title, threshold in (("dp_mw", "MW", "Change in active power by option", 0.01),
                                        ("dq_mvar", "Mvar", "Change in reactive power by option", 0.05)):
        units = []
        for o in options:
            for row in o.get("dispatch") or []:
                if abs(row.get(key) or 0.0) >= threshold and row.get("unit") not in units:
                    units.append(row.get("unit"))
        if not units:
            continue
        fig = go.Figure()
        for o in options:
            by_unit = {row.get("unit"): row.get(key) or 0.0 for row in o.get("dispatch") or []}
            fig.add_trace(go.Bar(
                name=o.get("label", o.get("id")), y=_safe_labels(units, "Unit"),
                x=[by_unit.get(u, 0.0) for u in units], orientation="h",
                marker={"color": OPTION_COLOURS.get(o.get("id"), _C.NEUTRAL), "cornerradius": 4,
                        **({"pattern": {"shape": OPTION_PATTERNS[o.get("id")], "fgcolor": "white", "size": 6}}
                           if o.get("id") in OPTION_PATTERNS else {})},
                hovertemplate="%{y}: %{x:+.2f} " + unit + "<extra>" + str(o.get("label", "")) + "</extra>"))
        fig.update_layout(**CHART_THEME, barmode="group", bargap=0.35, bargroupgap=0.08,
                          title=f"{title} ({unit}) — {when}", xaxis_title="", yaxis_title="",
                          height=max(260, 26 * len(units) * max(1, len(options)) + 160),
                          legend={"orientation": "h", "y": -0.12, "x": 0, "entrywidth": 150,
                                  "entrywidthmode": "pixels"})
        fig.add_vline(x=0, line={"color": _C.INK, "width": 1})
        figures.append(fig)
    # Last: the app shows charts two to a row, and the two power charts pair best.
    if result.get("near_optimal_ranges"):
        figures.append(_ranges_chart(result, when))
    return figures


_TABLE_CHARS = 175           # characters across a full-width table at 12 px (about 1250 px)


def _table_height(headers: list, rows: list, widths: list) -> int:
    """A table's height from its text: each row as tall as its longest cell
    wraps (an exported page shows the figure, which clips; the app shows a
    data frame, which does not)."""
    total = sum(widths) or 1.0

    def lines(cells) -> int:
        return max((math.ceil(len(str(c)) / max(_TABLE_CHARS * w / total - 2, 4)) for c, w in zip(cells, widths)),
                   default=1)
    head = 14 + 18 * max(lines(headers), 1)
    return 110 + head + sum(max(30, 12 + 18 * lines(r)) for r in rows)


def _plain_table(headers: list, rows: list, widths: list, title: str, row_height: int | None = None) -> go.Figure:
    """A copyable table (the app shows it as a data frame), titled."""
    fig = go.Figure(go.Table(
        columnwidth=widths,
        header={"values": [f"<b>{h}</b>" for h in headers], "fill_color": "#eef1f7", "align": "left",
                "font": {"size": 12}},
        cells={"values": [list(col) for col in zip(*rows)], "align": "left", "height": 26, "font": {"size": 12}}))
    extra = 15 * max(title.count("<br>") - 1, 0)            # a subtitle of several lines
    fig.update_layout(title=title, height=_table_height(headers, rows, widths) + extra,
                      **{**CHART_THEME, "margin": {"l": 10, "r": 10, "t": 70 + extra, "b": 10}})
    return fig


def _load_reduction_tables(options: list[dict], when: str) -> list[go.Figure]:
    """Which loads an option reduces, and by how much — instructions to
    customers (demand response) or involuntary relief."""
    rows = [[o.get("label", o.get("id")), a.get("load", ""), a.get("bus") or "—",
             f"{a.get('demand_response_mw', 0):.2f}", f"{a.get('load_relief_mw', 0):.2f}"]
            for o in options for a in o.get("load_actions") or []]
    if not rows:
        return []
    return [_plain_table(["Option", "Load", "Bus", "Demand response MW", "Load relief MW"], rows,
                         [1.4, 0.9, 0.9, 1.1, 1.0],
                         f"Load reductions at {when}<br><sup>Demand response: the share of a load reduced on "
                         "request, at its price · load relief: involuntary, the last resort (cost table, per load)"
                         "</sup>")]


def _switching_table(candidates: list[dict], when: str, after: str = "") -> go.Figure:
    rows = [["Open " + c.get("open", ""), "✓ clears every violation" if c.get("clears_all") else
             f"relieves {c.get('relieves_pct', 0)} %", "; ".join(c.get("remaining") or []) or "—"]
            for c in candidates]
    return _plain_table(["Switching", "Power flow", "Still out of limits"], rows, [1.6, 1.2, 2.6],
                        f"Switching that may help{after} at {when}<br><sup>Single openings an AC power flow found to "
                        "clear or relieve the violations · a screen, not an optimisation: check protection and "
                        "supply before switching</sup>")


def _ranges_chart(result: dict, when: str) -> go.Figure:
    """Each unit's range within the near-optimal margin, with where the
    cheapest puts it and its schedule: what is fixed and what is free."""
    ranges = result.get("near_optimal_ranges") or []
    units = [r["unit"] for r in ranges]
    fig = go.Figure()
    for r in ranges:
        fig.add_trace(go.Scatter(x=[r["min_mw"], r["max_mw"]], y=[r["unit"], r["unit"]], mode="lines",
                                 line={"color": _C.PRIMARY, "width": 10}, showlegend=False,
                                 hovertemplate=f"{r['unit']}: {r['min_mw']:.1f}–{r['max_mw']:.1f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["schedule_mw"] for r in ranges], y=units, mode="markers", name="Schedule",
                             marker={"color": "white", "size": 10, "line": {"color": _C.INK, "width": 2}},
                             hovertemplate="%{y} scheduled: %{x:.1f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["cheapest_mw"] for r in ranges], y=units, mode="markers", name="Cheapest",
                             marker={"color": OPTION_COLOURS["cheapest"], "size": 11, "symbol": "diamond",
                                     "line": {"color": "white", "width": 2}},
                             hovertemplate="%{y} in the cheapest: %{x:.1f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Range within the margin",
                             line={"color": _C.PRIMARY, "width": 10}))
    fig.update_layout(**CHART_THEME, title=(f"What each unit can do within {result.get('near_optimal_pct', 0):g} % "
                                            f"of the cheapest — {when}"),
                      xaxis_title="Active power (MW)", yaxis_title="", height=max(240, 46 * len(units) + 150),
                      legend={**_WIDE_LEGEND, "y": -0.3},
                      yaxis={"categoryorder": "array", "categoryarray": units, "autorange": "reversed"})
    return fig


def _tag(name) -> str:
    """"Bus_21 -> Bus_23 [L30]" -> "L30"; a name without a tag as it is."""
    found = re.search(r"\[([^\]]+)\]\s*$", str(name))
    return found.group(1) if found else str(name)


def _n1_tables(options: list[dict], study: dict, when: str) -> list[go.Figure]:
    """An N-1 set: each option's N-1 screen and what its study secured, then
    the corrective plan of a corrective option — what to do after each outage."""
    def table(headers, rows, widths, title, row_height=None):
        return _plain_table(headers, rows, widths, title)

    studied = study.get("outages_studied") or []
    figures, rows = [], []
    for o in options:
        n1 = o.get("n1") or {}
        screen = n1.get("screen") or {}
        if not screen:
            continue
        found = (f"{screen.get('outages_with_violations', '—')} of {screen.get('outages_tested', '—')}"
                 + (" before its action" if n1.get("screen_is") else ""))
        if "secured" in n1:
            secured = f"{n1['secured']} of {len(n1.get('studied') or studied)}"
            missing = ", ".join(_tag(x.get("outage")) for x in n1.get("not_secured") or []) or "none"
        else:
            secured, missing = "not studied", "—"
        cost = o.get("cost_eur_per_h")
        gap = o.get("above_cheapest_eur_per_h")
        rows.append([o.get("label", o.get("id")), found, str(screen.get("violations", "—")), secured, missing,
                     f"{cost:.2f}" if isinstance(cost, (int, float)) else "—",
                     f"{gap:+.2f}" if isinstance(gap, (int, float)) else "—"])
    if rows:
        band = study.get("post_outage_band_pu") or []
        limits = (f"after an outage up to {study.get('emergency_loading_pct', 100):g} % of rating"
                  + (f", voltages {band[0]:g}–{band[1]:g} p.u." if len(band) == 2 else ""))
        figures.append(table(
            ["Option", "Outages with violations", "Violations", "Studied outages secured", "Not secured",
             "Cost €/h", "Above cheapest €/h"], rows, [1.4, 1.3, 0.8, 1.1, 1.3, 0.8, 1.0],
            f"N-1 screen of each option at {when}<br><sup>Every line and transformer out in turn; {limits} · "
            "a corrective option is screened before its action</sup>", 28))
    for o in options:
        plan = (o.get("n1") or {}).get("corrective_plan") or []
        if not plan:
            continue
        units = sorted({str(a.get("unit")) for p in plan for a in p.get("actions") or []},
                       key=lambda u: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", u)])

        def cell(p, unit):
            a = next((a for a in p.get("actions") or [] if a.get("unit") == unit), None)
            if a is None:
                return "—"
            parts = [f"{a['dp_mw']:+.2f} MW"] if abs(a.get("dp_mw") or 0.0) >= 0.005 else []
            if "setpoint_vm_pu_after" in a:
                parts.append(f"V {a['setpoint_vm_pu_before']:.3f}→{a['setpoint_vm_pu_after']:.3f}")
            return ", ".join(parts) or "—"

        def outcome(p):
            if p.get("no_action_needed"):
                return "✓ no action needed"
            if p.get("checked_by_power_flow"):
                return "✓ confirmed"
            first = (p.get("after_action") or ["not confirmed"])[0]
            return "✗ " + (first if len(first) <= 40 else first[:39] + "…")

        def loads(p):
            parts = []
            for a in p.get("loads") or []:
                where = a.get("bus") or a.get("load")
                bits = ([f"−{a['demand_response_mw']:.2f} DR"] if a.get("demand_response_mw") else []) + (
                    [f"−{a['load_relief_mw']:.2f} relief"] if a.get("load_relief_mw") else [])
                parts.append(f"{where} {', '.join(bits)}")
            return "; ".join(parts) or "—"
        with_loads = any(p.get("loads") for p in plan)
        rows = [[p.get("outage", "")] + [cell(p, u) for u in units] + ([loads(p)] if with_loads else [])
                + [outcome(p)] for p in plan]
        reach = study.get("reach_pct")
        figures.append(table(
            ["After the outage of"] + units + (["Load reduced (MW)"] if with_loads else []) + ["Power flow"], rows,
            [1.6] + [1.0] * len(units) + ([2.0] if with_loads else []) + [1.4],
            f"Corrective plan of {o.get('label', o.get('id'))} at {when}<br><sup>What to do after each studied "
            f"outage: active power change{f' (at most {reach:g} % of capacity)' if reach else ''}, voltage "
            f"set-points{', load reduced when moving the units is not enough' if with_loads else ''} · each "
            "checked by a power flow without the outaged branch</sup>",
            48 if any("V " in c for r in rows for c in r[1:-1]) else 28))
    # What else might help where an outage could not be secured: switching.
    rows = [[o.get("label", o.get("id")), x.get("outage", ""), "; ".join((x.get("remaining") or [])[:2]),
             "; ".join(("open " + c["open"] + (" ✓ clears all" if c.get("clears_all") else
                                               f" (−{c.get('relieves_pct', 0)} %)")) for c in x.get("switching") or [])]
            for o in options for x in (o.get("n1") or {}).get("not_secured") or [] if x.get("switching")]
    if rows:
        figures.append(table(["Option", "Outage not secured", "Still out of limits", "Switching that helps"], rows,
                             [1.2, 1.5, 2.2, 2.4],
                             f"Outages the dispatch cannot secure — switching that may help, at {when}<br><sup>Single "
                             "openings after the outage, each checked by an AC power flow · a screen: check "
                             "protection and supply before switching</sup>", 40))
    return figures


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
    if result.get("mode") == "options" and result.get("feasible") is not False:
        return _render_options(result)
    # Any failed solve, not just the one spelled "infeasible": an iteration
    # limit or a capped solve that cannot meet its cap used to fall through to
    # "No dispatch data — run the optimizer first".
    if result.get("status") == "infeasible" or result.get("feasible") is False:
        reason = result.get("message") or "violations cannot be resolved by local flexibility"
        if str(reason).strip().rstrip(".").lower() == "infeasible":
            reason = "violations cannot be resolved by local flexibility"
        figures = [_empty_figure(f"❌ Optimizer did not find a usable dispatch — {reason}", color=_C.VIOLATION)]
        # Beyond redispatch: what reducing load — and switching — would take.
        rescue = result.get("last_resort") or {}
        if rescue.get("options"):
            figures += _render_options({**rescue, "timestamp": result.get("timestamp"),
                                        "outage": result.get("outage")})
        elif rescue.get("switching"):
            figures.append(_switching_table(rescue["switching"], str(result.get("timestamp") or "")[:16]))
        return figures

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
# 4.4 — Key performance indicators: the schedule against a corrective action
# ---------------------------------------------------------------------------

# How each indicator's number reads: (format, suffix).
_KPI_FORMATS = {"%": (".1f", " %"), "MW": (".1f", " MW"), "p.u.": ("+.3f", " p.u."), "pp": ("+.1f", " pp"),
                "€/h": (",.2f", " €/h"), "": ("d", "")}
# The tiles, in reading order: N-0 security, N-1 and what the action prevents,
# then flexibility (what there is, what is needed, what is used).
_KPI_TILES = ("violations", "vm_margin_pu", "loading_headroom_pct",
              "n1_outages_with_violations", "kpi_3_prevented_violation_ratio_pct", "cost_eur_per_h",
              "kpi_1_target_demand_flex_pct", "deliverable_up_mw", "deliverable_down_mw",
              "required_mw", "moved_mw", "kpi_2_flex_utilization_pct", "n1_security_eur_per_h")


# A tile is a third of a half-width chart: a short name, and what it counts.
_KPI_TILE_LABELS = {
    "violations": ("Violations", "N-0 · count"), "vm_margin_pu": ("Voltage margin", "p.u. to the nearest limit"),
    "loading_headroom_pct": ("Loading headroom", "pp to the limit"),
    "n1_outages_with_violations": ("N-1 outages", "with violations"),
    "kpi_3_prevented_violation_ratio_pct": ("KPI-3", "violations prevented"),
    "cost_eur_per_h": ("Action cost", "€/h, cost table"),
    "kpi_1_target_demand_flex_pct": ("KPI-1", "target demand flexibility"),
    "deliverable_up_mw": ("Deliverable up", "MW within limits"),
    "deliverable_down_mw": ("Deliverable down", "MW within limits"),
    "required_mw": ("Required", "fewest MW to clear"), "moved_mw": ("Activated", "MW the action moves"),
    "kpi_2_flex_utilization_pct": ("KPI-2", "flexibility utilisation"),
    "n1_security_eur_per_h": ("N-1 security", "€/h above the cheapest"),
}


def _kpi_value(value, unit: str) -> str:
    v = _float_or_none(value)
    if v is None:
        return "—"
    fmt, suffix = _KPI_FORMATS.get(unit, (".2f", f" {unit}"))
    return f"{(round(v) if fmt == 'd' else round(v, 3)) + 0:{fmt}}{suffix}"


def _kpi_compared(result: dict) -> str:
    """What the indicators compare the schedule with, in a few words."""
    after_is = result.get("after_is")
    if after_is == "schedule":
        return "no corrective action needed: the schedule's own indicators"
    if after_is == "none":
        return "no corrective action found"
    held = f", held from {_hhmm(result.get('held_since'))[-5:]}" if result.get("held") else ""
    return f"schedule → {result.get('after_label', 'the action')} ({result.get('after_ref', '')}){held}"


def _kpi_tiles(result: dict, rows: list[dict], when: str) -> go.Figure:
    """One tile per indicator: its value with the action, and the change from
    the schedule — ▲/▼ coloured by whether that way is better."""
    by_key = {r["key"]: r for r in rows}
    tiles = [by_key[k] for k in _KPI_TILES if k in by_key]
    cols = 3
    n_rows = max(1, math.ceil(len(tiles) / cols))
    fig = go.Figure()
    for i, r in enumerate(tiles):
        after, before = _float_or_none(r.get("after")), _float_or_none(r.get("before"))
        fmt, suffix = _KPI_FORMATS.get(r.get("unit", ""), (".2f", ""))
        good, bad = (_C.OK, _C.VIOLATION)
        up_good = r.get("better") == "higher"
        show_delta = (after is not None and before is not None and abs(after - before) > 1e-9
                      and r["key"] not in ("cost_eur_per_h", "moved_mw"))
        name, sub = _KPI_TILE_LABELS.get(r["key"], (r["name"], r.get("unit") or "count"))
        fig.add_trace(go.Indicator(
            mode="number+delta" if show_delta else "number",
            value=after if after is not None else float("nan"),
            number={"valueformat": fmt, "suffix": suffix, "font": {"size": 21, "color": _C.INK}},
            delta={"reference": before, "valueformat": fmt if fmt != "d" else "d", "relative": False,
                   "increasing": {"color": good if up_good else bad},
                   "decreasing": {"color": bad if up_good else good}, "font": {"size": 12}} if show_delta else None,
            title={"text": f"<b>{name}</b><br><span style='font-size:11px;color:#6B7280'>{sub}"
                           + ("" if after is not None else " · no result") + "</span>",
                   "font": {"size": 12}},
            domain={"row": i // cols, "column": i % cols}))
    fig.update_layout(**{**CHART_THEME, "margin": {"l": 10, "r": 10, "t": 118, "b": 10}},
                      grid={"rows": n_rows, "columns": cols, "pattern": "independent"},
                      height=138 + 125 * n_rows,
                      title=f"Key performance indicators at {when}<br><sup>{_kpi_compared(result)}"
                            + ("" if result.get("after_is") in ("schedule", "none") else
                               "<br>with the action · ▲▼ change from the schedule, green better, red worse")
                            + "</sup>")
    return fig


def _kpi_deliverable_chart(result: dict, when: str) -> go.Figure | None:
    """How far each unit can move within every limit (each alone, the others
    held), against the spare capacity it has: what the network lets through."""
    found = (result.get("deliverable") or {}).get("with_action")
    if not isinstance(found, dict) or not found.get("units"):
        return None
    units = list(found["units"])[::-1]                     # the first unit on top
    get = lambda u, k: _float_or_none((found["units"][u] or {}).get(k))  # noqa: E731
    fig = go.Figure()
    fig.add_trace(go.Bar(y=units, x=[get(u, "spare_mw") for u in units], orientation="h",
                         name="Spare capacity (KPI-1)", marker={"color": "rgba(154,163,178,0.28)",
                                                                "line": {"color": _C.NEUTRAL, "width": 1}},
                         hovertemplate="%{y}: %{x:.1f} MW spare<extra></extra>"))
    fig.add_trace(go.Bar(y=units, x=[get(u, "up_mw") for u in units], orientation="h", name="Up, within limits",
                         marker={"color": _C.TEAL, "cornerradius": 4}, width=0.45,
                         text=[_kpi_value(get(u, "up_mw"), "MW") for u in units], textposition="outside",
                         textfont={"color": _C.INK, "size": 11}, cliponaxis=False,
                         hovertemplate="%{y}: up %{x:.1f} MW<extra></extra>"))
    fig.add_trace(go.Bar(y=units, x=[-(get(u, "down_mw") or 0) if get(u, "down_mw") is not None else None
                                     for u in units], orientation="h", name="Down, within limits",
                         marker={"color": _C.PURPLE, "cornerradius": 4}, width=0.45,
                         text=[_kpi_value(get(u, "down_mw"), "MW") for u in units], textposition="outside",
                         textfont={"color": _C.INK, "size": 11}, cliponaxis=False,
                         customdata=[get(u, "down_mw") for u in units],
                         hovertemplate="%{y}: down %{customdata:.1f} MW<extra></extra>"))
    missing = [u for u in units if get(u, "up_mw") is None]
    for u in missing:
        fig.add_annotation(y=u, x=0, text="no range within the limits", showarrow=False, xanchor="left",
                           font={"size": 11, "color": _C.INK})
    total = (f"all together +{_kpi_value(found.get('up_mw'), 'MW')} / −{_kpi_value(found.get('down_mw'), 'MW')}"
             if found.get("up_mw") is not None else "no joint range")
    fig.add_vline(x=0, line={"color": _C.INK, "width": 1})
    # Room for the labels outside the bars, on both sides.
    reach_up = max([get(u, k) or 0 for u in units for k in ("up_mw", "spare_mw")] + [1.0])
    reach_down = max([get(u, "down_mw") or 0 for u in units] + [1.0])
    state = (f"with {result['after_label']}" if result.get("after_is") not in ("schedule", "none", None)
             else "the schedule")
    fig.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": 95, "l": 90}},
                      xaxis={"range": [-reach_down * 1.4, reach_up * 1.4], "title": "MW"},
                      barmode="overlay", height=165 + 46 * len(units), yaxis_title="",
                      legend=_WIDE_LEGEND,
                      title=f"How far each unit can move<br><sup>{when} · {state} · within every limit<br>"
                            f"each alone, the others held · {total}</sup>")
    return fig


def _kpi_table(result: dict, rows: list[dict], when: str) -> go.Figure:
    """Every indicator as a copyable table: schedule, with the action, change, what it measures."""
    body = []
    for r in rows:
        b, a = _float_or_none(r.get("before")), _float_or_none(r.get("after"))
        change = "—" if a is None or b is None else _kpi_value(a - b, r["unit"]) if r["unit"] not in ("", "%") \
            else (f"{a - b:+.1f} pp" if r["unit"] == "%" else f"{a - b:+g}")
        body.append([r["name"], _kpi_value(r.get("before"), r["unit"]), _kpi_value(r.get("after"), r["unit"]),
                     change, r.get("what", "")])
    notes = [d.get("note") for d in ((result.get("kpi_details") or {}).get(k) or {} for k in ("kpi_2", "kpi_3"))
             if d.get("note")]
    return _plain_table(["Indicator", "Schedule", "With the action", "Change", "What it measures"], body,
                        [1.5, 0.8, 0.9, 0.7, 2.4],
                        f"Indicators at {when}<br><sup>{_kpi_compared(result)}"
                        + "".join(f"<br>{n[:1].upper()}{n[1:]}" for n in notes) + "</sup>")


def render_kpis(result: dict) -> list[go.Figure]:
    """The indicators of one moment: tiles (with the action, change from the
    schedule), each unit's deliverable range against its spare capacity, and
    the table."""
    rows = [r for r in result.get("indicators") or [] if isinstance(r, dict) and r.get("key")]
    if not rows:
        return [_empty_figure(str(result.get("message") or result.get("error") or
                                  "No indicators — run evaluate_kpis first"))]
    when = _hhmm(result.get("timestamp"))
    figures = [_kpi_tiles(result, rows, when)]
    deliverable = _kpi_deliverable_chart(result, when)
    if deliverable is not None:
        figures.append(deliverable)
    figures.append(_kpi_table(result, rows, when))
    return figures


def render_kpis_over_time(result: dict) -> list[go.Figure]:
    """The indicators over a window, each moment with its own corrective
    action: what the units can deliver, what is required and activated, the
    three indicators of the project notebook, N-1 security, and the cost."""
    rows = [r for r in result.get("moments") or [] if isinstance(r, dict) and r.get("timestamp")]
    if not any(r.get("values") for r in rows):
        return [_empty_figure(str(result.get("message") or "Nothing to show: no moment has indicators"))]
    x = _times(r["timestamp"] for r in rows)
    summary = result.get("summary") or {}
    totals = summary.get("totals") or {}
    span = f"{_hhmm(rows[0]['timestamp'])} – {_hhmm(rows[-1]['timestamp'])}"   # the first subtitle line

    def series(key, which=1):
        return [_float_or_none(((r.get("values") or {}).get(key) or [None, None])[which]) for r in rows]

    def layout(fig, title, sub, unit, height=340):
        top = 80 + 15 * sub.count("<br>")                    # a subtitle of two lines
        fig.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": top}},
                          title=f"{title}<br><sup>{sub}</sup>", xaxis=_TIME_AXIS, yaxis_title=unit,
                          height=height, legend={**_WIDE_LEGEND, "y": -0.3})   # clear of the dates
        return fig

    def line(fig, y, name, colour, dash=None, unit="", fmt=".1f"):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers", connectgaps=False,
                                 line={"color": colour, "width": 2, **({"dash": dash} if dash else {})},
                                 marker={"size": 7 if not dash else 5},
                                 hovertemplate="%{x|%d %b %H:%M}: %{y:" + fmt + "} " + unit
                                               + "<extra>" + name + "</extra>"))

    figures = []
    # The project notebook's three, with the action.
    kpis = go.Figure()
    # KPI-2 where flexibility was required or activated (elsewhere it is a trivial
    # 100 %); KPI-3 where the schedule breaks a limit (elsewhere it has no value).
    acted = [(rq or 0) > 0 or (mv or 0) > 0 for rq, mv in zip(series("required_mw"), series("moved_mw"))]
    line(kpis, series("kpi_1_target_demand_flex_pct"), "KPI-1 Target demand flexibility", _C.PRIMARY, unit="%")
    line(kpis, [v if a else None for v, a in zip(series("kpi_2_flex_utilization_pct"), acted)],
         "KPI-2 Flexibility utilisation", _C.MAGENTA, unit="%")
    line(kpis, series("kpi_3_prevented_violation_ratio_pct"), "KPI-3 Prevented violations", _C.OLIVE, unit="%")
    kpis.update_layout(yaxis={"range": [-5, 108]})
    figures.append(layout(kpis, "KPI-1 to KPI-3", f"{span} · with each moment's action<br>KPI-2 where MW move · "
                                                  "KPI-3 where limits break", "%", height=355))
    # What the units can deliver within every limit, against the spare capacity they have.
    deliver = go.Figure()
    # Spare capacity as a wide band under the deliverable line: where the network
    # carries all of it, the line runs inside the band; where not, the band shows above.
    deliver.add_trace(go.Scatter(x=x, y=series("spare_up_mw"), name="Spare upward capacity", mode="lines",
                                 line={"color": "rgba(154,163,178,0.45)", "width": 9},
                                 hovertemplate="%{x|%d %b %H:%M}: %{y:.1f} MW spare<extra></extra>"))
    line(deliver, series("deliverable_up_mw"), "Deliverable up", _C.TEAL, unit="MW")
    line(deliver, series("deliverable_down_mw"), "Deliverable down", _C.PURPLE, unit="MW")
    worst = (summary.get("indicators") or {}).get("deliverable_up_mw", {}).get("worst")
    if worst and worst.get("value") is not None:
        when = _times([worst["timestamp"]])[0]
        deliver.add_trace(go.Scatter(x=[when], y=[worst["value"]], mode="markers", showlegend=False,
                                     marker={"size": 14, "color": "rgba(0,0,0,0)",
                                             "line": {"color": _C.WARN, "width": 2.5}}, hoverinfo="skip"))
        deliver.add_annotation(x=when, y=worst["value"], text="least upward", showarrow=False, yshift=-18,
                               font={"size": 10, "color": _C.INK})
    figures.append(layout(deliver, "Flexibility the units can deliver",
                          f"{span} · within every limit<br>grey band: spare capacity, deliverable or not", "MW",
                          height=355))
    # What each moment requires, and what its action moves.
    need = go.Figure()
    for key, name, colour in (("required_mw", "Required (fewest MW)", _C.PRIMARY),
                              ("moved_mw", "Activated by the action", _C.MAGENTA)):
        need.add_trace(go.Bar(x=x, y=series(key), name=name, marker={"color": colour, "cornerradius": 4},
                              hovertemplate="%{x|%d %b %H:%M}: %{y:.1f} MW<extra>" + name + "</extra>"))
    need.update_layout(barmode="group", bargap=0.3)
    moved = (f"<br>{totals['mwh_moved']:.1f} MWh activated, {totals['mwh_required']:.1f} MWh required over the "
             f"{_window_hours(result)}" if "mwh_moved" in totals and "mwh_required" in totals else "")
    figures.append(layout(need, "Flexibility required and activated", span + moved, "MW", height=355))
    # N-1 security, and the schedule's N-0 violations.
    n1 = go.Figure()
    line(n1, series("n1_outages_with_violations", 0), "N-1, as scheduled", _C.NEUTRAL, dash="dot", fmt="d")
    line(n1, series("n1_outages_with_violations"), "N-1, with the action", _C.PRIMARY, fmt="d")
    n1.add_trace(go.Scatter(x=x, y=[v if v else None for v in series("violations", 0)],
                            name="N-0 violations, as scheduled", mode="markers",
                            marker={"symbol": "x", "size": 9, "color": _C.VIOLATION},
                            hovertemplate="%{x|%d %b %H:%M}: %{y} violations<extra>as scheduled</extra>"))
    tested = next(((r.get("schedule") or {}).get("n1_outages_tested") for r in rows
                   if isinstance(r.get("schedule"), dict)), None)
    figures.append(layout(n1, "Security", f"{span}<br>single outages with violations"
                                          f"{f', of {tested} tested' if tested else ''}", "count", height=355))
    # The cost of each moment's action.
    cost = go.Figure(go.Bar(x=x, y=series("cost_eur_per_h"), name="Cost of the action",
                            marker={"color": OPTION_COLOURS["cheapest"], "cornerradius": 4},
                            hovertemplate="%{x|%d %b %H:%M}: %{y:,.2f} €/h<extra></extra>"))
    cost.add_hline(y=0, line={"color": _C.INK, "width": 1})
    net = totals.get("cost_eur")
    figures.append(layout(cost, "Cost of the corrective action",
                          f"{span} · at the cost table's prices<br>" + (
                              "not priced" if net is None else
                              f"{net:,.0f} € over the {_window_hours(result)}" if net >= 0 else
                              f"{-net:,.0f} € saved over the {_window_hours(result)} (negative: cheaper)"),
                          "€/h", height=345))
    return figures


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


def render_network_comparison(result: dict) -> list[go.Figure]:
    """compare_networks: what a proposal changes against the base.

    Security assessment — the largest voltage changes and the largest loading
    changes (proposal − base), one chart each. N-1 screen — outages that cause
    violations on one network and not the other. Increases are teal, decreases
    purple; the title names the proposal and the timestamp, so a chart can
    never be read as the base on its own.
    """
    if result.get("several"):
        return _render_several_comparisons(result)
    net = result.get("network") or {}
    ref = net.get("compared_with") or "base"
    where = f"{net.get('id', 'proposal')} − {ref} at {str(result.get('timestamp', '?'))[:16]}"
    figures: list[go.Figure] = []

    def bars(rows, title, unit, fmt):
        rows = [r for r in rows or [] if _float_or_none(r.get("change")) is not None]
        if not rows:
            return
        rows = sorted(rows, key=lambda r: abs(r["change"]))
        fig = go.Figure(go.Bar(
            x=[r["change"] for r in rows], y=_safe_labels([r["element"] for r in rows], "Element"),
            orientation="h",
            marker_color=[_C.TEAL if r["change"] > 0 else _C.PURPLE for r in rows],
            customdata=[[r.get("base"), r.get("proposal")] for r in rows],
            hovertemplate="%{y}: %{customdata[0]:" + fmt + "} → %{customdata[1]:" + fmt
                          + "} " + unit + "<extra></extra>",
        ))
        fig.update_layout(**CHART_THEME, title=f"{title} ({where})",
                          xaxis_title=f"Change ({unit})", yaxis_title="",
                          height=max(220, 40 * len(rows) + 120))
        figures.append(fig)

    if result.get("study") == "scan":
        series = result.get("series") or {}
        stamps = series.get("timestamps") or []
        if not stamps:
            return [_empty_figure(f"No ticks scanned ({where})")]
        limits = result.get("thresholds_used") or {}
        window = result.get("window") or {}
        span = f"{net.get('id', 'proposal')} vs {ref} · {window.get('start', '?')[:16]} → {window.get('end', '?')[:16]}"
        for key, title, unit, limit_key in (
                ("min_vm", "Lowest bus voltage", "p.u.", "vm_lower_pu"),
                ("max_line_loading", "Highest line loading", "%", "max_line_loading_pct")):
            fig = go.Figure([
                go.Scatter(x=stamps, y=series.get(f"base_{key}"), name=ref, mode="lines",
                           line=dict(color=_C.NEUTRAL, width=1.5)),
                go.Scatter(x=stamps, y=series.get(f"proposal_{key}"), name=net.get("id", "proposal"),
                           mode="lines", line=dict(color=_C.PRIMARY, width=2)),
            ])
            limit = _float_or_none(limits.get(limit_key))
            if limit is not None:
                fig.add_hline(y=limit, line=dict(color=_C.VIOLATION, dash="dash", width=1),
                              annotation_text="limit", annotation_position="top left")
            fig.update_layout(**CHART_THEME, title=f"{title} over time ({span})",
                              yaxis_title=unit, xaxis_title="", hovermode="x unified")
            figures.append(fig)
        return figures

    if result.get("study") == "opf":
        units = result.get("units") or []
        if not units:
            b, p = result.get("base") or {}, result.get("proposal") or {}
            text = (f"No unit is moved on either network ({where})" if b.get("feasible") and p.get("feasible")
                    else f"Corrective dispatch — base: {b.get('status')}, proposal: {p.get('status')} ({where})")
            return [_empty_figure(text)]
        names = _safe_labels([u["element"] for u in units], "Unit")
        fig = go.Figure([
            go.Bar(name=ref, y=names, x=[u["base_dP_mw"] for u in units], orientation="h",
                   marker_color=_C.NEUTRAL),
            go.Bar(name=net.get("id", "proposal"), y=names, x=[u["proposal_dP_mw"] for u in units],
                   orientation="h", marker_color=_C.PRIMARY),
        ])
        fig.update_layout(**CHART_THEME, barmode="group",
                          title=f"Corrective redispatch by unit ({where})",
                          xaxis_title="Change in active power (MW)", yaxis_title="",
                          height=max(240, 46 * len(units) + 140))
        return [fig]

    if result.get("study") == "n1":
        def per_outage(rows):
            counts: dict[str, int] = {}
            for r in rows or []:
                counts[str(r.get("outage"))] = counts.get(str(r.get("outage")), 0) + 1
            return counts
        new, gone = per_outage(result.get("new_violations")), per_outage(result.get("resolved_violations"))
        outages = sorted(set(new) | set(gone), key=lambda o: new.get(o, 0) - gone.get(o, 0))
        if not outages:
            return [_empty_figure(f"Same N-1 violations on both networks ({where})", _C.OK)]
        fig = go.Figure([
            go.Bar(name="New on the proposal", y=outages, x=[new.get(o, 0) for o in outages],
                   orientation="h", marker_color=_C.VIOLATION),
            go.Bar(name="Resolved by the proposal", y=outages, x=[-gone.get(o, 0) for o in outages],
                   orientation="h", marker_color=_C.OK),
        ])
        fig.update_layout(**CHART_THEME, barmode="relative",
                          title=f"N-1 violations by outage ({where})",
                          xaxis_title="Violations (new → right, resolved ← left)", yaxis_title="",
                          height=max(240, 40 * len(outages) + 140))
        return [fig]

    bars(result.get("largest_voltage_changes"), "Largest voltage changes", "p.u.", ".4f")
    bars((result.get("largest_line_loading_changes") or []) + (result.get("largest_trafo_loading_changes") or []),
         "Largest loading changes", "%", ".1f")
    return figures or [_empty_figure(f"No measurable change ({where})")]


def _render_several_comparisons(result: dict) -> list[go.Figure]:
    """compare_networks with several proposals: the ranking table as bars
    (one chart per figure, every network side by side, the reference grey),
    or — for a scan — every network's curve over the window on one chart."""
    ref = result.get("compared_with") or "base"
    comparisons = [c for c in result.get("comparisons") or [] if isinstance(c, dict)]
    study = result.get("study")
    when = str(result.get("timestamp") or "")[:16]
    if study == "opf":
        return [fig for c in comparisons for fig in render_network_comparison(c)]
    if study == "scan":
        if not comparisons:
            return [_empty_figure("No proposals compared")]
        first = comparisons[0].get("series") or {}
        limits = comparisons[0].get("thresholds_used") or {}
        figures = []
        for key, title, unit, limit_key in (
                ("min_vm", "Lowest bus voltage", "p.u.", "vm_lower_pu"),
                ("max_line_loading", "Highest line loading", "%", "max_line_loading_pct")):
            traces = [go.Scatter(x=first.get("timestamps"), y=first.get(f"base_{key}"), name=ref, mode="lines",
                                 line=dict(color=_C.NEUTRAL, width=1.5))]
            for c in comparisons:
                series = c.get("series") or {}
                traces.append(go.Scatter(x=series.get("timestamps"), y=series.get(f"proposal_{key}"),
                                         name=(c.get("network") or {}).get("id", "proposal"), mode="lines"))
            fig = go.Figure(traces)
            limit = _float_or_none(limits.get(limit_key))
            if limit is not None:
                fig.add_hline(y=limit, line=dict(color=_C.VIOLATION, dash="dash", width=1),
                              annotation_text="limit", annotation_position="top left")
            fig.update_layout(**CHART_THEME, title=f"{title} over time (proposals vs {ref})",
                              yaxis_title=unit, xaxis_title="", hovermode="x unified")
            figures.append(fig)
        return figures
    rows = [r for r in result.get("table") or [] if isinstance(r, dict)]
    if not rows:
        return [_empty_figure("No proposals compared")]
    names = _safe_labels([str(r.get("network")) for r in rows], "Network")
    colours = [_C.NEUTRAL if r.get("role") == "reference" else _C.PRIMARY for r in rows]
    metrics = (("total_outages_causing_violations", "Outages causing violations", ""),
               ("total_violations", "Violations under single outages", "")) if study == "n1" else (
               ("total_violations", "Violations", ""),
               ("min_vm_pu", "Lowest bus voltage", "p.u."),
               ("max_line_loading_pct", "Highest line loading", "%"))
    figures = []
    for key, title, unit in metrics:
        values = [_float_or_none(r.get(key)) for r in rows]
        if all(v is None for v in values):
            continue
        fig = go.Figure(go.Bar(x=names, y=values, marker_color=colours,
                               hovertemplate="%{x}: %{y}" + (f" {unit}" if unit else "") + "<extra></extra>"))
        fig.update_layout(**CHART_THEME, title=f"{title} by network ({study.upper()} at {when})",
                          yaxis_title=unit or "count", xaxis_title="")
        if key == "min_vm_pu":
            fig.update_yaxes(range=[min(v for v in values if v is not None) - 0.02,
                                    max(v for v in values if v is not None) + 0.01])
        figures.append(fig)
    return figures or [_empty_figure("No figures to compare")]



def _hhmm(ts) -> str:
    """"2000-01-06 20:15:00" -> "06 Jan 20:15" (a window inside one week reads so)."""
    try:
        import pandas as pd
        return pd.Timestamp(ts).strftime("%d %b %H:%M")
    except Exception:  # noqa: BLE001
        return str(ts)[:16]


def _window_hours(result: dict) -> str:
    """ "8 h window" from a window study's `hours_each` (they sum to the window)."""
    try:
        return f"{sum(float(h) for h in result.get('hours_each') or []):g} h window"
    except (TypeError, ValueError):
        return "window"


def _times(values) -> list:
    """Timestamps as datetimes, for a real time axis (sparse, readable ticks —
    17 quarter-hours as categories crowded a half-width chart)."""
    import pandas as pd
    try:
        return [pd.Timestamp(v).to_pydatetime() for v in values]
    except Exception:  # noqa: BLE001
        return list(values)


# A time axis within a day or two: hours, the date beneath.
_TIME_AXIS = {"tickformat": "%H:%M\n%d %b", "nticks": 7, "title": ""}


def _render_hold(result: dict) -> list[go.Figure]:
    """A corrective option held over the next hours, against the schedule
    untouched: the highest loading and the lowest voltage at each moment, the
    limits, and the first moment it no longer holds."""
    hours = result.get("hours") or []
    if not hours:
        return [_empty_figure("Nothing to show: no moments were studied")]
    option = result.get("option") or {}
    limits = result.get("thresholds_used") or {}
    x = _times(h["timestamp"] for h in hours)
    fail = result.get("first_failure")
    until = result.get("holds_until")
    colour = OPTION_COLOURS.get(str(option.get("ref", "")).split(":")[-1], _C.PRIMARY)
    # Short: the chart is half a row wide.
    verdict = f"out of limits from {_hhmm(fail)[-5:]}" if fail else "within limits throughout"
    figures = []
    for key, title, unit, limit, fmt in (
            ("max_line_loading_pct", "Highest line loading", "%", limits.get("max_line_loading_pct"), ".1f"),
            ("min_vm_pu", "Lowest voltage", "p.u.", limits.get("vm_lower_pu"), ".3f")):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=[h["schedule"].get(key) for h in hours], name="Schedule, untouched",
                                 mode="lines", line={"color": _C.NEUTRAL, "width": 2, "dash": "dot"},
                                 hovertemplate="%{x|%H:%M}: %{y:" + fmt + "} " + unit + "<extra>schedule</extra>"))
        fig.add_trace(go.Scatter(x=x, y=[h["with_option"].get(key) for h in hours],
                                 name=f"{option.get('label', 'Option')} held", mode="lines+markers",
                                 line={"color": colour, "width": 2}, marker={"size": 7},
                                 hovertemplate="%{x|%H:%M}: %{y:" + fmt + "} " + unit + "<extra>held</extra>"))
        if isinstance(limit, (int, float)):
            fig.add_hline(y=limit, line={"color": _C.VIOLATION, "width": 1.5, "dash": "dash"},
                          annotation_text=f"limit {limit:g}", annotation_position="top left",
                          annotation_font={"color": _C.INK, "size": 11})
        if fail:
            when = _times([fail])[0]
            fig.add_shape(type="line", x0=when, x1=when, y0=0, y1=1, yref="paper",
                          line={"color": _C.WARN, "width": 1.5})
            fig.add_annotation(x=when, y=1, yref="paper", text="first out of limits", showarrow=False,
                               xanchor="left", yanchor="bottom", font={"size": 10, "color": _C.INK})
        fig.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": 80}},
                          title=f"{title} ({unit})<br><sup>{option.get('label', 'Option')} ({option.get('ref', '')}) "
                                f"held from {_hhmm(option.get('timestamp'))[-5:]} · {verdict}</sup>",
                          xaxis=_TIME_AXIS, yaxis_title="", height=340, legend=_WIDE_LEGEND)
        figures.append(fig)
    if any("n1_outages_with_violations" in h["with_option"] for h in hours):
        fig = go.Figure(go.Bar(x=x, y=[h["with_option"].get("n1_outages_with_violations") for h in hours],
                               marker={"color": colour, "cornerradius": 4},
                               hovertemplate="%{x|%H:%M}: %{y} outages with violations<extra></extra>"))
        fig.update_layout(**CHART_THEME, title="N-1: outages with violations, option held", height=300,
                          xaxis=_TIME_AXIS, yaxis_title="")
        figures.append(fig)
    return figures


def _render_security_over_time(result: dict) -> list[go.Figure]:
    """What N-1 security costs at each moment of a window, above the cheapest
    corrective action, and the outages each leaves with violations."""
    rows = [r for r in result.get("moments") or [] if isinstance(r, dict)]
    if not rows:
        return [_empty_figure("Nothing to show: no moments were studied")]
    x = _times(r["timestamp"] for r in rows)
    modes = result.get("modes") or ["preventive"]
    series = [("cheapest", "Cheapest (N-0)")] + [(m, f"N-1 {m}") for m in modes]

    def get(r, key, field):
        part = r.get(key)
        return part.get(field) if isinstance(part, dict) else None
    cost = go.Figure()
    for key, name in series:
        cost.add_trace(go.Bar(x=x, y=[get(r, key, "cost_eur_per_h") for r in rows], name=name,
                              marker={"color": OPTION_COLOURS.get("cheapest" if key == "cheapest" else f"n1_{key}"),
                                      "cornerradius": 4},
                              hovertemplate="%{x|%d %b %H:%M}: %{y:.2f} €/h<extra>" + name + "</extra>"))
    totals = result.get("totals") or {}
    total = " · ".join(f"N-1 {m}: {t['security_cost_eur']:,.0f} € above the cheapest" for m, t in totals.items())
    cost.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": 80}}, barmode="group",
                       bargap=0.3, height=360, xaxis=_TIME_AXIS, yaxis_title="€/h",
                       title=f"Cost of N-1 security, {_hhmm(rows[0]['timestamp'])} – {_hhmm(rows[-1]['timestamp'])}"
                             f"<br><sup>{total or 'not priced'} over the {_window_hours(result)}</sup>",
                       legend=_WIDE_LEGEND)
    cost.add_hline(y=0, line={"color": _C.INK, "width": 1})
    outages = go.Figure()
    short = sum(1 for r in rows if r.get("schedule_meets_limits") is False)
    outages.add_trace(go.Scatter(x=x, y=[r.get("schedule_n1_outages_with_violations") for r in rows],
                                 name="Schedule (reference)" if short else "Schedule", mode="lines",
                                 line={"color": _C.NEUTRAL, "dash": "dot", "width": 2}))
    for key, name in series:
        outages.add_trace(go.Scatter(
            x=x, y=[get(r, key, "n1_outages_with_violations") for r in rows], name=name, mode="lines+markers",
            marker={"size": 8}, line={"width": 2, "color": OPTION_COLOURS.get(
                "cheapest" if key == "cheapest" else f"n1_{key}")}))
    tested = next((r.get("outages_tested") for r in rows if r.get("outages_tested")), None)
    notes = " · ".join(([f"the schedule breaks normal limits at {short} of {len(rows)} moments:"
                         " compare with the cheapest"] if short else [])
                       + (["a corrective option before its action"] if "corrective" in modes else []))
    outages.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": 80}}, height=360,
                          xaxis=_TIME_AXIS, yaxis_title="outages",
                          title=f"N-1: outages with violations{f' (of {tested})' if tested else ''}"
                                + (f"<br><sup>{notes}</sup>" if notes else ""),
                          legend=_WIDE_LEGEND)
    return [cost, outages]


def _render_plan(result: dict) -> list[go.Figure]:
    """One corrective plan across a window: each unit's change at each moment,
    as a step (an instruction holds until the next), and the comparison with
    solving each moment on its own."""
    plan = result.get("plan") or {}
    steps = plan.get("steps") or []
    units = result.get("units") or []
    if not steps or not units:
        return [_empty_figure("Nothing to show: no plan was found")]
    x = _times(s["timestamp"] for s in steps)
    fig = go.Figure()
    for n, unit in enumerate(units):
        fig.add_trace(go.Scatter(x=x, y=[s["dp_mw"].get(unit, 0.0) for s in steps], name=unit,
                                 mode="lines+markers", line={"shape": "hv", "width": 2,
                                                             "color": UNIT_COLOURS[n] if n < len(UNIT_COLOURS)
                                                             else _C.NEUTRAL},
                                 marker={"size": 7},
                                 hovertemplate="%{x|%H:%M}: %{y:+.2f} MW<extra>" + unit + "</extra>"))
    fig.add_hline(y=0, line={"color": _C.INK, "width": 1})
    alone = result.get("each_moment_alone") or {}
    versus = (f"{plan.get('instructions_issued')} instructions against {alone.get('instructions_issued')} solving "
              "each moment on its own" if alone else f"{plan.get('instructions_issued')} instructions")
    fig.update_layout(**{**CHART_THEME, "margin": {**CHART_THEME.get("margin", {}), "t": 80}}, height=380,
                      xaxis=_TIME_AXIS, yaxis_title="Change from schedule (MW)",
                      title=f"Corrective plan, {_hhmm(steps[0]['timestamp'])} – {_hhmm(steps[-1]['timestamp'])[-5:]}"
                            f"<br><sup>{versus} · {plan.get('mwh_moved', 0):.1f} MWh moved · "
                            f"{str(result.get('objective', '')).replace('_', ' ')}"
                            f"{f' · ramp {result['ramp_pct_per_h']:g} %/h' if result.get('ramp_pct_per_h') else ''}"
                            "</sup>", legend=_WIDE_LEGEND)
    rows = [[_hhmm(s["timestamp"])] + [f"{s['dp_mw'].get(u, 0.0):+.2f}" for u in units]
            + [", ".join(s.get("changed") or []) or "—", f"{s.get('cost_eur_per_h', 0):.2f}",
               "✓ confirmed" if s.get("power_flow_check") else "✗ not confirmed"] for s in steps]
    table = _plain_table(["Moment"] + units + ["Instructions changed", "Cost €/h", "Power flow"], rows,
                         [1.1] + [0.8] * len(units) + [1.8, 0.8, 1.0],
                         "The plan, moment by moment<br><sup>Change from the schedule, MW · an instruction holds "
                         "until it changes · each moment checked by a power flow</sup>")
    return [fig, table]


def render_over_time(result: dict) -> list[go.Figure]:
    """flexibility_over_time: a held option, a plan, or the cost of security."""
    mode = result.get("mode")
    if mode == "hold":
        return _render_hold(result)
    if mode == "plan":
        return _render_plan(result)
    if mode == "security_over_time":
        return _render_security_over_time(result)
    return [_empty_figure(str(result.get("message") or "Nothing to show"))]

RENDERER_MAP: dict[str, Callable] = {
    "run_rsa": render_rsa,                              # returns list of 3 figs
    "simulate_contingency": render_contingency_violations,
    "simulate_all_contingencies": render_contingency_violations,
    "optimize_flexibility": render_dispatch,
    "optimize_contingency": render_dispatch,
    "evaluate_kpis": render_kpis,
    "kpis_over_time": render_kpis_over_time,
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
    "compare_networks": render_network_comparison,
    "flexibility_over_time": render_over_time,
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


# A change, not a level: purple for a decrease, teal for an increase, white
# for none — the palette's own meanings, so a drop never reads as a violation.
_DIFF_SCALE = [[0.0, _C.PURPLE], [0.5, "#f4f5f7"], [1.0, _C.TEAL]]


def _diff_colour(delta: float, span: float) -> str:
    from plotly.colors import sample_colorscale
    t = 0.5 + 0.5 * max(-1.0, min(1.0, delta / span))
    return sample_colorscale(_DIFF_SCALE, [t])[0]


def render_network_diff(base: NetworkMap, proposal: NetworkMap, positions: dict, *,
                        new_violations: set | None = None, subtitle: str = "") -> go.Figure:
    """What a proposal changes, drawn on the network: proposal − base.

    Buses are filled by the change in voltage and branches coloured and sized
    by the change in loading, on one symmetric scale each so the largest
    change sets the range. Branches the proposal opens are dashed grey;
    elements violating on the proposal and not on the base are ringed; added
    demand and generation sit beside the bus they were added to. Hover gives
    both values.
    """
    new_violations = new_violations or set()
    drawn = [n for n in proposal.nodes.values() if n.index in positions]
    if not drawn:
        return _empty_figure("No network to draw")
    xs = [positions[n.index][0] for n in drawn]
    ys = [positions[n.index][1] for n in drawn]
    span = max(max(ys) - min(ys), max(xs) - min(xs)) or 1.0
    per_px = span / _PLOT_PX
    offset = 0.022 * span

    def dv(node):
        b = base.nodes.get(node.index)
        return (None if b is None or b.vm_pu is None or node.vm_pu is None
                else node.vm_pu - b.vm_pu)

    base_edges = {(e.kind, e.index): e for e in base.edges}

    def dl(edge):
        b = base_edges.get((edge.kind, edge.index))
        if b is None or b.loading_pct is None or edge.loading_pct is None:
            return None
        return edge.loading_pct - b.loading_pct

    v_span = max([abs(d) for d in map(dv, drawn) if d is not None] + [0.005])
    l_span = max([abs(d) for d in map(dl, proposal.edges) if d is not None] + [1.0])

    traces: list[go.Scatter] = []
    hx, hy, ht = [], [], []
    for edge in proposal.edges:
        if edge.from_bus not in positions or edge.to_bus not in positions:
            continue
        (x0, y0), (x1, y1) = positions[edge.from_bus], positions[edge.to_bus]
        b = base_edges.get((edge.kind, edge.index))
        opened = b is not None and b.in_service and not edge.in_service
        d = dl(edge)
        if edge.kind == "ext_grid":
            colour, width, dash, detail = _EXTERNAL, 3.0, "solid", "external connection"
        elif b is None:
            # Built by the proposal: there is nothing to subtract.
            load = "" if edge.loading_pct is None else f" · {edge.loading_pct:.1f}% loaded"
            colour, width, detail = _C.PRIMARY, 3.0, "added by the proposal" + load
            dash = "dot" if edge.kind == "trafo" else "solid"
        elif opened:
            colour, width, dash, detail = _OUT_OF_SERVICE, 2.0, "dash", "opened by the proposal"
        elif d is None:
            colour, width, dash, detail = _UNMEASURED_EDGE, 1.4, "solid", "not measured"
        else:
            colour = _diff_colour(d, l_span)
            width = 1.4 + 4.6 * min(abs(d) / l_span, 1.0)
            dash = "dot" if edge.kind == "trafo" else "solid"
            detail = f"{b.loading_pct:.1f}% → {edge.loading_pct:.1f}% ({d:+.1f} pts)"
        if edge.name in new_violations:
            colour, width = _OVER_LIMIT_EDGE, max(width, 5.5)
            detail += " — newly over its limit"
        traces.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip",
                                 showlegend=False, line=dict(color=colour, width=width, dash=dash)))
        for f in (0.2, 0.35, 0.5, 0.65, 0.8):
            hx.append(x0 + (x1 - x0) * f)
            hy.append(y0 + (y1 - y0) * f)
            ht.append(f"<b>{edge.name}</b><br>{edge.kind} · {detail}")
    if hx:
        traces.append(go.Scatter(x=hx, y=hy, mode="markers", showlegend=False,
                                 marker=dict(size=14, color="rgba(0,0,0,0)"),
                                 hoverinfo="text", hovertext=ht))

    fills, outlines, widths, sizes, hovers, symbols = [], [], [], [], [], []
    for node in drawn:
        d = dv(node)
        flagged = node.name in new_violations
        b = base.nodes.get(node.index)
        new_bus = b is None and node.kind != "external"
        if node.kind == "external":
            symbols.append("square")
            fills.append(_EXTERNAL)
        elif new_bus:
            symbols.append("diamond")
            fills.append("white")
        elif d is None:
            symbols.append("circle-open")
            fills.append(_UNMEASURED_EDGE)
        else:
            symbols.append("circle")
            fills.append(_diff_colour(d, v_span))
        outlines.append("#111111" if flagged else _C.PRIMARY if new_bus else "#5b5b5b")
        widths.append(3.0 if flagged or new_bus else 1.0)
        sizes.append(MARKER_PX.get(node.kind, 10) * proposal.marker_scale + (3 if flagged else 0))
        if new_bus:
            voltage = "added by the proposal · " + (
                "not measured" if node.vm_pu is None else f"{node.vm_pu:.4f} pu")
        else:
            voltage = ("not measured" if d is None
                       else f"{b.vm_pu:.4f} → {node.vm_pu:.4f} pu ({d:+.4f})")
        hovers.append(f"<b>{node.name}</b><br>bus {node.index} · {node.vn_kv:g} kV<br>V {voltage}"
                      + ("<br><b>newly violating</b>" if flagged else ""))
    names = [n.name for n in drawn]
    ranked = [(0 if n.name in new_violations else 1 if n.kind == "substation" else 2,
               -abs(dv(n) or 0.0), i) for i, n in enumerate(drawn)]
    obstacles = [(xs[i] - (sizes[i] / 2 + 3) * per_px, ys[i] - (sizes[i] / 2 + 3) * per_px,
                  xs[i] + (sizes[i] / 2 + 3) * per_px, ys[i] + (sizes[i] / 2 + 3) * per_px)
                 for i in range(len(drawn))]
    labels, placements, dropped = _place_labels(xs, ys, names, sizes, ranked, obstacles, per_px)
    traces.append(go.Scatter(
        x=xs, y=ys, mode="markers+text", showlegend=False,
        marker=dict(size=sizes, color=fills, symbol=symbols,
                    line=dict(color=outlines, width=widths)),
        text=labels, textposition=placements, textfont=dict(size=_FONT_PX, color="#333"),
        hoverinfo="text", hovertext=hovers))

    # Added demand and generation, beside the bus they were added to; sized by
    # the square root of the change, as the power marks on the map are.
    def change_of(node, attr):
        b = base.nodes.get(node.index)
        return getattr(node, attr) - (getattr(b, attr) if b else 0.0)

    peak = max([abs(change_of(n, a)) for n in drawn for a in ("gen_mw", "load_mw")] + [1e-9])
    for attr, symbol, colour, label, sign in (("gen_mw", "triangle-up", _GENERATION, "added generation", 1),
                                              ("load_mw", "triangle-down", _DEMAND, "added demand", -1)):
        px, py, ps, ph = [], [], [], []
        for node in drawn:
            change = change_of(node, attr)
            if abs(change) < 0.01 or node.kind == "external":
                continue
            x, y = positions[node.index]
            px.append(x)
            py.append(y + sign * offset)
            ps.append(7 + 13 * (abs(change) / peak) ** 0.5)
            ph.append(f"<b>{node.name}</b><br>{label}: {change:+.2f} MW")
        if px:
            traces.append(go.Scatter(x=px, y=py, mode="markers", name=label, showlegend=True,
                                     marker=dict(size=ps, color=colour, symbol=symbol),
                                     hoverinfo="text", hovertext=ph))

    traces.append(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=0.1, color=[-v_span, v_span], colorscale=_DIFF_SCALE,
                    cmin=-v_span, cmax=v_span, showscale=True,
                    colorbar=dict(title=dict(text="ΔV (pu)", side="right"), thickness=12,
                                  len=0.38, x=1.02, y=0.0, yanchor="bottom",
                                  tickformat="+.3f"))))

    title = f"<b>{proposal.name}</b> — difference from the base"
    if subtitle:
        title += f"<br><span style='font-size:12px;color:#666'>{subtitle}</span>"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left"),
        hovermode="closest", plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=74, b=44), height=720,
        legend=dict(orientation="v", x=1.0, y=1.0, xanchor="left",
                    bgcolor="rgba(255,255,255,0.8)", borderwidth=0, font=dict(size=10)),
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1),
        annotations=[dict(
            text=(f"buses: voltage change (±{v_span:.3f} pu) · branches: loading change "
                  f"(±{l_span:.1f} pts, width = size) · purple decrease, teal increase · "
                  "dashed = opened · blue = added · dark ring/line = newly violating"
                  + (f" · {dropped} name(s) hidden" if dropped else "")),
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
