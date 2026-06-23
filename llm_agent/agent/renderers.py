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
import re

import plotly.graph_objects as go
import plotly.express as px

from .config import DEFAULT_GRID_CONSTANTS

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

CHART_THEME: dict = {
    "template": "plotly_white",
    "margin": {"l": 40, "r": 20, "t": 50, "b": 60},
}

_VM_LOWER = DEFAULT_GRID_CONSTANTS["vm_lower"]
_VM_UPPER = DEFAULT_GRID_CONSTANTS["vm_upper"]
_MAX_LOADING = DEFAULT_GRID_CONSTANTS["max_loading_pct"]


def _empty_figure(title: str, color: str = "#333") -> go.Figure:
    """Return a blank figure with a descriptive centred title."""
    fig = go.Figure()
    fig.update_layout(
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


def _compact_line_label(label: str, max_len: int = 28) -> str:
    """Build a compact axis label for line names while keeping stable identity.

    Expected long format: "CODE | From -> To [L12]".
    Compact format: "CODE [L12]". Falls back to truncation when pattern is absent.
    """
    raw = str(label).strip()
    if not raw:
        return raw

    # Keep stable line index when present.
    idx_match = re.search(r"\[L\d+\]$", raw)
    idx_suffix = f" {idx_match.group(0)}" if idx_match else ""

    code = re.sub(r"\s*\[L\d+\]$", "", raw.split("|", 1)[0]).strip()
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
    vm_lower = thresholds.get("vm_lower_pu", _VM_LOWER)
    vm_upper = thresholds.get("vm_upper_pu", _VM_UPPER)
    max_loading = thresholds.get("max_line_loading_pct", _MAX_LOADING)

    # ------------------------------------------------------------------
    # Figure 1: Bus voltages
    # ------------------------------------------------------------------
    all_voltages = result.get("all_voltages", [])
    if not all_voltages:
        fig_v = _empty_figure(f"No data — run the analysis first{title_suffix}")
    else:
        bus_names = _safe_labels([v.get("bus_name", "") for v in all_voltages], "Bus")
        vm_values = [v.get("vm_pu", 1.0) for v in all_voltages]
        colors = [
            "red" if (v < vm_lower or v > vm_upper) else "steelblue"
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
                line_color="red",
                annotation_text=label,
                annotation_position="right",
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
        line_names_axis = [_compact_line_label(n) for n in line_names_raw]
        line_names = _unique_labels(line_names_axis)
        line_vals = [l.get("loading_percent", 0.0) for l in all_line_loading]
        line_colors = ["red" if v > max_loading else "steelblue" for v in line_vals]

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
            line_color="red",
            annotation_text=f"{max_loading:.0f}% limit",
            annotation_position="right",
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
        trafo_names = _unique_labels(trafo_names_raw)
        trafo_vals = [t.get("loading_percent", 0.0) for t in all_trafo_loading]
        trafo_colors = [
            "red" if v > max_loading else "steelblue" for v in trafo_vals
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
            line_color="red",
            annotation_text=f"{max_loading:.0f}% limit",
            annotation_position="right",
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
    "bus_vm_pu": "orange",
    "line_loading": "red",
    "trafo_loading": "darkred",
}


def render_contingency_violations(result: dict) -> go.Figure:
    """
    Single figure for simulate_contingency or simulate_all_contingencies results.
    Returns a green placeholder if no violations.
    """
    is_full_sweep = "total_outages_tested" in result
    title_prefix = "N-1 Sweep — Violations by Element" if is_full_sweep else "Contingency Violations by Element"

    # Secure system
    if result.get("system_secure") or result.get("system_n1_secure"):
        fig = _empty_figure("✅ No violations detected", color="green")
        fig.update_layout(title={"text": "✅ No violations detected", "font": {"color": "green"}})
        return fig

    violations = result.get("violations", [])
    if not violations:
        return _empty_figure("No violation data available")

    total_count = result.get("total_violations", len(violations))

    # Count occurrences per element + violation_type
    element_counts: dict[tuple[str, str], int] = {}
    for v in violations:
        raw_name = v.get("element_name") or ""
        name = str(raw_name).strip() if str(raw_name).strip() else f"Element_{v.get('element_index', '?')}"
        key = (name, v.get("violation_type", "unknown"))
        element_counts[key] = element_counts.get(key, 0) + 1

    elements = [k[0] for k in element_counts]
    vtypes = [k[1] for k in element_counts]
    counts = list(element_counts.values())
    colors = [_VIOLATION_COLORS.get(vt, "grey") for vt in vtypes]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=elements,
            orientation="h",
            marker_color=colors,
            text=vtypes,
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": "white", "size": 12},
        )
    )
    subtitle = f"Total: {total_count} violation(s)"
    if is_full_sweep:
        subtitle += (
            f" | {result.get('total_outages_causing_violations', '?')} outages "
            f"out of {result.get('total_outages_tested', '?')} tested"
        )
    fig.update_layout(
        title=f"{title_prefix}<br><sup>{subtitle}</sup>",
        xaxis={"title": "Violation count"},
        yaxis={"title": "Element", "automargin": True},
        **CHART_THEME,
    )
    return fig


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
    vm_lower = result.get("opf_vm_lower_used", _VM_LOWER)
    vm_upper = result.get("opf_vm_upper_used", _VM_UPPER)

    # Dots are red when outside the OPF bounds (should not happen, but surface if so)
    colors = [
        "red" if (v < vm_lower - 1e-4 or v > vm_upper + 1e-4) else "#1f77b4"
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
        y=vm_upper, line_dash="dash", line_color="red",
        annotation_text=f"Upper {vm_upper} p.u.", annotation_position="top right",
    )
    fig.add_hline(
        y=vm_lower, line_dash="dash", line_color="red",
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
    if result.get("status") == "infeasible":
        fig = _empty_figure(
            "❌ Optimizer infeasible — violations cannot be resolved by local flexibility",
            color="red",
        )
        fig.update_layout(
            title={
                "text": "❌ Optimizer infeasible",
                "font": {"color": "red"},
            }
        )
        return [fig]

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
        ("green" if (n >= b) else "orange") if (n is not None and b is not None) else "grey"
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
                marker_color="lightgrey",
                opacity=0.9,
            )
        )

    fig.add_trace(
        go.Bar(
            name="P_base (MW)",
            x=all_elements,
            y=pg_base,
            marker_color="grey",
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
            "teal" if (v is not None and v >= 0) else "orange"
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
        dp_colors = ["#2ca02c" if v >= 0 else "#ff7f0e" for v in delta_p]
        fig_dp = go.Figure(go.Bar(
            x=delta_p_elems, y=delta_p,
            marker_color=dp_colors,
            text=[f"{v:+.4f}" if abs(v) > 1e-5 else "" for v in delta_p],
            textposition="outside",
        ))
        fig_dp.add_hline(y=0, line_color="black", line_width=0.8)
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
        dq_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in delta_q]
        fig_dq = go.Figure(go.Bar(
            x=delta_q_elems, y=delta_q,
            marker_color=dq_colors,
            text=[f"{v:+.4f}" if abs(v) > 1e-5 else "" for v in delta_q],
            textposition="outside",
        ))
        fig_dq.add_hline(y=0, line_color="black", line_width=0.8)
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
                dv_colors.append("#d62728" if dv > 0 else "#1f77b4")
        if dv_names and any(abs(v) > 1e-5 for v in dv_vals):
            fig_dv = go.Figure(go.Bar(
                x=dv_names, y=dv_vals,
                marker_color=dv_colors,
                text=[f"{v:+.5f}" if abs(v) > 1e-5 else "" for v in dv_vals],
                textposition="outside",
            ))
            fig_dv.add_hline(y=0, line_color="black", line_width=0.8)
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

    kpi1 = metrics.get("kpi_1_target_demand_flex_pct", 0.0)
    kpi2 = metrics.get("kpi_2_flex_utilization_pct", 0.0)
    kpi3 = metrics.get("kpi_3_prevented_violation_ratio_pct", 0.0)

    has_constrained = bool(constrained) and constrained.get("status") == "optimal"
    n_rows = 2 if has_constrained else 1

    specs = [[{"type": "indicator"}] * 3 for _ in range(n_rows)]
    subplot_titles = ["KPI-3: Violations Prevented", "KPI-2: Flex Utilization", "KPI-1: Flex Headroom"]
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
            title={"text": "KPI-3: Violations Prevented (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "green" if kpi3 == 100 else ("orange" if kpi3 > 50 else "red")
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
            title={"text": "KPI-2: Flex Utilization (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "green" if kpi2 > 90 else ("orange" if kpi2 > 50 else "red")
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
            title={"text": "KPI-1: Target Demand Flex (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "steelblue"},
            },
            number={"suffix": "%"},
        ),
        row=1, col=3,
    )

    # Row 2: constrained scenario (optional)
    if has_constrained:
        ck1 = constrained.get("kpi_1_target_demand_flex_pct", 0.0)
        ck2 = constrained.get("kpi_2_flex_utilization_pct", 0.0)
        ck3 = constrained.get("kpi_3_prevented_violation_ratio_pct", 0.0)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=ck3,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "green" if ck3 == 100 else ("orange" if ck3 > 50 else "red")
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
                        "color": "green" if ck2 > 90 else ("orange" if ck2 > 50 else "red")
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
                    "bar": {"color": "steelblue"},
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
            line={"color": "steelblue"},
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
    vm_lower = thresholds.get("vm_lower_pu", _VM_LOWER)
    vm_upper = thresholds.get("vm_upper_pu", _VM_UPPER)
    max_loading = thresholds.get("max_line_loading_pct", _MAX_LOADING)
    max_trafo_loading_pct = thresholds.get("max_trafo_loading_pct", _MAX_LOADING)

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
                line={"color": "steelblue", "dash": "dot", "width": 2},
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
                line={"color": "tomato", "dash": "dot", "width": 2},
                marker={"size": 5},
                name="Max Voltage (p.u.)",
            )
        )

    # Reference lines via shapes (reliable with overlaying dual-axis)
    max_violations = max(violations) if violations else 1
    voltage_fig.update_layout(
        title="Grid Security Evolution Over Time",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={
            "title": "Voltage (p.u.)",
            "range": [0.88, 1.12],
        },
        yaxis2={
            "title": "Violation Count",
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
                "line": {"dash": "dash", "color": "red", "width": 1},
            },
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": vm_lower, "y1": vm_lower,
                "line": {"dash": "dash", "color": "red", "width": 1},
            },
        ],
        annotations=[
            {
                "x": 1.01, "y": vm_upper, "xref": "paper", "yref": "y",
                "text": f"V_max {vm_upper:.2f}", "showarrow": False,
                "xanchor": "left", "font": {"color": "red"},
            },
            {
                "x": 1.01, "y": vm_lower, "xref": "paper", "yref": "y",
                "text": f"V_min {vm_lower:.2f}", "showarrow": False,
                "xanchor": "left", "font": {"color": "red"},
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
                line={"color": "darkorange", "width": 2},
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
                line={"color": "seagreen", "dash": "dot", "width": 2},
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
                "line": {"dash": "dash", "color": "darkorange", "width": 1},
            },
            {
                "type": "line", "xref": "paper", "x0": 0, "x1": 1,
                "yref": "y", "y0": max_trafo_loading_pct, "y1": max_trafo_loading_pct,
                "line": {"dash": "dash", "color": "seagreen", "width": 1},
            },
        ],
        annotations=[
            {
                "x": 1.01, "y": max_loading, "xref": "paper", "yref": "y",
                "text": f"Line limit {max_loading:.0f}%", "showarrow": False,
                "xanchor": "left", "font": {"color": "darkorange"},
            },
            {
                "x": 1.01, "y": max_trafo_loading_pct, "xref": "paper", "yref": "y",
                "text": f"Trafo limit {max_trafo_loading_pct:.0f}%", "showarrow": False,
                "xanchor": "left", "font": {"color": "seagreen"},
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
        pg_mw    = [g["Pg_mw"]      for g in generators]
        pg_max   = [g.get("Pg_max_mw") for g in generators]
        # Colour bars: green when producing, grey when idle (≈0)
        colors = ["#2ca02c" if p > 0.01 else "#aaaaaa" for p in pg_mw]

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
                marker={"symbol": "line-ew", "size": 12, "color": "red",
                        "line": {"width": 2, "color": "red"}},
            ))

        # Add ext grid as a separate bar at the end
        ext = result.get("ext_grid", {})
        if ext:
            ext_label = ext.get("name") or "Slack"
            fig_gen.add_trace(go.Bar(
                name=f"{ext_label} (MW)",
                x=[ext_label], y=[ext.get("P_import_mw", 0)],
                marker_color="#1f77b4",
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
            marker_color="steelblue",
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
            "tomato" if d < -0.01 else ("seagreen" if d > 0.01 else "#888888")
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
        fig_pg.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_width=1)
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
            "tomato" if d < -0.001 else ("seagreen" if d > 0.001 else "#888888")
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
        fig_qg.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_width=1)
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
            "tomato" if abs(d) > 0.005 else "steelblue" for d in deltas_vm
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
        fig_vm.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_width=1)
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
    vm_lower = thresholds.get("vm_lower_pu", _VM_LOWER)
    vm_upper = thresholds.get("vm_upper_pu", _VM_UPPER)

    max_violations = max(violations) if violations else 1

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
            line={"color": "steelblue", "dash": "dot", "width": 2},
            marker={"size": 4},
            name="Min Voltage (p.u.)",
        ))
    if max_v:
        fig.add_trace(go.Scatter(
            x=timestamps, y=max_v,
            mode="lines+markers",
            line={"color": "tomato", "dash": "dot", "width": 2},
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
                marker={"symbol": "star", "size": 16, "color": "gold",
                        "line": {"color": "darkorange", "width": 1.5}},
                text=[highlight_name],
                textposition="top center",
                textfont={"color": "darkorange", "size": 11},
                name="Worst point",
                yaxis="y2" if metric == "violations" else "y",
            ))

    # Build shapes and annotations lists (add_vline fails on string x-axis values)
    shapes = [
        {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
         "yref": "y", "y0": vm_upper, "y1": vm_upper,
         "line": {"dash": "dash", "color": "red", "width": 1}},
        {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
         "yref": "y", "y0": vm_lower, "y1": vm_lower,
         "line": {"dash": "dash", "color": "red", "width": 1}},
    ]
    annotations = [
        {"x": 1.01, "y": vm_upper, "xref": "paper", "yref": "y",
         "text": f"V_max {vm_upper:.2f}", "showarrow": False,
         "xanchor": "left", "font": {"color": "red"}},
        {"x": 1.01, "y": vm_lower, "xref": "paper", "yref": "y",
         "text": f"V_min {vm_lower:.2f}", "showarrow": False,
         "xanchor": "left", "font": {"color": "red"}},
    ]
    if worst_ts:
        shapes.append(
            {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
             "yref": "paper", "y0": 0, "y1": 1,
             "line": {"dash": "dash", "color": "darkorange", "width": 1.5}}
        )
        annotations.append(
            {"x": worst_ts, "y": 1, "xref": "x", "yref": "paper",
             "text": f"Worst ({metric})", "showarrow": False,
             "xanchor": "left", "font": {"color": "darkorange", "size": 10}}
        )

    fig.update_layout(
        title=f"Worst-case Scan — {n_scanned} timestamps ({metric})",
        xaxis={"title": "Timestamp", "tickangle": 45},
        yaxis={"title": "Voltage (p.u.)", "range": [0.88, 1.12]},
        yaxis2={
            "title": "Violation Count",
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
            line={"color": "steelblue", "width": 2},
            marker={"size": 4},
            name="External Grid import (MW)",
        ))
        if worst_ts and worst_ts in timestamps and metric == "slack_import":
            wi = timestamps.index(worst_ts)
            fig2.add_trace(go.Scatter(
                x=[worst_ts], y=[slack[wi]],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": "gold",
                        "line": {"color": "darkorange", "width": 1.5}},
                text=[f"Worst: {slack[wi]:.2f} MW"],
                textposition="top center",
                textfont={"color": "darkorange", "size": 11},
                name="Worst point",
            ))
        fig2_shapes = []
        if worst_ts and worst_ts in timestamps and metric == "slack_import":
            fig2_shapes.append(
                {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
                 "yref": "paper", "y0": 0, "y1": 1,
                 "line": {"dash": "dash", "color": "darkorange", "width": 1.5}}
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
            line={"color": "seagreen", "width": 2},
            marker={"size": 4},
            name="Max line loading (%)",
        ))
        if worst_ts and worst_ts in timestamps and metric == "max_loading":
            wi = timestamps.index(worst_ts)
            fig3.add_trace(go.Scatter(
                x=[worst_ts], y=[max_line[wi]],
                mode="markers+text",
                marker={"symbol": "star", "size": 16, "color": "gold",
                        "line": {"color": "darkorange", "width": 1.5}},
                text=[f"Worst: {max_line[wi]:.0f}%"],
                textposition="top center",
                textfont={"color": "darkorange", "size": 11},
                name="Worst point",
            ))
        fig3_shapes = [
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1,
             "yref": "y", "y0": max_load_pct, "y1": max_load_pct,
             "line": {"dash": "dash", "color": "red", "width": 1}},
        ]
        fig3_annotations = [
            {"x": 1.01, "y": max_load_pct, "xref": "paper", "yref": "y",
             "text": f"Limit {max_load_pct:.0f}%", "showarrow": False,
             "xanchor": "left", "font": {"color": "red"}},
        ]
        if worst_ts and worst_ts in timestamps and metric == "max_loading":
            fig3_shapes.append(
                {"type": "line", "xref": "x", "x0": worst_ts, "x1": worst_ts,
                 "yref": "paper", "y0": 0, "y1": 1,
                 "line": {"dash": "dash", "color": "darkorange", "width": 1.5}}
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
_SCENARIO_COLOURS = ["steelblue", "tomato", "seagreen", "darkorchid", "darkorange", "teal"]


def render_scenarios(result: dict) -> list[go.Figure]:
    """Multi-scenario RSA comparison charts.

    Returns up to 3 figures:
      [0] Violation count over time — one line per scenario
    [1] External-grid import (MW) over time — one line per scenario
      [2] Summary bar chart: total violations per scenario
    """
    if "error" in result:
        return [_empty_figure(f"Error: {result['error']}", color="red")]

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
        return [_empty_figure(f"Error: {result['error']}", color="red")]

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
        vm_upper = thresholds.get("vm_upper_pu", _VM_UPPER)
        vm_lower = thresholds.get("vm_lower_pu", _VM_LOWER)
        violation_mask = [v > vm_upper or v < vm_lower for v in vm]

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
            line={"color": "steelblue", "width": 2},
            marker={
                "color": ["red" if v else "steelblue" for v in violation_mask],
                "size": [9 if v else 4 for v in violation_mask],
            },
            name="vm_pu",
        ))
        fig.add_hline(y=vm_upper, line_dash="dash", line_color="red",
                      annotation_text=f"V_max {vm_upper:.2f}",
                      annotation_position="top right")
        fig.add_hline(y=vm_lower, line_dash="dash", line_color="red",
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
                line={"color": "darkorange", "width": 1.5},
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
            thresholds.get("max_line_loading_pct", _MAX_LOADING)
            if etype == "line"
            else thresholds.get("max_trafo_loading_pct", _MAX_LOADING)
        )
        violation_mask = [v > threshold_pct for v in loading]
        n_viol = sum(violation_mask)
        elem_label = "Line" if etype == "line" else "Transformer"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps, y=loading,
            mode="lines+markers",
            line={"color": "steelblue", "width": 2},
            marker={
                "color": ["red" if v else "steelblue" for v in violation_mask],
                "size": [9 if v else 4 for v in violation_mask],
            },
            fill="tozeroy",
            fillcolor="rgba(70,130,180,0.12)",
            name="Loading (%)",
        ))
        fig.add_hline(
            y=threshold_pct, line_dash="dash", line_color="red",
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
                line={"color": "darkorange", "width": 1.5},
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
        return [_empty_figure(f"Error: {result['error']}", color="red")]

    ts = result.get("timestamp", "")
    n_samples = result.get("n_samples", "?")
    n_converged = result.get("n_converged", "?")
    p_any = result.get("p_any_violation", 0.0)
    exp_viol = result.get("expected_violations", 0.0)
    sgen_sigma = result.get("samples_summary", {}).get("sgen_sigma", 0.15)
    thresholds = result.get("thresholds", {})
    vm_upper = thresholds.get("vm_upper_pu", _VM_UPPER)
    vm_lower = thresholds.get("vm_lower_pu", _VM_LOWER)

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
        colour_map = {"Bus": "steelblue", "Line": "darkorange", "Trafo": "seagreen"}
        colours = [colour_map.get(t, "grey") for t in etypes]
        fig_bar = go.Figure(go.Bar(
            x=names, y=probs,
            marker_color=colours,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
        ))
        fig_bar.add_hline(y=0.05, line_dash="dot", line_color="orange",
                          annotation_text="5%", annotation_position="top right")
        fig_bar.add_hline(y=0.20, line_dash="dot", line_color="red",
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
            color="green",
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
            marker={"color": "steelblue", "size": 8},
            name="P50 (median)",
        ))
        fig_box.add_trace(go.Scatter(
            x=bnames, y=p95_vals, mode="markers",
            marker={"color": "tomato", "size": 6, "symbol": "triangle-up"},
            name="P95",
        ))
        fig_box.add_hline(y=vm_upper, line_dash="dash", line_color="red",
                          annotation_text=f"{vm_upper} p.u. limit")
        fig_box.add_hline(y=vm_lower, line_dash="dash", line_color="red",
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
        colours = ["tomato" if c > 0 else "steelblue" for c in counts]
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
            marker_color="#d62728",
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
            marker_color="#1f77b4",
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
                point_colors.append("#d62728")   # red: constraint violated
            elif v > ub or v < lb:
                point_colors.append("#ff7f0e")   # amber: in the back-off margin
            else:
                point_colors.append("#2ca02c")   # green: within tightened bound

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
            line=dict(color="#ff7f0e", dash="dot", width=1.5),
            name="Tightened upper bound",
            hovertemplate="<b>%{x}</b><br>Tightened UB: %{y:.4f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=bus_names, y=tight_lbs,
            mode="lines",
            line=dict(color="#1f77b4", dash="dot", width=1.5),
            name="Tightened lower bound",
            hovertemplate="<b>%{x}</b><br>Tightened LB: %{y:.4f}<extra></extra>",
        ))
        # Global limits
        fig2.add_hline(y=vm_upper_global, line_dash="dash", line_color="red",
                       annotation_text=f"Original UB {vm_upper_global:.3f}", annotation_position="top right")
        fig2.add_hline(y=vm_lower_global, line_dash="dash", line_color="blue",
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
        bar_cols = ["#d62728"]
        text = [f"{values[0]:.1f}%"]
        if p_after_validation is not None:
            labels.append("After Robust OPF (Certified)")
            values.append(float(p_after_validation) * 100)
            bar_cols.append("#1f77b4")
            text.append(f"{values[-1]:.1f}%")
        elif p_after is not None:
            labels.append("After Robust OPF")
            values.append(float(p_after) * 100)
            bar_cols.append("#2ca02c")
            text.append(f"{values[-1]:.1f}%")
        else:
            labels.append("After Robust OPF")
            values.append(0.0)
            bar_cols.append("#9e9e9e")
            text.append("N/A")

        fig3 = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color=bar_cols,
            text=text,
            textposition="outside",
        ))
        fig3.add_hline(y=100 * (1 - confidence), line_dash="dash", line_color="orange",
                   annotation_text=f"Reference: {100*(1-confidence):.0f}% (not guaranteed system-wide)",
                       annotation_position="right")
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
        fig_split.add_trace(go.Bar(x=split_labels, y=split_before, name="Before", marker_color="#d62728"))
        fig_split.add_trace(go.Bar(x=split_labels, y=split_after, name="After", marker_color="#2ca02c"))
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
            marker=dict(color="#d62728"),
            line=dict(color="#d62728"),
            yaxis="y1",
        ))
        fig_loop.add_trace(go.Scatter(
            x=loop_iters,
            y=loop_curt,
            mode="lines+markers",
            name="Expected curtailment [MW]",
            marker=dict(color="#17becf"),
            line=dict(color="#17becf", dash="dot"),
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
            marker_color=["#9467bd", "#17becf", "#7f7f7f"],
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
        fig_bus.add_trace(go.Bar(x=buses, y=before_vals, name="Before", marker_color="#d62728"))
        fig_bus.add_trace(go.Bar(x=buses, y=after_vals, name="After", marker_color="#2ca02c"))
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
        fig_line.add_trace(go.Bar(x=lines, y=before_vals, name="Before", marker_color="#d62728"))
        fig_line.add_trace(go.Bar(x=lines, y=after_vals, name="After", marker_color="#2ca02c"))
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
        fig_trafo.add_trace(go.Bar(x=trafos, y=before_vals, name="Before", marker_color="#d62728"))
        fig_trafo.add_trace(go.Bar(x=trafos, y=after_vals, name="After", marker_color="#2ca02c"))
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
                            line=dict(color="black", width=1.5)),
                name="Base operating point",
            ))
    colorscale_feas = [
        [0.0,  "#aec7e8"],   # -1 → grey (no convergence)
        [0.49, "#aec7e8"],
        [0.50, "#d62728"],   #  0 → red (infeasible)
        [0.74, "#d62728"],
        [0.75, "#2ca02c"],   #  1 → green (feasible)
        [1.0,  "#2ca02c"],
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
    _add_overlays(fig2, star_color="black")
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
    _add_overlays(fig3, star_color="black")
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
        return [_empty_figure(str(result["error"]), color="red")]

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
        line=dict(color="#1f77b4", width=2),
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
        line_color="red",
        annotation_text="Limit boundary",
        annotation_position="right",
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
        marker_color=["#d62728", "#ff7f0e"],
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
            marker=dict(color="#9467bd", size=9),
            line=dict(color="#9467bd", dash="dot"),
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
            marker_color="#d62728",
            name="Exceedance [%]",
            text=[f"{v:.1f}%" for v in exc_vals],
            textposition="outside",
            offsetgroup="exceedance",
        ))
        fig_cond.add_trace(go.Bar(
            x=bin_labels,
            y=near_vals,
            marker_color="#ff7f0e",
            name="Near miss [%]",
            text=[f"{v:.1f}%" for v in near_vals],
            textposition="outside",
            offsetgroup="near_miss",
        ))
        fig_cond.add_trace(go.Scatter(
            x=bin_labels,
            y=n_steps,
            mode="lines+markers",
            line=dict(color="#1f77b4", dash="dot"),
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
        return [_empty_figure(str(result["error"]), color="red")]

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
            "unity": "#0b7285",
            "fixed_pf": "#2b8a3e",
            "reactive_proxy": "#c92a2a",
        }
        for m in available_modes:
            mode_map = rows_by_mode[m]
            x_vals = [float(mode_map[b]["hosting_capacity_mw"]) if b in mode_map else 0.0 for b in top_buses]
            bindings = [str(mode_map[b]["binding"]) if b in mode_map else "none" for b in top_buses]
            fig_rank.add_trace(go.Bar(
                x=x_vals,
                y=top_buses,
                orientation="h",
                marker_color=mode_colors.get(m, "#2b8a3e"),
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
                marker_color=["#0b7285", "#2b8a3e", "#c92a2a"][: len(per_mode_avg)],
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
        "unity": "#0b7285",
        "fixed_pf": "#2b8a3e",
        "reactive_proxy": "#c92a2a",
        "voltage_control": "#c92a2a",
    }

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=modes,
        y=capacities,
        marker_color=[mode_color_map.get(str(m).strip().lower(), "#2b8a3e") for m in modes],
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
        "unity": "#0b7285",
        "fixed_pf": "#2b8a3e",
        "reactive_proxy": "#c92a2a",
    }
    risk_mode_colors = [mode_color_map.get(str(m).strip().lower(), "#495057") for m in modes]

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(
        x=modes,
        y=risk_vals,
        marker={
            "color": risk_mode_colors,
            "pattern": {"shape": "/", "fgcolor": "#1f2937", "size": 7, "solidity": 0.22},
            "line": {"color": "#1f2937", "width": 1.0},
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
            "pattern": {"shape": "x", "fgcolor": "#111827", "size": 7, "solidity": 0.38},
            "line": {"color": "#111827", "width": 1.0},
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
            line_color="#c92a2a",
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
    "find_worst_case_timestamp": render_worst_case,
    "scan_scenarios": render_scenarios,
    "get_element_timeseries": render_element_timeseries,
    "run_probabilistic_rsa": render_probabilistic_rsa,
    "optimize_robust_flexibility": render_robust_flexibility,
    "compute_flexibility_envelope": render_flexibility_envelope,
    "compute_hosting_capacity": render_hosting_capacity,
    "compute_historical_risk": render_historical_risk,
}
