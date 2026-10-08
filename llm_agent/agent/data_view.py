"""
data_view.py — the Data tab: the loaded measurements and forecasts as they are.

Raw input data, not analysis: consumption and production per substation, as
loaded, before any power flow. Pure functions over `/api/data/series`
payloads (one per dataset), so they can be tested without the page.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .renderers import CHART_THEME, _C

TOTAL = "System total"
QUANTITIES = {"consumption": "Consumption", "production": "Production", "net": "Net load"}
_LINE_COLOURS = [_C.PRIMARY, _C.TEAL, _C.WARN, _C.INK, _C.OK, _C.VIOLATION]
_DATASET_LABELS = {"measured": "measured", "forecast": "forecast"}
_DATASET_COLOURS = {"measured": _C.PRIMARY, "forecast": _C.PURPLE}


def values(payload: dict | None, item: str, quantity: str) -> list[float | None]:
    """One series from a payload: `item` is TOTAL or a substation name;
    `quantity` is consumption, production or net (consumption − production)."""
    if not payload:
        return []
    source = payload.get("totals") if item == TOTAL else (payload.get("substations") or {}).get(item)
    if not source:
        return []
    cons, prod = source.get("consumption") or [], source.get("production") or []
    if quantity == "consumption":
        return list(cons)
    if quantity == "production":
        return list(prod)
    return [None if c is None or p is None else round(c - p, 4) for c, p in zip(cons, prod)]


def forecast_error(measured: dict | None, forecast: dict | None, item: str, quantity: str) -> dict | None:
    """Forecast against measured on the timestamps both carry.

    Mean error (bias), mean absolute error, and the mean absolute percentage
    error where the measured value is not near zero. None without overlap.
    """
    if not measured or not forecast:
        return None
    m = dict(zip(measured.get("timestamps") or [], values(measured, item, quantity)))
    f = dict(zip(forecast.get("timestamps") or [], values(forecast, item, quantity)))
    shared = sorted(t for t in set(m) & set(f) if m[t] is not None and f[t] is not None)
    if not shared:
        return None
    errors = [f[t] - m[t] for t in shared]
    pct = [abs(f[t] - m[t]) / abs(m[t]) for t in shared if abs(m[t]) > 1e-6]
    return {
        "n": len(shared),
        "first": shared[0],
        "last": shared[-1],
        "bias": round(sum(errors) / len(errors), 3),
        "mae": round(sum(abs(e) for e in errors) / len(errors), 3),
        "mape_pct": round(100 * sum(pct) / len(pct), 2) if pct else None,
    }


def figure(series: dict[str, dict | None], items: list[str], quantity: str,
           marker: str | None = None) -> go.Figure:
    """Lines per item, measured solid and forecast dashed.

    One item: blue measured, purple forecast — the operating-point card's
    colours. Several: one colour per item, the dataset told by the dash.

    `series` maps "measured" / "forecast" to a payload (or None). Where the two
    share timestamps the window is shaded; `marker` draws the operating point.
    """
    fig = go.Figure()
    for k, item in enumerate(items):
        for dataset in ("measured", "forecast"):
            colour = (_DATASET_COLOURS[dataset] if len(items) == 1
                      else _LINE_COLOURS[k % len(_LINE_COLOURS)])
            payload = series.get(dataset)
            ys = values(payload, item, quantity)
            if not ys:
                continue
            fig.add_trace(go.Scatter(
                x=payload["timestamps"], y=ys, mode="lines",
                name=f"{item} — {_DATASET_LABELS[dataset]}",
                line={"color": colour, "width": 1.6 if dataset == "measured" else 1.4,
                      "dash": "solid" if dataset == "measured" else "dash"},
                hovertemplate="%{x}<br>%{y:.2f} MW<extra>%{fullData.name}</extra>",
            ))

    measured, forecast = series.get("measured"), series.get("forecast")
    if measured and forecast:
        shared = sorted(set(measured.get("timestamps") or []) & set(forecast.get("timestamps") or []))
        if shared:
            fig.add_vrect(x0=shared[0], x1=shared[-1], fillcolor=_C.PURPLE, opacity=0.07,
                          line_width=0, layer="below")
            fig.add_annotation(x=shared[-1], y=0.99, yref="paper", text="overlap",
                               showarrow=False, xanchor="right", yanchor="top",
                               font={"size": 11, "color": _C.PURPLE})
    if marker:
        # A shape, not add_vline: Plotly's vline annotation fails on date axes.
        fig.add_shape(type="line", x0=marker, x1=marker, y0=0, y1=1, yref="paper",
                      line={"color": _C.INK, "width": 1, "dash": "dot"})
        fig.add_annotation(x=marker, y=1, yref="paper", text="operating point", showarrow=False,
                           xanchor="right", yanchor="bottom", font={"size": 11, "color": _C.INK})

    fig.update_layout(
        **{**CHART_THEME, "margin": {"l": 60, "r": 20, "t": 40, "b": 50}},
        height=420,
        yaxis_title=f"{QUANTITIES.get(quantity, quantity)} (MW)",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode="x unified",
    )
    return fig
