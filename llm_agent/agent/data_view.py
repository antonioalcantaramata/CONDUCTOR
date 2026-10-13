"""
data_view.py — the Data tab: the loaded measurements and forecasts as they are.

Raw input data, not analysis: consumption and production per substation, as
loaded, before any power flow. Pure functions over `/api/data/series`
payloads (one per dataset), so they can be tested without the page.

Plotly (through `renderers`) is imported only by `figure`: the series and
forecast-error functions are plain Python, and CI installs no UI packages.
"""

from __future__ import annotations

TOTAL = "System total"
QUANTITIES = {"consumption": "Consumption", "production": "Production", "net": "Net load"}
# Shown only when a dataset carries them (optional upload columns), in Mvar.
REACTIVE = {"consumption_q": "Reactive demand", "production_q": "Reactive generation"}


def quantities(*payloads) -> dict:
    """The quantities to offer: active power always; reactive when any of the
    payloads carries it."""
    out = dict(QUANTITIES)
    for p in payloads:
        for key, label in REACTIVE.items():
            if p and (p.get("totals") or {}).get(key):
                out[key] = label
    return out


def unit(quantity: str) -> str:
    return "Mvar" if quantity in REACTIVE else "MW"
_DATASET_LABELS = {"measured": "measured", "forecast": "forecast"}


def values(payload: dict | None, item: str, quantity: str) -> list[float | None]:
    """One series from a payload: `item` is TOTAL or a substation name;
    `quantity` is consumption, production or net (consumption − production)."""
    if not payload:
        return []
    source = payload.get("totals") if item == TOTAL else (payload.get("substations") or {}).get(item)
    if not source:
        return []
    if quantity in REACTIVE:
        return list(source.get(quantity) or [])
    cons, prod = source.get("consumption") or [], source.get("production") or []
    if quantity == "consumption":
        return list(cons)
    if quantity == "production":
        return list(prod)
    return [None if c is None or p is None else round(c - p, 4) for c, p in zip(cons, prod)]


def nominal_value(payload: dict | None, item: str, quantity: str) -> float | None:
    """The network's own value for `item` (TOTAL or a substation), from the
    payload's `nominal` block; None when there is none."""
    nominal = (payload or {}).get("nominal")
    if not nominal or quantity in REACTIVE:
        return None
    source = nominal.get("totals") if item == TOTAL else (nominal.get("substations") or {}).get(item)
    if not source:
        return None
    if quantity == "net":
        return round(source["consumption"] - source["production"], 4)
    return source.get(quantity)


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
           marker: str | None = None):
    """Lines per item, measured solid and forecast dashed.

    One item: blue measured, purple forecast — the operating-point card's
    colours. Several: one colour per item, the dataset told by the dash.

    `series` maps "measured" / "forecast" to a payload (or None). Where the two
    share timestamps the window is shaded; `marker` draws the operating point.
    The network's own (nominal) value is a flat dotted line, when there is one.
    """
    import plotly.graph_objects as go

    from .renderers import CHART_THEME, _C

    line_colours = [_C.PRIMARY, _C.TEAL, _C.WARN, _C.INK, _C.OK, _C.VIOLATION]
    dataset_colours = {"measured": _C.PRIMARY, "forecast": _C.PURPLE}
    fig = go.Figure()
    for k, item in enumerate(items):
        for dataset in ("measured", "forecast"):
            colour = (dataset_colours[dataset] if len(items) == 1
                      else line_colours[k % len(line_colours)])
            payload = series.get(dataset)
            ys = values(payload, item, quantity)
            if not ys:
                continue
            fig.add_trace(go.Scatter(
                x=payload["timestamps"], y=ys, mode="lines",
                name=f"{item} — {_DATASET_LABELS[dataset]}",
                line={"color": colour, "width": 1.6 if dataset == "measured" else 1.4,
                      "dash": "solid" if dataset == "measured" else "dash"},
                hovertemplate="%{x}<br>%{y:.2f} " + unit(quantity) + "<extra>%{fullData.name}</extra>",
            ))

    payload = next((p for p in series.values() if p), None)
    for k, item in enumerate(items):
        level = nominal_value(payload, item, quantity)
        if level is None or not payload.get("timestamps"):
            continue
        xs = [payload["timestamps"][0], payload["timestamps"][-1]]
        for other in series.values():
            if other and other.get("timestamps"):
                xs = [min(xs[0], other["timestamps"][0]), max(xs[1], other["timestamps"][-1])]
        colour = _C.INK if len(items) == 1 else line_colours[k % len(line_colours)]
        fig.add_trace(go.Scatter(
            x=xs, y=[level, level], mode="lines", name=f"{item} — nominal",
            line={"color": colour, "width": 1.2, "dash": "dot"}, opacity=0.75,
            hovertemplate=f"nominal {level:.2f} MW<extra>{item}</extra>",
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
        yaxis_title=f"{QUANTITIES.get(quantity) or REACTIVE.get(quantity, quantity)} ({unit(quantity)})",
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode="x unified",
    )
    return fig
