"""
synthetic_view.py — the Data tab's synthetic-data panel: how the data are
generated, in words, and the preview of other settings.

Pure functions over `/api/data/synthetic` and `/api/data/synthetic/preview`
payloads, so they can be tested without the page. Plotly is imported only by
`preview_figure`.
"""

from __future__ import annotations

# What a unit drawn with each shape does, for the line under the shape picker.
_SHAPE_WORDS = {"solar": ("follows the sun", "follow the sun"), "wind": ("follows the wind", "follow the wind"),
               "flat": ("gives a steady output", "give a steady output")}


def title(label: str) -> str:
    """'Residential' from 'Residential: a morning peak …'."""
    return label.split(":", 1)[0]


def detail(label: str) -> str:
    """'A morning peak …' from 'Residential: a morning peak …'."""
    rest = label.split(":", 1)[1].strip() if ":" in label else ""
    return rest[:1].upper() + rest[1:]


def sentence(text: str) -> str:
    """A settings summary as a sentence: capitalised, with a full stop."""
    text = (text or "").strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else ""


def units_line(payload: dict, default_shape: str) -> str:
    """How the units are drawn: those the network (or the operator) describes
    by their source, the rest by the default shape chosen."""
    units = payload.get("units") or {}
    stated = {k: v for k, v in (units.get("stated") or {}).items() if v}
    rest = int(units.get("default") or 0)
    parts = [f"{n} {_SHAPE_WORDS.get(shape, (shape, shape))[n != 1]}"
             for shape, n in sorted(stated.items(), key=lambda kv: -kv[1])]
    out = []
    if parts:
        out.append("As their source says: " + ", ".join(parts) + ".")
    if rest:
        out.append(f"{rest} unit{'s' if rest != 1 else ''} the network does not describe "
                   f"{'follow' if rest != 1 else 'follows'} this shape.")
    elif parts:
        out.append("Every unit's source is known, so this shape applies to none.")
    return " ".join(out) or "The network has no units."


def loads_line(payload: dict) -> str:
    """How the loads are drawn: those whose category the network (or the operator) gives by their own shape,
    the rest by the shape chosen."""
    loads = payload.get("loads") or {}
    stated = {k: v for k, v in (loads.get("stated") or {}).items() if v}
    rest = int(loads.get("default") or 0)
    out = []
    if stated:
        out.append("As their category says: " + ", ".join(
            f"{n} {'follows' if n == 1 else 'follow'} a {shape.replace('_', ' ')} shape"
            for shape, n in sorted(stated.items(), key=lambda kv: -kv[1])) + ".")
    if rest:
        out.append(f"{rest} load{'s' if rest != 1 else ''} whose category is not known "
                   f"{'follow' if rest != 1 else 'follows'} this shape.")
    elif stated:
        out.append("Every load's category is known, so this shape applies to none.")
    return " ".join(out)


def stale_line(stale: list[dict]) -> str:
    """Units whose declared source changed after the data were generated."""
    if not stale:
        return ""
    shown = [f"**{s['unit']}** (now {s['now'].replace('_', ' ')}, drawn as {s['drawn_as'].replace('_', ' ')})"
             for s in stale[:4]]
    more = f" and {len(stale) - 4} more" if len(stale) > 4 else ""
    what = "loads" if all(s.get("kind") == "load" for s in stale) else (
        "units" if not any(s.get("kind") == "load" for s in stale) else "units and loads")
    return (f"The data do not follow what these {what} are now: " + ", ".join(shown) + more
            + ". Generate the data again to follow them.")


_LABELS = {"n_days": "days", "resolution_min": "step", "load_profile": "load shape",
           "generation_profile": "shape of undescribed units", "peak_load_factor": "daily peak",
           "stress_events": "stress windows", "seed": "seed", "load_error": "load forecast error",
           "generation_error": "generation forecast error"}


def _value(key: str, value) -> str:
    if key == "resolution_min":
        return f"{value} min"
    if key in ("peak_load_factor", "load_error", "generation_error"):
        return f"{value * 100:.0f} %"
    if key == "stress_events":
        return "on" if value else "off"
    return str(value).replace("_", " ")


def differences(settings: dict, base: dict) -> list[str]:
    """What differs from the settings the network loads with, in words."""
    return [f"{_LABELS[k]} {_value(k, base.get(k))} → {_value(k, settings.get(k))}"
            for k in _LABELS if settings.get(k) != base.get(k)]


def comparable(preview: dict) -> bool:
    """Whether the data in use share a period with the preview: other dates
    (uploaded data of another year) on the same time axis would squash both."""
    ours, theirs = preview.get("timestamps") or [], (preview.get("current") or {}).get("timestamps") or []
    return bool(ours and theirs) and theirs[0] <= ours[-1] and ours[0] <= theirs[-1]


def other_period(preview: dict) -> str:
    """Why the data in use are not drawn beside the preview."""
    theirs = (preview.get("current") or {}).get("timestamps") or []
    if not theirs:
        return ""
    return f"The data in use ({theirs[0][:10]} → {theirs[-1][:10]}) cover another period, so they are not drawn."


def preview_figure(preview: dict):
    """The system totals with the new settings (solid) over the data in use
    (dotted, same colour) when they share a period: consumption blue,
    production teal, one MW axis."""
    import plotly.graph_objects as go

    from .renderers import CHART_THEME, _C

    fig = go.Figure()
    current = (preview.get("current") or {}) if comparable(preview) else {}
    for key, name, colour in (("consumption", "Consumption", _C.PRIMARY), ("production", "Production", _C.TEAL)):
        if current.get(key):
            fig.add_trace(go.Scatter(
                x=current["timestamps"], y=current[key], mode="lines", name=f"{name} — in use",
                line={"color": colour, "width": 1.2, "dash": "dot"}, opacity=0.55,
                hovertemplate="%{x}<br>%{y:.2f} MW<extra>" + f"{name}, in use</extra>"))
        fig.add_trace(go.Scatter(
            x=preview.get("timestamps") or [], y=preview.get(key) or [], mode="lines", name=f"{name} — new",
            line={"color": colour, "width": 2},
            hovertemplate="%{x}<br>%{y:.2f} MW<extra>" + f"{name}, new</extra>"))
    fig.update_layout(
        **{**CHART_THEME, "margin": {"l": 50, "r": 10, "t": 10, "b": 10}},
        height=360, yaxis_title="System total (MW)", hovermode="x unified",
        legend={"orientation": "h", "y": -0.12, "x": 0, "font": {"size": 11}},
    )
    fig.update_yaxes(rangemode="tozero")
    # Day and month: every tick is in the same year, and the year on its own line ran into the legend.
    fig.update_xaxes(tickformat="%-d %b")
    return fig
