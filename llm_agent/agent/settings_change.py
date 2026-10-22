"""
settings_change.py — what an answer changed in the session's settings.

Most answers change nothing: they study the grid. A few change the settings
every later study runs under — the load power factor, how a unit is modelled,
what it costs, what an element is called — because the operator asked, or,
rarely, because the model chose a setting as a workaround. Either way the
operator should see it under the answer and be able to put it back.

The app snapshots `/api/settings` before and after each answer; `lines()`
says, in the operator's words, what differs. The note is written here, not by
the model, so it appears whatever the answer says. Undo is the backend's
`/api/settings/revert`, which puts back only those items, and only if each is
still as the answer left it.

Pure data in, pure data out: no Streamlit, no HTTP.
"""

from __future__ import annotations

_CONTROL = {"voltage": "voltage control", "power_factor": "fixed power factor", "fixed_q": "fixed Q"}
_COSTS = {
    "cost_eur_per_mwh": ("production", "€/MWh"),
    "cost_eur_per_mw2h": ("quadratic", "€/MW²h"),
    "cost_eur_per_h": ("fixed", "€/h"),
    "up_eur_per_mwh": ("up", "€/MWh"),
    "down_eur_per_mwh": ("down", "€/MWh"),
    "q_eur_per_mvarh": ("reactive", "€/Mvarh"),
    "q_band_tan_phi": ("free band tan φ", ""),
    "q_penalty_eur_per_mvarh": ("beyond the band", "€/Mvarh"),
}
_KINDS = {"bus": "bus", "line": "line", "trafo": "transformer", "load": "load", "gen": "unit", "sgen": "unit"}


def _num(value) -> str:
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _model_field(field: str, value) -> str:
    if field == "control":
        return _CONTROL.get(value, str(value))
    if field == "vm_pu":
        return f"set-point {_num(value)} p.u."
    if field == "pf":
        return f"pf {_num(value)}"
    if field == "absorbing":
        return "absorbing" if value else "injecting"
    if field == "q_mvar":
        return f"Q {_num(value)} Mvar"
    if field == "q_min_mvar":
        return f"Q min {_num(value)} Mvar"
    if field == "q_max_mvar":
        return f"Q max {_num(value)} Mvar"
    if field == "dispatchable":
        return "dispatchable" if value else "not dispatchable"
    return f"{field} {_num(value)}"


def _cost_field(field: str, value) -> str:
    label, unit = _COSTS.get(field, (field, ""))
    if value is None:
        return f"{label} back to default"
    return f"{label} {_num(value)}" + (f" {unit}" if unit else "")


def _pf(value, default) -> str:
    return f"{_num(value)}" if value is not None else f"{_num(default)} (default)"


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def lines(before: dict | None, after: dict | None) -> list[str]:
    """One line per changed setting, in the operator's words; [] when
    nothing changed (or either snapshot is missing)."""
    if not before or not after:
        return []
    out: list[str] = []
    default = after.get("default_load_power_factor") or before.get("default_load_power_factor") or 0.95
    if before.get("load_power_factor") != after.get("load_power_factor"):
        out.append(f"load power factor {_pf(before.get('load_power_factor'), default)} → "
                   f"{_pf(after.get('load_power_factor'), default)}")

    units_now = {**(before.get("unit_names_now") or {}), **(after.get("unit_names_now") or {})}
    for key, label in (("generator_model", None), ("unit_costs", "costs")):
        b, a = before.get(key) or {}, after.get(key) or {}
        # Units given the same change share one line: "Gen_bus1 and Gen_bus2
        # costs: reactive 0 €/Mvarh", not the line once per unit.
        said: dict[str, list[str]] = {}
        for unit in sorted(set(b) | set(a)):
            if b.get(unit) == a.get(unit):
                continue
            new, old = a.get(unit) or {}, b.get(unit) or {}
            if key == "generator_model":
                parts = [_model_field(f, v) for f, v in new.items() if old.get(f) != v]
                what = ": " + ", ".join(parts or ["model changed"]) if new else " back to as loaded"
            else:
                parts = [_cost_field(f, new.get(f)) for f in list(old) + [f for f in new if f not in old]
                         if old.get(f) != new.get(f)]
                what = f" {label}: " + ", ".join(parts)
            said.setdefault(what, []).append(units_now.get(unit, unit))
        out += [_and(names) + what for what, names in said.items()]

    flat = lambda m: {f"{t}:{k}": n for t, names in (m or {}).items() for k, n in (names or {}).items()}  # noqa: E731
    b, a = flat(before.get("element_names")), flat(after.get("element_names"))
    loaded = {**(before.get("names_as_loaded") or {}), **(after.get("names_as_loaded") or {})}
    for item in sorted(set(b) | set(a)):
        if b.get(item) == a.get(item):
            continue
        table, index = item.split(":", 1)
        was = b.get(item) or loaded.get(item) or f"{_KINDS.get(table, table)} {index}"
        if a.get(item):
            out.append(f"{_KINDS.get(table, table)} {was} renamed {a[item]}")
        else:
            out.append(f"{was} back to {loaded.get(item) or 'its name as loaded'}")
    return out


def summary(changes: list[str], shown: int = 4) -> str:
    """The changes as one line: the first few, then how many more."""
    if len(changes) <= shown:
        return " · ".join(changes)
    return " · ".join(changes[:shown]) + f" · and {len(changes) - shown} more"
