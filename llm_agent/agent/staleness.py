"""
staleness.py — is an earlier answer still computed under today's settings?

An answer's figures depend on the session's assumptions — the load power
factor, how each generator is modelled — and on the data loaded. Change one
afterwards (System tab, CONDUCTOR, an upload) and the earlier charts silently
disagree with what the System tab shows. So each answer is stamped with what
its studies ran under (`stamp`), and compared with the current settings when
it is drawn (`reasons`); the app then offers to run it again.

The stamp comes from the results themselves where it can: every result that
assigned a tick says which power factor it used and which units were modelled
differently (`load_power_factor` / `generator_model` blocks). A study given its
own power factor is not compared on it — it never used the setting.

Pure data in, pure data out: no Streamlit, no HTTP.
"""

from __future__ import annotations


def stamp(tool_results: list, assumptions: dict | None) -> dict | None:
    """What an answer's studies ran under; None when it ran no study.

    `tool_results`: the turn's (tool name, result) pairs. `assumptions`: the
    backend's `/api/assumptions` payload read after the turn (for the data key).
    """
    results = [r for _, r in tool_results or [] if isinstance(r, dict) and "error" not in r]
    if not results:
        return None
    pf, gm = None, None
    for result in results:
        block = result.get("load_power_factor")
        if not isinstance(block, dict):
            continue
        block_gm = result.get("generator_model") or {}
        # Keyed by the units' names as loaded, like the settings: a rename is not a model change.
        model = block_gm.get("changed_units_as_loaded", block_gm.get("changed_units")) or {}
        gm = model                                     # every tick-assigning result says it
        if block.get("source") != "this study" and block.get("value") is not None:
            pf = float(block["value"])
    return {"pf": pf, "gm": gm, "data_key": (assumptions or {}).get("data_key"),
            "names": (assumptions or {}).get("names_key")}


def current(assumptions: dict | None) -> dict | None:
    """Today's settings in the stamp's shape, from `/api/assumptions`."""
    if not assumptions:
        return None
    value = (assumptions.get("load_power_factor") or {}).get("value")
    return {"pf": None if value is None else float(value),
            "gm": assumptions.get("generator_model") or {},
            "data_key": assumptions.get("data_key"),
            "names": assumptions.get("names_key")}


def reasons(then: dict | None, now: dict | None) -> list[str]:
    """What changed since the answer, in words; empty when nothing did (or
    when either side is unknown — never a guess)."""
    if not then or not now:
        return []
    out = []
    if then.get("pf") is not None and now.get("pf") is not None and abs(then["pf"] - now["pf"]) > 1e-9:
        out.append(f"load power factor {then['pf']:.2f} → {now['pf']:.2f}")
    if then.get("gm") is not None and now.get("gm") is not None and then["gm"] != now["gm"]:
        units = set(then["gm"]) | set(now["gm"])
        n = sum(1 for u in units if then["gm"].get(u) != now["gm"].get(u))
        out.append(f"generator model changed ({n} unit{'s' if n != 1 else ''})")
    if then.get("data_key") and now.get("data_key") and then["data_key"] != now["data_key"]:
        out.append("other data loaded")
    return out


def notes(then: dict | None, now: dict | None) -> list[str]:
    """What changed since the answer without changing its figures — a
    rename: the answer is still right, under the names it ran with."""
    if not then or not now or not then.get("names") or not now.get("names"):
        return []
    return ["elements renamed since — this answer shows the names it ran with"] \
        if then["names"] != now["names"] else []


def rerun_prompt(question: str, why: list[str]) -> str:
    """The question again, asking for fresh studies under today's settings."""
    return (f"Run this again with the current settings ({'; '.join(why)} since your earlier answer) — "
            f"re-run the studies, do not reuse the earlier results: {question}")
