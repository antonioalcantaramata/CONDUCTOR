"""
resolve.py — one way to find a bus, a line, a transformer or a unit by what
someone calls it.

Eight lookups had grown, one per endpoint, each accepting different spellings:
"Bus_7" found a unit in one study and nothing in another, "line Bus_8 -> Bus_9"
found no line because the label carries "[L12]", "external grid" was "not a
bus, line or transformer". Each failure cost the agent a step, and some sent it
to ask the operator something it could have read off the network. This module
is the one lookup they all use.

What it accepts, for every element:

- its index (an integer, or a string of digits);
- its name, its name as loaded, and the label every result shows
  ("Bus_0 -> Bus_1 [L0]"), with or without the "[L0]" tag;
- the tag alone ("L12", "T1") and "line 12" / "transformer 1";
- a branch by its ends, in either order, joined by any arrow, dash, "to" or
  "and" ("Bus_8 - Bus_9", "between Bus_8 and Bus_9");
- a leading kind word ("line …", "the bus …", "transformer …");
- a unit by its name, "Gen_bus7", or its bus ("Bus_7") when one unit is there.

Spelling is compared loosely: case, spaces, underscores, dashes and the break
between letters and digits do not count ("bus7", "Bus 7", "BUS_7" are one).
Letters of another script stuck to the end of a name (a slip of a model's
output, "Bus_4На") are dropped.

It never guesses. Two elements matching is an ambiguity, listed; nothing
matching comes back with the closest names, so the caller can correct itself.
A name that is not an element but that operators use — the external grid, the
slack — comes back saying what it is and where it connects.

Pure functions of a pandapower network: no app state, no HTTP.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import NamedTuple

import numpy as np

import generators as gm
from element_names import bus_display_name, clean_label, line_display_name, trafo_display_name

KINDS = ("bus", "line", "trafo")
UNIT_TABLES = ("gen", "sgen")
_KIND_WORDS = {"bus": "bus", "busbar": "bus", "node": "bus", "substation": "bus",
               "line": "line", "cable": "line", "feeder": "line", "circuit": "line",
               "trafo": "trafo", "transformer": "trafo", "tx": "trafo",
               "branch": None}
_PLURAL = {"bus": "buses", "line": "lines", "trafo": "transformers", "gen": "units", "sgen": "units"}
_WORD = {"bus": "bus", "line": "line", "trafo": "transformer", "gen": "unit", "sgen": "unit"}
# Names for the connection to the outside grid, which is not a bus, a line or a transformer.
_EXTERNAL = {"external grid", "ext grid", "extgrid", "slack", "slack bus", "grid connection",
             "external connection", "upstream grid", "upstream network", "external network",
             "the grid", "grid supply point", "point of connection"}


class Match(NamedTuple):
    etype: str        # bus | line | trafo | gen | sgen
    idx: int
    label: str        # the name every result shows
    how: str          # how it was found: name, index, tag, ends, bus, partial, slip


class ResolveError(ValueError):
    """Nothing, or more than one thing, matches; or the name is not an element.

    `reason` is "not_found", "ambiguous" or "not_an_element"; `candidates` the
    elements that matched (ambiguous); `suggestions` the closest names (not
    found)."""

    def __init__(self, message: str, reason: str, candidates: list[str] | None = None,
                 suggestions: list[str] | None = None):
        super().__init__(message)
        self.reason = reason
        self.candidates = candidates or []
        self.suggestions = suggestions or []


# ---------------------------------------------------------------------------
# Spelling
# ---------------------------------------------------------------------------

def _script(ch: str) -> str:
    return unicodedata.name(ch, "").split(" ")[0] if ch.isalpha() else ""


def foreign_tail(name: str) -> str | None:
    """The name without up to three trailing letters of a script the rest of
    it does not use ('Bus_4На' → 'Bus_4'); None when there is no such tail."""
    head = str(name or "").rstrip()
    tail_script = _script(head[-1]) if head else ""
    if not tail_script:
        return None
    end = len(head)
    while end and _script(head[end - 1]) == tail_script:
        end -= 1
    rest = head[:end]
    scripts = {_script(c) for c in rest if c.isalpha()}
    if not rest.strip() or tail_script in scripts or not scripts or len(head) - end > 3:
        return None
    return rest


def key(text) -> str:
    """A name as compared: its runs of letters and of digits, lower case,
    one space apart — 'Bus_0 -> Bus_1 [L0]' is 'bus 0 bus 1 l 0'."""
    return " ".join(re.findall(r"[^\W\d_]+|\d+", str(text or "").casefold()))


# ---------------------------------------------------------------------------
# What each element is called
# ---------------------------------------------------------------------------

def _raw_names(tbl, idx: int) -> list[str]:
    out = []
    for col in ("name", "loaded_name", "substation_name", "loaded_unit"):
        if col in tbl.columns and isinstance(tbl.at[idx, col], str) and clean_label(tbl.at[idx, col]):
            out.append(clean_label(tbl.at[idx, col]))
    return out


def display(net, etype: str, idx: int) -> str:
    """The name every result shows for the element."""
    if etype == "bus":
        return bus_display_name(net, idx)
    if etype == "line":
        return line_display_name(net, idx)
    if etype == "trafo":
        return trafo_display_name(net, idx)
    return gm.unit_name(net, etype, idx)


def labels(net, etype: str, idx: int) -> list[str]:
    """Every name the element answers to."""
    tbl = net[etype]
    out = [display(net, etype, idx), *_raw_names(tbl, idx)]
    if etype == "bus":
        out += [f"Bus_{idx}"]
    elif etype in ("line", "trafo"):
        tag = "L" if etype == "line" else "T"
        shown = display(net, etype, idx)
        stamped = shown.split(" | ")[-1]
        out += [stamped, stamped.rsplit(" [", 1)[0], f"{tag}{idx}", f"{etype} {idx}"]
        if etype == "trafo":
            out += [f"transformer {idx}"]
        a, b = (("from_bus", "to_bus") if etype == "line" else ("hv_bus", "lv_bus"))
        ends = [bus_display_name(net, int(tbl.at[idx, a])), bus_display_name(net, int(tbl.at[idx, b]))]
        out += [f"{ends[1]} -> {ends[0]}"]
    else:
        out += [f"Gen_bus{int(tbl.at[idx, 'bus'])}"]
    return list(dict.fromkeys(x for x in out if x))


def _index(net, tables: tuple[str, ...], text) -> Match | None:
    """An element by its index: an integer, or a string of digits."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, np.integer)) or (isinstance(text, str) and text.strip().isdigit()):
        i = int(text)
        for etype in tables:
            if i in net[etype].index:
                return Match(etype, i, display(net, etype, i), "index")
    return None


def _exact(net, tables: tuple[str, ...], wanted: str) -> list[tuple[str, int]]:
    k = key(wanted)
    if not k:
        return []
    return [(etype, int(i)) for etype in tables for i in net[etype].index
            if any(key(lab) == k for lab in labels(net, etype, int(i)))]


def _ends(net, tables: tuple[str, ...], wanted: str) -> list[tuple[str, int]]:
    """Branches named by their two ends, in either order."""
    branch_tables = [t for t in tables if t in ("line", "trafo")]
    if not branch_tables:
        return []
    text = re.sub(r"^\s*between\s+", "", wanted, flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:<->|->|<-|→|←|↔|—|–)\s*|\s+-\s+|\s+(?:to|and)\s+", text, flags=re.IGNORECASE)
    parts = [p for p in parts if p.strip()]
    if len(parts) != 2:
        return []
    try:
        a, b = (find(net, p, ("bus",)).idx for p in parts)
    except ResolveError:
        return []
    out = []
    for etype in branch_tables:
        tbl = net[etype]
        c1, c2 = ("from_bus", "to_bus") if etype == "line" else ("hv_bus", "lv_bus")
        out += [(etype, int(i)) for i in tbl.index if {int(tbl.at[i, c1]), int(tbl.at[i, c2])} == {a, b}]
    return out


def _partial(net, tables: tuple[str, ...], wanted: str) -> list[tuple[str, int]]:
    """Elements one of whose names holds the words asked for, in order
    ('Oscar' → 'Oscar substation'). One such element is taken only for four
    characters or more; several are an ambiguity at any length ('Bus', 'OLS-'),
    listed rather than refused as unknown."""
    k = key(wanted)
    if not k:
        return []
    pat = re.compile(r"(?:^| )" + re.escape(k) + r"(?: |$)")
    hits = [(etype, int(i)) for etype in tables for i in net[etype].index
            if any(pat.search(key(lab)) for lab in labels(net, etype, int(i)))]
    return hits if len(hits) > 1 or len(k.replace(" ", "")) >= 4 else []


def _external(net, wanted: str) -> str | None:
    """What the external grid is, when that is what was asked for."""
    if key(wanted) not in {key(x) for x in _EXTERNAL}:
        return None
    if not len(getattr(net, "ext_grid", [])):
        return "This network has no connection to an external grid."
    parts = []
    for i in net.ext_grid.index:
        bus = int(net.ext_grid.at[i, "bus"])
        via = [line_display_name(net, int(j)) for j in net.line.index
               if bus in (int(net.line.at[j, "from_bus"]), int(net.line.at[j, "to_bus"]))]
        via += [trafo_display_name(net, int(j)) for j in net.trafo.index
                if bus in (int(net.trafo.at[j, "hv_bus"]), int(net.trafo.at[j, "lv_bus"]))]
        parts.append(f"at {bus_display_name(net, bus)}" + (f", reached through {', '.join(via[:6])}" if via else ""))
    return ("The external grid (the slack) is not a bus, a line or a transformer: it is the connection "
            + "; ".join(parts) + ". Name that bus or one of those branches.")


def _suggest(net, tables: tuple[str, ...], wanted: str, n: int = 5) -> list[str]:
    """The closest names, as results show them."""
    by_key: dict[str, str] = {}
    for etype in tables:
        for i in net[etype].index:
            for lab in labels(net, etype, int(i)):
                by_key.setdefault(key(lab), display(net, etype, int(i)))
    close = difflib.get_close_matches(key(wanted), list(by_key), n=n * 2, cutoff=0.5)
    return list(dict.fromkeys(by_key[c] for c in close))[:n]


def _one(net, hits: list[tuple[str, int]], how: str, wanted: str) -> Match | None:
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        etype, i = hits[0]
        return Match(etype, i, display(net, etype, i), how)
    if len(hits) > 1:
        names = [f"{display(net, t, i)} ({_WORD[t]}, index {i})" for t, i in hits[:12]]
        kinds = {t for t, _ in hits}
        what = _PLURAL[next(iter(kinds))] if len(kinds) == 1 else "elements"
        raise ResolveError(f"{wanted!r} matches {len(hits)} {what}: {'; '.join(names)}. Name one of them exactly, "
                           "or give its index.", "ambiguous", candidates=[display(net, t, i) for t, i in hits])
    return None


# ---------------------------------------------------------------------------
# The lookups
# ---------------------------------------------------------------------------

def find(net, wanted, kinds: tuple[str, ...] = KINDS) -> Match:
    """The bus, line or transformer (of `kinds`) named `wanted`; raises
    ResolveError when none, or more than one, matches."""
    tables = tuple(k for k in kinds if k in net and len(net[k]))
    by_index = _index(net, tables, wanted)
    if by_index:
        return by_index
    text = str(wanted if wanted is not None else "").strip().strip("'\"`")
    if not text:
        raise ResolveError("A name is required.", "not_found")
    if isinstance(wanted, (int, np.integer)) and not isinstance(wanted, bool):
        raise ResolveError(f"There is no {'/'.join(_WORD[k] for k in kinds)} with index {wanted}.", "not_found")
    said = _external(net, text)
    if said:
        raise ResolveError(said, "not_an_element")

    for attempt, how in ((text, "name"), (foreign_tail(text), "slip")):
        if not attempt:
            continue
        found = _one(net, _exact(net, tables, attempt), how, text)
        if found:
            return found
        # A leading kind word: "line Bus_8 -> Bus_9", "the transformer T1", "bus 7".
        words = attempt.split()
        if words and words[0].casefold() == "the":
            words = words[1:]
        if len(words) > 1 and words[0].casefold().rstrip(":") in _KIND_WORDS:
            kind = _KIND_WORDS[words[0].casefold().rstrip(":")]
            narrowed = tuple(t for t in tables if kind is None or t == kind) or tables
            rest = " ".join(words[1:])
            found = (_index(net, narrowed, rest)
                     or _one(net, _exact(net, narrowed, rest), how, text)
                     or _one(net, _ends(net, narrowed, rest), "ends", text))
            if found:
                return found
        found = _one(net, _ends(net, tables, attempt), "ends", text)
        if found:
            return found
        found = _one(net, _partial(net, tables, attempt), "partial", text)
        if found:
            return found

    words = [_WORD[k] for k in kinds]
    kinds_said = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]
    suggestions = _suggest(net, tables, text)
    # Nothing close: some real names, so the caller sees how they are spelled.
    examples = [] if suggestions else [display(net, t, int(i)) for t in tables for i in net[t].index][:8]
    raise ResolveError(f"There is no {kinds_said} named {text!r}."
                       + (f" Closest: {'; '.join(suggestions)}." if suggestions else
                          f" Names here look like: {'; '.join(examples)}." if examples else ""),
                       "not_found", suggestions=suggestions or examples)


def find_unit(net, wanted, tables: tuple[str, ...] = UNIT_TABLES) -> Match:
    """The generating unit (of `tables`) named `wanted` — by name, 'Gen_bus7',
    or its bus ('Bus_7', 'Gen 7', 'generator at bus 7') when exactly one unit
    is there."""
    present = tuple(t for t in tables if t in net and len(net[t]))
    text = str(wanted if wanted is not None else "").strip().strip("'\"`")
    if not text:
        raise ResolveError("A unit name is required.", "not_found")
    for attempt, how in ((text, "name"), (foreign_tail(text), "slip")):
        if not attempt:
            continue
        found = _one(net, _exact(net, present, attempt), how, text)
        if found:
            return found
        # Its bus, said as a bus ("Bus_7") or after a unit word ("Gen 7",
        # "generator at bus 7", "the unit at Bus_7").
        rest = re.sub(r"^(?:the\s+)?(?:gen|generator|unit|machine|plant)\b\s*(?:at|on|in)?\s*", "", attempt,
                      flags=re.IGNORECASE)
        bus = None
        for said in dict.fromkeys((attempt, rest)):
            try:
                bus = find(net, said, ("bus",)).idx
                break
            except ResolveError:
                continue
        if bus is not None:
            at_bus = [(t, int(i)) for t in present for i in net[t].index if int(net[t].at[i, "bus"]) == bus]
            found = _one(net, at_bus, "bus", text)
            if found:
                return found
    units = [display(net, t, int(i)) for t in present for i in net[t].index]
    suggestions = _suggest(net, present, text)
    raise ResolveError(f"There is no unit named {text!r}. Units: {units[:20]}."
                       + (f" Closest: {'; '.join(suggestions)}." if suggestions else ""),
                       "not_found", candidates=units, suggestions=suggestions)
