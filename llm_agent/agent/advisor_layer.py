"""
advisor_layer.py — the advisor (recommender.py) and the CIM layer.

The advisor runs nothing and states no figure of its own; what it says about an element — that a unit is wind,
that a bus is at the end of a feeder — was until now unchecked, and it saw no more of the layer than the answer's
own words. Here, for any network:

1. **What it is shown** (`context_block`): what the layer says about the elements the latest results name, through
   their links (the turn's `cim_context`, `grid_context.py`) — feeder and where on it, voltage, what is connected,
   what a branch cuts off, each with its basis — with how to use it. Sent with the session only when there is
   something to show, so the advisor's system prompt does not grow.
2. **What it says, checked** (`layer_flags`): its comment, suggestions, replies and notes go through the same
   statement check as the answers (`entities.check_answer`): a source, a category, a feeder or a voltage the layer
   contradicts or does not support, an element the network does not have.
3. **What the network does not say, as a question** (`fact_gaps`, `gap_block`, `gap_suggestion`): when the
   reading check found that the conclusion moves with what units the layer does not know are (counted as wind or
   solar, or not), the advisor asks the operator to say what they are — it never assumes it, and declaring is the
   operator's alone. If its reply has no such suggestion, one is added.
4. **Groups by meaning, resolved** (`group_phrases`): a study it proposes may name a group as the tools accept it
   ("the solar units on feeder Line 1-2"); each such phrase is resolved against the layer before it is shown —
   the elements it means, or why the layer cannot resolve it.

Pure where it can be: the backend is read through `_get` (the layer, a phrase's selection), injectable in tests.
"""

from __future__ import annotations

import re
from urllib.parse import urlencode

from .grid_context import describe

# How much of the turn's layer context the advisor is shown.
_MAX_ELEMENTS = 8
_MAX_PHRASES = 6
_SHOWN_NAMES = 6
_GAP_STATUSES = ("changes", "shifts")
_MAX_UNITS_NAMED = 20


def _get(endpoint: str):
    from .tools import _get as get
    return get(endpoint)


# --- 1. What the advisor is shown ------------------------------------------------------------------------------

def context_block(record: dict | None) -> str:
    """The session part on the CIM layer: one line per element the latest results name, where they sit together,
    and the rules for using it — "" when the turn linked nothing."""
    ctx = (record or {}).get("cim_context") or {}
    elements = list((ctx.get("elements") or {}).values())[:_MAX_ELEMENTS]
    if not elements:
        return ""
    lines = [f"- {str(f.get('name', '')).split(' | ')[0]}: {describe(f)}" for f in elements if isinstance(f, dict)]
    where = ctx.get("where") or []
    if ctx.get("studied"):
        lines.insert(0, "Studied (the element the study was about): " + ", ".join(
            str(n).split(" | ")[0] for n in ctx["studied"]))
    from .grid_context import tie_lines
    lines += [f"- feeder {f}, normally open points: {t}" for f, t in tie_lines(ctx.get("ties")).items()]
    out = ["## What the CIM layer says about the elements the latest results name",
           "Each fact with its basis (stated: the network says so; inferred: from the topology; named: read from a "
           "name; unknown: the network does not say). What an element is — its energy source, a load's category, "
           "its feeder, its voltage — comes only from here or the turn record, never from its name or a guess; "
           "what you say about elements is checked against the layer. To name a group in a study, you may write it "
           "by meaning, as the tools accept it: \"the solar units on feeder Line 1-2\", \"the loads at 20 kV\", "
           "\"the units in substation Bus 5\" — with feeders, substations and levels this layer names; each such "
           "phrase is resolved against the layer and shown to the operator with the elements it means.",
           *lines]
    if where:
        out.append("Together: " + "; ".join(where))
    if ctx.get("changed"):
        out.append("The layer has changed since " + ", ".join(ctx["changed"]) + " ran (a declaration): the facts "
                   "above are as it is now.")
    return "\n".join(out)


# --- 3. What the network does not say -------------------------------------------------------------------------

def fact_gaps(record: dict | None) -> list[dict]:
    """The facts the conclusion moved with: each reading-check outcome on what units are (`unit_sources`) whose
    conclusion changed or a figure shifted — the units, the reading, the study's verdict before and after."""
    out = []
    for o in ((record or {}).get("reading_check") or {}).get("outcomes") or []:
        if not isinstance(o, dict) or o.get("reading") != "unit_sources" or o.get("status") not in _GAP_STATUSES:
            continue
        units = sorted(((o.get("overrides") or {}).get("assume_unit_sources") or {}))
        if units:
            out.append({"tool": o.get("tool"), "units": units, "label": o.get("label"), "status": o.get("status"),
                        "assumed": o.get("assumed"), "before": o.get("before"), "after": o.get("after")})
    return out


def _names(units: list[str]) -> str:
    shown = units[:_SHOWN_NAMES]
    text = ", ".join(shown[:-1]) + (" and " if len(shown) > 1 else "") + shown[-1] if shown else ""
    return text + (f" (and {len(units) - len(shown)} more)" if len(units) > len(shown) else "")


def gap_block(gaps: list[dict]) -> str:
    """The session part on what the conclusion depends on that the network does not say, with the rule."""
    if not gaps:
        return ""
    # Every unit by name: the reading's label shortens a long list ("… and 2 more"), and the advisor must name them.
    lines = [f"- {g['tool']}: with {g['label']} (the answer took them as {g['assumed']}), the conclusion "
             f"{'changes' if g['status'] == 'changes' else 'holds but a figure moves'}: {g['before']} → {g['after']}. "
             f"The units: {', '.join(g['units'][:_MAX_UNITS_NAMED])}"
             + (f" and {len(g['units']) - _MAX_UNITS_NAMED} more" if len(g['units']) > _MAX_UNITS_NAMED else "")
             for g in gaps]
    return ("## What the conclusion depends on that the network does not say (CIM layer, reading check)\n"
            + "\n".join(lines) + "\nOne suggestion (step_back) — or, in a discussion, your reply — must ask the "
            "operator to say what these units are, naming them and what the study found under each reading as "
            "given here; its prompt asks CONDUCTOR to record what the operator states (declare_network_facts) and "
            "to rerun the same study. Never assume what they are: only the operator can say.")


def gap_suggestion(gaps: list[dict], record: dict | None):
    """The suggestion added when the advisor's reply did not ask about the units the conclusion depends on."""
    from .recommender import Suggestion
    if not gaps:
        return None
    units = sorted({u for g in gaps for u in g["units"]})
    g = next((x for x in gaps if x["status"] == "changes"), gaps[0])
    timestamp = next((str(r.get("timestamp")) for r in _results(record) if isinstance(r, dict) and r.get("timestamp")),
                     "")
    rerun = f"rerun {g['tool']}" + (f" at {timestamp}" if timestamp else "") + " with the same settings"
    return Suggestion(
        angle="step_back", title=f"Say what {_names(units)} {'is' if len(units) == 1 else 'are'}",
        prompt=(f"If you know what {_names(units)} {'is' if len(units) == 1 else 'are'} (wind, solar or a "
                f"conventional unit), declare it for each, then {rerun}."),
        why=(f"The network does not say what {'it is' if len(units) == 1 else 'they are'}, and the reading check "
             f"found that with {g['label']} the conclusion "
             f"{'changes' if g['status'] == 'changes' else 'moves'}: {g['before']} → {g['after']}."))


def _results(record: dict | None) -> list:
    return [e.get("result") for e in (record or {}).get("tool_results") or [] if isinstance(e, dict)]


def asks_about(text: str, gaps: list[dict]) -> bool:
    """Whether a suggestion or reply already asks about the units the conclusion depends on."""
    units = {u for g in gaps for u in g["units"]}
    return bool(units) and any(re.search(r"(?<![\w])" + re.escape(u) + r"(?![\w])", text or "") for u in units)


# --- 2. What the advisor says, checked -------------------------------------------------------------------------

class Review:
    """The layer's review of one reply of the advisor: the facts of every element its texts name, fetched once
    (`loop._network_facts`: the base and the proposals the turn used), then each text checked on them; and the
    places a group phrase may name, fetched when first needed."""

    def __init__(self, texts: list[str], record: dict | None):
        from .loop import _network_facts
        self.record = record or {}
        joined = "\n".join(t for t in texts if t)
        self.facts = _network_facts(self.record.get("tool_results") or [], joined) if joined.strip() else {}
        self._vocab: dict | None = None

    def flags(self, *texts: str) -> tuple[str, ...]:
        """What the CIM layer contradicts or does not support in these texts (the answers' statement check)."""
        text = "\n".join(t for t in texts if t)
        if not text.strip() or not self.facts:
            return ()
        from .entities import check_answer
        issues = check_answer(text, {"network_facts": self.facts, "user": self.record.get("user") or ""})
        return tuple(dict.fromkeys(f"\"{i.text[:100]}\" {i.note}" for i in issues))

    def groups(self, *texts: str) -> tuple[dict, ...]:
        if self._vocab is None:
            self._vocab = _vocabulary()
        return group_phrases(list(texts), self._vocab)


def layer_flags(texts: list[str], record: dict | None) -> tuple[str, ...]:
    """What the CIM layer contradicts or does not support in the advisor's text (one-off `Review`)."""
    return Review(texts, record).flags(*texts)


# --- 4. Groups by meaning ----------------------------------------------------------------------------------------

_KINDS = r"(?:units?|generators?|loads?|lines?|transformers?|buses)"
_LEAD = {"the", "all", "of", "study", "curtail", "check", "compare", "run", "with", "and", "for", "from", "to",
         "only", "its", "their", "a", "an", "on", "in", "at", "every", "each", "whether", "which", "out", "take",
         "drop", "raise", "lower", "scale", "disable", "then", "also", "next", "first", "these", "those", "same",
         "both", "plus", "versus", "vs", "or", "by", "curtailing", "scaling", "studying", "checking"}


def _vocabulary() -> dict:
    """The places a phrase may name, from the layer: feeders (short names), substations, voltage levels."""
    facts = _get("/api/network/semantics?about=facts")
    if not isinstance(facts, dict) or "error" in facts:
        return {}
    return {"feeders": sorted({str(f).split(" | ")[0] for f in facts.get("feeders") or []}, key=len, reverse=True),
            "substations": sorted({str(b.get("substation")) for b in (facts.get("buses") or {}).values()
                                   if b.get("substation")}, key=len, reverse=True),
            "levels": [float(v) for v in facts.get("voltage_levels_kv") or []]}


def candidates(text: str, vocab: dict) -> list[str]:
    """Phrases in `text` that name a group by its place: a kind of element on a feeder, in a substation or at a
    voltage level the layer has — "the solar units on feeder Line 1-2", "loads at 20 kV" — with at most two
    qualifying words before the kind."""
    feeders = [re.escape(f) for f in vocab.get("feeders") or []]
    subs = [re.escape(s) for s in vocab.get("substations") or []]
    # A place either way round — "feeder Line 1-2" or "the Line 1-2 feeder" — written back the first way.
    places = [r"feeder\s+(?P<f1>" + "|".join(feeders) + ")"] if feeders else []
    places += [r"(?P<f2>" + "|".join(feeders) + r")\s+feeder"] if feeders else []
    places += [r"substation\s+(?P<s1>" + "|".join(subs) + ")", r"(?P<s2>" + "|".join(subs) + r")\s+substation"] \
        if subs else []
    places.append(r"(?P<kv>\d+(?:\.\d+)?\s*kV)(?:\s+level)?")
    pattern = re.compile(r"(?P<head>(?:[\w&/-]+\s+){0,2}" + _KINDS + r")\s+(?P<prep>on|in|at|of)\s+(?:the\s+)?(?:"
                         + "|".join(places) + r")(?![\w-])", re.IGNORECASE)
    out = []
    for m in pattern.finditer(text or ""):
        # The qualifier is the words right before the kind, back to the first article or connector: "report the
        # loads" → "loads", "curtailing the solar units" → "solar units".
        head = m.group("head").split()
        words = [head[-1]]
        for w in reversed(head[:-1]):
            if w.casefold() in _LEAD:
                break
            words.insert(0, w)
        g = m.groupdict()
        place = (f"feeder {g.get('f1') or g.get('f2')}" if g.get("f1") or g.get("f2") else
                 f"substation {g.get('s1') or g.get('s2')}" if g.get("s1") or g.get("s2") else g["kv"])
        phrase = f"{' '.join(words)} {m.group('prep')} {place}"
        if phrase not in out:
            out.append(phrase)
    return out[:_MAX_PHRASES]


def group_phrases(texts: list[str], vocab: dict | None = None) -> tuple[dict, ...]:
    """Each group phrase in `texts`, resolved against the layer (`describe_network(about="select")`): {"phrase",
    "means", "count", "elements"} — or {"phrase", "error"} when the layer cannot resolve it (no such units there,
    no such level). A phrase the selection cannot read at all is left out: it may be prose, not a group."""
    text = "\n".join(t for t in texts if t)
    if not text.strip():
        return ()
    vocab = _vocabulary() if vocab is None else vocab
    out = []
    for found in candidates(text, vocab or {}):
        got, phrase = _select(found)
        if isinstance(got, dict) and got.get("elements") is not None and "error" not in got:
            names = [e.get("name") for e in got["elements"]]
            out.append({"phrase": phrase, "means": got.get("means") or phrase, "count": len(names),
                        "elements": names[:_SHOWN_NAMES]})
            continue
        # The layer's reason (a 400's detail); none when the backend could not be reached — then nothing is said.
        detail = str((got or {}).get("detail") or "") if isinstance(got, dict) else ""
        if detail and not detail.startswith("Could not read"):
            out.append({"phrase": phrase, "error": detail})
    return tuple(out)


def _select(phrase: str):
    """(the layer's selection, the phrase it was given) — when the layer cannot read the qualifier (a verb taken
    for one: "report solar units on feeder …"), the same phrase with one word fewer before the kind, until the
    kind comes first."""
    words = phrase.split()
    while True:
        said = " ".join(words)
        got = _get("/api/network/semantics?" + urlencode({"about": "select", "element": said}))
        unread = isinstance(got, dict) and str(got.get("detail") or "").startswith("Could not read")
        if not unread or re.fullmatch(_KINDS, words[0], re.IGNORECASE):
            return got, said
        words = words[1:]


def group_line(g: dict) -> str:
    """One resolved phrase, as the operator reads it under a suggestion."""
    if g.get("error"):
        return f"“{g['phrase']}” — the CIM layer cannot resolve it: {g['error']}"
    more = g["count"] - len(g.get("elements") or [])
    return (f"“{g['phrase']}” — {g['count']} element(s) by the CIM layer: {', '.join(g.get('elements') or [])}"
            + (f" and {more} more" if more > 0 else ""))
