"""
entities.py — what an answer says the network's elements are, against what
the network says they are.

Provenance checks every figure against the study that produced it. A claim
about an element is checked here the same way, against the network's semantic
layer (backend/semantics.py) as the turn's record keeps it (`network_facts`):

- **a unit's energy source** — "Gen_bus7, a wind farm": unsupported when the
  network does not say what drives it, contradicted when it says otherwise;
  a load's category the same way ("Load 5, an industrial customer");
- **a group** — "the renewable generation", "the wind units": unsupported when
  no unit of the network is stated (or named) to be of that kind;
- **a voltage** — "the 132 kV substation", "Bus_9 at 20 kV": contradicted when
  the network states another, unsupported when the network has no such level
  (a per-unit case has none) — unless the figure is the operator's own;
- **a relation or a kind** — a branch "between A and B" whose ends are others,
  a line called a transformer (or the reverse), a line said to carry the
  external grid's power that does not;
- **an element that does not exist** — "Load R9" in a network whose loads are
  named "Load R…" but have no R9 (`names_not_in_network`, found when the turn
  is recorded);
- **a value called measured that is not** — "Load 5's measured reactive
  demand" when it is derived from the load power factor, or a unit's output
  that is one share of its bus's measurement.

A sentence that holds a claim back ("if", "would", "not stated", "unknown")
or denies it ("no wind units") is not a claim. Contradicted is a failure;
unsupported a warning — the network may know more than it states.

Built from the turn's record only, so the check runs again on old logs.
"""

from __future__ import annotations

import re
from typing import NamedTuple

UNSUPPORTED, CONTRADICTED = "entity_unsupported", "entity_contradicted"
VARIABLE = {"wind", "solar"}
RENEWABLE = {"wind", "solar", "hydro", "bioenergy"}


class Issue(NamedTuple):
    kind: str          # entity_unsupported | entity_contradicted
    text: str          # the sentence
    note: str


# A sentence ends at . ! ? before a space — also when markdown closes after it ("0.942 p.u.** I can't…": left
# joined, the next sentence's denial held back a claim in this one; found by error injection).
_SENTENCE_RE = re.compile(r"(?<=[.!?])[*_`]*\s+|\n+")
# Holding back or denying: not a claim about what the network is.
_HELD_RE = re.compile(
    r"\b(if|would|could|might|suppose|assum\w*|scenario\w*|hypothetical\w*|declar\w*|unless|whether|"
    r"not (?:stated|known|say|described|specified|identified|classified)|unknown|does not (?:say|state|describe)|"
    r"doesn['’]t (?:say|state)|no (?:\w+[- ])?(?:units?|generators?|generation|farms?|parks?|plants?)\b|none|neither|nor|without|"
    r"not (?:wind|solar|renewable)|isn['’]t|aren['’]t)\b", re.IGNORECASE)
_DENIED_RE = re.compile(r"\b(?:no|not|never|nor|without|isn['’]t|doesn['’]t|does not|has no)\b", re.IGNORECASE)
_SOURCE_WORDS = {
    "wind": r"\bwind(?:[- ]?(?:farm|park|turbine|unit|generator|power|plant)s?)?\b",
    "solar": r"\b(?:solar|photovoltaic|pv)(?:[- ]?(?:farm|park|plant|unit|array|generator)s?)?\b",
    "hydro": r"\bhydro\w*\b",
    "thermal": r"\b(?:gas|coal|diesel|thermal|ccgt|ocgt|chp|lignite)(?:[- ]?(?:turbine|plant|unit|fired))?\b",
    "nuclear": r"\bnuclear\b",
    "storage": r"\b(?:battery|batteries|storage)\b",
    "condenser": r"\b(?:synchronous )?condensers?\b",
}
_CATEGORY_WORDS = {
    "residential": r"\b(?:residential|household|domestic|housing)\b",
    "commercial": r"\b(?:commercial|office|retail)\b",
    "industrial": r"\b(?:industr\w*|factory|manufactur\w*)\b",
    "rural": r"\brural\b",
    "agricultural": r"\b(?:agricultur\w*|irrigation)\b",
    "ev_charging": r"\b(?:ev|electric[- ]vehicle|charging)\b",
    "data_centre": r"\bdata ?cent(?:re|er)s?\b",
    "heat_pump": r"\bheat ?pumps?\b",
}
_ALSO = {"commercial_industrial": {"commercial", "industrial"}}
# The head of a list of units by source: "10 solar units:", "the solar PV units are", "solar units —", "the
# storage unit, Battery Storage 3,". Only with
# the marker: "the eight PV units and WKA 7" is not a list of PV units that WKA 7 belongs to.
_LIST_HEAD_RE = re.compile(
    r"\b(?P<src>wind|solar|pv|photovoltaic|hydro|thermal|nuclear|storage|battery)\b(?:\s+(?:pv|photovoltaic))?"
    r"\s+(?:units?|generators?|plants?|farms?|parks?)\b(?:\s+(?:are|is|include|includes)\b\s*:?|\s*[:—–,])",
    re.IGNORECASE)
_GROUP_RE = re.compile(
    r"\b(?P<kind>renewables?|wind|solar|pv|photovoltaic)\b(?:[- ](?:generation|output|production|capacity|power|"
    r"units?|farms?|parks?|plants?|generators?|in[- ]feed))", re.IGNORECASE)
_STUDY_TERM = r"[- ]?(?:uncertainty|case|scenario|variability|forecast error)\b"
_KV_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?kV\b")
_BETWEEN_RE = re.compile(r"\b(?:between|connecting|joining|linking)\s+(?P<a>\S+(?: \S+)?)\s+(?:and|to|with)\s+"
                         r"(?P<b>\S+(?: \S+)?)", re.IGNORECASE)


# A clause ends at these: "WKA 7 is measured directly; PV 9 is allocated" says nothing measured about PV 9.
_CLAUSE_RE = re.compile(r"[;:—–]|,\s|\s(?:but|while|whereas|so)\s")
# The word that says what a value is: a clause that already says it ("allocated", "set") is right, whatever else it
# says is measured ("allocated as a share of the measured total").
_KIND_WORDS = re.compile(r"\b(?:allocat\w*|estimat\w*|derived|deriv\w*|solved|set|set-point|setpoint|assum\w*|nominal|"
                         r"shares?|split)\b", re.IGNORECASE)
_NOT_MEASURED = re.compile(r"\b(?:rather than|instead of|not|never|isn['’]t|aren['’]t|wasn['’]t)\s+(?:\w+\s+)?"
                           r"(?:measured|metered)\b", re.IGNORECASE)


def _clauses_naming(sentence: str, name: str) -> list[str]:
    parts, start = [], 0
    for m in _CLAUSE_RE.finditer(sentence):
        parts.append(sentence[start:m.start()])
        start = m.end()
    parts.append(sentence[start:])
    return [p for p in parts if _mentions(p, name)]


def _calls_measured(clause: str) -> bool:
    return bool(re.search(r"\b(?:measured|metered)\b", clause, re.IGNORECASE)) \
        and not _NOT_MEASURED.search(clause) and not _KIND_WORDS.search(clause)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]


def _clean(text: str) -> str:
    """The sentence without markdown emphasis. Underscores stay: they are part of names ("Gen_bus12", "Bus_25"),
    and stripping them made every such name unmatchable (found by error injection, evaluation/injection.py)."""
    return re.sub(r"[*`]+", "", text or "")


def _mentions(sentence: str, name: str) -> list[int]:
    return [m.start() for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"(?![\w])", sentence)]


def _tag_of(label: str) -> str | None:
    m = re.search(r"\[([LT]\d+)\]$", label)
    return m.group(1) if m else None


def _branch_spans(sentence: str, label: str) -> list[tuple[int, int]]:
    """Where the sentence names the branch, as (start, end): its full label, its short name ("Line 1-2", as the
    tables and most answers write it), or its tag ("L0")."""
    forms = [label]
    short = label.split(" | ")[0]
    if short != label and re.search(r"\d", short):
        forms.append(short)
    spans = {(m, m + len(form)) for form in forms for m in _mentions(sentence, form)}
    tag = _tag_of(label)
    if tag:
        spans |= {(m.start(), m.end()) for m in re.finditer(r"(?<![\w\[])" + tag + r"(?![\w])", sentence)}
    return sorted(spans)


def _template(name: str) -> str:
    return re.sub(r"\d+", "#", name)


def names_not_in_network(answer: str, names) -> list[str]:
    """Names in the answer shaped like the network's own but not among them:
    "Load R9" where the loads are "Load R1"…"Load R14" without an R9 (observed
    live). A shape counts when at least three names share it and it has a
    word in it, so a stray "Phase 2" or a bare number is not an element."""
    names = {n for n in names if n}
    shapes: dict[str, int] = {}
    for n in names:
        if re.search(r"\d", n):
            shapes[_template(n)] = shapes.get(_template(n), 0) + 1
    out = []
    folded = {n.casefold() for n in names}
    for shape, count in shapes.items():
        # A shape needs a word: bare numbers ("1", "2" — pandapower's case30
        # buses) would make every figure an element name.
        if count < 3 or not re.search(r"[A-Za-z]{1,}", shape):
            continue
        pattern = r"(?<![\w])" + re.escape(shape).replace("\\#", "#").replace("#", r"\d+") + r"(?![\w])"
        for m in re.finditer(pattern, answer or ""):
            if m.group(0).casefold() not in folded:
                out.append(m.group(0))
    return sorted(set(out))


def check_answer(answer: str, record: dict) -> list[Issue]:
    """Where the answer says an element is something the network does not
    say (or says otherwise). [] when the record holds no network facts."""
    facts_by_net = record.get("network_facts") or {}
    if not facts_by_net or not answer:
        return []
    question = record.get("user") or ""
    units = {k: v for f in facts_by_net.values() for k, v in (f.get("units") or {}).items()}
    loads = {k: v for f in facts_by_net.values() for k, v in (f.get("loads") or {}).items()}
    buses = {k: v for f in facts_by_net.values() for k, v in (f.get("buses") or {}).items()}
    branches = {k: v for f in facts_by_net.values() for k, v in (f.get("branches") or {}).items()}
    known_by_source: dict[str, set] = {}
    for f in facts_by_net.values():
        for src, names in (f.get("units_by_source") or {}).items():
            known_by_source.setdefault(src, set()).update(names)
    levels = {float(v) for f in facts_by_net.values() for v in f.get("voltage_levels_kv") or []}
    per_unit = all(f.get("per_unit_case") for f in facts_by_net.values())
    ext_branches = {b for f in facts_by_net.values() for e in f.get("external_grids") or [] for b in e["branches"]}
    issues: list[Issue] = []

    def _operator_said(name: str, said: list[str]) -> bool:
        """The operator's own statement in the question ("Gen_bus12 is a wind farm"): theirs to make."""
        return bool(re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", question)) and any(
            re.search(_SOURCE_WORDS[s], question, re.IGNORECASE) for s in said if s in _SOURCE_WORDS)

    # Every name the answer uses, longest first: blanked out of a window before
    # its words are read, so "PV 3, PV 4, WKA 7" says nothing about WKA 7
    # (observed live: a list of units flagged WKA 7 as said to be solar).
    names = sorted({*units, *loads, *buses, *branches}, key=len, reverse=True)

    def masked(sentence: str) -> str:
        """The sentence with every name blanked, positions kept (a window
        must not cut a name in two and read half of it)."""
        for other in names:
            sentence = re.sub(r"(?<![\w])" + re.escape(other) + r"(?![\w])",
                              lambda m: " " * len(m.group(0)), sentence)
        return sentence

    def attached(sentence: str, at: int, name: str, patterns: dict) -> list[str]:
        """What the sentence says the element is — only a word attached to
        its name: "WKA 7, a wind turbine", "WKA 7 (wind)", "WKA 7 is a wind
        turbine", "the wind turbine WKA 7". A word that is merely near it — in
        a list of units, beside another's name — says nothing about it
        (observed live: "PV 3–6, …, CHP diesel 1, … Batteries 1 and 2")."""
        m = masked(sentence)
        after = m[at + len(name): at + len(name) + 60]
        before = m[max(0, at - 40): at]
        said = []
        for value, pat in patterns.items():
            # A conjunction ends the description: "PV 11 — and the wind unit WKA 7" says nothing
            # about PV 11 (observed: a false "PV 11 is wind").
            # A closing quote may come first: '"Wind Park West" is listed as solar'.
            follows = re.match(r"[\s\"”’']*(?:,\s*(?:an?|the)\s+|\(\s*|(?:is|was|are|were)\s+(?:an?\s+|the\s+)?|"
                               r"[—–-]\s*(?:an?\s+|the\s+)?)(?:(?!(?:and|or|but|while|whereas)\b)[\w-]+\s+){0,2}?"
                               r"(?:" + pat + r")", after, re.IGNORECASE)
            precedes = re.search(r"(?:" + pat + r")(?:\s+(?:unit|farm|park|plant|generator|turbine|array|"
                                 r"station|load|customer|site)s?)?\s*$", before, re.IGNORECASE)
            if follows or precedes:
                said.append(value)
        return said

    for raw in _sentences(answer):
        sentence = _clean(raw)
        held = bool(_HELD_RE.search(sentence))

        # 1. A unit's energy source.
        for name, unit in units.items():
            for at in _mentions(sentence, name):
                said = attached(sentence, at, name, _SOURCE_WORDS)
                if not said or held or _operator_said(name, said):
                    continue
                if unit["basis"] == "unknown":
                    issues.append(Issue(UNSUPPORTED, raw[:240], f"says {name} is {said[0]}, but the network does not "
                                                                 "say what drives it (describe_network; the operator "
                                                                 "can declare it)"))
                elif unit["source"] not in said and not (unit["source"] == "condenser" and "condenser" in said):
                    issues.append(Issue(CONTRADICTED, raw[:240], f"says {name} is {said[0]}, but it is "
                                                                  f"{unit['source']} ({unit['basis']})"))
                break

        # 1a. Units listed under a source: "10 solar units: PV 3, …, Solar Farm North" claims each named one
        # is solar — up to the next source word ("…PV 11 — and the wind unit WKA 7"). Observed: a CHP
        # unit renamed "Solar Farm North" listed among the solar units.
        if not held:
            ms = masked(sentence)
            for m in _LIST_HEAD_RE.finditer(ms):
                src = next((k for k, pat in _SOURCE_WORDS.items()
                            if re.fullmatch(pat, m.group("src"), re.IGNORECASE)), None)
                # A mixed head ("the solar and wind units:") claims no one source.
                if src is None or re.search(r"(?:" + "|".join(_SOURCE_WORDS.values()) + r")\s*(?:,|and|or|/)\s*$",
                                            ms[max(0, m.start() - 30):m.start()], re.IGNORECASE):
                    continue
                tail = sentence[m.end():]
                # The list ends at the next source word or at a closing dash ("—PV 3 … PV 11—and WKA 7").
                ends = [n.start() for pat in _SOURCE_WORDS.values() for n in re.finditer(pat, masked(tail), re.IGNORECASE)]
                ends += [n.start() for n in re.finditer(r"[—–;]|\.[\"”’']*\s", tail) if n.start() > 0]
                stop = min(ends or [len(tail)])
                for name, unit in units.items():
                    if not _mentions(tail[:stop], name) or _operator_said(name, [src]):
                        continue
                    # Its own description wins over the list's ("WKA 7 (wind)").
                    if any(attached(sentence, at, name, _SOURCE_WORDS) for at in _mentions(sentence, name)):
                        continue
                    if unit["basis"] != "unknown" and unit["source"] != src:
                        issues.append(Issue(CONTRADICTED, raw[:240], f"lists {name} among the {src} units, but it "
                                                                      f"is {unit['source']} ({unit['basis']})"))

        # 1b. A load's category.
        for name, load in loads.items():
            for at in _mentions(sentence, name):
                said = attached(sentence, at, name, _CATEGORY_WORDS)
                if not said or held:
                    continue
                if load["basis"] == "unknown":
                    issues.append(Issue(UNSUPPORTED, raw[:240], f"says {name} is {said[0]} demand, but the network "
                                                                 "does not say what kind of demand it is"))
                elif load["category"] not in said and not set(said) & _ALSO.get(load["category"], set()):
                    issues.append(Issue(CONTRADICTED, raw[:240], f"says {name} is {said[0]} demand, but it is "
                                                                  f"{load['category']} ({load['basis']})"))
                break

        # 2. A group of units by source — not when the sentence denies it ("the network does not identify
        # any solar unit") or the operator's question spoke of it ("if wind and solar output dropped").
        if not held and not _DENIED_RE.search(sentence):
            for m in _GROUP_RE.finditer(sentence):
                kind = m.group("kind").casefold()
                if re.search(r"\b" + re.escape(kind.rstrip("s")), question, re.IGNORECASE):
                    continue
                # A study's setting or operation, not a claim about the units: "renewable-output uncertainty was
                # disabled", "the 50 % renewable-output case", "studies that scale wind or solar output".
                if re.match(_STUDY_TERM, sentence[m.end():], re.IGNORECASE) or re.search(
                        r"\bscal\w*\s+(?:\w+\s+){0,2}$", sentence[max(0, m.start() - 30):m.start()], re.IGNORECASE):
                    continue
                wanted = RENEWABLE if kind.startswith("renewable") else (
                    {"solar"} if kind in ("solar", "pv", "photovoltaic") else {"wind"})
                if not any(known_by_source.get(src) for src in wanted):
                    issues.append(Issue(UNSUPPORTED, raw[:240],
                                        f"speaks of {m.group(0)}, but no unit of this network is stated or named to "
                                        f"be {'/'.join(sorted(wanted))}"))
                    break

        # 3. Voltages.
        for m in _KV_RE.finditer(sentence):
            kv = float(m.group(1))
            if re.search(r"(?<![\d.])" + re.escape(m.group(1)) + r"\s?kV", question):
                continue                                            # the operator's own figure
            named = [(n, b) for n, b in buses.items() if _mentions(sentence, n)]
            stated = [(n, b) for n, b in named if b.get("kv_basis") == "stated" and b.get("kv") is not None]
            # Attached to the bus: "Bus 3 is at 110 kV", "Bus 3 (20 kV)", "the 20 kV Bus 9". Merely near it, the
            # level may be another's: "Bus 0 feeds the 20 kV network through T0" (Bus 0 is the 110 kV side).
            near = [(n, b) for n, b in stated if any(
                re.fullmatch(r"[\s,(:—–-]*(?:(?:is|was|are|were|at|of|rated|nominal)\s+){0,3}[\s(]*",
                             sentence[at + len(n):m.start()], re.IGNORECASE)
                or (at >= m.end() and re.fullmatch(r"\s*(?:kV\s+)?(?:bus(?:bar)?\s+)?", sentence[m.end():at],
                                                   re.IGNORECASE))
                for at in _mentions(sentence, n))]
            if near and all(abs(float(b["kv"]) - kv) > 1e-6 for _n, b in near):
                n, b = near[0]
                issues.append(Issue(CONTRADICTED, raw[:240], f"says {kv:g} kV beside {n}, which is at "
                                                              f"{float(b['kv']):g} kV"))
            elif not held and not any(abs(lv - kv) < 1e-6 for lv in levels) \
                    and not any(abs(float(b["kv"]) - kv) < 1e-6 for _n, b in stated):
                issues.append(Issue(UNSUPPORTED, raw[:240],
                                    ("speaks of " + f"{kv:g} kV, but this network's voltages are per-unit "
                                     "placeholders (no level is stated)") if per_unit else
                                    f"speaks of {kv:g} kV, but the network has no such level "
                                    f"({', '.join(f'{v:g}' for v in sorted(levels, reverse=True))} kV)"))
            break

        # 4. Relations and kinds of branches.
        for label, br in branches.items():
            spans = _branch_spans(sentence, label)
            if not spans:
                continue
            before = " ".join(sentence[max(0, s - 14): s] for s, _e in spans).casefold()
            # Or after it: "Line 1-2 is a transformer", "Line 1-2 (transformer)".
            after = [sentence[e:e + 40] for _s, e in spans]

            def called(word: str) -> bool:
                return any(re.match(r"\s*(?:\(\s*|,\s*(?:a|an|the)\s+|(?:is|was)\s+(?:a|an|the)\s+)(?:[\w-]+\s+)?"
                                    + word + r"s?\b", a, re.IGNORECASE) for a in after)
            if br["type"] == "line" and ((re.search(r"\btransformers?\s*$|\btransformers? ", before)
                                          and not re.search(r"\blines?\b", before)) or called("transformer")):
                issues.append(Issue(CONTRADICTED, raw[:240], f"calls {label} a transformer; it is a line"))
            if br["type"] == "trafo" and (re.search(r"\blines?\s*$", before) or called("line")):
                issues.append(Issue(CONTRADICTED, raw[:240], f"calls {label} a line; it is a transformer"))
            for m in _BETWEEN_RE.finditer(sentence):
                if abs(m.start() - spans[0][0]) > 60:
                    continue
                ends = {e.casefold() for e in br["ends"]}
                said = {m.group("a").strip(" ,.;:()").casefold(), m.group("b").strip(" ,.;:()").casefold()}
                bus_said = {s for s in said if s in {b.casefold() for b in buses} or re.fullmatch(r"bus[_ ]\d+", s)}
                if len(bus_said) == 2 and bus_said != ends:
                    issues.append(Issue(CONTRADICTED, raw[:240],
                                        f"places {label} between {m.group('a')} and {m.group('b')}; its ends are "
                                        f"{br['ends'][0]} and {br['ends'][1]}"))
                break
            if re.search(r"\b(?:from|carr\w*|feed\w*|suppl\w*)\b[^.]{0,30}\bexternal grid\b|\bexternal grid\b[^.]{0,20}"
                         r"\b(?:through|via|over)\b", sentence, re.IGNORECASE) and ext_branches \
                    and label not in ext_branches and not held \
                    and not re.search(r"\b(?:route|path)\b|→|->", sentence, re.IGNORECASE):
                # A route from the external grid ("Trafo 0-1 → Line 1-2 → Line 2-3 …") names every branch on it,
                # not only the first; all of them carry its power.
                if re.search(r"\b(?:line|branch|transformer)s?\b", sentence, re.IGNORECASE):
                    issues.append(Issue(CONTRADICTED, raw[:240],
                                        f"has {label} carrying the external grid's power; the branches that do are "
                                        f"{', '.join(sorted(ext_branches)[:4])}"))
    # 5b. A value called measured that is not: derived from an assumption, solved by the power flow, set, or
    # the case's own (contradicted); a share of its bus's measurement (unsupported: the bus total is measured,
    # the element's share is an estimate). Only where the record kept where each value comes from.
    for raw in _sentences(answer):
        sentence = _clean(raw)
        if not re.search(r"\b(?:measured|metered)\b", sentence, re.IGNORECASE) or _HELD_RE.search(sentence) \
                or _DENIED_RE.search(sentence):
            continue
        reactive = bool(re.search(r"\breactive\b|\bq\b|\bmvar\b", sentence, re.IGNORECASE))
        q = "q" if reactive else "p"
        for name, e in {**units, **loads}.items():
            kind = e.get(f"{q}_kind")
            if not kind or not any(_calls_measured(clause) for clause in _clauses_naming(sentence, name)):
                continue
            what = "reactive" if reactive else "active"
            src = e.get(f"{q}_from") or ""
            if kind in ("derived", "solved", "set", "nominal"):
                issues.append(Issue(CONTRADICTED, raw[:240], f"calls {name}'s {what} power measured, but it is "
                                                              f"{kind}: {src}"))
            elif kind == "allocated":
                issues.append(Issue(UNSUPPORTED, raw[:240], f"calls {name}'s {what} power measured, but it is a "
                                                             f"share of its bus's measurement: {src}"))

    # 5. An element the network does not have.
    # Missing from every network the turn studied: a proposal's new bus exists in the proposal.
    lists = [set(f.get("names_not_in_network") or []) for f in facts_by_net.values()]
    missing = set.intersection(*lists) if lists else set()
    # A hedge does not excuse it ("if Load R9 trips", "the sources of Gen_bus62 are unknown" still name an element
    # that is not there); only denying the element itself does ("there is no Load R9", "Load R9 does not exist").
    # The sources' wider exemption (any hedge or denial in the sentence) let 31 of 144 injected names through.
    for raw in _sentences(answer):
        sentence = _clean(raw)
        for name in sorted(missing):
            if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", question):
                continue                                    # the operator's own name
            for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"(?![\w])", sentence):
                denied = re.search(r"\b(?:no|not|never|nor|isn['’]t|aren['’]t|without)\s+(?:[\w-]+\s+){0,2}$",
                                   sentence[max(0, m.start() - 40):m.start()], re.IGNORECASE) \
                    or re.match(r"[\s,]*(?:does not|doesn['’]t|do not|don['’]t|did not|didn['’]t) exist",
                                sentence[m.end():], re.IGNORECASE)
                if not denied:
                    issues.append(Issue(CONTRADICTED, raw[:240], f"names {name}, which this network does not have"))
                    break
    # One issue per sentence and kind.
    seen, out = set(), []
    for i in issues:
        if (i.kind, i.text) not in seen:
            seen.add((i.kind, i.text))
            out.append(i)
    return out


def as_conflicts(answer: str, record: dict) -> list:
    """The issues as characterisation conflicts (path `entities`), so the
    answer review, the audit panel and the advisor show them with the others."""
    from .characterisation import Conflict
    return [Conflict(0.0, None, i.text, "entities", i.kind, i.note, study=True)
            for i in check_answer(answer, record)]
