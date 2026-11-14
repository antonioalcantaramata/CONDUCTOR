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
  that is one share of its bus's measurement;
- **a fact read only from a name, said as a fact** — "Solar Plant A is the
  solar unit" when nothing but its name says so (basis `named`): unsupported,
  unless the sentence says it is read from the name.

A unit's source is read where the answer says it: beside the name, in a list
under a source ("The wind units are …", or a bulleted list under "Solar
units:"), in a table row with a source or type column, or as the answer to a
question about a source ("Where is the storage?" — "Battery Storage 3, at Bus
5", whose name says storage but which is thermal).

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
NAMED = "named"
# The sentence says the fact is read from the name: "going by its name", "presumably", "its name suggests".
_FROM_NAME_RE = re.compile(
    r"\b(?:(?:its|the|their) names?|by (?:its )?name|from (?:its |the )?name|names? (?:suggests?|says|indicates?|implies)|"
    r"going by|judging by|according to (?:its|the|their) names?|presumably|apparently|appears? to be|seems? to be|"
    r"probably|likely|named as)\b", re.IGNORECASE)
# A table column that says what a unit is.
_SOURCE_COLUMN_RE = re.compile(r"\b(?:source|type|technology|kind|fuel|energy|generation)\b", re.IGNORECASE)
# A question about units of one source: "Which units are wind turbines?", "Where is the storage?".
_ASKS_RE = re.compile(r"\b(?:which|where|what|list|name|identify|how many)\b", re.IGNORECASE)
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
            # A location may come between: "Solar Plant A at Bus_1 is the solar unit" (the bus's name is blanked).
            follows = re.match(r"[\s\"”’']*(?:(?:at|on)\s+(?:[\w-]+\s+)?)?"
                               r"(?:,\s*(?:an?|the)\s+|\(\s*|(?:is|was|are|were)\s+(?:an?\s+|the\s+)?|"
                               r"[—–-]\s*(?:an?\s+|the\s+)?)(?:(?!(?:and|or|but|while|whereas)\b)[\w-]+\s+){0,2}?"
                               r"(?:" + pat + r")", after, re.IGNORECASE)
            precedes = re.search(r"(?:" + pat + r")(?:\s+(?:unit|farm|park|plant|generator|turbine|array|"
                                 r"station|load|customer|site)s?)?\s*$", before, re.IGNORECASE)
            if follows or precedes:
                said.append(value)
        return said

    def listed(sentence: str, raw: str, held: bool) -> None:
        """1a. Units listed under a source: "10 solar units: PV 3, …, Solar Farm North" claims each named one is solar."""
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
                tail = sentence[m.start() + len(m.group(0).rstrip()):]
                # The list ends at the next source word or at a closing dash ("—PV 3 … PV 11—and WKA 7").
                # A source word that starts a name ("PV 61", one the network does not have) is not a new list.
                mt = masked(tail)
                ends = [n.start() for pat in _SOURCE_WORDS.values() for n in re.finditer(pat, mt, re.IGNORECASE)
                        if not re.match(r"\s*\d", mt[n.end():])]
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
                    elif unit["basis"] == NAMED and not _FROM_NAME_RE.search(sentence):
                        issues.append(Issue(UNSUPPORTED, raw[:240], _name_only(name, src)))


    # The feeder each bus, unit and load is on, over every network the turn studied (a proposal may switch
    # differently): a claim stands when any of them agrees.
    feeders = {f for nf in facts_by_net.values() for f in nf.get("feeders") or []}
    on_feeder: dict[str, set] = {}
    for nf in facts_by_net.values():
        for key in ("buses", "units", "loads"):
            for n, e in (nf.get(key) or {}).items():
                if "feeder" in e:
                    on_feeder.setdefault(n, set()).add(e["feeder"])

    def feeder_claim(sentence: str, raw: str) -> None:
        """3b. The element a feeder is said of — the nearest name before "feeder F" (or "the F feeder") in the
        same clause, joined by a word of place ("on", "of", "at the end of", "supplied by") with no other name
        between — against the feeder the layer infers for it."""
        for f in sorted(feeders, key=len, reverse=True):
            for m in re.finditer(r"\bfeeders?\s+" + re.escape(f) + r"(?![\w-])|(?<![\w-])" + re.escape(f)
                                 + r"\s+feeder\b", sentence, re.IGNORECASE):
                best = None
                for n in on_feeder:
                    for at in _mentions(sentence, n):
                        end = at + len(n)
                        if end <= m.start() and (best is None or end > best[1]):
                            best = (n, end)
                if best is None:
                    continue
                name, end = best
                gap = sentence[end:m.start()]
                if len(gap) > 60 or re.search(r"[;.]|\bnot\b|\bno\b|\bwhile\b|\bwhereas\b|\bbut\b", gap) or \
                        not re.search(r"\b(?:on|of|in|along|at|by|via|from|downstream)\b", gap, re.IGNORECASE) or \
                        any(_mentions(gap, other) for other in names if other != name):
                    continue
                known = on_feeder[name]
                if f in known:
                    continue
                if known == {None}:
                    issues.append(Issue(UNSUPPORTED, raw[:240], f"puts {name} on feeder {f}, but the CIM layer puts "
                                                                 "it on no feeder (at a substation's busbar, or in a "
                                                                 "meshed voltage level)"))
                else:
                    issues.append(Issue(CONTRADICTED, raw[:240], f"puts {name} on feeder {f}, but it is on feeder "
                                                                  f"{' or '.join(sorted(k for k in known if k))} "
                                                                  "(inferred from the topology as switched now)"))
                return

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
                elif unit["basis"] == NAMED and unit["source"] in said:
                    if not _FROM_NAME_RE.search(sentence):
                        issues.append(Issue(UNSUPPORTED, raw[:240], _name_only(name, said[0])))
                elif unit["source"] not in said and not (unit["source"] == "condenser" and "condenser" in said):
                    issues.append(Issue(CONTRADICTED, raw[:240], f"says {name} is {said[0]}, but it is "
                                                                  f"{unit['source']} ({unit['basis']})"))
                break

        listed(sentence, raw, held)

        # 1b. A load's category.
        for name, load in loads.items():
            for at in _mentions(sentence, name):
                said = attached(sentence, at, name, _CATEGORY_WORDS)
                if not said or held:
                    continue
                if load["basis"] == "unknown":
                    issues.append(Issue(UNSUPPORTED, raw[:240], f"says {name} is {said[0]} demand, but the network "
                                                                 "does not say what kind of demand it is"))
                elif load["basis"] == NAMED and load["category"] in said:
                    if not _FROM_NAME_RE.search(sentence):
                        issues.append(Issue(UNSUPPORTED, raw[:240], _name_only(name, f"{said[0]} demand")))
                elif load["category"] not in said and not set(said) & _ALSO.get(load["category"], set()):
                    issues.append(Issue(CONTRADICTED, raw[:240], f"says {name} is {said[0]} demand, but it is "
                                                                  f"{load['category']} ({load['basis']})"))
                break

        # 2. A group of units by source — not when the sentence denies it ("the network does not identify
        # any solar unit") or the operator's question spoke of it ("if wind and solar output dropped").
        # Not where a unit in the sentence was already flagged: "Gen_bus22, a wind farm" is one claim, not also
        # one about "the wind farms" of the network.
        if not held and not _DENIED_RE.search(sentence) and not any(i.text == raw[:240] for i in issues):
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

        # 3b. A feeder: "Bus 14, at the end of feeder Line 1-2", "Load R8 on the Line 12-13 feeder".
        if feeders and not held:
            feeder_claim(sentence, raw)

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
    # Bulleted lists under a head that lists one source, joined: through the list check only.
    for raw in _joined_lists(answer):
        listed(_clean(raw), raw, bool(_HELD_RE.search(_clean(raw))))

    # 1c. Table rows: a unit's name in one cell and its source in a source or type column.
    for row, name, said in _table_claims(answer, units):
        unit = units[name]
        if _operator_said(name, [said]):
            continue
        if unit["basis"] == "unknown":
            issues.append(Issue(UNSUPPORTED, row[:240], f"gives {name} as {said}, but the network does not say what "
                                                        "drives it (describe_network; the operator can declare it)"))
        elif unit["basis"] == NAMED and unit["source"] == said:
            issues.append(Issue(UNSUPPORTED, row[:240], _name_only(name, said)))
        elif unit["source"] != said and not (unit["source"] == "condenser" and said == "condenser"):
            issues.append(Issue(CONTRADICTED, row[:240], f"gives {name} as {said}, but it is {unit['source']} "
                                                         f"({unit['basis']})"))

    # 1d. The answer to a question about one source, given by a unit whose name says that source but which is
    # another: "Where is the storage?" — "Battery Storage 3 is at Bus 5" (a CHP unit). The name answered.
    asked = [src for src, pat in _SOURCE_WORDS.items() if re.search(pat, question, re.IGNORECASE)] \
        if _ASKS_RE.search(question) else []
    # Read where the answer first names the unit, as its subject — not "the active power of Wind Park West".
    flagged = {i.text for i in issues}
    for src in asked:
        for name, unit in units.items():
            if unit["basis"] in ("unknown", NAMED) or unit["source"] == src or \
                    not re.search(_SOURCE_WORDS[src], name, re.IGNORECASE):
                continue
            first = next((raw for raw in _sentences(answer) if _mentions(_clean(raw), name)), None)
            if first is None or first[:240] in flagged:
                continue
            sentence = _clean(first)
            at = _mentions(sentence, name)[0]
            if _HELD_RE.search(sentence) or _DENIED_RE.search(sentence) \
                    or re.search(r"\b(?:of|for|from|by|to|with|than)\s+(?:the\s+)?$", sentence[:at], re.IGNORECASE) \
                    or re.search(_SOURCE_WORDS.get(unit["source"], r"(?!)"), masked(sentence), re.IGNORECASE):
                continue
            issues.append(Issue(CONTRADICTED, first[:240], f"answers a question about {src} with {name}, whose "
                                                            f"name says {src}, but it is {unit['source']} "
                                                            f"({unit['basis']})"))

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
    # Each problem once. Two different ones in one sentence are both kept: one per sentence and kind hid an
    # invented "Bus 64" behind the sentence's other finding (error injection).
    seen, out = set(), []
    for i in issues:
        if (i.kind, i.text, i.note) not in seen:
            seen.add((i.kind, i.text, i.note))
            out.append(i)
    return out


def _name_only(name: str, said: str) -> str:
    return (f"says {name} is {said}, but only its name says so: the network does not state it (say it is read from "
            "the name; the operator can declare it)")


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def _joined_lists(answer: str) -> list[str]:
    """Bulleted lists with the line that heads them, as one sentence each: "There are 10 solar units:" over
    "- PV 3, …" "- Solar Farm North" reads "There are 10 solar units: PV 3, …, Solar Farm North" — a list under
    a source, which line by line was never one (found in the CIM layer paper's records)."""
    out, head, items = [], None, []
    for line in (answer or "").splitlines() + [""]:
        bullet = _BULLET_RE.match(line)
        if bullet and head is not None:
            items.append(_clean(bullet.group(1)).strip().rstrip(".;,"))
            continue
        if not line.strip() and head is not None and not items:
            continue                                   # a blank line between the head and its list
        if head is not None and items:
            out.append(head + " " + ", ".join(items) + ".")
        head, items = None, []
        stripped = _clean(line).strip()
        # Only a head that lists one source ("There are 10 solar units:"): under "15 units:", each bullet is its own
        # claim ("- 8 solar units: …", "- 1 wind unit: WKA 7"), read as a sentence of its own.
        if stripped.endswith(":") and _LIST_HEAD_RE.search(stripped):
            head = stripped
    return out


def _cells(line: str) -> list[str]:
    return [_clean(c).strip() for c in line.strip().strip("|").split("|")]


def _table_claims(answer: str, units: dict) -> list[tuple[str, str, str]]:
    """(row, unit, source said) for each markdown table row that names a unit in one cell and gives a source in
    a column headed source, type, technology, kind or fuel."""
    out, header, source_cols = [], None, []
    for line in (answer or "").splitlines():
        if not line.strip().startswith("|"):
            header, source_cols = None, []
            continue
        if re.fullmatch(r"\s*\|[\s:|\-]+\|\s*", line):
            continue                                   # the rule under the header
        cells = _cells(line)
        if header is None:
            header = cells
            source_cols = [i for i, c in enumerate(cells) if _SOURCE_COLUMN_RE.search(c)]
            continue
        names = [c for c in cells if c in units]
        if not names or not source_cols:
            continue
        for i in source_cols:
            if i >= len(cells):
                continue
            said = next((src for src, pat in _SOURCE_WORDS.items()
                         if re.fullmatch(r"(?:" + pat + r")(?:\s*\([^)]*\))?", cells[i], re.IGNORECASE)), None)
            if said:
                out.append((line.strip(), names[0], said))
                break
    return out


def as_conflicts(answer: str, record: dict) -> list:
    """The issues as characterisation conflicts (path `entities`), so the
    answer review, the audit panel and the advisor show them with the others."""
    from .characterisation import Conflict
    return [Conflict(0.0, None, i.text, "entities", i.kind, i.note, study=True)
            for i in check_answer(answer, record)]
