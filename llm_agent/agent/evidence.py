"""
evidence.py — what backs each claim in an answer, and claims worded beyond it.

An answer mixes statements of very different standing: "no dispatch of the
units can clear these violations" can be a mathematical certificate, "L0 is at
114.7 %" an AC power flow of the network, "the cheapest option costs 26.48 €/h"
a local optimum, "a 3 % chance of a violation" a statistic over sampled
scenarios, "at power factor 0.95" a setting the agent chose, and "probably the
evening peak" the model's own reading. Written as prose, all of them sound
equally sure. This module gives each claim its evidence level, from the turn
record alone:

  proved       a certificate inside a result: a convex relaxation's lower bound
               on a cost, or its proof that no dispatch exists (relaxation.py)
  power flow   computed by an AC power flow of the network — physics, for the
               data given (an assessment, an N-1 screen, an option's check)
  optimised    found by the optimiser: a local optimum (the cheapest option
               carries a bound on how far it can be from the best)
  estimated    a statistic over sampled scenarios (Monte Carlo)
  input        read back: the question, the network's data, the measurements,
               or a setting the study ran with (assumed when the operator did
               not give it)
  derived      arithmetic in the answer on figures it quotes
  unsupported  in no result of this turn

Figures are graded where provenance traced them (provenance.py: the path of the
value they match); the tool's kind is its row in `tool_registry`, refined by
where in the result the value sits — a bound inside an options result is a
proof, an option's power-flow check is a power flow, an echoed setting is an
input. Sentences without figures are graded by what they assert: a proof
("proved", "impossible", "best possible"), safety ("secure", "no
violations"), the model's interpretation ("because", "probably") or advice
("should", "consider").

An assertion of proof with no certificate behind it, or of safety with no power
flow behind it, is **overstated** — the one finding this module raises. It is
the failure that makes an answer more dangerous than no answer: the solver
found no dispatch, and the prose says none exists.

Like the other checks it describes, never certifies: a figure graded "power
flow" is the power flow's figure, which is only as right as the data and the
network model; the levels say what kind of evidence stands behind a claim, not
that the claim is true.
"""

from __future__ import annotations

import html
import re
from typing import Any, Iterable, NamedTuple

from . import tool_registry as reg
from .provenance import (
    _NUMBER_RE,
    NK_RE,
    ELEMENT_NO_RE,
    AGENT_WRITTEN,
    DERIVED as _DERIVED_STATUS,
    IMPRECISE,
    MISATTRIBUTED,
    UNGROUNDED,
    _strip_noise,
    _tolerance,
    check_turn as _check_provenance,
    collect_sources,
    without_own_notes,
)

PROVED = reg.PROVED
POWER_FLOW = reg.POWER_FLOW
OPTIMISED = reg.OPTIMISED
ESTIMATED = reg.ESTIMATED
INPUT = reg.INPUT
DERIVED = "derived"
UNSUPPORTED = "unsupported"
INTERPRETATION = "interpretation"     # the model's reading: no study asserts it
ADVICE = "advice"                     # a recommendation: the operator's call

# Strongest first; the order the answer's summary is shown in.
LEVELS = (PROVED, POWER_FLOW, OPTIMISED, ESTIMATED, INPUT, DERIVED, INTERPRETATION, ADVICE, UNSUPPORTED)

LABELS = {
    PROVED: "Proved",
    POWER_FLOW: "Power flow",
    OPTIMISED: "Optimised",
    ESTIMATED: "Estimated",
    INPUT: "Input",
    DERIVED: "Derived",
    INTERPRETATION: "Interpretation",
    ADVICE: "Advice",
    UNSUPPORTED: "Unsupported",
}

MEANINGS = {
    PROVED: "A certificate from a convex relaxation of the same problem: holds for every possible dispatch.",
    POWER_FLOW: "Computed by an AC power flow of the network: physics, for the data and network model given.",
    OPTIMISED: "Found by the optimiser: a good solution (a local optimum), not proved to be the best.",
    ESTIMATED: "A statistic over sampled scenarios: an estimate with sampling error.",
    INPUT: "Read back: your question, the network's data, the measurements or a setting the study used.",
    DERIVED: "Computed in the answer from figures it quotes.",
    INTERPRETATION: "The agent's own reading of the results: no study states it.",
    ADVICE: "A recommendation: weigh it yourself.",
    UNSUPPORTED: "Found in no result of this turn.",
}


class ClaimEvidence(NamedTuple):
    """One claim in the answer and what backs it."""

    kind: str               # "figure" | "assertion"
    text: str               # the figure as written, or the sentence
    level: str
    source: str             # in words: "Security assessment (RSA) · violations · value"
    context: str            # the phrase around it
    position: int           # in the cleaned answer (answer order)
    note: str = ""          # "assumed", "on forecast data", why it is overstated …
    overstated: bool = False


class Ledger(NamedTuple):
    claims: tuple[ClaimEvidence, ...]

    @property
    def figures(self) -> tuple[ClaimEvidence, ...]:
        return tuple(c for c in self.claims if c.kind == "figure")

    @property
    def assertions(self) -> tuple[ClaimEvidence, ...]:
        return tuple(c for c in self.claims if c.kind == "assertion")

    @property
    def overstated(self) -> tuple[ClaimEvidence, ...]:
        return tuple(c for c in self.claims if c.overstated)

    @property
    def assumed(self) -> tuple[ClaimEvidence, ...]:
        return tuple(c for c in self.claims if c.note.startswith("assumed"))

    def counts(self) -> dict[str, int]:
        """Claims per level, strongest first, levels with none left out. An
        overstated assertion counts as unsupported: that is its standing."""
        out = {level: 0 for level in LEVELS}
        for claim in self.claims:
            out[UNSUPPORTED if claim.overstated else claim.level] += 1
        return {level: n for level, n in out.items() if n}

    def headline(self) -> str:
        """"2 proved · 6 power flow · 3 optimised · ⚠ 1 overstated"."""
        parts = [f"{n} {LABELS[level].lower()}" for level, n in self.counts().items()]
        if self.overstated:
            parts.append(f"⚠ {len(self.overstated)} overstated")
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_TOOL_RE = re.compile(r"^([a-z_]+)")
# Inside a result: values that are proofs, power flows, or settings read back,
# whatever the tool. First match wins.
_PATH_RULES = (
    (re.compile(r"optimality\.(lower_bound_eur_per_h|gap_eur_per_h)"), PROVED, "bound from a convex relaxation"),
    (re.compile(r"power_flow_check|\.n1\.screen|\.schedule\.|n1_outages_with_violations"), POWER_FLOW,
     "checked by an AC power flow"),
    (re.compile(r"p_any|probabilit|percentile|histogram|expected_violations"), ESTIMATED,
     "over sampled scenarios"),
    # An investigation's classical views: a linear approximation, and an
    # allocation convention — beside the exact shares, never in their place.
    (re.compile(r"classical\.linear"), ESTIMATED, "a linear estimate from sensitivities"),
    (re.compile(r"load_power_factor|thresholds?_used|_used\b|slack_max_mw|current_safety_margin|"
                r"costs\.prices|\.limits?\b|band"), INPUT, "a setting the study used"),
)


def _humanise(path: str) -> str:
    """"optimize_flexibility.options[1].cost_eur_per_h" → "options · cost eur per h"."""
    tail = re.sub(r"\[\d+\]", "", path.split(".", 1)[1]) if "." in path else ""
    parts = [p for p in tail.split(".") if p][-3:]
    return " · ".join(p.replace("_", " ") for p in parts)


def _results_by_tool(record: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for entry in record.get("tool_results") or []:
        if (isinstance(entry, dict) and isinstance(entry.get("result"), dict)
                and entry.get("name") not in AGENT_WRITTEN):
            out.setdefault(str(entry.get("name")), []).append(entry["result"])
    return out


def _forecast_tools(record: dict) -> set[str]:
    """Tools whose every call in the turn ran on the forecast series."""
    seen: dict[str, set] = {}
    for call in record.get("tool_calls") or []:
        if isinstance(call, dict):
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            seen.setdefault(str(call.get("name")), set()).add(args.get("data_source") or "measurements")
    return {name for name, sources in seen.items() if sources == {"forecasts"}}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _prompt_numbers(record: dict) -> set[float]:
    return {float(m.group(0).replace(",", "")) for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", record.get("user") or "")
            if m.group(0).replace(",", "").replace(".", "", 1).isdigit()}


# Fields whose numbers are prose or labels: the same value elsewhere in the
# result, in a field of its own, is the one to grade by.
_PROSE_RE = re.compile(r"\.(message|how|meaning|note|notes|text|explanation|why|label|headline|summary)\b|"
                       r"\.(bus|index|idx|element_index|name|id|ref|set_id|unit|units|outage|element)$|_index$|_idx$|_name\b|\[\d+\]$")
# The unit a field's name implies, and what the answer must write beside it.
_FIELD_UNITS = ((re.compile(r"eur"), ("€", "eur")), (re.compile(r"_mvar|mvar"), ("mvar",)),
                (re.compile(r"_mw\b|_mw_|mwh"), ("mw",)), (re.compile(r"_pct|percent|loading"), ("%",)),
                (re.compile(r"_pu\b|vm_|voltage"), ("p.u", "pu")))
_STRENGTH = {PROVED: 0, POWER_FLOW: 1, OPTIMISED: 2, ESTIMATED: 3, INPUT: 4}


def _units_agree(path: str, claim) -> bool:
    """Whether the unit a field implies is the one written with the figure
    (beside it, or in the few characters before it: "€22.48/h")."""
    field = path.rsplit(".", 1)[-1].lower()
    written = f"{claim.unit or ''} {claim.context}".lower()
    for pattern, marks in _FIELD_UNITS:
        if pattern.search(field):
            return any(m in written for m in marks)
    return True


def _best_source(claim, path: str, sources: list) -> str:
    """The source to grade a traced figure by. Provenance keeps the first value
    that matches; the same number can also sit in a prose field (a result's
    message repeating its bound) or in a label (bus 4 for a gap of 4.00 €/h).
    Among the turn's values that match, fields of their own with a unit that
    agrees come first, then the strongest evidence."""
    tolerance = _tolerance(claim.decimals)
    matches = [src.path for src in sources if abs(src.value - claim.value) <= tolerance
               and not src.path.startswith(("prompt", "grid", "args:"))]
    clean = [p for p in matches if not _PROSE_RE.search(p) and _units_agree(p, claim)]
    if not clean:
        return path
    return min(clean, key=lambda p: (_STRENGTH.get(_path_level(p), 5), 0 if p == path else 1))


def _path_level(path: str) -> str:
    match = _TOOL_RE.match(path)
    tool = match.group(1) if match else ""
    for pattern, level, _note in _PATH_RULES:
        if pattern.search(path):
            return level
    return reg.evidence(tool)


def _figure_level(verdict, record: dict, proofs: set, forecast: set, given: set,
                  sources: list | None = None) -> tuple[str, str, str]:
    """(level, source in words, note) of one provenance verdict."""
    if verdict.status in (UNGROUNDED, MISATTRIBUTED):
        why = ("in no result of this turn" if verdict.status == UNGROUNDED
               else f"real, but it belongs to {verdict.belongs_to or 'another element'}")
        return UNSUPPORTED, "—", why
    if verdict.status == _DERIVED_STATUS:
        return DERIVED, verdict.source or "arithmetic on figures in the answer", ""
    path = verdict.source or ""
    if path == "prompt" or path.startswith("prompt"):
        return INPUT, "your question", ""
    if path.startswith("grid"):
        return INPUT, "the network's data", ""
    if path.startswith("args:"):
        tool = path[5:].split(".", 1)[0]
        assumed = verdict.claim.value not in given
        return (INPUT, f"{reg.label(tool)} · {_humanise(path[5:]) or 'argument'}",
                "assumed: a setting the agent chose" if assumed else "given in your question")
    if sources:
        path = _best_source(verdict.claim, path, sources)
    match = _TOOL_RE.match(path)
    tool = match.group(1) if match else ""
    source = f"{reg.label(tool)} · {_humanise(path)}".rstrip(" ·")
    if "no_dispatch_check" in path:
        return (PROVED if "no_dispatch" in proofs else reg.evidence(tool)), source, ""
    for pattern, level, note in _PATH_RULES:
        if pattern.search(path):
            if level == INPUT:                      # a setting the study ran with, echoed back
                note = ("given in your question" if verdict.claim.value in given
                        else "assumed: a setting the question did not give")
            return level, source, note
    level = reg.evidence(tool)
    note = "on forecast data" if tool in forecast and level in (POWER_FLOW, OPTIMISED, ESTIMATED) else ""
    return level, source, note


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

# A sentence ends at a full stop (or ! ?) before a space and a capital, or at a
# line end — never at a decimal point ("3.88 MW").
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])|\n+")
_CODE_RE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
# A negation earlier in the same clause hedges a claim: "not an N-1 screen or a
# guarantee of security" asserts no guarantee.
_NEGATION_RE = re.compile(r"\b(not|no|never|cannot|can't|isn't|wasn't|without|nor|unless|neither)\b[^,;:()]{0,80}$",
                          re.IGNORECASE)

# A proof: holds for every possible dispatch.
_PROOF_RE = re.compile(r"\b(prov(?:ed|en|es)|proof|guarantee[sd]?|certain(?:ly)?|definitely|impossible)\b",
                       re.IGNORECASE)
_IMPOSSIBLE_RE = re.compile(
    r"\b(no (?:feasible |possible )?(?:dispatch|redispatch|solution)(?: of the units)? (?:exists|is possible|can)|"
    r"cannot be (?:cleared|solved|secured|resolved|fixed) by (?:any )?(?:re)?dispatch|"
    r"there is no way to|impossible to)\b", re.IGNORECASE)
_BEST_RE = re.compile(r"\b(best possible|globally optimal|the (?:global )?optimum|no cheaper (?:dispatch|option|"
                      r"solution)|cannot be (?:done )?cheaper)\b", re.IGNORECASE)
# A verdict on the grid, either way: "secure", "no violations" — and "not
# secure", "insecure", which are claims too (a negation hedges a proof, not this).
_SAFE_RE = re.compile(r"\b(n-1 secure|secure|insecure|no (?:limit )?violations?|within (?:all |every )?(?:its )?limits|"
                      r"all limits (?:are )?(?:met|respected))\b", re.IGNORECASE)
_INTERPRET_RE = re.compile(r"\b(because|due to|likely|probably|suggests?|indicates?|presumably|it seems|appears to|"
                           r"most likely)\b", re.IGNORECASE)
_ADVICE_RE = re.compile(r"\b(i (?:would )?recommend|recommend(?:ed|s)?|you should|should consider|consider|"
                        r"i suggest|it is advisable)\b", re.IGNORECASE)


def _turn_facts(record: dict) -> tuple[set, str | None]:
    """(certificates the turn holds, the strongest study that ran: a power
    flow, an optimiser, a statistic over sampled scenarios — or None)."""
    proofs: set = set()
    studied: set = set()
    for name, results in _results_by_tool(record).items():
        for result in results:
            if "error" in result:
                continue
            if reg.evidence(name) in (POWER_FLOW, OPTIMISED, ESTIMATED):
                studied.add(reg.evidence(name))
            for node in _walk(result):
                check = node.get("no_dispatch_check")
                if isinstance(check, dict) and check.get("verdict") == "no_dispatch_exists":
                    proofs.add("no_dispatch")
                bound = node.get("optimality")
                if isinstance(bound, dict) and bound.get("lower_bound_eur_per_h") is not None:
                    proofs.add("bound")
                    if bound.get("best_possible"):
                        proofs.add("best_possible")
                # An option's power-flow check, or an N-1 screen inside the
                # result (the KPI study's): power flows ran.
                if isinstance(node.get("power_flow_check"), dict) or "n1_outages_with_violations" in node:
                    studied.add(POWER_FLOW)
    strongest = next((level for level in (POWER_FLOW, OPTIMISED, ESTIMATED) if level in studied), None)
    return proofs, strongest


def _hedged(sentence: str, start: int) -> bool:
    return bool(_NEGATION_RE.search(sentence[:start]))


def _assertion(sentence: str, proofs: set, studied: str | None) -> tuple[str, str, str, bool] | None:
    """(level, source, note, overstated) of one sentence, or None when it
    asserts nothing this grades."""
    for pattern, needs, what in ((_IMPOSSIBLE_RE, "no_dispatch", "that no dispatch exists"),
                                 (_BEST_RE, "best_possible", "that nothing is cheaper"),
                                 (_PROOF_RE, None, "a proof")):
        m = pattern.search(sentence)
        if not m or _hedged(sentence, m.start()):
            continue
        held = proofs if needs is None else proofs & {needs}
        if needs == "best_possible" and not held and "bound" in proofs and re.search(
                r"\bno (?:other )?dispatch costs less\b|\bnone saves more\b", sentence, re.IGNORECASE):
            held = {"bound"}                                     # the bound itself, quoted
        if held:
            return PROVED, "a certificate from the convex relaxation", "", False
        why = {"no_dispatch": "the solver found no dispatch, which does not show that none exists",
               "best_possible": "no bound shows that nothing is cheaper",
               None: "no certificate stands behind it in this turn"}[needs]
        return UNSUPPORTED, "—", f"asserts {what}, but {why}", True
    m = _SAFE_RE.search(sentence)
    if m:
        if studied:
            return studied, {POWER_FLOW: "an AC power flow of this turn", OPTIMISED: "the optimiser's solution",
                             ESTIMATED: "power flows over sampled scenarios"}[studied], "", False
        return UNSUPPORTED, "—", "gives a verdict on the grid's security, but no study ran in this turn", True
    if _ADVICE_RE.search(sentence):
        return ADVICE, "the agent", "", False
    if _INTERPRET_RE.search(sentence):
        return INTERPRETATION, "the agent", "", False
    return None


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def build(record: dict) -> Ledger:
    """Every claim of one turn's answer, graded. Pure: the turn record in."""
    answer = record.get("assistant") or ""
    proofs, studied = _turn_facts(record)
    forecast = _forecast_tools(record)
    given = _prompt_numbers(record)
    sources = collect_sources(record)
    claims: list[ClaimEvidence] = []
    for verdict in _check_provenance(record).verdicts:
        level, source, note = _figure_level(verdict, record, proofs, forecast, given, sources)
        if verdict.status == IMPRECISE and not note:
            note = "rounded differently in the text"
        shown = f"{verdict.claim.value:g}" + (f" {verdict.claim.unit}" if verdict.claim.unit else "")
        claims.append(ClaimEvidence("figure", shown, level, source, verdict.claim.context,
                                    verdict.claim.position, note))
    # Sentences of the answer as written (dates kept, so they read whole), less
    # code, CONDUCTOR's own notes, emphasis and table rules.
    prose = _CODE_RE.sub(" ", without_own_notes(answer)).replace("*", "")
    position = 0
    for raw in _SENTENCE_END_RE.split(prose):
        sentence = raw.strip(" |-#>\t")
        position += len(raw) + 1
        if len(sentence) < 12 or set(sentence) <= set("|-: "):
            continue
        graded = _assertion(sentence, proofs, studied)
        if graded is None:
            continue
        level, source, note, overstated = graded
        claims.append(ClaimEvidence("assertion", sentence, level, source, sentence, position, note, overstated))
    return Ledger(tuple(claims))


def check_answer(answer: str, record: dict) -> list[ClaimEvidence]:
    """The overstated claims of a draft answer (for reflection and the graders)."""
    return list(build({**record, "assistant": answer}).overstated)


def as_conflicts(answer: str, record: dict) -> list:
    """The overstated claims as characterisation conflicts (path `evidence`),
    so the answer review, the audit panel and the advisor show them with the
    other findings."""
    from .characterisation import Conflict
    return [Conflict(0.0, None, c.text[:240], "evidence", c.level, c.note, study=True)
            for c in check_answer(answer, record)]


def format_overstated(claims: Iterable[ClaimEvidence]) -> str:
    """Overstatements in the wording the agent is shown (reflection)."""
    lines = [f"- “{c.text[:160]}” {c.note}." for c in claims]
    if not lines:
        return ""
    return ("Some sentences claim more certainty than the results give:\n" + "\n".join(lines)
            + "\nWord each claim by what backs it: proved only with a certificate in the results "
              "(`optimality`, `no_dispatch_check`), \"the power flow shows\" for assessments, \"the "
              "optimiser found\" for dispatches.")


class Group(NamedTuple):
    """Figures of one evidence level from one study, for display."""

    level: str
    study: str                       # "Flexibility OPF (deterministic)", "your question"
    fields: tuple[str, ...]          # what they are: "cost eur per h", "lower bound eur per h"
    figures: tuple[ClaimEvidence, ...]


def groups(ledger: Ledger) -> list[Group]:
    """The ledger's figures grouped by level and study, strongest first (the
    unsupported last), each figure once. Inputs are grouped by their note too,
    so what the question gave and what was assumed are told apart."""
    by: dict[tuple[str, str, str], list[ClaimEvidence]] = {}
    for claim in ledger.figures:
        study = claim.source.split(" · ", 1)[0] if claim.source != "—" else "this turn's results"
        by.setdefault((claim.level, study, claim.note if claim.level == INPUT else ""), []).append(claim)
    out = []
    for (level, study, _note), claims in by.items():
        seen, unique = set(), []
        for c in claims:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)
        fields = tuple(dict.fromkeys(c.source.split(" · ", 1)[1] for c in unique if " · " in c.source))
        out.append(Group(level, study, fields, tuple(unique)))
    return sorted(out, key=lambda g: (LEVELS.index(g.level), -len(g.figures)))


# What the evidence levels cannot tell, for the card's limits.
BLIND_SPOT = ("An evidence level says what kind of evidence stands behind a claim, not that the claim is true: "
              "a power flow's figure is only as right as the data and the network model. Proofs and safety "
              "claims are recognised by their words.")

# ---------------------------------------------------------------------------
# The answer with its figures marked
# ---------------------------------------------------------------------------

def mark_figures(answer: str, ledger: Ledger) -> str:
    """The answer with each graded figure wrapped in a span of its level
    (`cd-ev cd-ev-<level>`, its evidence as the tooltip), for display. Figures
    inside code spans are left alone; text that cannot be matched back stays
    as it is."""
    cleaned = _strip_noise(answer or "")
    # A figure's position is in the answer without its dates and names, so it
    # is found again in the answer itself — as a number of its own, never part
    # of another (observed live: the "0" of a figure marked inside "[L30]"),
    # nor in code, nor in a criterion's name ("N-1").
    names = [m.span() for pattern in (NK_RE, ELEMENT_NO_RE) for m in pattern.finditer(answer or "")]
    out, cursor = [], 0
    for claim in ledger.figures:
        m = _NUMBER_RE.match(cleaned, claim.position)
        if not m:
            continue
        literal = m.group(0).rstrip()
        alone = re.compile(r"(?<![A-Za-z0-9_.,])" + re.escape(literal) + r"(?![A-Za-z0-9_]|\.\d|,\d{3})")
        found = -1
        for hit in alone.finditer(answer, cursor):
            at = hit.start()
            if answer[:at].count("`") % 2 or any(a <= at < b for a, b in names):
                continue
            found = at
            break
        if found == -1:
            continue
        tip = f"{LABELS[claim.level]} — {claim.source}" + (f" ({claim.note})" if claim.note else "")
        out.append(answer[cursor:found])
        out.append(f'<span class="cd-ev cd-ev-{claim.level}" title="{html.escape(tip, quote=True)}">'
                   f"{literal}</span>")
        cursor = found + len(literal)
    out.append(answer[cursor:])
    return "".join(out)
