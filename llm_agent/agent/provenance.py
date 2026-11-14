"""
provenance.py — does every number in the answer come from a tool result?

The third failure channel. Computation is guarded by construction (the agent
never computes engineering quantities itself) and parameterisation is guarded
by `validators.py`. Interpretation — the prose the model writes over the tool
output — has had no guard at all, and the submitted paper's claim that "every
number it returns comes from a validated solver" was asserted rather than
measured.

This measures it. Each numeric claim in the answer is matched against the
values the turn actually produced: tool results, the arguments the tools ran
with, and the numbers the user themself supplied. What matches is *grounded*;
what does not is reported with the nearest source value so it can be judged.

Scope, stated plainly:

  Catches   figures that appear in no tool output; figures carried stale from
            an earlier turn (the source set is turn-scoped); arithmetic the
            model performed itself.

  Misses    a correct number given a wrong label. The inverted-feasibility
            failure wrote "6.24 MW (feasible)" about a value the engine had
            flagged IMPOSSIBLE — 6.24 *was* in the result, so provenance
            cannot see it. That belongs to deterministic resolution in the
            engine, and the two guards are complementary rather than
            overlapping.

This is a measuring instrument, not a filter: nothing here modifies an answer.
Its findings are put to the model only in reflection's `inform` mode
(`reflection.py`); otherwise they are recorded and shown to the operator.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterable, NamedTuple

from .grid_context import LINKS_KEY

# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

GROUNDED = "grounded"
# Traceable to a source, but not to the precision the answer states — the
# model truncated where it should have rounded. Separated from `ungrounded`
# because the two demand different responses: one is a transcription slip, the
# other is a number that exists nowhere in the evidence.
IMPRECISE = "imprecise"
# Traceable, correctly rounded, and attached to the wrong subject. Observed
# live: asked why Alpha was violating, the answer quoted the 3.35 MW that
# clears *Oscar* — same source, different bus, and a movement that does
# not clear the bus under discussion. Flat grounding cannot see this; the
# figure is in the payload and correctly transcribed.
MISATTRIBUTED = "misattributed"
# Not in the evidence, but obtainable by one arithmetic step from figures that
# are, and shown as such in the prose: "176.7793 / 220.1206 ≈ 0.803101". The
# model still computed it — that remains worth counting, and the offline
# grader counts it — but it is a different act from producing a number out of
# nowhere, because the operands and the operation are both on the page and an
# operator can check them.
#
# Separated after a live run in which the check flagged a ratio *and* the
# division that produced it, so an agent asked to show its work was penalised
# for doing so. Left folded into `ungrounded` that pressure is real: the
# cheapest way to reduce findings becomes hiding the arithmetic.
DERIVED = "derived"
UNGROUNDED = "ungrounded"

# Units the agent actually writes. Captured alongside the value because a
# figure matching a `_mvar` field but written as MW is a real defect that the
# bare number cannot reveal.
# Longest first: MWh before MW, MVAr before MVA. `%` and `p.u.` end in
# non-word characters, so neither can carry a trailing word boundary.
_UNIT_RE = re.compile(
    r"\s*(%|p\.u\.|pu\b|MWh\b|MVAr\b|MVA\b|MW\b|kW\b|kV\b|MVar\b)",
    re.IGNORECASE,
)

# A number that is not glued to a word (so `10kV` inside a bus name and the
# `L0` in a line label are not read as claims) and not part of a longer
# numeric literal.
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.,])"
    r"(-?\d{1,3}(?:,\d{3})+|-?\d+)"
    r"(?:\.(\d+))?"
    r"(?![A-Za-z0-9_])"
)

_SIGNED_CURRENCY_RE = re.compile(r"-\s?[€$£]\s?$")

# A security criterion's name: "N-1", "N-0", "N-1-1", "N–2". Its digits are a
# name, not a figure (observed live: the "0" of "an N-0 assessment" traced to
# "0 violations", and the "1" of "N-1" marked as an input).
# An element named by a kind word and a number — "Bus 7", "line 12",
# "transformer 1", "Gen 2": a name, not a figure. So is a branch named by its
# ends, and a range of elements — "Line 2-3", "Trafo 0-1" (CIGRE's names),
# "Bus 4–11" (observed live: the "-3" of "Line 2-3" flagged as a figure no
# result held). A unit after it ("line 95 %") makes it a figure again.
# A full stop or comma after the name ends the sentence, not the number: the
# guard against a decimal ("Bus 7.5") once refused "Line 14-8." whole and
# matched "Line 14" alone, leaving "-8" to be read as a figure (14 of 60
# untraced figures sampled in the CIM layer paper's runs were such fragments).
ELEMENT_NO_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:bus|line|transformer|trafo|gen|generator|unit|node|feeder|substation|outage)"
    r"\s?#?\s?\d+(?:\s?[-\u2010\u2011\u2013]\s?\d+)?(?!\d|[.,]\d|\s?[-\u2010\u2011\u2013]\s?\d)"
    r"(?!\s?(?:%|MW|Mvar|MVAr|MVA|kV|p\.?u\.?|€|EUR|h\b|min\b))", re.IGNORECASE)

NK_RE = re.compile(r"(?<![A-Za-z0-9_])[Nn]\s?[-\u2010\u2011\u2013]\s?\d+(?:\s?[-\u2010\u2011\u2013]\s?\d+)?"
                   r"(?!\d|\.\d)")

# An identifier with an underscore — a proposal's name, "gen_bus4_minus10_2000-01-06-20" — is a label, and so are
# its digits. Removing the date inside it left "-20" standing alone, read as a figure.
_IDENTIFIER_RE = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z0-9]*_(?:[A-Za-z0-9_]|[-\u2010\u2011](?=\w))*")

# An exponent in written mathematics: the "-1" of "\cos^{-1}" is not a figure.
_EXPONENT_RE = re.compile(r"\^\{?-?\d+\}?")

# Digit groups anywhere, used to mine numerals out of *strings* in tool
# results. Element names carry them ("Oscar 10kV", "Foxtrot Trf 1") and the
# answer quotes them back, so without this every such mention reads as
# fabricated.
_DIGITS_RE = re.compile(r"\d+(?:\.\d+)?")

# Removed before extraction: these are numerals that were never claims.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_TABLE_RULE_RE = re.compile(r"^\s*\|[\s:|\-]+\|\s*$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^(\s*)\d+\.(\s)", re.MULTILINE)
# The model also writes a date with non-breaking hyphens (U+2010, U+2011) and a
# no-break space before the time ("2000‑01‑01 00:00").
_DASH = "[-\u2010\u2011]"
_DATE_TIME_SEP = "[ T\u00a0\u202f]"
_DATETIME_RE = re.compile(
    rf"\d{{4}}{_DASH}\d{{2}}{_DASH}\d{{2}}(?:{_DATE_TIME_SEP}\d{{2}}:\d{{2}}(?::\d{{2}})?)?")

# Dates and times as prose writes them: "18:15", "18:15:00", "6 January",
# "January 6", "6 Jan", "06 Jan 2000". Their digits are no more a claim than an
# ISO timestamp's. Month names are matched capitalised only, so "6 may" in a
# sentence is left alone.
# A colon after the time is punctuation ("07:45–09:45: peaks"), not seconds.
_CLOCK_RE = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}(?::\d{2})?(?!\d|:\d)")
_MONTH = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
          r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?")
_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
# A span of days is a date too: "January 1–7, 2000" left "7" and "2000" as figures.
_DAYS = rf"{_DAY}(?:\s?[-‐‑–]\s?{_DAY})?"
_YEAR = r"(?:,?\s+\d{4}\b)?"
_DAY_MONTH_RE = re.compile(
    rf"(?<![\w.]){_DAYS}\s+(?:of\s+)?{_MONTH}{_YEAR}"
    rf"|\b{_MONTH}\s+{_DAYS}\b{_YEAR}"
)

# A timestamp inside a result or argument string, as the tools write it —
# "2000-01-09 18:15:00", the prefix "2000-01-09 18", an ISO form with
# fraction or offset. Its digits are not sources: a fabricated "9 violations"
# was grounded by the day of "2000-01-09 18:15:00", and "18 buses" by its hour.
_SOURCE_DATETIME_RE = re.compile(
    rf"\d{{4}}{_DASH}\d{{2}}{_DASH}\d{{2}}(?:{_DATE_TIME_SEP}\d{{2}}(?::\d{{2}}(?::\d{{2}}(?:\.\d+)?)?)?(?!\d))?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
)

# CONDUCTOR's own notes appended to an answer (loop.py): the step limit it
# stopped at, the call it stopped on, or what re-running the studies under
# other readings found (robustness.py). They are the system speaking, not the
# model, and their figures are settings or the re-runs' own, not claims.
_OWN_NOTE_RE = re.compile(
    r"CONDUCTOR (?:used all \d+ steps allowed for one question|stopped this question:|"
    r"re-ran the studies under readings)[^\n]*")


def without_own_notes(text: str) -> str:
    """The answer without the notes CONDUCTOR itself appended to it."""
    return _OWN_NOTE_RE.sub(" ", text or "")


def _without_dates(text: str) -> str:
    """Text with its dates and clock times removed, ISO or written out."""
    text = _SOURCE_DATETIME_RE.sub(" ", text)
    text = _DAY_MONTH_RE.sub(" ", text)
    return _CLOCK_RE.sub(" ", text)


# Words that state which way power is moving, or which way a value changed.
# Where the prose names a direction, the sign is carried by the word and the
# magnitude by the number,
# so a value may legitimately appear negated: the payload reports
# `slack_import_mw = -1.3146` and the answer says "exporting 1.31 MW". Both
# describe the same flow, and reading only the signed value called the second
# one fabricated.
_DIRECTION_RE = re.compile(
    r"\b(export|exports|exporting|import|imports|importing|inject|injects|"
    r"injecting|injection|absorb|absorbs|absorbing|draw|draws|drawing|"
    r"supply|supplies|supplying|consume|consumes|consuming|deliver|delivers|"
    r"delivering|withdraw|withdraws|withdrawing|feed|feeds|feeding|"
    # A change said as a fall: "fell by 0.00165 p.u." over `change: -0.00165`.
    r"fell|fall|falls|fallen|dropped|drop|drops|declined|declines|decreased|decrease|decreases|"
    r"lowered|reduced|down by|"
    # The same, said of an action or a comparison: "reduce Gen_bus12 5.66 MW"
    # over `dp_mw: -5.66`, "16.42 MW lower than in the baseline".
    r"reduce|reduces|reducing|lower|lowers|less|"
    # A flow said against its sign: "carries 0.52 MW back toward Bus 1" over `p_from_mw: -0.5233`.
    r"back|reverse|reversed)\b",
    re.IGNORECASE,
)


class Claim(NamedTuple):
    """A number as the answer states it."""

    value: float
    decimals: int
    unit: str | None
    context: str
    position: int = 0

    def render(self) -> str:
        shown = f"{self.value:g}" + (f" {self.unit}" if self.unit else "")
        return f"{shown} — “{self.context}”"


class ClaimVerdict(NamedTuple):
    claim: Claim
    status: str
    source: str | None = None
    nearest: tuple[float, str] | None = None
    subject: str | None = None
    belongs_to: str | None = None
    inverted: bool = False

    def render(self) -> str:
        if self.status == MISATTRIBUTED:
            return (f"{self.claim.render()}  ← {self.source}, which belongs to "
                    f"{self.belongs_to or 'another element'}")
        if self.status in (GROUNDED, IMPRECISE):
            sign = " (sign read from the wording)" if self.inverted else ""
            return f"{self.claim.render()}  ← {self.source}{sign}"
        near = ""
        if self.nearest is not None:
            near = f"  (nearest source {self.nearest[0]:g} at {self.nearest[1]})"
        return f"{self.claim.render()}{near}"


class ProvenanceReport(NamedTuple):
    verdicts: tuple[ClaimVerdict, ...]
    n_sources: int

    @property
    def grounded(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == GROUNDED]

    @property
    def imprecise(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == IMPRECISE]

    @property
    def misattributed(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == MISATTRIBUTED]

    @property
    def derived(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == DERIVED]

    @property
    def ungrounded(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == UNGROUNDED]

    @property
    def rate(self) -> float:
        """Share of numeric claims traceable to a source. 1.0 if there are none.

        Imprecise claims count as traceable: the figure came from the
        evidence, it was just transcribed at the wrong precision.

        Derived claims do not. The model computed them, which is the thing
        this measures, and folding them in would let a turn raise its own
        score by showing more arithmetic. They are reported separately by
        `derived_rate` instead.
        """
        if not self.verdicts:
            return 1.0
        return (len(self.grounded) + len(self.imprecise)) / len(self.verdicts)

    @property
    def derived_rate(self) -> float:
        """Share of claims the model computed from evidence in view.

        Reported alongside `rate` rather than inside it: the interesting
        quantity for the paper is how much of an answer is arithmetic the
        tools did not do, and that is exactly what gets lost if it is either
        counted as grounded or counted as invention.
        """
        if not self.verdicts:
            return 0.0
        return len(self.derived) / len(self.verdicts)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _strip_noise(text: str, keep_names: bool = False) -> str:
    """Remove the parts of the markdown whose digits are never claims.

    Element numbers ("Bus 11", "Line 2-3", "N-1") are blanked with spaces of the same length, so a text cleaned
    with `keep_names` — the same, with the names left in — has every position in the same place: claims are read
    from the one, the element each is about from the other. (Blanking the names outright made every name written
    with a space invisible to the misattribution check — "Bus 11 had the lowest voltage: 0.944" had no subject;
    found by error injection, evaluation/injection.py.)"""
    # The model writes the typographic minus (U+2212), not a hyphen. Left
    # alone, "−1.35 MW" parses as +1.35 and a correctly reported export reads
    # as fabricated. Only U+2212 is converted: the en-dash it resembles is a
    # range separator here ("0.95–1.045") and must not become a sign.
    text = text.replace("\u2212", "-")
    text = without_own_notes(text)
    text = _FENCED_CODE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _TABLE_RULE_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub(r"\1\2", text)
    text = _EXPONENT_RE.sub(lambda m: " " * len(m.group(0)), text)
    if not keep_names:
        text = _IDENTIFIER_RE.sub(lambda m: " " * len(m.group(0)), text)
    text = _DATETIME_RE.sub(" ", text)
    text = _DAY_MONTH_RE.sub(" ", text)
    text = _CLOCK_RE.sub(" ", text)
    if not keep_names:
        text = NK_RE.sub(lambda m: " " * len(m.group(0)), text)
        text = ELEMENT_NO_RE.sub(lambda m: " " * len(m.group(0)), text)
    # Emphasis markers sit between a value and its unit ("**3.35 MW**").
    return text.replace("*", "")


def _context(text: str, start: int, end: int, width: int = 55) -> str:
    snippet = text[max(0, start - width):min(len(text), end + width)]
    return " ".join(snippet.split())


def extract_claims(answer: str) -> list[Claim]:
    """Every number the answer asserts, with its unit and surrounding phrase."""
    if not answer:
        return []

    cleaned = _strip_noise(answer)
    claims: list[Claim] = []

    for match in _NUMBER_RE.finditer(cleaned):
        integer, fraction = match.group(1), match.group(2)
        try:
            value = float(integer.replace(",", "") + ("." + fraction if fraction else ""))
        except ValueError:  # pragma: no cover — regex guarantees the shape
            continue
        # A sign written before the currency symbol: "-€24.27/h" is a saving
        # of 24.27, not a cost (seen live, flagged as invented).
        if value > 0 and _SIGNED_CURRENCY_RE.search(cleaned, max(0, match.start() - 4), match.start()):
            value = -value

        unit_match = _UNIT_RE.match(cleaned, match.end())
        unit = unit_match.group(1).strip() if unit_match else None

        claims.append(Claim(
            value=value,
            decimals=len(fraction) if fraction else 0,
            unit=unit or None,
            context=_context(cleaned, match.start(), match.end()),
            position=match.start(),
        ))

    return claims


# ---------------------------------------------------------------------------
# Sources
#
# A number is not just a value and a path: it belongs to something. The tool
# payloads are already partitioned by subject — a violation carries its
# element name, a driver its source name, a per-bus dict its bus names — and
# carrying that partition through is what lets a claim be checked against the
# thing it is talking about rather than against the turn as a whole.
# ---------------------------------------------------------------------------

# Keys whose string value names the subject of the subtree they sit in.
# `source` is included so a driver's numbers stay attributable to their unit,
# but only element keys seed the subject vocabulary below.
_ELEMENT_KEYS = ("element", "bus_name", "line_name", "trafo_name")
_NAME_KEYS = _ELEMENT_KEYS + ("source",)
# Keys whose string value is a label rather than a statement.
_NAME_SUFFIXES = ("name", "element", "source")


class Source(NamedTuple):
    """One number the turn had access to, and what it belongs to."""

    value: float
    path: str
    scopes: frozenset = frozenset()


def _scopes_of(source: Any) -> frozenset:
    """Scopes of a source, tolerating the plain (value, path) pairs in tests."""
    return getattr(source, "scopes", frozenset())


def _entity_name(node: dict) -> str | None:
    for key in _NAME_KEYS:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _walk(node: Any, path: str, out: list[Source],
          scopes: frozenset = frozenset(), vocabulary: frozenset = frozenset()) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(Source(float(node), path, scopes))
        return
    if isinstance(node, str):
        # Digits inside an identifier — "Oscar 10kV", "[L9]" — are real
        # sources, since the answer quotes those labels back, but they say
        # nothing about the element they name. Scoping them would blame a
        # figure on whichever label happened to contain the same digits.
        # Prose fields (`text`, `message`) keep their scope: the engine's
        # resolved sentence is genuinely about its own violation.
        keeps_scope = not path.rsplit(".", 1)[-1].endswith(_NAME_SUFFIXES)
        for hit in _DIGITS_RE.findall(_without_dates(node)):
            try:
                out.append(Source(float(hit), path, scopes if keeps_scope else frozenset()))
            except ValueError:  # pragma: no cover
                continue
        return
    if isinstance(node, dict):
        name = _entity_name(node)
        inner = scopes | {name} if name else scopes
        for key, value in node.items():
            # A study's links to the CIM layer are identifiers (mRIDs, a version digest): their digits are no
            # figure. What the links resolve to is a source of its own (`cim_context`, below).
            if key == LINKS_KEY:
                continue
            # A dict keyed by entity names — `bus_violation_probability`,
            # `voltage_percentiles` — partitions its values the same way a
            # list of violations does.
            keyed = inner | {key} if key in vocabulary else inner
            _walk(value, f"{path}.{key}" if path else str(key), out, keyed, vocabulary)
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, f"{path}[{index}]", out, scopes, vocabulary)


def _names_under(record: dict, keys: tuple[str, ...]) -> frozenset:
    found: set[str] = set()

    def scan(node: Any) -> None:
        if isinstance(node, dict):
            for key in keys:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    found.add(value.strip())
            for value in node.values():
                scan(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                scan(value)

    for entry in record.get("tool_results") or []:
        if isinstance(entry, dict):
            scan(entry.get("result"))
    return frozenset(found)


def element_names(record: dict) -> frozenset:
    """Every bus, line or transformer the turn named.

    Collected in a first pass because dictionaries keyed by these names carry
    no marker of their own — only knowing the vocabulary reveals that
    `voltage_percentiles` is partitioned by bus.
    """
    return _names_under(record, _ELEMENT_KEYS)


def subject_names(record: dict) -> frozenset:
    """Names the prose can be *about*: violated elements, and only those.

    Two narrowings, both forced by measurement on real logs.

    Admitting every named thing produced false attribution on nearly every
    figure. A parenthetical gloss hijacked the subject of the table row it sat
    in, and source names are actors in a sentence rather than its subject.

    Restricting to the `element` key was still not enough: the dispatch table
    names its rows `element` too, so the *generator* "Alpha" entered the
    vocabulary and shadowed the violated bus "Alpha 10.5 kV" that the
    sentence was actually about. Only elements inside a `violations` list
    qualify, and any name contained in a longer one is dropped as ambiguous —
    prose that says "Alpha" has not said which Alpha.
    """
    found: set[str] = set()

    def scan(node: Any) -> None:
        if isinstance(node, dict):
            violations = node.get("violations")
            if isinstance(violations, list):
                for item in violations:
                    if isinstance(item, dict):
                        name = item.get("element")
                        if isinstance(name, str) and name.strip():
                            found.add(name.strip())
            for value in node.values():
                scan(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                scan(value)

    for entry in record.get("tool_results") or []:
        if isinstance(entry, dict):
            scan(entry.get("result"))

    return frozenset(
        name for name in found
        if not any(other != name and name in other for other in found)
    )


# Tools whose content the agent itself wrote (the investigation record: its
# hypotheses and findings). Their numbers are the answer's own, restated, so
# they are never a source — or a figure could be "traced" to the agent's note
# about it.
AGENT_WRITTEN = frozenset({"record_investigation"})


def collect_sources(record: dict) -> list[Source]:
    """Every number the turn had legitimate access to, with what it belongs to.

    Three origins, all of them fair: what the tools returned, what they were
    called with, and what the user wrote. Omitting the last two would flag the
    user's own voltage limit, quoted back in the answer, as an invention.
    """
    sources: list[Source] = []
    vocabulary = element_names(record)

    for entry in record.get("tool_results") or []:
        # A malformed entry costs its own sources, not the whole grading pass.
        if not isinstance(entry, dict):
            continue
        name = entry.get("name", "tool")
        if name in AGENT_WRITTEN:
            continue
        _walk(entry.get("result"), name, sources, frozenset(), vocabulary)

    # What the conversation's previous turns returned: the model has them in its history and a follow-up quotes
    # them. Turn-level, like the arguments: they are not this turn's subject partition.
    for entry in record.get("earlier_results") or []:
        if isinstance(entry, dict) and entry.get("name") not in AGENT_WRITTEN:
            _walk(entry.get("result"), f"earlier:{entry.get('name', 'tool')}", sources)

    # The grid facts the system prompt stated. A figure quoted from them —
    # how many buses the network has, its voltage limits — came from the
    # evidence the model was given, and reading only tool results called those
    # fabricated. Turn-level, since they describe the network, not an element.
    _walk(record.get("grid") or {}, "grid", sources)

    # What the CIM layer says about the elements the studies named, through their links (`grid_context`): the
    # model was told it with the result — a voltage level, a feeder position, the loads a branch cuts off.
    # Scoped to its element, like a result's own row.
    for facts in ((record.get("cim_context") or {}).get("elements") or {}).values():
        if isinstance(facts, dict) and facts.get("name"):
            names = frozenset({facts["name"], facts["name"].split(" | ")[0]})
            _walk(facts, f"cim layer:{facts['name']}", sources, names, vocabulary)

    # Arguments and the prompt belong to the turn, not to any one element.
    for call in record.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("name", "tool")
        if name in AGENT_WRITTEN:
            continue
        _walk(call.get("args"), f"args:{name}", sources)

    _walk(record.get("user") or "", "prompt", sources)
    sources += _bus_sums(record)
    return sources


# A figure the answer adds up from the rows at one bus: "loads at Bus 3 (0.23 MW)" over two loads there, 0.2143
# and 0.0153 MW. The commonest false alarm left in the CIM layer paper's labelled findings. Kept apart (`SUM_PREFIX`):
# a figure that matches only such a sum is `derived`, never `grounded`.
SUM_PREFIX = "sum:"
_GROUP_KEYS = ("bus", "bus_name")
# A list's total is read only for power: a total of maximum voltages or severities means nothing.
_POWER_FIELD_RE = re.compile(r"(?:^|_)(?:mw|mvar)(?:$|_)", re.IGNORECASE)


def _bus_sums(record: dict) -> list[Source]:
    """Per bus, the sum of each numeric field over the rows of a tool's list
    that sit at that bus — only where two rows or more do."""
    out: list[Source] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{path}.{key}")
        elif isinstance(node, list):
            rows = [r for r in node if isinstance(r, dict)]
            key = next((k for k in _GROUP_KEYS if rows and all(isinstance(r.get(k), str) for r in rows)), None)
            if len(rows) >= 2:
                # The whole list's total of a field, said as a total ("1.40 MW total", "plus").
                fields = {f for r in rows for f, v in r.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool) and not f.endswith("index")}
                for field in sorted(f for f in fields if _POWER_FIELD_RE.search(f)):
                    out.append(Source(sum(float(r.get(field) or 0.0) for r in rows),
                                      f"{SUM_PREFIX}{path}[*].{field}"))
            if key:
                groups: dict[str, list[dict]] = {}
                for r in rows:
                    groups.setdefault(r[key], []).append(r)
                for bus, members in groups.items():
                    if len(members) < 2:
                        continue
                    fields = {f for r in members for f, v in r.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool) and not f.endswith("index")}
                    for field in sorted(fields):
                        total = sum(float(r.get(field) or 0.0) for r in members)
                        out.append(Source(total, f"{SUM_PREFIX}{path}[{key}={bus}].{field}", frozenset({bus})))
            for index, value in enumerate(node):
                visit(value, f"{path}[{index}]")

    for entry in record.get("tool_results") or []:
        if isinstance(entry, dict) and entry.get("name") not in AGENT_WRITTEN:
            visit(entry.get("result"), entry.get("name", "tool"))
    return out


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _tolerance(decimals: int) -> float:
    """Half a unit in the last stated place.

    The answer rounds: the prose says 1.0456 where the solver returned
    1.0456116504256603. A claim is grounded when some source value rounds to
    it at the precision the claim itself was written to — which is the
    definition of "this number came from that one", under any rounding
    convention.
    """
    return 0.5 * (10.0 ** -decimals) + 1e-9


def check_claim(claim: Claim, sources: Iterable[Any], subject: str | None = None,
                vocabulary: frozenset | None = None,
                answer: str | None = None) -> ClaimVerdict:
    """Grade one claim against the turn's evidence.

    Two tolerance bands. Half a unit in the last place is a correct rounding.
    A full unit additionally admits truncation, which the model does: it wrote
    1.0466 for a solver value of 1.046666, where rounding gives 1.0467. That
    is worth seeing and is not worth confusing with a fabricated figure.

    `subject` is the violated element the surrounding prose is about. When it
    is known, a figure that matches only under some *other* violated element
    is misattributed rather than grounded — right number, wrong subject, which
    is the failure a flat check cannot see.

    `vocabulary` is the set of names that count as elements for that purpose.
    Only those partition the evidence. A dispatch row or a generator scope is
    treated as turn-level, because a sentence about a unit's new setpoint is
    not a claim about whichever bus the paragraph opened with — measured, that
    distinction removed every false positive on the logs to hand while keeping
    the real one.
    """
    exact = _tolerance(claim.decimals)
    # Truncation is only a thing a *decimal* can suffer. For a whole number
    # the band would reach the neighbouring integers, and a claim of "24
    # lines" then matched an unrelated bus index of 23 and was excused as a
    # transcription slip. Whole numbers get the rounding band and nothing
    # more: 24 is either in the evidence or the model produced it.
    loose = (10.0 ** -claim.decimals + 1e-9) if claim.decimals else exact

    # A percentage in the prose is routinely a probability in the result: the
    # risk tools return p_any_violation_after = 0.49 and the answer says
    # "49%". The tolerance scales with the value, so precision is preserved.
    targets = [(claim.value, 1.0, False)]
    if claim.unit == "%":
        targets.append((claim.value / 100.0, 100.0, False))
    # Only where the surrounding words state a direction. Accepting a negated
    # match everywhere would ground any fabricated figure that happened to
    # equal some value's opposite.
    if _DIRECTION_RE.search(claim.context):
        targets.append((-claim.value, 1.0, True))
        if claim.unit == "%":
            targets.append((-claim.value / 100.0, 100.0, True))

    matches: list[tuple[Any, bool]] = []
    truncated: Any = None
    nearest: tuple[float, str] | None = None
    smallest = float("inf")

    for source in sources:
        value, path = source[0], source[1]
        if str(path).startswith(SUM_PREFIX):
            continue                                  # read by `_sum_steps`, where the words say it is a sum
        for target, scale, inverted in targets:
            difference = abs(value - target)
            if difference <= exact / scale:
                matches.append((source, inverted))
                break
            if truncated is None and difference <= loose / scale:
                truncated = source
        difference = abs(value - claim.value)
        if difference < smallest:
            smallest, nearest = difference, (value, path)

    if matches:
        if subject:
            def element_scope(pair) -> frozenset:
                scopes = _scopes_of(pair[0])
                return scopes if vocabulary is None else scopes & vocabulary

            on_subject = [m for m in matches if subject in element_scope(m)]
            if on_subject:
                return ClaimVerdict(claim, GROUNDED, source=on_subject[0][0][1],
                                    subject=subject, inverted=on_subject[0][1])
            # Totals, thresholds, dispatch rows and the turn's own arguments
            # belong to no violated element; quoting them under any subject is
            # fine.
            unscoped = [m for m in matches if not element_scope(m)]
            if unscoped:
                return ClaimVerdict(claim, GROUNDED, source=unscoped[0][0][1],
                                    subject=subject, inverted=unscoped[0][1])
            elsewhere = sorted(element_scope(matches[0]))
            return ClaimVerdict(
                claim, MISATTRIBUTED, source=matches[0][0][1], subject=subject,
                belongs_to=", ".join(elsewhere) or None, inverted=matches[0][1],
            )
        return ClaimVerdict(claim, GROUNDED, source=matches[0][0][1],
                            inverted=matches[0][1])

    if truncated is not None:
        return ClaimVerdict(claim, IMPRECISE, source=truncated[1], nearest=nearest,
                            subject=subject)

    # The whole answer, not the claim's own window: a percentage is routinely
    # stated a sentence after the division that produced it, and the 55-char
    # context is cut off well before it.
    derived = _derivation(claim, sources, answer) or _share_of(claim, sources, answer)
    if derived is not None:
        return ClaimVerdict(claim, DERIVED, source=derived, nearest=nearest,
                            subject=subject)

    return ClaimVerdict(claim, UNGROUNDED, nearest=nearest, subject=subject)


def is_plausible_nearest(nearest: tuple[float, str] | None, value: float) -> bool:
    """Is the nearest source close enough to be worth showing as a suggestion.

    The verdict always carries `nearest`: for the offline grader it is
    diagnostic context, and the distance is part of what a reader is judging —
    a claim of 6.24 MW against an available 3.54 MW is exactly the sort of gap
    worth seeing.

    Presenting it *to the agent* is a different act. There it reads as "did you
    mean this?", and it only means that when the two figures could be
    transcriptions of each other. Reported live: a claim of 0.803101 was
    offered `0.94` — the network's lower voltage limit — which is not a
    suggestion anyone should act on, and a wrong pointer is worse than none
    when the agent is being asked to judge.

    5% of the claim, roughly the reach of a mistyped or transposed digit.
    """
    if nearest is None:
        return False
    return abs(nearest[0] - value) <= max(abs(value), 1.0) * 0.05


# Two numbers with a division between them, as the model actually writes it.
#
# Symbolic: plain `176.7793 / 220.1206`, LaTeX `\frac{176.7793}{220.1206}` and
# `\div`, and the Unicode division sign `÷`.
#
# Prose: "dividing 176.7793 by 220.1206", "176.7793 divided by 220.1206". These
# were missed by the symbolic form alone, and the omission mattered more than
# it looks: an answer that explained its arithmetic in words was scored as
# having invented the result, while the same arithmetic written with a slash
# was not. That is the transparency penalty `DERIVED` exists to remove,
# reappearing one level down.
#
# The connecting text is bounded and must not contain a digit, so
# "12.4 MW divided between Alpha and Bravo, leaving 3.1 MW" cannot pair 12.4
# with 3.1 across an unrelated clause.
_DIVISION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:/|÷|\\div|}\s*\{)\s*(\d+(?:\.\d+)?)"
    r"|(?:divid\w+)\s+(\d+(?:\.\d+)?)\s*(?:[^\d\n]{0,12}?\bby)\s*(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*(?:[^\d\n]{0,12}?\bdivided by)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _derivation(claim: Claim, sources: Iterable[Any],
                answer: str | None = None) -> str | None:
    """The shown division that produces this claim, if both operands are evidence.

    Deliberately narrow. Only division, only operands the answer writes out,
    and only when each of them is itself traceable. That covers the case this
    exists for — a share or percentage the tools do not return — without
    becoming a licence to accept any number that happens to be expressible as
    a ratio of two others in the payload.

    Returns a description of the derivation, or None.
    """
    text = answer if answer is not None else (claim.context or "")
    values = [(s[0], s[1]) for s in sources]

    def trace(operand: float) -> str | None:
        for value, path in values:
            if abs(value - operand) <= max(abs(operand), 1.0) * 1e-6:
                return path
        return None

    for groups in _DIVISION_RE.findall(text):
        # One alternative matched; the rest are empty. Take the populated pair.
        pairs = [(groups[i], groups[i + 1]) for i in range(0, len(groups), 2)]
        numerator, denominator = next(
            ((a, b) for a, b in pairs if a and b), ("", ""))
        if not numerator:
            continue
        try:
            top, bottom = float(numerator), float(denominator)
        except ValueError:
            continue
        if bottom == 0:
            continue

        top_path, bottom_path = trace(top), trace(bottom)
        if not (top_path and bottom_path):
            continue

        quotient = top / bottom
        # Two units in the last place. The model both truncates its own
        # arithmetic and rounds its operands before dividing, so the quotient
        # it writes can sit a little further from the exact one than a
        # transcription band allows: 0.803101 was written for 0.8031020, which
        # a one-unit band misses by a hair. The looseness is safe here because
        # a derivation must already have matched both operands exactly.
        band = 2 * 10.0 ** -claim.decimals + 1e-9 if claim.decimals else _tolerance(0)
        # The claim is the quotient itself, or the quotient as a percentage.
        for candidate in (quotient, quotient * 100.0):
            if abs(candidate - claim.value) <= band:
                return f"{top_path} / {bottom_path}"
    return None


# "190 of 200 samples": a count written as a share of a total.
_SHARE_OF_RE = re.compile(r"(\d[\d,]*)\s+(?:of|out of)\s+(\d[\d,]*)\b")
_PERCENT_NEAR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%|\b(0\.\d+)\b")


def _share_of(claim: Claim, sources: Iterable[Any], answer: str | None) -> str | None:
    """The count `claim` as a written share of a written total: "95 % (190 of
    200 samples)" — 190 is 0.95 × 200. As narrow as `_derivation`: the claim is
    the K of "K of N", N traces to a result, and K/N equals a share written
    within the same stretch of text that also traces (0.95, or 95 %)."""
    if claim.decimals or not answer:
        return None
    values = [(s[0], s[1]) for s in sources]

    def trace(value: float, tol: float) -> str | None:
        return next((path for v, path in values if abs(v - value) <= tol), None)

    for m in _SHARE_OF_RE.finditer(answer):
        try:
            k, n = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        if n <= 0 or abs(k - claim.value) > 1e-9:
            continue
        n_path = trace(n, 1e-9)
        if not n_path:
            continue
        window = answer[max(0, m.start() - 80):m.end() + 80]
        for p in _PERCENT_NEAR_RE.finditer(window):
            share = float(p.group(1)) / 100.0 if p.group(1) else float(p.group(2))
            if abs(share - k / n) > 0.5 / n + 1e-9:
                continue
            share_path = trace(share, 0.5 / n) or trace(share * 100.0, 50.0 / n)
            if share_path:
                return f"{share_path} × {n_path}"
    return None


# ---------------------------------------------------------------------------
# Subjects — what the prose around a figure is talking about
# ---------------------------------------------------------------------------

# A heading or a rule ends the current subject. Without this the last element
# named in section 1 would still be the subject halfway through section 3.
# So does a paragraph break: "…the assessment also found Bus_29 undervoltage at
# 0.943 p.u." closed a paragraph, and the next three, about the overloaded line,
# had all thirteen of their figures read as written about Bus_29 (the CIM layer paper's runs).
_SECTION_BREAK_RE = re.compile(r"^(#{1,6}\s|-{3,}\s*$|_{3,}\s*$|[ \t]*$)", re.MULTILINE)


def _section_start(cleaned: str, position: int) -> int:
    last = 0
    for match in _SECTION_BREAK_RE.finditer(cleaned):
        if match.start() > position:
            break
        last = match.start()
    return last


def _cell_bounds(line: str, column: int) -> tuple[int, int] | None:
    """Offsets (within `line`) of the `column`-th cell of a markdown table row."""
    bars = [i for i, ch in enumerate(line) if ch == "|"]
    if len(bars) < column + 2:
        return None
    return bars[column] + 1, bars[column + 1]


def _table_subject(cleaned: str, position: int, occurrences: list[tuple[int, str]]) -> tuple[bool, str | None]:
    """For a figure in a markdown table cell: (True, the element its column's
    header names — else the one its row's first cell names — else None).
    (False, None) outside a table.

    Prose's rule (the element named last before the figure) reads a table
    wrongly: in "| | Bus_29 voltage | L0 loading |" over "| Forecast | 0.939
    p.u. | 121.4 % |", the last name before 0.939 is L0, from the header's last
    column, and a correct voltage was reported as belonging to the line."""
    start = cleaned.rfind("\n", 0, position) + 1
    end = cleaned.find("\n", position)
    end = len(cleaned) if end == -1 else end
    line = cleaned[start:end]
    if not line.lstrip().startswith("|"):
        return False, None
    column = line[:position - start].count("|") - 1
    # The table's first row: walk up through its rows and its rule row (which
    # `_strip_noise` blanks to a single space; a paragraph break is empty).
    header_start, cursor = start, start
    while cursor > 0:
        prev_start = cleaned.rfind("\n", 0, cursor - 1) + 1
        prev = cleaned[prev_start:cursor - 1]
        if prev.lstrip().startswith("|"):
            header_start = prev_start
        elif prev != " ":
            break
        cursor = prev_start
    header_end = cleaned.find("\n", header_start)
    header_end = len(cleaned) if header_end == -1 else header_end

    def named_in(line_start: int, text: str, col: int) -> str | None:
        bounds = _cell_bounds(text, col)
        if bounds is None:
            return None
        lo, hi = line_start + bounds[0], line_start + bounds[1]
        names = [name for offset, name in occurrences if lo <= offset < hi]
        return names[-1] if names else None

    if header_start != start:
        found = named_in(header_start, cleaned[header_start:header_end], column)
        if found:
            return True, found
    return True, named_in(start, line, 0)


def _subject_at(cleaned: str, position: int, occurrences: list[tuple[int, str]]) -> str | None:
    """The element most recently named before this figure, in its own section
    — or, in a table, the element its column or its row names.

    Nearest-preceding rather than nearest-either-side: prose introduces its
    subject and then reports figures about it, and a lookahead would attach a
    figure to whatever element the next sentence happens to mention.
    """
    in_table, subject = _table_subject(cleaned, position, occurrences)
    if in_table:
        return subject
    boundary = _section_start(cleaned, position)
    subject = None
    for offset, name in occurrences:
        if offset > position:
            break
        if offset >= boundary:
            subject = name
    return subject


# The arrow of a branch label as prose rewrites it: "Bus_0 -> Bus_1 [L0]" came
# back as "Bus_0–Bus_1 [L0]", was not recognised, and its loading was reported
# as written about the bus named before it (seen live, then repeated by the
# advisor as a mistake in a correct answer).
_ARROW_RE = r"\s*(?:->|→|–|—|-|\bto\b)\s*"


def _name_pattern(name: str) -> re.Pattern:
    """An element name as the prose may write it."""
    if " -> " not in name:
        return re.compile(re.escape(name))
    left, right = name.split(" -> ", 1)
    return re.compile(re.escape(left) + _ARROW_RE + re.escape(right))


_BRANCH_RE = re.compile(r"^(?:.* \| )?(?P<ends>.+? -> .+?) \[(?P<tag>[LT]\d+)\]$")


def _short_patterns(name: str) -> list[tuple[str, re.Pattern]]:
    """Shorter ways prose names a branch: its ends alone ("Bus_0–Bus_1") or its
    tag alone ("L0"), as in "Line L0 (Bus_0–Bus_1)". Each is keyed so a form
    two branches share (parallel lines) can be dropped as ambiguous."""
    match = _BRANCH_RE.match(name)
    if not match:
        return []
    left, right = match["ends"].split(" -> ", 1)
    ends = rf"(?<![\w]){re.escape(left)}{_ARROW_RE}{re.escape(right)}(?![\w])"
    return [(f"ends:{left}|{right}", re.compile(ends)),
            (f"tag:{match['tag']}", re.compile(rf"\b{match['tag']}\b"))]


def _occurrences(cleaned: str, names: Iterable[str]) -> list[tuple[int, str]]:
    """Where each element name appears, longest names first.

    Longest-first so "Echo 10kV" is not shadowed by a shorter name it
    contains; overlapping shorter hits at the same offset are dropped. A
    branch's short forms (`_short_patterns`) come after every full name, and
    only when one branch alone answers to them.
    """
    names = sorted(names, key=len, reverse=True)
    spans: list[tuple[int, int, str]] = []

    def add(name: str, pattern: re.Pattern) -> None:
        for match in pattern.finditer(cleaned):
            if not any(start <= match.start() < end for start, end, _ in spans):
                spans.append((match.start(), match.end(), name))

    for name in names:
        add(name, _name_pattern(name))
    short: dict[str, list[tuple[str, re.Pattern]]] = {}
    for name in names:
        for key, pattern in _short_patterns(name):
            short.setdefault(key, []).append((name, pattern))
    for owners in short.values():
        if len(owners) == 1:
            add(*owners[0])
    return sorted((start, name) for start, _, name in spans)


def check_answer(answer: str, sources: list[Any],
                 subjects: Iterable[str] = ()) -> ProvenanceReport:
    """Grade every figure in an answer.

    `subjects` are the element names the prose can be about. Without them the
    check is flat: every figure is graded against the turn as a whole, which
    is the behaviour before scoping and still the right one when the payload
    has no subject structure.
    """
    claims = extract_claims(answer)

    # A subject is only usable if the payload actually partitions by it.
    scoped = {name for source in sources for name in _scopes_of(source)}
    # Names are found in the text that keeps them; positions match the claims' (see `_strip_noise`).
    cleaned = _strip_noise(answer or "", keep_names=True)
    occurrences = _occurrences(cleaned, [s for s in subjects if s in scoped])

    vocabulary = frozenset(subjects)
    verdicts = [
        check_claim(claim, sources,
                    subject=_subject_at(cleaned, claim.position, occurrences),
                    vocabulary=vocabulary,
                    answer=answer or "")
        for claim in claims
    ]
    verdicts = _with_unshown_steps(verdicts, cleaned, sources)
    verdicts = _sum_steps(verdicts, cleaned, sources)
    return ProvenanceReport(verdicts=tuple(verdicts), n_sources=len(sources))


def _sum_steps(verdicts: list[ClaimVerdict], cleaned: str, sources: list[Any]) -> list[ClaimVerdict]:
    """Untraced figures that are a sum the tools' rows give: the rows at one bus, where that bus is named on the
    claim's own line ("Bus 3 (0.23 MW)", a table row); or a list's total power, where the words say it is a total
    ("1.40 MW total"). Measured against error injection: a figure changed to another bus's sum, or to a total of
    an unrelated field, stays ungrounded."""
    sums = [s for s in sources if str(s[1]).startswith(SUM_PREFIX)]
    if not sums or not any(v.status == UNGROUNDED for v in verdicts):
        return verdicts
    out = []
    for v in verdicts:
        claim = v.claim
        if v.status != UNGROUNDED:
            out.append(v)
            continue
        band = _tolerance(claim.decimals)
        start = cleaned.rfind("\n", 0, claim.position) + 1
        end = cleaned.find("\n", claim.position)
        line = cleaned[start:end if end >= 0 else len(cleaned)]
        total = bool(_TOTAL_RE.search(cleaned[max(0, claim.position - _STEP_WINDOW):claim.position + _STEP_WINDOW]))
        step = None
        for value, path, *rest in sums:
            if abs(value - claim.value) > band:
                continue
            scopes = rest[0] if rest else frozenset()
            if scopes and any(re.search(r"(?<![\w])" + re.escape(b) + r"(?![\w])", line) for b in scopes):
                step = "sum of " + str(path)[len(SUM_PREFIX):]
            elif not scopes and total:
                step = "total of " + str(path)[len(SUM_PREFIX):]
            if step:
                break
        out.append(v._replace(status=DERIVED, source=step) if step else v)
    return out


# The limits a study applied, by the unit a margin to them is stated in: "0.943 p.u., 0.007 p.u. below the limit".
_LIMIT_KEYS = {"vm_lower_pu": "p.u.", "vm_upper_pu": "p.u.", "max_line_loading_pct": "%", "max_trafo_loading_pct": "%",
               "max_loading_pct": "%", "vm_lower": "p.u.", "vm_upper": "p.u."}
# A sentence ends at a stop followed by a space or a line break; the stops inside "p.u." and "e.g." do not end one.
_SENTENCE_END_RE = re.compile(r"[.!?;](?=\s|$)|\n")
_ABBREVIATION_RE = re.compile(r"\b(?:p\.u|e\.g|i\.e|approx|vs)\.", re.IGNORECASE)
# The words that say a figure is a step from others — a margin, a change, a total, a share. Without them a figure
# that merely equals some difference was excused (error injection: a changed import of 32.19 MW "derived" as
# 3 + 29.41).
_STEP_WORDS_RE = re.compile(
    r"\b(?:below|above|over|under|beyond|margin|headroom|short|by|less|more|lower|higher|increases?|increased|"
    r"decreases?|decreased|rises?|rose|falls?|fell|drops?|dropped|reduction|reduced|difference|leaving|leaves|"
    r"violation of|exceed\w*|total|totals|plus|together|combined|altogether|in all|sum|half|halved|halving|"
    # A change written as its two ends: "from 29.41 MW to 29.87 MW (about 0.46 MW)", "shifts by", "→".
    r"from|shifts?|shifted|chang\w*|compared)\b|\+|→|->",
    re.IGNORECASE)
_STEP_WINDOW = 45
# How far back a step's operands may be: the same paragraph, up to three sentences before the claim ("41.42 MW,
# compared with 41.51 MW … Both peaks occurred at … The proposal's peak was about 0.09 MW lower").
_STEP_SENTENCES = 3


def _sentence_starts(cleaned: str) -> list[int]:
    masked = _ABBREVIATION_RE.sub(lambda m: "x" * len(m.group(0)), cleaned)
    return [0] + [m.end() for m in _SENTENCE_END_RE.finditer(masked)]


def _quantity(claim: Claim) -> bool:
    """A figure that can be an operand: one with a unit or decimals. A bare integer is too often a name or a count
    ("PV 3, 4, 5", "Load 16") to be one."""
    return claim.unit is not None or claim.decimals > 0


def _same_unit(claim: Claim, operand: Claim) -> bool:
    if claim.unit is None:                        # "14.7 percentage points" from two percentages
        return operand.unit in (None, "%")
    return (operand.unit or "").casefold() == claim.unit.casefold()


def _with_unshown_steps(verdicts: list[ClaimVerdict], cleaned: str, sources: list[Any]) -> list[ClaimVerdict]:
    """Untraced figures that are one step of arithmetic from figures beside them.

    The model states a margin without the subtraction — "Bus_29 was at 0.943 p.u., 0.007 p.u. below its 0.950
    p.u. limit", "114.7 %, 14.7 points over" — or a total after its parts. Both operands are on the page and
    traced, and the operator can check the step; reading such figures as invented was the commonest false alarm
    in the CIM layer paper's runs. As narrow as `_derivation`, and measured against error injection: one difference or sum
    of two traced figures in the same sentence or the one before, in the claim's unit (or of one of them and a
    limit the study applied in that unit), exact on the figures as written to the claim's precision, and only where
    the words beside the claim say it is a step ("below", "by", "an increase of", "together")."""
    if not any(v.status == UNGROUNDED for v in verdicts):
        return verdicts
    starts = _sentence_starts(cleaned)

    def sentence_of(position: int) -> int:
        return max(i for i, start in enumerate(starts) if start <= position)

    traced = [v.claim for v in verdicts if v.status != UNGROUNDED and _quantity(v.claim)]
    breaks = [0] + [m.start() for m in _SECTION_BREAK_RE.finditer(cleaned)]
    limits = {(round(s[0], 6), unit) for s in sources
              if (unit := _LIMIT_KEYS.get(str(s[1]).rsplit(".", 1)[-1]))}
    # Figures the operator wrote in the question: their limit, quoted as "your limit".
    asked = sorted({round(s[0], 6) for s in sources if str(s[1]) == "prompt"})
    out = []
    for v in verdicts:
        claim = v.claim
        if v.status != UNGROUNDED or not _quantity(claim) or not _STEP_WORDS_RE.search(
                cleaned[max(0, claim.position - _STEP_WINDOW):claim.position + _STEP_WINDOW]):
            out.append(v)
            continue
        here = sentence_of(claim.position)
        paragraph = max(b for b in breaks if b <= claim.position)
        operands = [t for t in traced if here - _STEP_SENTENCES <= sentence_of(t.position) <= here
                    and t.position >= paragraph and _same_unit(claim, t)]
        window = cleaned[starts[max(here - 1, 0)]:starts[here + 1] if here + 1 < len(starts) else len(cleaned)]
        step = _share_or_total(claim, operands, window)
        band = _tolerance(claim.decimals)
        for i, a in enumerate(operands if step is None else ()):
            pool = [(b.value, f"{b.value:g}") for b in operands[i + 1:]]
            pool += [(limit, f"limit {limit:g}") for limit, unit in limits if unit == (a.unit or "").casefold()
                     or (unit == "p.u." and (a.unit or "").casefold() in ("p.u.", "pu"))]
            pool += [(value, f"your {value:g}") for value in asked]
            for b, name in pool:
                if abs(abs(a.value - b) - abs(claim.value)) <= band:
                    step = f"{a.value:g} - {name} (stated)"
                elif abs(a.value + b - claim.value) <= band:
                    step = f"{a.value:g} + {name} (stated)"
                if step:
                    break
            if step:
                break
        out.append(v._replace(status=DERIVED, source=step) if step else v)
    for _ in range(2):
        out = _count_steps(out, cleaned, starts)
    return out


# Counts, and what is computed from them. A bare integer is too often a name or an index to be an operand in
# general (`_quantity`); these steps read one only where the words say what it counts.
_NUMBER_WORDS = {w: n for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_OTHER_RE = re.compile(r"\b(?:the\s+)?(?:other|remaining|rest of the)\s*$", re.IGNORECASE)
_COUNT_NOUN_RE = re.compile(r"\s*(?:[\w-]+\s+){0,3}?(intervals?|ticks|steps|periods|quarter-hours|lines|outages|"
                            r"units|buses|loads|branches|transformers|hours|days|moments|samples|cases|options)\b",
                            re.IGNORECASE)
_RESOLUTION_RE = re.compile(r"\b(\d+)[- ]?min(?:ute)?s?\b|\b(quarter[- ]hour)|\b(half[- ]hour)", re.IGNORECASE)
_HOURS_RE = re.compile(r"\s*(?:h\b|hours?\b|hrs?\b)(?:\s*(?:and\s+)?(\d{1,2})\s*min(?:ute)?s?\b)?", re.IGNORECASE)
_MINUTES_AFTER_HOURS_RE = re.compile(r"(\d{1,3})\s*(?:h|hours?)\s*(?:and\s+)?$", re.IGNORECASE)
_PF_RE = re.compile(r"power factor(?: of)?\s*(0\.\d+)|\b(0\.\d+)\s*(?:lagging|leading)?\s*power factor",
                    re.IGNORECASE)


def _end_of(cleaned: str, claim: Claim) -> int:
    """Where the figure ends as written ("2.50", not "2.5")."""
    m = _NUMBER_RE.match(cleaned, claim.position)
    return m.end() if m else claim.position + len(f"{claim.value:g}")


def _noun_after(cleaned: str, claim: Claim) -> str | None:
    """The noun a count is of: "29 intervals", "41 lines", "10 quarter-hour intervals"."""
    m = _COUNT_NOUN_RE.match(cleaned, _end_of(cleaned, claim))
    return m.group(1).casefold().rstrip("s") if m else None


def _count_steps(verdicts: list[ClaimVerdict], cleaned: str, starts: list[int]) -> list[ClaimVerdict]:
    """Untraced figures computed from counts the answer states, where the words say so:

    - "the other 33" — the difference of two stated counts ("screened all 41 lines … these eight outages");
    - a total of stated counts of the same thing ("29 intervals … 27 intervals … together, 56 intervals");
    - a duration from a count of intervals at the step the answer states ("10 quarter-hour intervals (2.5
      hours)", "27 intervals, about 6 hours 45 minutes");
    - a share of a stated count in % ("56 intervals, or 8.3 %" of the "672 15-minute intervals");
    - an apparent power from a stated active power and power factor ("0.8 MW at 0.95 … about 0.84 MVA").

    Each operand must be traced (or itself derived), or, for the difference of counts, written as a word."""
    if not any(v.status == UNGROUNDED for v in verdicts):
        return verdicts
    counts = [v.claim for v in verdicts if v.status != UNGROUNDED and v.claim.decimals == 0 and v.claim.unit is None]
    known = [v.claim for v in verdicts if v.status != UNGROUNDED]
    resolution = None
    m = _RESOLUTION_RE.search(cleaned)
    if m:
        resolution = float(m.group(1)) if m.group(1) else (15.0 if m.group(2) else 30.0)
    words = [(_NUMBER_WORDS[w.group(0).casefold()], w.start())
             for w in re.finditer(r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", cleaned, re.IGNORECASE)]

    def sentence_of(position: int) -> int:
        return max(i for i, start in enumerate(starts) if start <= position)

    out = []
    for v in verdicts:
        claim, step = v.claim, None
        if v.status != UNGROUNDED:
            out.append(v)
            continue
        band = _tolerance(claim.decimals)
        before = cleaned[max(0, claim.position - 30):claim.position]
        after = cleaned[_end_of(cleaned, claim):_end_of(cleaned, claim) + 40]
        here = sentence_of(claim.position)
        near = [c for c in counts if c is not claim and here - _STEP_SENTENCES <= sentence_of(c.position) <= here]
        noun = _noun_after(cleaned, claim) if claim.decimals == 0 and claim.unit is None else None
        if claim.decimals == 0 and claim.unit is None and _OTHER_RE.search(before):
            pool = [(c.value, f"{c.value:g}") for c in counts if c is not claim] + \
                   [(float(n), word) for n, at in words for word in [cleaned[at:at + 12].split()[0]]]
            step = next((f"{a:g} - {name_b}" for (a, _na), (b, name_b) in itertools.permutations(pool, 2)
                         if a > b and abs(a - b - claim.value) < 1e-9), None)
        elif noun and (_TOTAL_RE.search(cleaned[max(0, claim.position - 45):claim.position + 45])
                       or re.search(r"\beither\b[^.]{0,60}\bor\b", cleaned[starts[here]:claim.position])):
            same = [c for c in near if _noun_after(cleaned, c) == noun]
            for r in (2, 3):
                hit = next((combo for combo in itertools.combinations(same, r)
                            if abs(sum(c.value for c in combo) - claim.value) < 1e-9), None)
                if hit:
                    step = " + ".join(f"{c.value:g}" for c in hit) + f" {noun}s"
                    break
        if step is None and resolution and _HOURS_RE.match(after):
            intervals = [c for c in near if _noun_after(cleaned, c) in ("interval", "tick", "step", "period",
                                                                         "quarter-hour")]
            minutes = _HOURS_RE.match(after).group(1)
            for c in intervals:
                total = c.value * resolution
                if abs(total / 60.0 - claim.value) <= band or (
                        minutes and abs(total - (claim.value * 60 + float(minutes))) < 1e-9):
                    step = f"{c.value:g} intervals × {resolution:g} min"
                    break
        if step is None and resolution and claim.unit is None and claim.decimals == 0 and \
                re.match(r"\s*min(?:ute)?s?\b", after, re.IGNORECASE):
            hours = _MINUTES_AFTER_HOURS_RE.search(before)
            if hours:
                for c in near:
                    if _noun_after(cleaned, c) in ("interval", "tick", "step", "period") and \
                            abs(c.value * resolution - (float(hours.group(1)) * 60 + claim.value)) < 1e-9:
                        step = f"{c.value:g} intervals × {resolution:g} min"
                        break
        if step is None and claim.unit == "%":
            for a in counts:
                if not (0 <= claim.position - _end_of(cleaned, a) <= 40):
                    continue
                noun_a = _noun_after(cleaned, a)
                for b in counts:
                    if b.value <= a.value or not (noun_a and _noun_after(cleaned, b) == noun_a
                                                  or re.search(r"\b(?:of|out of)\s*$",
                                                               cleaned[max(0, b.position - 8):b.position])):
                        continue
                    if abs(100.0 * a.value / b.value - claim.value) <= band:
                        step = f"{a.value:g} / {b.value:g}"
                        break
                if step:
                    break
        if step is None and (claim.unit or "").casefold() == "mva":
            window = cleaned[starts[max(here - 1, 0)]:starts[here + 1] if here + 1 < len(starts) else len(cleaned)]
            pfs = [float(m.group(1) or m.group(2)) for m in _PF_RE.finditer(window)]
            for p in [k for k in known if (k.unit or "").casefold() == "mw"
                      and here - _STEP_SENTENCES <= sentence_of(k.position) <= here]:
                for pf in pfs:
                    if 0.5 <= pf <= 1.0 and abs(p.value / pf - claim.value) <= band:
                        step = f"{p.value:g} MW / {pf:g}"
                        break
                if step:
                    break
        out.append(v._replace(status=DERIVED, source=step) if step else v)
    return out


_HALF_RE = re.compile(r"\b(?:half|halved|halving|halve)\b", re.IGNORECASE)
_PERCENT_IN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_TOTAL_RE = re.compile(r"\b(?:together|in total|total|combined|altogether|in all|sum)\b", re.IGNORECASE)


def _share_or_total(claim: Claim, operands: list[Claim], window: str) -> str | None:
    """A stated share of a stated figure ("24.39 MW … a 50 % reduction … 12.19 MW", "halved to 12.19 MW"), or a
    total of three stated figures where the words say it is one ("together, that is 14.806 points")."""
    shares = [float(m.group(1)) / 100 for m in _PERCENT_IN_RE.finditer(window) if 0 < float(m.group(1)) < 100]
    if _HALF_RE.search(window):
        shares.append(0.5)
    for a in operands:
        if a.unit == "%" or claim.unit is None:
            continue
        for share in shares:
            for part in (share, 1 - share):
                if abs(a.value * part - claim.value) <= _tolerance(claim.decimals) + _tolerance(a.decimals) * part:
                    return f"{part:g} × {a.value:g} (stated)"
    if _TOTAL_RE.search(window) and len(operands) >= 3:
        for trio in itertools.combinations(operands, 3):
            if abs(sum(o.value for o in trio) - claim.value) <= _tolerance(claim.decimals):
                return " + ".join(f"{o.value:g}" for o in trio) + " (stated)"
    return None


# ---------------------------------------------------------------------------
# A question put to the operator
# ---------------------------------------------------------------------------

# Tools that only look something up. A turn that called nothing else and asks
# the operator to choose has not answered yet: the figures in its choices are
# proposals ("1. 80 % (suggested)  2. 90 %"), not results, and flagging them
# would put a false "not in any tool output" under every clarifying question.
LOOKUP_TOOLS = frozenset({"locate_network_element", "get_current_timestamp", "get_current_conditions",
                          "get_element_timeseries", "get_data_quality"})
_CHOICE_LINE = re.compile(r"^[ \t]*(?:[-*][ \t]*)?(?:\*\*)?\(?[1-9][.)]")


def is_clarifying_question(answer: str, calls: Iterable[dict] = ()) -> bool:
    """Whether a turn asks the operator to choose before anything ran: only
    lookups were called, and a question comes before a numbered list of
    choices. The order is the test — a list of results followed by "which do
    you prefer?" is an answer, and its figures stay checked."""
    if not isinstance(calls, (list, tuple)) or any(
            not isinstance(c, dict) or c.get("name") not in LOOKUP_TOOLS for c in calls):
        return False                      # a malformed record is never exempt
    asked = False
    for line in (answer or "").splitlines():
        if _CHOICE_LINE.match(line):
            return asked
        asked = asked or line.rstrip(" *_").endswith("?")
    return False


def _proposes_choices(answer: str, calls: Iterable[dict] = ()) -> bool:
    """A wider reading of `is_clarifying_question`, for the figure check only (the graders keep the narrow one):
    the question may sit mid-line ("What should it mean? Choose one:"), and a turn that ran no tool at all and asks
    is proposing whatever figures it states ("I suggest 80 %; alternatives are 70 % or 90 %")."""
    if not isinstance(calls, (list, tuple)) or any(
            not isinstance(c, dict) or c.get("name") not in LOOKUP_TOOLS for c in calls):
        return False
    asked = False
    for line in (answer or "").splitlines():
        if _CHOICE_LINE.match(line):
            return asked
        asked = asked or "?" in line
    return not calls and asked


# A sentence that offers the operator values to choose from: "I suggest 80 %; you can choose 85 % or 90 % instead",
# "80 % is a reasonable target; I can also use 70 %". Its figures are proposals, not results — six of the false
# alarms left in the CIM layer paper's labelled findings.
_OFFER_RE = re.compile(
    r"\b(?:I (?:would )?suggest|suggested|you (?:can|could|may) (?:choose|pick|use|set)|I can (?:also )?use|"
    r"or another (?:limit|value|threshold)|reasonable\b[^.?;]{0,40}\b(?:target|limit|choice|threshold))\b",
    re.IGNORECASE)
# A sentence within a line; a stop inside a figure ("77.9") does not end it.
_OFFER_SENTENCE_RE = re.compile(r"[^.?!]*(?:\.(?=\d)[^.?!]*)*(?:[.?!]+|$)")


def _without_offers(answer: str) -> str:
    """The answer without the sentences that offer values (blanked, so positions and lines stay), when it asks
    the operator something."""
    if "?" not in (answer or ""):
        return answer or ""
    return "\n".join("".join(" " * len(part) if _OFFER_RE.search(part) else part
                              for part in _OFFER_SENTENCE_RE.findall(line))
                      for line in answer.split("\n"))


def answer_to_check(answer: str, calls: Iterable[dict] = ()) -> str:
    """The text whose figures are checked: a clarifying question without its
    choices (the question's own sentences are still checked), and without the
    sentences that offer the operator values to choose from."""
    if not (is_clarifying_question(answer, calls) or _proposes_choices(answer, calls)):
        return _without_offers(answer or "")
    if not any(_CHOICE_LINE.match(line) for line in (answer or "").splitlines()):
        return ""                         # proposed inline: every figure is a choice
    return "\n".join(line for line in (answer or "").splitlines() if not _CHOICE_LINE.match(line))


def check_turn(record: dict) -> ProvenanceReport:
    """Grade one session-log turn record."""
    return check_answer(
        answer_to_check(record.get("assistant") or "", record.get("tool_calls") or ()),
        collect_sources(record),
        subjects=subject_names(record),
    )
