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

This is a measuring instrument, not a filter: nothing here modifies an answer
or is fed back to the model.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, NamedTuple

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
_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")

# Words that state which way power is moving. Where the prose names a
# direction, the sign is carried by the word and the magnitude by the number,
# so a value may legitimately appear negated: the payload reports
# `slack_import_mw = -1.3146` and the answer says "exporting 1.31 MW". Both
# describe the same flow, and reading only the signed value called the second
# one fabricated.
_DIRECTION_RE = re.compile(
    r"\b(export|exports|exporting|import|imports|importing|inject|injects|"
    r"injecting|injection|absorb|absorbs|absorbing|draw|draws|drawing|"
    r"supply|supplies|supplying|consume|consumes|consuming|deliver|delivers|"
    r"delivering|withdraw|withdraws|withdrawing|feed|feeds|feeding)\b",
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
    def ungrounded(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status == UNGROUNDED]

    @property
    def rate(self) -> float:
        """Share of numeric claims traceable to a source. 1.0 if there are none.

        Imprecise claims count as traceable: the figure came from the
        evidence, it was just transcribed at the wrong precision.
        """
        if not self.verdicts:
            return 1.0
        return (len(self.grounded) + len(self.imprecise)) / len(self.verdicts)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _strip_noise(text: str) -> str:
    """Remove the parts of the markdown whose digits are never claims."""
    # The model writes the typographic minus (U+2212), not a hyphen. Left
    # alone, "−1.35 MW" parses as +1.35 and a correctly reported export reads
    # as fabricated. Only U+2212 is converted: the en-dash it resembles is a
    # range separator here ("0.95–1.045") and must not become a sign.
    text = text.replace("\u2212", "-")
    text = _FENCED_CODE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _TABLE_RULE_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub(r"\1\2", text)
    text = _DATETIME_RE.sub(" ", text)
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
        for hit in _DIGITS_RE.findall(node):
            try:
                out.append(Source(float(hit), path, scopes if keeps_scope else frozenset()))
            except ValueError:  # pragma: no cover
                continue
        return
    if isinstance(node, dict):
        name = _entity_name(node)
        inner = scopes | {name} if name else scopes
        for key, value in node.items():
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
        scan(entry.get("result"))

    return frozenset(
        name for name in found
        if not any(other != name and name in other for other in found)
    )


def collect_sources(record: dict) -> list[Source]:
    """Every number the turn had legitimate access to, with what it belongs to.

    Three origins, all of them fair: what the tools returned, what they were
    called with, and what the user wrote. Omitting the last two would flag the
    user's own voltage limit, quoted back in the answer, as an invention.
    """
    sources: list[Source] = []
    vocabulary = element_names(record)

    for entry in record.get("tool_results") or []:
        name = entry.get("name", "tool")
        _walk(entry.get("result"), name, sources, frozenset(), vocabulary)

    # The grid facts the system prompt stated. A figure quoted from them —
    # how many buses the network has, its voltage limits — came from the
    # evidence the model was given, and reading only tool results called those
    # fabricated. Turn-level, since they describe the network, not an element.
    _walk(record.get("grid") or {}, "grid", sources)

    # Arguments and the prompt belong to the turn, not to any one element.
    for call in record.get("tool_calls") or []:
        name = call.get("name", "tool")
        _walk(call.get("args"), f"args:{name}", sources)

    _walk(record.get("user") or "", "prompt", sources)
    return sources


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
                vocabulary: frozenset | None = None) -> ClaimVerdict:
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
    return ClaimVerdict(claim, UNGROUNDED, nearest=nearest, subject=subject)


# ---------------------------------------------------------------------------
# Subjects — what the prose around a figure is talking about
# ---------------------------------------------------------------------------

# A heading or a rule ends the current subject. Without this the last element
# named in section 1 would still be the subject halfway through section 3.
_SECTION_BREAK_RE = re.compile(r"^(#{1,6}\s|-{3,}\s*$|_{3,}\s*$)", re.MULTILINE)


def _section_start(cleaned: str, position: int) -> int:
    last = 0
    for match in _SECTION_BREAK_RE.finditer(cleaned):
        if match.start() > position:
            break
        last = match.start()
    return last


def _subject_at(cleaned: str, position: int, occurrences: list[tuple[int, str]]) -> str | None:
    """The element most recently named before this figure, in its own section.

    Nearest-preceding rather than nearest-either-side: prose introduces its
    subject and then reports figures about it, and a lookahead would attach a
    figure to whatever element the next sentence happens to mention.
    """
    boundary = _section_start(cleaned, position)
    subject = None
    for offset, name in occurrences:
        if offset > position:
            break
        if offset >= boundary:
            subject = name
    return subject


def _occurrences(cleaned: str, names: Iterable[str]) -> list[tuple[int, str]]:
    """Where each element name appears, longest names first.

    Longest-first so "Echo 10kV" is not shadowed by a shorter name it
    contains; overlapping shorter hits at the same offset are dropped.
    """
    found: dict[int, str] = {}
    for name in sorted(names, key=len, reverse=True):
        start = cleaned.find(name)
        while start != -1:
            if not any(o <= start < o + len(n) for o, n in found.items()):
                found[start] = name
            start = cleaned.find(name, start + 1)
    return sorted(found.items())


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
    cleaned = _strip_noise(answer or "")
    occurrences = _occurrences(cleaned, [s for s in subjects if s in scoped])

    vocabulary = frozenset(subjects)
    verdicts = [
        check_claim(claim, sources,
                    subject=_subject_at(cleaned, claim.position, occurrences),
                    vocabulary=vocabulary)
        for claim in claims
    ]
    return ProvenanceReport(verdicts=tuple(verdicts), n_sources=len(sources))


def check_turn(record: dict) -> ProvenanceReport:
    """Grade one session-log turn record."""
    return check_answer(
        record.get("assistant") or "",
        collect_sources(record),
        subjects=subject_names(record),
    )
