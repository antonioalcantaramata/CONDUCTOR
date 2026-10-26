"""
investigation_record.py — the agent's record of an investigation, held to the
studies it cites, and its answer held to the record.

The agent leads an investigation: it forms hypotheses and chooses how to test
them. What it may not do is conclude past the studies. Two checks, both in
code:

1. **The record against the studies** (`validate`, run by the
   `record_investigation` tool before it accepts anything):
   - a supported or refuted hypothesis must cite tools that returned a result
     in this question;
   - when a hypothesis is one of `investigate`'s factors (named in `factor`,
     or written in its text), its verdict must agree with that factor's:
     supported only if the study found it a cause or a contributor, refuted
     only if it found it ruled out or counteracting;
   - an untested hypothesis must describe a tool CONDUCTOR does not have
     (naming an existing one means a step not taken, not a missing tool).
   A record that fails is refused with the reasons, for the agent to correct.

2. **The answer against the record** (`check_answer`, run by the answer review,
   the audit panel and the grader):
   - an untested hypothesis stated as a finding ("the hot weather caused the
     overload", with nothing hedging it);
   - untested hypotheses with no word in the answer saying they were not
     tested;
   - a supported hypothesis whose elements the answer never names.

Both read words and structure, not meaning: the first matches factor names, the
second looks for a causal verb without a hedge near the hypothesis's own
distinctive words. A miss is possible either way; a finding names the sentence
so a reader can judge it.
"""

from __future__ import annotations

import re
from typing import Iterable, NamedTuple

VERDICTS = ("supported", "refuted", "untested")
MAX_RECORDED = 10
# What a study's factor verdict allows the agent to record.
_AGREES = {"supported": {"cause", "contributes"}, "refuted": {"ruled_out", "counteracts"}}
# A hypothesis that the change behind a factor is a fault in the data, not
# something that happened: decided by the factor's data check, not its share.
_DATA_FAULT_RE = re.compile(r"\b(anomal\w*|artefacts?|artifacts?|faulty|fault in the data|data[- ]quality|data errors?|"
                            r"bad data|measurement errors?|meters?|telemetry|spikes?|frozen|glitch\w*)\b", re.IGNORECASE)


def existing_tools_named(text: str) -> list[str]:
    """CONDUCTOR tools a "missing tool" description names, by their function
    name or their label ("Investigation (why)")."""
    from .tool_registry import REGISTRY

    lowered = text.casefold()
    return sorted(name for name, row in REGISTRY.items()
                  if name != "record_investigation"
                  and (re.search(rf"\b{re.escape(name)}\b", lowered) or row.label.casefold() in lowered))


def _factor_key(name: str) -> str:
    """A factor's name without its trailing count ("Demand near Bus_29 (2 buses)"
    → "demand near bus_29"), case-folded, spaces collapsed."""
    name = re.sub(r"\s*\([^()]*\)\s*$", "", str(name or ""))
    return " ".join(name.casefold().split())


def study_factors(results: Iterable[tuple[str, dict]]) -> list[dict]:
    """Every factor an `investigate` call returned in this question:
    {name, key, verdict, share_pct, reference, data_suspicious} — the last
    None when the factor's measured changes were not checked."""
    out = []
    for name, result in results:
        if name != "investigate" or not isinstance(result, dict) or result.get("error"):
            continue
        reference = (result.get("reference") or {}).get("label")
        groups = [(reference, result.get("factors") or [])]
        # A recurring investigation keeps each episode's factors with the episode.
        groups += [(f"the episode at {str(e.get('start'))[:16]}", e.get("factors") or [])
                   for e in result.get("events") or []]
        for ref, rows in groups:
            for f in rows:
                check = f.get("data_check")
                out.append({"name": f.get("factor", ""), "key": _factor_key(f.get("factor", "")),
                            "verdict": f.get("verdict"), "share_pct": f.get("share_pct"), "reference": ref,
                            "data_suspicious": bool(check.get("suspicious")) if isinstance(check, dict) else None})
    return out


def _deciding(text: str, factor: str, factors: list[dict]) -> tuple[list[dict], bool]:
    """The study factors a hypothesis is about, and whether it named one
    explicitly. By name when given; otherwise every factor whose name (six
    characters or more) its text contains."""
    if factor:
        key = _factor_key(factor)
        return [f for f in factors if f["key"] == key], True
    lowered = " ".join(text.casefold().split())
    return [f for f in factors if len(f["key"]) >= 6 and f["key"] in lowered], False


def validate(hypotheses: list | None, called: set[str], factors: list[dict]) -> tuple[list[dict], list[str]]:
    """(rows, problems) for a record; it is accepted only with no problems."""
    problems, rows = [], []
    given = [h for h in hypotheses or [] if isinstance(h, dict)]
    if not given:
        problems.append("Record at least one hypothesis.")
    if len(given) > MAX_RECORDED:
        problems.append(f"Record at most {MAX_RECORDED} hypotheses: the ones the operator needs.")
    for n, h in enumerate(given, 1):
        text = str(h.get("hypothesis") or "").strip()
        verdict = str(h.get("verdict") or "").strip().lower()
        evidence = [str(e).strip() for e in h.get("evidence") or [] if str(e).strip()]
        missing_tool = str(h.get("missing_tool") or "").strip()
        missing_kind = str(h.get("missing_kind") or "").strip().lower()
        factor = str(h.get("factor") or "").strip()
        if not text:
            problems.append(f"Hypothesis {n} has no text.")
        if verdict not in VERDICTS:
            problems.append(f"Hypothesis {n}: verdict must be one of {list(VERDICTS)}.")
        elif verdict == "untested":
            if not missing_tool:
                problems.append(f"Hypothesis {n} is untested: say which tool would test it (missing_tool).")
            existing = existing_tools_named(missing_tool)
            if existing:
                # Observed live: hypotheses left for after the operator's answer
                # were recorded as untested, "needing" investigate — a step not
                # yet taken, not a tool CONDUCTOR lacks.
                problems.append(f"Hypothesis {n}: {existing} already exist — run them, or record the "
                                "investigation once it is done. Untested is for what no CONDUCTOR tool can test.")
            if missing_kind not in MISSING_KINDS:
                missing_kind = kind_of({"missing_tool": missing_tool})
            can = existing_capabilities(missing_tool) if missing_kind == ANALYSIS and not existing else []
            if can:
                problems.append(f"Hypothesis {n}: CONDUCTOR already computes that ({', '.join(can)}) — run it and "
                                "record what it found, or say what that study lacks for this hypothesis.")
            evidence, factor = [], ""
        else:
            if not evidence:
                problems.append(f"Hypothesis {n} is {verdict}: name the tools whose results decide it.")
            absent = [e for e in evidence if e not in called]
            if absent:
                problems.append(f"Hypothesis {n} cites {absent}, which returned no result in this question "
                                f"(called: {sorted(called) or 'nothing'}). Run the study, or mark it untested.")
            if "investigate" in evidence or factor:
                matched, named = _deciding(text, factor, factors)
                if named and not matched:
                    problems.append(f"Hypothesis {n} names the factor {factor!r}, which no investigation in this "
                                    f"question returned (factors: {sorted({f['name'] for f in factors})}).")
                checked = [f for f in matched if f["data_suspicious"] is not None]
                if checked and _DATA_FAULT_RE.search(text):
                    # Observed live: "the demand change is a data anomaly", refuted by
                    # a clean data check, was refused for disagreeing with the share.
                    if all(f["data_suspicious"] != (verdict == "supported") for f in checked):
                        problems.append(
                            f"Hypothesis {n} is recorded as {verdict}, but the data check of {checked[0]['name']!r} "
                            + ("found something unusual" if checked[0]["data_suspicious"] else "found nothing unusual")
                            + ". Record what the check found.")
                    elif not factor:
                        factor = checked[0]["name"]
                    matched = []
                disagree = [f for f in matched if f["verdict"] not in _AGREES[verdict]]
                if matched and len(disagree) == len(matched):
                    f = disagree[0]
                    problems.append(
                        f"Hypothesis {n} is recorded as {verdict}, but the investigation found "
                        f"{f['name']!r} {str(f['verdict']).replace('_', ' ')} ({f['share_pct']} % of the change"
                        + (f", against {f['reference']}" if f.get("reference") else "") + "). Record what the "
                        "study found, or test it another way and cite that.")
                if matched and not disagree:
                    factor = factor or matched[0]["name"]
        rows.append({"hypothesis": text, "verdict": verdict, "evidence": evidence,
                     "finding": str(h.get("finding") or "").strip(),
                     **({"factor": factor} if factor else {}),
                     **({"missing_tool": missing_tool, "missing_kind": missing_kind}
                        if verdict == "untested" else {})})
    return rows, problems


# ---------------------------------------------------------------------------
# The answer against the record
# ---------------------------------------------------------------------------

class Issue(NamedTuple):
    kind: str          # untested_as_finding | untested_unsaid | hypothesis_unsaid | coverage_unsaid | data_doubt_unsaid
    hypothesis: str
    text: str          # the sentence, or the hypothesis when nothing in the answer is at fault
    note: str


# Saying something was not tested, in the words answers use.
_NOT_TESTED_RE = re.compile(
    r"\b(not (?:been )?(?:tested|established|verified|confirmed|assessed|determined|testable)|untested|"
    r"(?:cannot|can['’]t|could not|couldn['’]t) (?:be )?(?:test|tested|assess|assessed|establish|established|"
    r"determine|determined|verify|verified|confirm|confirmed|tell|say)|"
    r"(?:contains?|has|have|had|with) no \w+ (?:data|information|record|records|measurements?|telemetry)|"
    r"no (?:tool|data|record|measurement|telemetry|model|information)s?\b|would (?:require|need)|"
    # Observed live: "a stability study would be needed to assess…", "the analysis
    # does not establish whether weather…", "nor does it assess…".
    r"would be (?:needed|required)|(?:is|are) (?:needed|required) to|"
    r"(?:does|do|did) not (?:establish|show|test|determine|assess|tell|say|explain)|"
    r"nor (?:does it |do they )?(?:establish|assess|test|determine)|"
    r"not (?:available|possible to (?:test|assess))|remains? (?:open|untested|unknown|unverified)|unverified|"
    r"outside what conductor)", re.IGNORECASE)
# Stating a cause.
_CAUSAL_RE = re.compile(r"\b(caused|causes|causing|explains?|explained|drove|drives|driven by|due to|because of|"
                        r"responsible for|led to|leads to|resulted in|is the (?:main |root )?cause|was the "
                        r"(?:main |root )?cause|triggered)\b", re.IGNORECASE)
# Holding a statement back.
_HEDGE_RE = re.compile(r"\b(could|may|might|possibly|perhaps|plausibl\w*|whether|if|not|no|cannot|can['’]t|unknown|untested|"
                       r"unclear|would need|would be needed|without|hypothes\w*|suspect\w*|speculat\w*)\b",
                       re.IGNORECASE)
# Words too common in grid answers to identify a hypothesis by.
_COMMON = {"demand", "loads", "load", "overload", "overloaded", "voltage", "limit", "limits", "line", "lines",
           "network", "grid", "change", "changes", "changed", "caused", "explain", "explains", "higher",
           "lower", "increase", "increased", "rising", "reduced", "output", "units", "unit", "evening",
           "moment", "external", "contributed", "contribute", "contributes", "possible", "through", "their",
           "there", "which", "would", "could", "about", "other", "between", "because", "before", "after",
           "below", "above", "loading", "rating", "itself", "directly", "indirectly", "independently",
           # Observed live: a sentence on recurrence ("demand was the main cause in all
           # 10 episodes") tied to "the undervoltage indicates Bus_29 is close to collapse".
           "undervoltage", "overvoltage", "episode", "episodes", "indicates", "measurements", "measured",
           "recurring", "operating",
           # Function words long enough to pass the length filter (observed live: "during"
           # tied a sentence on recurrence to a hypothesis about ramps).
           "during", "while", "where", "these", "those", "since", "until", "under", "across", "within",
           "without", "should", "might", "still", "already", "quickly", "enough", "available"}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ELEMENT_RE = re.compile(r"\b(?:Bus_\d+|Gen_\w+|L\d+|T\d+)\b")


def _distinctive(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{4,}", text.casefold()) if w not in _COMMON}


def _record_of(record: dict) -> dict | None:
    found = None
    for entry in record.get("tool_results") or []:
        result = entry.get("result") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and entry.get("name") == "record_investigation" \
                and isinstance(result, dict) and result.get("status") == "recorded":
            found = result
    return found


# Saying the agent's hypotheses leave part of the change unexplained.
_UNEXPLAINED_RE = re.compile(r"\b(other changes|unexplained|not explained|unaccounted|remaining|the rest of the "
                             r"change|did not name|not named|left (?:over|out))\b", re.IGNORECASE)


# Saying a cause may be a fault in the data.
_DATA_DOUBT_RE = re.compile(r"\b(data|measurement|measured value|meter|telemetry|artefact|artifact|reading error|"
                            r"sensor|faulty)\b", re.IGNORECASE)


def _data_doubts(record: dict) -> list[str]:
    """The members an investigation in this turn flagged as possibly a fault in the data."""
    out = []
    for entry in record.get("tool_results") or []:
        result = entry.get("result") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and entry.get("name") == "investigate" and isinstance(result, dict):
            out += [d.get("member", "") for d in (result.get("data_check") or {}).get("suspicious") or []]
    return out


def _thin_coverage(record: dict) -> float | None:
    """The lowest share of a change the agent's own hypotheses explained in this
    turn, when an investigation said it was too low."""
    low = None
    for entry in record.get("tool_results") or []:
        result = entry.get("result") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and entry.get("name") == "investigate" and isinstance(result, dict) \
                and result.get("coverage_note") and result.get("hypotheses_coverage_pct") is not None:
            pct = float(result["hypotheses_coverage_pct"])
            low = pct if low is None else min(low, pct)
    return low


def check_answer(answer: str, record: dict) -> list[Issue]:
    """Where the answer departs from the turn's investigation record (if any),
    or leaves unsaid how much its own hypotheses left unexplained."""
    if not answer:
        return []
    from .provenance import without_own_notes  # local: avoids an import cycle

    text = without_own_notes(answer)
    issues: list[Issue] = []
    thin = _thin_coverage(record)
    if thin is not None and not _UNEXPLAINED_RE.search(text):
        issues.append(Issue("coverage_unsaid", "", f"its hypotheses explain {thin:.0f} % of the change",
                            f"the investigation's own hypotheses explained only {thin:.0f} % of the change, and "
                            "the answer does not say what remains unexplained"))
    doubts = _data_doubts(record)
    if doubts and not _DATA_DOUBT_RE.search(text):
        issues.append(Issue("data_doubt_unsaid", "", f"{', '.join(doubts[:3])} may be a fault in the data",
                            "the investigation flagged a cause that may be a fault in the data, and the answer "
                            "does not say so"))
    trail = _record_of(record)
    if not trail:
        return issues
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    untested = [h for h in trail.get("hypotheses") or [] if h.get("verdict") == "untested"]
    for h in untested:
        words = _distinctive(h.get("hypothesis", ""))
        # One word in common is chance when the hypothesis has several.
        needed = 1 if len(words) <= 2 else 2
        for s in sentences:
            if len(words & _distinctive(s)) >= needed and _CAUSAL_RE.search(s) and not _HEDGE_RE.search(s) \
                    and not _NOT_TESTED_RE.search(s):
                issues.append(Issue("untested_as_finding", h["hypothesis"], s[:240],
                                    "states as a finding a hypothesis the investigation recorded as untested"))
                break
    if untested and not _NOT_TESTED_RE.search(text):
        issues.append(Issue("untested_unsaid", untested[0]["hypothesis"], untested[0]["hypothesis"][:240],
                            f"{len(untested)} hypothesis(es) could not be tested, and the answer does not say so"))
    for h in trail.get("hypotheses") or []:
        if h.get("verdict") != "supported":
            continue
        named = set(_ELEMENT_RE.findall(h.get("hypothesis", "")))
        if named and not any(n in text for n in named):
            issues.append(Issue("hypothesis_unsaid", h["hypothesis"], h["hypothesis"][:240],
                                "was recorded as supported, but the answer never names it"))
    return issues


def as_conflicts(answer: str, record: dict) -> list:
    """The issues as characterisation conflicts (path `investigation`), so the
    answer review, the audit panel and the advisor show them with the others."""
    from .characterisation import Conflict
    return [Conflict(0.0, None, i.text, "investigation", i.kind, i.note, study=True)
            for i in check_answer(answer, record)]


# ---------------------------------------------------------------------------
# The tools investigations lacked, across sessions
# ---------------------------------------------------------------------------

REQUESTS_FILE = "tool_requests.jsonl"
_FILLER = {"a", "an", "the", "of", "for", "to", "and", "or", "with", "that", "which", "would", "tool", "data",
           "per", "by", "on", "in", "at", "its", "it"}


def _request_key(text: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in _FILLER]
    return " ".join(sorted(set(words)))


def read_requests(path=None) -> list[dict]:
    """The requests file's rows (local cache by default); [] when there is none."""
    import json
    import pathlib

    if path is None:
        from .usage import cache_dir
        path = cache_dir() / REQUESTS_FILE
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("missing_tool"):
            rows.append(row)
    return rows


# What an untested hypothesis lacks: an ANALYSIS of the grid CONDUCTOR could
# compute and does not (a tool to build), or DATA it does not have (an input to
# integrate). Observed live: weather, calendar and customer records were all
# "tools needed" — inputs, not tools.
ANALYSIS, DATA = "analysis", "data"
MISSING_KINDS = (ANALYSIS, DATA)


def _patterns(*pairs: tuple[str, str]) -> tuple[tuple[str, re.Pattern], ...]:
    return tuple((label, re.compile(pattern, re.IGNORECASE)) for label, pattern in pairs)


# The needs, as the developers would build them. The agent words one need many
# ways ("weather/temperature time series", "matched temperature observations…"):
# grouped by wording alone, one need read as several.
NEEDS: dict[str, tuple[tuple[str, re.Pattern], ...]] = {
    ANALYSIS: _patterns(
        ("Voltage stability margin", r"\b(continuation|PV[- ]curves?|nose curves?|voltage (?:collapse|stability)|"
                                     r"stability margins?|loadability)\b"),
        ("Dynamic simulation", r"\b(dynamics?|dynamic simulation|transients?|time[- ]domain|motor stall\w*|"
                               r"electromechanical|RMS simulation|EMT)\b"),
        ("Time-coupled studies", r"\b(ramp\w*|time[- ]coupled|multi[- ]period|inter[- ]?temporal|unit commitment)\b"),
        ("Cost of each cause", r"\b(costs?|economic|prices?)\b"),
        ("Short circuit and protection", r"\b(short[- ]circuit|fault currents?|protection coordination|"
                                         r"relay settings?)\b"),
        ("Multiple contingencies", r"\b(N-2|N-k|double contingenc\w*|multiple contingenc\w*|common[- ]mode)\b"),
        ("Harmonics and power quality", r"\b(harmonics?|flicker|power quality|unbalance)\b"),
        ("Thermal and rating models", r"\b(thermal models?|conductor temperatures?|ampacity|dynamic line rating|"
                                      r"rating models?)\b"),
        ("Reactive capability", r"\b(capability curves?|reactive capability|PQ (?:curves?|charts?))\b"),
    ),
    DATA: _patterns(
        ("Weather data", r"\b(weather|ambient|temperatures?|wind|irradiance|meteorolog\w*|heat ?wave)\b"),
        ("Calendar and events", r"\b(calendar|holidays?|week ?days?|week ?ends?|special events?|public events?)\b"),
        ("Customer and consumption data", r"\b(customers?|consumers?|smart[- ]meters?|end[- ]use\w*|tariffs?)\b"),
        ("Operator and dispatch logs", r"\b(operator(?:'s)?[- ](?:events?|actions?|decisions?|logs?|records?)|"
                                       r"dispatch(?:er)? logs?|switching logs?|maneuvers?|manoeuvres?|"
                                       r"control[- ]room|schedule changes?)\b"),
        ("Protection and fault events", r"\b(protection|relays?|trips?|fault records?|breakers?|alarms?|"
                                        r"SCADA events?)\b"),
        ("Equipment history (taps, regulators, capacitors)", r"\b(taps?|tap[- ]changers?|regulators?|"
                                                             r"capacitor banks?)\b"),
        ("Line ratings", r"\b(ratings?|ampacity|thermal (?:limit|capacity|capability)|dynamic line)\b"),
        ("Market and prices", r"\b(prices?|market)\b"),
    ),
}


# A clause saying why CONDUCTOR cannot do it, not what is needed: the needs it
# names are only mentioned ("…; no CONDUCTOR tool provides weather or customer data").
_JUSTIFICATION_RE = re.compile(
    r"\b(no (?:available |current )?CONDUCTOR (?:tool|study)|(?:the )?available (?:grid )?(?:tools?|studies|data)|"
    r"not available|(?:does|do|did) not (?:provide|include|have|contain|retrieve|expose|attribute)|cannot|can['’]t|"
    r"only (?:exposes?|provides?|has|have))\b", re.IGNORECASE)


def _what_is_needed(missing_tool: str) -> str:
    """A missing tool's description without its justification clauses."""
    clauses = re.split(r";|\s[—–-]\s|\.\s+", missing_tool or "")
    kept = [c for c in clauses if not _JUSTIFICATION_RE.search(c)]
    return " ; ".join(kept) if kept else (missing_tool or "")


def needs_of(missing_tool: str, kind: str = DATA) -> list[str]:
    """The needs of that kind a missing tool's description asks for (NEEDS), in
    the order it names them — not those only mentioned in saying why CONDUCTOR
    cannot do it."""
    text = _what_is_needed(missing_tool)
    found = [(m.start(), label) for label, pattern in NEEDS[kind] if (m := pattern.search(text))]
    return [label for _, label in sorted(found)]


# What CONDUCTOR already computes, as an analysis request might describe it
# without naming the tool: a request for one of these alone is a study to run,
# not a tool to build. With something new beside it ("an OPF with ramp limits
# over time"), it is a tool to build.
CAPABILITIES = _patterns(
    ("simulate_all_contingencies", r"\b(N-1|single contingenc\w*|contingency (?:screen\w*|analys\w*|stud\w*))\b"),
    ("optimize_flexibility", r"\b(optimal power flow|OPF|redispatch\w*)\b"),
    ("compute_hosting_capacity", r"\bhosting capacity\b"),
    ("compute_violation_attribution", r"\b(sensitivit\w*)\b"),
    ("run_probabilistic_rsa", r"\b(probabilistic|monte carlo)\b"),
    ("edit_proposal", r"\b(shunt|capacitor|reactor|new line|added line|add(?:ing)? a (?:line|generator|load|"
                      r"transformer))\b"),
)


def existing_capabilities(missing_tool: str) -> list[str]:
    """The tools that already compute what an analysis request describes, when
    it describes nothing new (no ANALYSIS need) beside them."""
    if needs_of(missing_tool, ANALYSIS):
        return []
    return [tool for tool, pattern in CAPABILITIES if pattern.search(missing_tool or "")]


def kind_of(row: dict) -> str:
    """What a request lacks: as recorded, or — for a request recorded before the
    kind was asked for — data when it names a data need and no analysis."""
    if row.get("missing_kind") in MISSING_KINDS:
        return row["missing_kind"]
    text = row.get("missing_tool") or ""
    return DATA if needs_of(text, DATA) and not needs_of(text, ANALYSIS) else ANALYSIS


def summarise_requests(rows: list[dict]) -> list[dict]:
    """Requests grouped by what they lack (an analysis to build, data to
    integrate), then by each need of that kind they ask for (NEEDS) — a request
    for a stability margin and a ramp study counts under both — or, asking for
    none, by their wording, loosely (case, punctuation and filler words do not
    count); most needed first:
    {kind, need, tool, for (its hypothesis), count, hypotheses, wordings}."""
    groups: dict[str, dict] = {}
    for row in rows:
        tool, kind = row["missing_tool"], kind_of(row)
        for key, need in ([(f"{kind}:{n}", n) for n in needs_of(tool, kind)]
                          or [(f"{kind}:{_request_key(tool)}", None)]):
            group = groups.setdefault(key, {"kind": kind, "need": need, "tool": tool, "count": 0,
                                            "hypotheses": [], "wordings": []})
            group["count"] += 1
            if tool not in group["wordings"]:
                group["wordings"].append(tool)
            # Shown: the most specific wording (fewest needs named), the latest of those.
            if len(needs_of(tool, kind)) <= len(needs_of(group["tool"], kind)):
                group["tool"], group["for"] = tool, row.get("hypothesis") or group.get("for", "")
            if row.get("hypothesis") and row["hypothesis"] not in group["hypotheses"]:
                group["hypotheses"].append(row["hypothesis"])
    return sorted(groups.values(), key=lambda g: (-g["count"], g["need"] is None))
