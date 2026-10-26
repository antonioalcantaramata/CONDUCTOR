"""
robustness.py — does the answer depend on how the question was read?

A question leaves things open. "Is the grid secure now?" does not say at which
load power factor, against which loading limit, on the measured data or the
forecast. The agent then reads the question one way — a default, the session's
setting, the network's own limits — and the answer is only as good as that
reading. Asking every time would be tiresome (and the clarification rule asks
only when the reading changes the result *and* no default fits); answering
silently from one reading hides that the conclusion may hang on it.

This is the third way: after the answer, CONDUCTOR re-runs the turn's decisive
studies under the other readings the question left open, and compares their
conclusions with the one the answer gave:

  holds     the same verdict, and its figure within 10 %
  shifts    the same verdict, a figure that moves by more (the cheapest
            option at 26.48 €/h under one reading, 31.20 €/h under another)
  changes   another verdict (secure under one reading, not under another)

A reading is open only when the operator did not settle it: the question does
not mention it, and the study ran with the default (`readings`). The agent
passes defaults explicitly, so an argument settles nothing by being there. The
alternatives are the ones an operator would plausibly have meant:

  load power factor   0.90 and 1.00, beside the value the study used
  loading limit       90 % of the rating against 100 % (assessments), the full
                      rating against the 90 % margin (optimisations)
  voltage band        the common ±5 % band, when the network's own differs
  data                the forecast for a measured moment that also has one
                      (and the measurements for a forecast one)

The studies re-run are those with one clear verdict and no side effects: the
security assessment, single and full N-1 screens, and the two optimisations
(with nothing kept on the backend — `tools.unrecorded`). Two studies at most,
the answer's main one (the last) first with every reading, at most eight
re-runs and a time budget, so a check never takes longer than the answer.

The verdicts compared are the studies' own fields (`secure`,
`system_n1_secure`, a dispatch's existence, the cheapest option's cost), never
the model's prose: nothing here is model output. When a reading changes the
conclusion, CONDUCTOR says so in its own words: a note kept with the answer
(the model reads it in the next question; the app shows it in the answer's
card rather than in the text).
"""

from __future__ import annotations

import re
import time
from typing import Any, NamedTuple

from .tool_registry import label as _label
from .validators import _ACTUALS_WORD_RE, _FORECAST_WORD_RE, _PF_WORD_RE, requested_limits

HOLDS, SHIFTS, CHANGES, FAILED = "holds", "shifts", "changes", "failed"

STUDIES = ("run_rsa", "simulate_contingency", "simulate_all_contingencies", "optimize_flexibility",
           "optimize_contingency", "investigate")
_ASSESSMENTS = ("run_rsa", "simulate_contingency", "simulate_all_contingencies", "investigate")
MAX_RERUNS = 8
MAX_STUDIES = 2
TIME_BUDGET_S = 25.0
SHIFT_SHARE = 0.10            # a figure moving by more than this share "shifts"
PF_ALTERNATIVES = (0.90, 1.00)
STANDARD_BAND = (0.95, 1.05)

# A loading limit the question sets: a rating or limit named, or a percentage
# about lines, transformers or their loading — not "loads at 125 %", a load
# scaling.
_LOADING_STATED_RE = re.compile(
    r"\b(?:rating|thermal (?:limit|margin)|loading limit|line limit|branch limit|overload(?:ed|ing)? limit)\b|"
    r"\b(?:lines?|branch(?:es)?|transformers?|trafos?)\b[^.?!]{0,25}\d+(?:\.\d+)?\s*%|"
    r"\d+(?:\.\d+)?\s*%[^.?!]{0,25}\b(?:rating|limit|loading)\b", re.IGNORECASE)
_VOLTAGE_STATED_RE = re.compile(r"\bp\.?\s?u\.?\b|\bvoltage (?:band|limit)s?\b|±\s*\d", re.IGNORECASE)


class Reading(NamedTuple):
    """One other way to read the question."""

    key: str                 # "load_power_factor" | "loading_limit" | "voltage_band" | "data_source"
    overrides: dict          # the arguments the study is re-run with
    label: str               # "loads at power factor 0.90"
    assumed: str             # what the answer used instead: "0.95 (the default)"


class Outcome(NamedTuple):
    tool: str
    reading: Reading
    status: str              # holds | shifts | changes | failed
    before: str              # the answer's verdict, in words
    after: str               # under this reading
    seconds: float = 0.0


class Report(NamedTuple):
    outcomes: tuple[Outcome, ...]
    skipped: tuple[str, ...] = ()      # why a study or reading was not re-run

    @property
    def judged(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.status != FAILED)

    @property
    def changes(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.status == CHANGES)

    @property
    def shifts(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.status == SHIFTS)

    @property
    def robust(self) -> bool:
        return bool(self.judged) and not self.changes

    def headline(self) -> str:
        if not self.judged:
            return "No other reading could be checked."
        n = len(self.judged)
        if self.changes:
            return (f"Depends on the reading: the conclusion changes under {len(self.changes)} of "
                    f"{n} other reading{'s' if n != 1 else ''}.")
        if self.shifts:
            return (f"The conclusion holds under {n} other reading{'s' if n != 1 else ''}; "
                    f"{len(self.shifts)} move{'s' if len(self.shifts) == 1 else ''} a figure by more than "
                    f"{SHIFT_SHARE:.0%}.")
        return f"The conclusion holds under {n} other reading{'s' if n != 1 else ''}."

    def as_record(self) -> dict:
        return {
            "headline": self.headline(),
            "outcomes": [{"tool": o.tool, "reading": o.reading.key, "label": o.reading.label,
                          "assumed": o.reading.assumed, "overrides": o.reading.overrides,
                          "status": o.status, "before": o.before, "after": o.after,
                          "seconds": round(o.seconds, 2)} for o in self.outcomes],
            **({"skipped": list(self.skipped)} if self.skipped else {}),
        }


def from_record(data: dict | None) -> Report | None:
    """A report back from a turn record's `reading_check` (the app's history)."""
    if not isinstance(data, dict):
        return None
    outcomes = []
    for o in data.get("outcomes") or []:
        if not isinstance(o, dict):
            continue
        reading = Reading(str(o.get("reading", "")), dict(o.get("overrides") or {}), str(o.get("label", "")),
                          str(o.get("assumed", "")))
        outcomes.append(Outcome(str(o.get("tool", "")), reading, str(o.get("status", FAILED)),
                                str(o.get("before", "")), str(o.get("after", "")), float(o.get("seconds") or 0)))
    return Report(tuple(outcomes), tuple(data.get("skipped") or ()))


# ---------------------------------------------------------------------------
# What a study concluded
# ---------------------------------------------------------------------------

class Verdict(NamedTuple):
    key: Any                 # the conclusion: compared for equality
    figure: float | None     # the figure compared for a shift
    text: str                # in words


def _cheapest(result: dict) -> float | None:
    for option in result.get("options") or []:
        if isinstance(option, dict) and option.get("id") == "cheapest":
            return option.get("cost_eur_per_h")
    bound = result.get("optimality") or {}
    return bound.get("cheapest_eur_per_h")


def _moved(result: dict) -> float:
    return round(sum(abs(float(r.get("Pg_up") or 0)) + abs(float(r.get("Pg_down") or 0))
                     for r in result.get("activated_resources") or [] if isinstance(r, dict)
                     and r.get("type") != "External Grid"), 2)


def verdict(tool: str, result: Any) -> Verdict | None:
    """The conclusion of one study, from its own fields."""
    if not isinstance(result, dict) or "error" in result:
        return None
    if tool == "run_rsa":
        n = int(result.get("total_violations") or 0)
        worst = (result.get("violations") or [{}])[0] if n else {}
        what = worst.get("element") if isinstance(worst, dict) else None
        return Verdict(bool(result.get("secure", n == 0)), float(n),
                       "secure, no violations" if n == 0 else
                       f"not secure, {n} violation{'s' if n != 1 else ''}" + (f" (first at {what})" if what else ""))
    if tool == "simulate_all_contingencies":
        bad = int(result.get("total_outages_causing_violations") or 0)
        tested = result.get("total_outages_tested")
        cut = int(result.get("total_outages_disconnecting_load") or 0)
        secure = bool(result.get("system_n1_secure", bad == 0 and cut == 0))
        return Verdict(secure, float(bad), "N-1 secure" if secure else
                       f"not N-1 secure, {bad} of {tested} outages with violations"
                       + (f", {cut} cutting load off" if cut else ""))
    if tool == "simulate_contingency":
        n = int(result.get("total_violations") or 0)
        secure = bool(result.get("system_secure", n == 0))
        return Verdict(secure, float(n), "secure after the outage" if secure else
                       f"not secure after the outage, {n} violation{'s' if n != 1 else ''}")
    if tool == "investigate":
        # Its conclusion is the main cause: another reading that names another
        # one, or finds nothing wrong, changes it; one that moves its share shifts it.
        status = result.get("status")
        if status == "explained":
            main = result.get("main_cause")
            row = next((f for f in result.get("factors") or [] if f.get("factor") == main), {})
            share = row.get("share_pct")
            # Compared by what kind of cause it is, not by its group's label: the
            # same demand can come back as "near the element" under one reading
            # and as all of it under another (observed with the forecast).
            kind = {"demand": "demand", "units": "units", "unit": "units", "renewables": "renewables",
                    "setpoint": "setpoint", "added": "added"}.get(row.get("kind"))
            key = "explained:" + (kind or re.sub(r"\s*\([^()]*\)\s*$", "", str(main or "")).casefold())
            return Verdict(key, float(share) if share is not None else None,
                           f"main cause {main} ({share:.0f} % of the change)" if main and share is not None
                           else "no single main cause")
        said = {"no_symptom": "nothing outside its limits", "choose": "several elements outside their limits",
                "no_reference": "no moment without the problem", "no_change": "no change to explain",
                "not_decomposed": "the change could not be divided"}
        return Verdict(status, None, said[status]) if status in said else None
    if tool in ("optimize_flexibility", "optimize_contingency"):
        status = result.get("status")
        if status == "options":
            last = result.get("set") == "last_resort"
            cost = _cheapest(result)
            if last:
                return Verdict("last_resort", None, "only load reduction clears the violations")
            return Verdict("options", float(cost) if cost is not None else None,
                           f"corrective options; the cheapest at {cost:.2f} €/h" if cost is not None
                           else "corrective options")
        if status == "no_action_needed":
            return Verdict("no_action", 0.0, "no corrective action needed")
        if result.get("feasible"):
            moved = _moved(result)
            return Verdict("dispatch", moved, f"a corrective dispatch moving {moved:.1f} MW")
        return Verdict("none", None, "no corrective dispatch found" if status != "cap_infeasible" else
                       "no dispatch within the exchange cap")
    return None


def compare(before: Verdict, after: Verdict) -> str:
    if before.key != after.key:
        return CHANGES
    if before.figure is not None and after.figure is not None:
        scale = max(abs(before.figure), 1.0)
        if abs(after.figure - before.figure) > SHIFT_SHARE * scale:
            return SHIFTS
    return HOLDS


# ---------------------------------------------------------------------------
# The readings a question left open
# ---------------------------------------------------------------------------

def _stated(prompt: str) -> set[str]:
    out = set()
    if _PF_WORD_RE.search(prompt):
        out.add("load_power_factor")
    if _FORECAST_WORD_RE.search(prompt) or _ACTUALS_WORD_RE.search(prompt):
        out.add("data_source")
    if requested_limits(prompt) or _VOLTAGE_STATED_RE.search(prompt):
        out.add("voltage_band")
    if _LOADING_STATED_RE.search(prompt):
        out.add("loading_limit")
    return out


def _forecast_covers(grid: dict, timestamp: str | None) -> bool:
    fc = (grid or {}).get("forecasts") or {}
    first, last = fc.get("first_shared"), fc.get("last_shared")
    return bool(timestamp and first and last and str(first) <= str(timestamp) <= str(last))


def _near(a: Any, b: Any, tol: float) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(float(a) - float(b)) <= tol


def readings(tool: str, args: dict, result: dict, prompt: str, grid: dict | None,
             settings: dict | None = None) -> list[Reading]:
    """The other readings of the question this study can be re-run under.

    Open means the operator left it open: the question does not mention it,
    and the study ran with the default — the network's own limits, the default
    load power factor (no session setting), the optimiser's margin, the
    measurements. The agent passes those defaults explicitly, so an argument
    being present settles nothing; a value other than the default does (it
    came from earlier in the conversation, or from a setting the operator
    chose), and is left alone."""
    grid, settings = grid or {}, settings or {}
    stated = _stated(prompt or "")
    out: list[Reading] = []
    timestamp = result.get("timestamp")
    source = (args or {}).get("data_source") or "measurements"
    if "data_source" not in stated and source == "measurements" and _forecast_covers(grid, timestamp):
        out.append(Reading("data_source", {"data_source": "forecasts", "timestamp": timestamp},
                           f"the forecast for {timestamp}", "the measurements"))
    pf = result.get("load_power_factor") if isinstance(result.get("load_power_factor"), dict) else {}
    used_pf = pf.get("value")
    default_pf = settings.get("default_load_power_factor", 0.95)
    if ("load_power_factor" not in stated and settings.get("load_power_factor") is None
            and _near(used_pf, default_pf, 0.005)):
        for alt in PF_ALTERNATIVES:
            if abs(alt - used_pf) > 0.005:
                out.append(Reading("load_power_factor", {"load_power_factor": alt},
                                   f"loads at power factor {alt:.2f}", f"{used_pf:.2f} (the default)"))
    if "loading_limit" not in stated:
        if tool in _ASSESSMENTS:
            used = (result.get("thresholds_used") or {}).get("max_line_loading_pct")
            if _near(used, grid.get("max_loading_pct", 100.0), 0.01):
                alt = 90.0 if used >= 99.0 else 100.0
                out.append(Reading("loading_limit", {"max_line_loading_pct": alt, "max_trafo_loading_pct": alt},
                                   f"branches judged at {alt:g} % of their rating",
                                   f"{used:g} % (the network's own limit)"))
        else:
            used = (args or {}).get("opf_current_safety_margin", result.get("current_safety_margin", 0.9))
            if _near(used, 0.9, 0.001):
                out.append(Reading("loading_limit", {"opf_current_safety_margin": 1.0},
                                   "the dispatch planned to the full rating",
                                   "90 % of the rating (the optimisation's margin)"))
    if "voltage_band" not in stated:
        used = result.get("thresholds_used") or {}
        lo = used.get("vm_lower_pu", result.get("opf_vm_lower_used"))
        hi = used.get("vm_upper_pu", result.get("opf_vm_upper_used"))
        own = (grid.get("vm_lower"), grid.get("vm_upper"))
        if (_near(lo, own[0], 0.001) and _near(hi, own[1], 0.001)
                and max(abs(lo - STANDARD_BAND[0]), abs(hi - STANDARD_BAND[1])) > 0.004):
            out.append(Reading("voltage_band", {"vm_lower_pu": STANDARD_BAND[0], "vm_upper_pu": STANDARD_BAND[1]},
                               "the common ±5 % voltage band (0.95–1.05 p.u.)",
                               f"{lo:g}–{hi:g} p.u. (the network's own band)"))
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _decisive(record: dict) -> list[tuple[str, dict, dict]]:
    """The studies to re-run: the last call of each kind, at most two, the
    later ones first (the answer's conclusion is usually the last study's)."""
    calls = record.get("tool_calls") or []
    results = record.get("tool_results") or []
    seen, out = set(), []
    for call, entry in reversed(list(zip(calls, results))):
        if not isinstance(call, dict) or not isinstance(entry, dict):
            continue
        name = call.get("name")
        result = entry.get("result")
        if name not in STUDIES or name in seen or verdict(name, result) is None or call.get("reused_from"):
            continue
        args = dict(call.get("args") or {})
        if name == "optimize_flexibility" and args.get("n1"):
            continue                                     # an N-1 study: too slow to re-run
        if name == "investigate" and (args.get("start_timestamp") or args.get("end_timestamp")
                                      or args.get("recurring")):
            continue                                     # a window or every episode: too slow to re-run
        seen.add(name)
        out.append((name, args, result))
        if len(out) == MAX_STUDIES:
            break
    return out


def check(record: dict, grid: dict | None = None, dispatch: dict | None = None,
          on_event=None, settings: dict | None = None) -> Report | None:
    """Re-run the turn's decisive studies under the readings its question left
    open. None when there is nothing to re-run. `dispatch`: tool name →
    function (default: the agent's tools); `settings`: the session's settings
    (default: read from the backend)."""
    studies = _decisive(record)
    if not studies:
        return None
    if dispatch is None:
        from .tool_schemas import TOOL_DISPATCH as dispatch
    from .tools import unrecorded

    if settings is None:
        from .tools import _get
        settings = _get("/api/settings")
        settings = settings if isinstance(settings, dict) and "error" not in settings else {}
    plan = []
    for tool, args, result in studies:
        for reading in readings(tool, args, result, record.get("user") or "", grid, settings):
            plan.append((tool, args, result, reading))
    if not plan:
        return Report((), ("the question settled every reading these studies take",))
    # The answer's main study (the last one) first, every reading of it, then
    # the other: a cap that cut the main study's readings short once skipped
    # the one reading that changed its conclusion.
    order = plan
    skipped = [f"{_label(t)}: {r.label}" for t, _, _, r in order[MAX_RERUNS:]]
    outcomes: list[Outcome] = []
    started = time.perf_counter()
    for tool, args, result, reading in order[:MAX_RERUNS]:
        if time.perf_counter() - started > TIME_BUDGET_S:
            skipped.append(f"{_label(tool)}: {reading.label} (time budget)")
            continue
        before = verdict(tool, result)
        if on_event:
            on_event("reading_check", {"name": tool, "reading": reading.label})
        t0 = time.perf_counter()
        try:
            with unrecorded():
                again = dispatch[tool](**{**args, **reading.overrides})
        except Exception as exc:  # noqa: BLE001 — a failed re-run is reported, never fatal
            again = {"error": str(exc)}
        after = verdict(tool, again)
        took = time.perf_counter() - t0
        if after is None:
            why = (again.get("error") if isinstance(again, dict) else None) or "no result"
            outcomes.append(Outcome(tool, reading, FAILED, before.text, f"could not be re-run ({str(why)[:120]})",
                                    took))
            continue
        outcomes.append(Outcome(tool, reading, compare(before, after), before.text, after.text, took))
    return Report(tuple(outcomes), tuple(skipped))


# The studies as a sentence names them.
NOUNS = {"run_rsa": "security assessment", "simulate_contingency": "single-outage study",
         "simulate_all_contingencies": "N-1 screen", "optimize_flexibility": "flexibility study",
         "optimize_contingency": "post-outage optimisation", "investigate": "investigation"}


# What the reading check cannot tell, for the card's limits.
BLIND_SPOT = ("The reading check varies only the load power factor, the loading limit, the voltage band and "
              "measured or forecast data, for the assessment, the N-1 screens and the two optimisations; a "
              "question open in another way is not re-checked.")

# How CONDUCTOR's note under an answer begins (the app sets it apart).
NOTE_PREFIX = "**Reading check.** CONDUCTOR re-ran"


def note(report: Report | None) -> str:
    """CONDUCTOR's own note under an answer whose conclusion depends on the
    reading ('' otherwise). One line: provenance leaves it out of the claims."""
    if report is None or not report.changes:
        return ""
    def finds(after: str) -> str:
        # "finds it not secure" for a verdict on the grid; "finds a corrective dispatch" for a finding.
        return f"finds it {after}" if after.startswith(("secure", "not secure", "N-1", "not N-1")) else f"finds {after}"
    parts = [f"with {o.reading.label} instead of {o.reading.assumed}, the {NOUNS.get(o.tool, _label(o.tool))} "
             f"{finds(o.after)}" for o in report.changes]
    return (f"{NOTE_PREFIX} the studies under readings the question left open, and the "
            "conclusion changes: " + "; ".join(parts) + ". Say which reading you meant.")
