"""
validators.py — parameter and result integrity for LLM-chosen tool calls.

Every number CONDUCTOR reports comes from a validated solver, but a validated
solver given invalid parameters returns a confidently wrong answer. Schema
validation catches type errors; it does not catch a voltage band whose lower
bound exceeds its upper bound, a load-scaling factor outside what the network
supports, or an OPF that reports success while leaving buses outside the very
bounds it claims to have enforced.

This module guards two seams:

  * `validate_call`   — before dispatch, on the arguments the model chose.
  * `validate_result` — after execution, checking a result against itself.

Severity policy
---------------
`reject` is reserved for the logically impossible or self-contradictory:
requests no parameterisation could satisfy. These are returned to the model as
structured errors so it can correct itself, which it does reliably.

`warn` covers the unusual-but-legal. These execute and the result is annotated,
so capability is preserved while the parameterisation stays auditable.
Over-rejecting would destroy the flexibility that makes the agent useful, so
anything merely surprising warns rather than blocks.

Result checks never drop data — they annotate. A model that has already seen a
result should be told the result is suspect, not have it silently removed.
"""

from __future__ import annotations

import re

from typing import Any, Iterable, NamedTuple

from .config import DEFAULT_GRID_CONSTANTS, last_grid_constants_status

REJECT = "reject"
WARN = "warn"

# Voltages outside this band are not an operating condition; they indicate a
# broken model or corrupt data.
_VM_PLAUSIBLE = (0.5, 1.5)
# Voltages legal but far from anything an operator would normally request.
_VM_TYPICAL = (0.80, 1.20)
# Tolerance when checking a solver's output against its own declared bounds.
_BOUND_TOL = 1e-4


class Issue(NamedTuple):
    severity: str
    code: str
    field: str
    message: str

    def render(self) -> str:
        return f"[{self.severity}] {self.field}: {self.message}"


class Verdict(NamedTuple):
    issues: tuple[Issue, ...] = ()

    @property
    def rejected(self) -> bool:
        return any(i.severity == REJECT for i in self.issues)

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARN]

    @property
    def rejections(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == REJECT]

    def as_tool_error(self, tool_name: str) -> dict:
        """A structured error the model can read and correct from."""
        problems = [i.render() for i in self.rejections]
        return {
            "error": (
                f"{tool_name} was not executed: the requested parameters are "
                f"invalid. " + " ".join(i.message for i in self.rejections)
            ),
            "invalid_parameters": [i.field for i in self.rejections],
            "problems": problems,
        }


# ---------------------------------------------------------------------------
# Range rules, keyed by parameter name.
#
# Parameter names carry the same meaning across every tool that accepts them
# (`load_sigma` means the same thing in all three of its tools), so rules are
# defined once per name rather than per tool.
# ---------------------------------------------------------------------------

# Closed [0, 1]: fractions and standard deviations expressed as fractions.
_UNIT_CLOSED = {
    "load_sigma", "sgen_sigma", "added_gen_sigma", "allowed_violation_fraction",
    "near_miss_band", "min_improvement",
}
# Open (0, 1): probabilities that cannot be certainty or impossibility.
_UNIT_OPEN = {"risk_target", "risk_threshold", "alpha", "beta", "confidence", "target_p_any"}
# (0, 1]: power factors.
_POWER_FACTOR = {"opf_min_power_factor", "power_factor"}
# Strictly positive integers — counts and resolutions.
_POSITIVE_INT = {
    "n_samples", "n_scenarios", "n_steps", "n_bins", "resolution", "step_size",
    "max_iter", "max_workers", "validation_samples", "scenario_k_cap", "worst_n",
    "top_n_sources", "top_n_drivers",
}
# Physically non-negative quantities.
_NON_NEGATIVE = {
    "slack_max_mw", "slack_q_max_mvar", "tolerance_mw", "p_max_search_mw",
    "opf_lambda_p", "opf_lambda_q", "delta_mw", "delta_mvar",
}
# Voltage magnitudes in per-unit.
_VOLTAGE_PU = {"vm_lower_pu", "vm_upper_pu", "opf_vm_lower", "opf_vm_upper"}

# Ordered pairs that must not be inverted.
_ORDERED_PAIRS = (
    ("vm_lower_pu", "vm_upper_pu", "voltage band"),
    ("opf_vm_lower", "opf_vm_upper", "OPF voltage envelope"),
    ("p_min_mw", "p_max_mw", "active power range"),
    ("q_min_mvar", "q_max_mvar", "reactive power range"),
)

# Known-good enum values. These WARN rather than reject: the backend is the
# authority on what it accepts, and a wrongly-rejected valid value would break
# a working capability. The warning still surfaces a likely typo.
_ENUMS = {
    "data_source": {"measurements", "forecasts"},
    "element_type": {"bus", "line", "trafo"},
    "metric": {"violations", "slack_import", "max_voltage", "min_voltage", "max_loading"},
    "q_mode": {"unity", "fixed_pf", "reactive_proxy", "all"},
    "robust_method": {"heuristic", "scenario"},
    "pf_sign": {"absorbing", "injecting"},
    "uncertainty_scope": {"added_generation_only", "added_generation_plus_load"},
    "condition": {"hour", "load", "slack_import", "slack", "slack_import_mw", "cable_flow"},
}


def _grid(grid: dict | None) -> dict:
    if grid is not None:
        return grid
    status = last_grid_constants_status()
    return status.values if status is not None else DEFAULT_GRID_CONSTANTS


def _numeric(value: Any) -> float | None:
    """Coerce to float, or None when the value is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _check_ranges(kwargs: dict) -> Iterable[Issue]:
    for field, raw in kwargs.items():
        value = _numeric(raw)

        if field in _UNIT_CLOSED and value is not None and not 0.0 <= value <= 1.0:
            yield Issue(REJECT, "out_of_range", field,
                        f"{field}={raw} must be a fraction between 0 and 1.")
        if field in _UNIT_OPEN and value is not None and not 0.0 < value < 1.0:
            yield Issue(REJECT, "out_of_range", field,
                        f"{field}={raw} must be a probability strictly between 0 and 1.")
        if field in _POWER_FACTOR and value is not None and not 0.0 < value <= 1.0:
            yield Issue(REJECT, "out_of_range", field,
                        f"{field}={raw} must be greater than 0 and at most 1.")
        if field in _POSITIVE_INT and value is not None and value < 1:
            yield Issue(REJECT, "out_of_range", field,
                        f"{field}={raw} must be at least 1.")
        if field in _NON_NEGATIVE and value is not None and value < 0:
            yield Issue(REJECT, "out_of_range", field,
                        f"{field}={raw} cannot be negative.")

        if field in _VOLTAGE_PU and value is not None:
            low, high = _VM_PLAUSIBLE
            if not low <= value <= high:
                yield Issue(REJECT, "implausible_voltage", field,
                            f"{field}={raw} p.u. is outside the physically "
                            f"plausible range {low}–{high}.")
            else:
                tlow, thigh = _VM_TYPICAL
                if not tlow <= value <= thigh:
                    yield Issue(WARN, "unusual_voltage", field,
                                f"{field}={raw} p.u. is far outside normal "
                                f"operating limits ({tlow}–{thigh}).")

        if field in _ENUMS and isinstance(raw, str) and raw not in _ENUMS[field]:
            yield Issue(WARN, "unknown_value", field,
                        f"{field}={raw!r} is not a recognised value "
                        f"({', '.join(sorted(_ENUMS[field]))}).")


def _check_pairs(kwargs: dict) -> Iterable[Issue]:
    for low_field, high_field, label in _ORDERED_PAIRS:
        low, high = _numeric(kwargs.get(low_field)), _numeric(kwargs.get(high_field))
        if low is not None and high is not None and low >= high:
            yield Issue(REJECT, "inverted_range", low_field,
                        f"{label} is inverted: {low_field}={low} is not below "
                        f"{high_field}={high}.")


def _check_against_network(kwargs: dict, grid: dict) -> Iterable[Issue]:
    """Checks that depend on the network actually loaded."""
    scaling = _numeric(kwargs.get("load_scaling_factor"))
    if scaling is not None:
        lo = grid.get("load_scaling_min", 0.0)
        hi = grid.get("load_scaling_max", 4.0)
        if not lo <= scaling <= hi:
            yield Issue(REJECT, "out_of_range", "load_scaling_factor",
                        f"load_scaling_factor={scaling} is outside the range "
                        f"{lo}–{hi} supported by this network.")

    slack = _numeric(kwargs.get("slack_max_mw"))
    if slack is not None:
        hi = grid.get("slack_max_mw_max")
        if hi is not None and slack > hi:
            yield Issue(REJECT, "out_of_range", "slack_max_mw",
                        f"slack_max_mw={slack} exceeds this network's maximum "
                        f"interconnection capacity of {hi} MW.")

    # Requested limits far from the network's own are legal but worth flagging:
    # they are usually a misread of the grid's actual security limits.
    for field, key in (("vm_lower_pu", "vm_lower"), ("vm_upper_pu", "vm_upper")):
        value = _numeric(kwargs.get(field))
        reference = grid.get(key)
        if value is not None and reference is not None and abs(value - reference) > 0.1:
            yield Issue(WARN, "far_from_network_limits", field,
                        f"{field}={value} differs from this network's {key}="
                        f"{reference} by more than 0.1 p.u.")


def _check_overrides(kwargs: dict) -> Iterable[Issue]:
    """Generator capability overrides must be non-negative and correctly ordered."""
    pg_max = kwargs.get("pg_max_overrides") or {}
    pg_min = kwargs.get("pg_min_overrides") or {}

    for field, mapping in (("pg_max_overrides", pg_max), ("pg_min_overrides", pg_min)):
        if not isinstance(mapping, dict):
            continue
        for gen, raw in mapping.items():
            value = _numeric(raw)
            if value is not None and value < 0:
                yield Issue(REJECT, "negative_capacity", field,
                            f"{field}[{gen!r}]={raw} — generator capacity "
                            "cannot be negative.")

    if isinstance(pg_max, dict) and isinstance(pg_min, dict):
        for gen in set(pg_min) & set(pg_max):
            lo, hi = _numeric(pg_min[gen]), _numeric(pg_max[gen])
            if lo is not None and hi is not None and lo > hi:
                yield Issue(REJECT, "inverted_range", "pg_min_overrides",
                            f"{gen!r}: minimum output {lo} MW exceeds maximum "
                            f"{hi} MW — no dispatch can satisfy both.")


def validate_call(tool_name: str, kwargs: dict, grid: dict | None = None) -> Verdict:
    """Check the arguments the model chose, before the tool runs."""
    constants = _grid(grid)
    issues = [
        *_check_ranges(kwargs),
        *_check_pairs(kwargs),
        *_check_against_network(kwargs, constants),
        *_check_overrides(kwargs),
    ]
    return Verdict(tuple(issues))


# ---------------------------------------------------------------------------
# Result checks — a result contradicting itself is the strongest available
# signal that a solve went wrong while still reporting success.
# ---------------------------------------------------------------------------


def _voltages(entries: Any) -> list[tuple[str, float]]:
    """Extract (name, vm_pu) pairs from a bus-voltage list."""
    out: list[tuple[str, float]] = []
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        vm = _numeric(entry.get("vm_pu"))
        if vm is not None:
            out.append((str(entry.get("bus_name", entry.get("bus", "?"))), vm))
    return out


def _check_physical_plausibility(result: dict) -> Iterable[Issue]:
    low, high = _VM_PLAUSIBLE
    for key in ("bus_voltages_post_opf", "bus_voltages_base", "all_voltages"):
        for name, vm in _voltages(result.get(key)):
            if not low <= vm <= high:
                yield Issue(WARN, "implausible_result", key,
                            f"{name} reports {vm} p.u., outside the physically "
                            f"plausible range {low}–{high}. Treat this result as "
                            "unreliable and report the anomaly.")
                return  # one is enough; the whole solve is suspect


def _check_opf_self_consistency(result: dict) -> Iterable[Issue]:
    """An OPF reporting success must respect the bounds it says it enforced."""
    if str(result.get("status", "")).lower() not in {"optimal", "solved", "success"}:
        return

    lower = _numeric(result.get("opf_vm_lower_used"))
    upper = _numeric(result.get("opf_vm_upper_used"))
    if lower is None or upper is None:
        return

    excluded = {
        str(e.get("bus_name", e.get("bus")))
        for e in result.get("excluded_buses_post_opf") or []
        if isinstance(e, dict)
    }

    for name, vm in _voltages(result.get("bus_voltages_post_opf")):
        if name in excluded:
            continue
        if vm < lower - _BOUND_TOL or vm > upper + _BOUND_TOL:
            yield Issue(WARN, "opf_bound_violation", "bus_voltages_post_opf",
                        f"The optimisation reported success while leaving {name} "
                        f"at {vm} p.u., outside the bounds it enforced "
                        f"({lower}–{upper}). Do not present this dispatch as "
                        "secure without flagging the inconsistency.")
            return


def _check_violation_accounting(result: dict) -> Iterable[Issue]:
    """The violation count must match the violation list and the voltage table."""
    total = result.get("total_violations")
    violations = result.get("violations")

    if isinstance(total, int) and isinstance(violations, list) and len(violations) != total:
        yield Issue(WARN, "inconsistent_count", "total_violations",
                    f"total_violations={total} but the violations list holds "
                    f"{len(violations)} entries.")

    thresholds = result.get("thresholds_used")
    if not (isinstance(total, int) and total == 0 and isinstance(thresholds, dict)):
        return

    lower = _numeric(thresholds.get("vm_lower_pu"))
    upper = _numeric(thresholds.get("vm_upper_pu"))
    if lower is None or upper is None:
        return

    for name, vm in _voltages(result.get("all_voltages")):
        if vm < lower - _BOUND_TOL or vm > upper + _BOUND_TOL:
            yield Issue(WARN, "unreported_violation", "total_violations",
                        f"The assessment reported no violations, but {name} is at "
                        f"{vm} p.u., outside the thresholds it applied "
                        f"({lower}–{upper}).")
            return


def validate_result(tool_name: str, result: Any, grid: dict | None = None) -> list[Issue]:
    """Check a tool result against itself. Never modifies or discards data."""
    if not isinstance(result, dict) or "error" in result:
        return []
    return [
        *_check_physical_plausibility(result),
        *_check_opf_self_consistency(result),
        *_check_violation_accounting(result),
    ]


def annotate(result: Any, issues: list[Issue]) -> Any:
    """Attach integrity findings to a result so the model must account for them."""
    if not issues or not isinstance(result, dict):
        return result
    return {
        **result,
        "_integrity_warnings": [
            {"code": i.code, "field": i.field, "message": i.message} for i in issues
        ],
    }

# ---------------------------------------------------------------------------
# Cross-tool consistency within a turn
#
# A single answer is often assembled from several tool calls, and the model has
# to carry the operating point from one to the next. It does not do this
# reliably: in testing it passed an explicit timestamp to the security
# assessment and omitted it from the follow-up attribution, which silently fell
# back to the simulation clock. Both results were individually correct, the
# composition was not, and nothing in the output revealed it.
#
# Prompt instructions to "propagate constraints" do not make this reliable, so
# the mismatch is detected here instead.
# ---------------------------------------------------------------------------

# Fields describing *which* operating point a result belongs to.
_CONTEXT_LABELS = {
    "timestamp": "operating point",
    "data_source": "dataset",
}
_THRESHOLD_FIELDS = ("vm_lower_pu", "vm_upper_pu",
                     "max_line_loading_pct", "max_trafo_loading_pct")


def _normalise_timestamp(value: str) -> str:
    """Compare to the minute: tools echo timestamps with and without seconds."""
    return value.strip()[:16]


_REQUESTED_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?")


def requested_timestamps(text: str) -> frozenset:
    """Operating points the user named, normalised to the minute.

    A turn that runs tools at two moments is drift when nobody asked for two,
    and exactly right when somebody did. The question is in the prompt, so the
    check reads it rather than guessing.
    """
    return frozenset(
        _normalise_timestamp(m.group(0).replace("T", " "))
        for m in _REQUESTED_TS_RE.finditer(text or "")
    )


# A voltage limit the user states in words. Deliberately narrow: a bare "1.045"
# somewhere in a sentence is not a limit, and reading it as one would silently
# re-run every study against a number the operator never meant as a ceiling.
_REQUESTED_LIMIT_RES = (
    ("vm_upper_pu", re.compile(
        r"(?:max(?:imum)?|upper|no (?:more|higher) than|below|under|ceiling|cap(?:ped)?)"
        r"[^.\n]{0,40}?\b(0?\.\d+|1\.\d+)\b", re.IGNORECASE)),
    ("vm_lower_pu", re.compile(
        r"(?:min(?:imum)?|lower|no (?:less|lower) than|above|at least|floor)"
        r"[^.\n]{0,40}?\b(0?\.\d+|1\.\d+)\b", re.IGNORECASE)),
)

# Plausible p.u. voltages. A "maximum of 100" is a loading percentage, and a
# limit parser that accepted it would propagate nonsense into every study.
_PU_RANGE = (0.80, 1.20)


def requested_limits(text: str) -> dict:
    """Voltage limits the user stated in the prompt itself.

    Inheritance carries a limit forward from an earlier *call*, which leaves
    the first call of a turn uncovered. Observed live: asked a question that
    set a 1.045 ceiling, the agent ran a week-long scan first — with nothing
    established yet, so at the tool default of 1.06 — and then three studies at
    1.045. The consistency guard correctly reported that these were not one
    assessment, which is a warning about a gap rather than a fix for it.

    The ceiling is in the prompt. Reading it there means the first call is
    covered too, on the same rule the timestamps already use.
    """
    found: dict[str, float] = {}
    for field, pattern in _REQUESTED_LIMIT_RES:
        for match in pattern.finditer(text or ""):
            try:
                value = float(match.group(1))
            except ValueError:  # pragma: no cover
                continue
            if _PU_RANGE[0] <= value <= _PU_RANGE[1]:
                found.setdefault(field, value)
                break
    return found


def operating_context(result: Any) -> dict:
    """What operating point a tool result was actually computed at."""
    if not isinstance(result, dict):
        return {}

    context: dict[str, Any] = {}
    for field in _CONTEXT_LABELS:
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            context[field] = (
                _normalise_timestamp(value) if field == "timestamp" else value.strip()
            )

    thresholds = result.get("thresholds_used")
    if isinstance(thresholds, dict):
        used = {k: thresholds.get(k) for k in _THRESHOLD_FIELDS if k in thresholds}
        if used:
            context["thresholds"] = used
    return context


class TurnConsistency:
    """
    Accumulates the operating context of each result within one turn and
    reports when a later tool disagrees with an earlier one.

    Only the *first* observation of a field establishes the reference, so a
    single stray call is reported against the established context rather than
    silently redefining it.

    `expected` are the operating points the user asked about. Comparing two
    moments is a normal request, and flagging it as drift told the model its
    correct behaviour was inconsistent — so timestamps the user named are not
    reported against each other. A tool that wanders to a *third* moment
    nobody asked for still is.
    """

    def __init__(self, expected: frozenset | None = None) -> None:
        self._seen: dict[str, tuple[Any, str]] = {}
        self._expected = frozenset(expected or ())

    def observe(self, tool_name: str, result: Any) -> list[Issue]:
        context = operating_context(result)
        issues: list[Issue] = []

        for field, value in context.items():
            if field not in self._seen:
                self._seen[field] = (value, tool_name)
                continue

            established, source = self._seen[field]
            if value == established:
                continue

            if (field == "timestamp"
                    and value in self._expected and established in self._expected):
                # Both were asked for; this is a comparison, not a slip.
                continue

            if field == "thresholds":
                differing = [
                    k for k in value
                    if k in established and value[k] != established[k]
                ]
                if not differing:
                    continue
                detail = ", ".join(
                    f"{k}={established[k]} in {source} but {value[k]} in {tool_name}"
                    for k in differing
                )
                issues.append(Issue(
                    WARN, "inconsistent_thresholds", field,
                    f"Analyses in this answer used different limits: {detail}. "
                    "They cannot be presented as one assessment.",
                ))
            else:
                label = _CONTEXT_LABELS[field]
                issues.append(Issue(
                    WARN, "inconsistent_context", field,
                    f"This answer combines analyses of different {label}s: "
                    f"{source} used {established}, {tool_name} used {value}. "
                    f"Results from different {label}s are not comparable — say "
                    "which each figure came from, or re-run so they agree.",
                ))

        return issues

# ---------------------------------------------------------------------------
# Turn-scoped operating point
#
# `TurnConsistency` above detects a chain that mixed two operating points. This
# prevents it. Measured on the live system, the agent dropped an explicitly
# stated timestamp from the second call of a two-tool chain in 4 of 10 runs at
# temperature 0 — the answer then described a different day while claiming the
# requested one.
#
# Requiring the argument would force the model to supply *a* timestamp, not the
# right one, and would break the common case where omitting it correctly means
# "the current operating point". Instead the loop resolves it: the first
# explicit value in a turn defines that turn's operating point, and later calls
# that omit it inherit rather than silently falling back to the simulation
# clock. An explicit value always wins, so a deliberate two-timestamp
# comparison still works.
# ---------------------------------------------------------------------------

# Arguments that identify *which* operating point a call runs against.
_OPERATING_POINT_ARGS = ("timestamp", "data_source")

# Limits the user set for this turn. A request like "would it still be secure
# with a maximum voltage of 1.045, and if contingencies happen?" decomposes
# into two studies, and the ceiling belongs to both: an N-1 screen run at the
# default 1.05 after an N-0 run at 1.045 answers a question nobody asked, and
# reports a security margin the operator did not ask about.
#
# The submitted paper's Act 1 turns on the agent propagating this correctly.
# It did — and nothing in the system required it to, which is the same gap the
# operating point had before it was carried in code rather than in the prompt.
_CONSTRAINT_ARGS = (
    "vm_upper_pu",
    "vm_lower_pu",
    "max_line_loading_pct",
    "max_trafo_loading_pct",
)

_INHERITED_ARGS = _OPERATING_POINT_ARGS + _CONSTRAINT_ARGS

# The same quantity under a different name.
#
# The tools once spelled the OPF envelope `opf_vm_upper` while everything else
# said `vm_upper_pu`, so a ceiling the operator set for an assessment did not
# reach the optimisation — one limit with two names is a propagation failure
# waiting to happen, and it happened. The tool layer now uses one name
# everywhere and translates at the wire, which is the real fix: an alias the
# guards must know about is a guard that can be forgotten.
#
# This mapping stays as a backstop. Nothing in the current schemas emits the
# old spelling, but a model that has seen it, a hand-written call, or a future
# tool that reintroduces it would otherwise silently bypass propagation.
_ALIASES = {
    "opf_vm_upper": "vm_upper_pu",
    "opf_vm_lower": "vm_lower_pu",
}


class TurnOperatingPoint:
    """Carries the operating point and the turn's limits across its tool calls.

    Named for the operating point because that is what it originally carried;
    it now also carries user-set thresholds, on the same rule — the first
    explicit value in a turn defines it, and later calls that leave the field
    empty inherit rather than falling back to a default.
    """

    def __init__(self, requested: dict | None = None) -> None:
        self._established: dict[str, tuple[Any, str]] = {}
        # Limits the user stated in the prompt establish the turn's reference
        # before any tool runs, so the first call is covered like the rest.
        for field, value in (requested or {}).items():
            self._established[field] = (value, "the request")

    def resolve(
        self, tool_name: str, kwargs: dict, accepts: set[str] | None = None
    ) -> tuple[dict, dict]:
        """
        Return (kwargs, inherited) for a call about to be dispatched.

        `accepts` is the set of argument names this tool actually takes; an
        argument is never injected into a tool that would reject it. Passing
        None means "inject nothing", which keeps the behaviour safe for callers
        that cannot determine the tool's signature.
        """
        resolved = dict(kwargs)
        inherited: dict[str, Any] = {}

        # An alias carries the same quantity under a different name, so an
        # explicit `opf_vm_upper` establishes the turn's `vm_upper_pu` and vice
        # versa. Without this the OPF is an island: it neither inherits the
        # ceiling the operator set nor contributes its own to later calls.
        for alias, canonical in _ALIASES.items():
            explicit = resolved.get(alias)
            if explicit not in (None, ""):
                self._established.setdefault(canonical, (explicit, tool_name))

        for field in _INHERITED_ARGS:
            explicit = resolved.get(field)
            if explicit not in (None, ""):
                # First explicit value in the turn defines the operating point.
                self._established.setdefault(field, (explicit, tool_name))
                continue

            if accepts is None or field not in accepts:
                continue
            if field not in self._established:
                continue

            value, source = self._established[field]
            resolved[field] = value
            inherited[field] = {"value": value, "from": source}

        # Fill an alias this tool accepts from the canonical value, so a limit
        # set as `vm_upper_pu` reaches a tool that spells it `opf_vm_upper`.
        for alias, canonical in _ALIASES.items():
            if resolved.get(alias) not in (None, ""):
                continue
            if accepts is None or alias not in accepts:
                continue
            if canonical not in self._established:
                continue
            value, source = self._established[canonical]
            resolved[alias] = value
            inherited[alias] = {"value": value, "from": source}

        return resolved, inherited
