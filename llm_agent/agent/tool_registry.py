"""
tool_registry.py — what CONDUCTOR knows about each of its tools, in one table.

The tool's *schema* (what the model may send) lives in `tool_schemas.py` and
its *function* in `tools.py`. Everything else about a tool is here, one row
per tool, and every other module reads it instead of keeping its own list:

- `label`         the name operators see (live status, audit panel, captions)
- `group`         its section in the documentation's tool reference
- `effect`        what calling it changes — nothing, the session's settings, a
                  proposal, or the simulation clock (deduplication, docs, and
                  any future confirmation step read this)
- `data_source`, `timestamp`, `network`, `load_pf`
                  the shared arguments `tool_schemas.py` adds to its schema
- `over_time`     it studies more than one moment (the advisor's and the
                  graders' "looked beyond one moment" check)
- `long_series`   its result carries per-tick series the model is not sent
- `evidence`      what backs a figure from its result (evidence.py): an AC
                  power flow, an optimiser's local optimum, a statistic over
                  sampled scenarios, or data read back

Adding a tool: one row here, its schema, its function, and (optionally) a
renderer. `tests/test_tool_registry.py` fails while any of them is missing.
"""

from __future__ import annotations

from typing import NamedTuple

# What a call changes.
READS = "reads"                          # reads data or runs a study; changes nothing
CHANGES_SETTINGS = "changes_settings"    # the session's settings: power factor, generator models, costs, names
CHANGES_PROPOSAL = "changes_proposal"    # a proposal (never the loaded network)
MOVES_CLOCK = "moves_clock"              # the simulation clock

EFFECT_TEXT = {
    READS: "Reads only: changes nothing.",
    CHANGES_SETTINGS: "Changes the session's settings, for every later study.",
    CHANGES_PROPOSAL: "Creates or changes a proposal; the loaded network is never modified.",
    MOVES_CLOCK: "Moves the simulation clock.",
}

# What backs a figure a tool returns (evidence.py), strongest first. "proved"
# is never a tool's default: only certificates inside a result (a relaxation's
# bound, its proof that no dispatch exists) are proofs.
PROVED = "proved"
POWER_FLOW = "power_flow"     # an AC power flow of the network: physics, for the data given
OPTIMISED = "optimised"       # an optimiser's solution: a local optimum
ESTIMATED = "estimated"       # a statistic over sampled scenarios
INPUT = "input"               # read back: data, the network, settings
EVIDENCE = (PROVED, POWER_FLOW, OPTIMISED, ESTIMATED, INPUT)

# Which datasets a tool can read: none, the measured or forecast series, or
# also the network's own nominal point (point-in-time studies).
SERIES, POINT = "series", "point"

# The documentation's sections, in order.
GROUPS = (
    "Reading the grid and the data",
    "Security at one moment",
    "Over time, scenarios and risk",
    "Optimisation and flexibility",
    "Key performance indicators",
    "What-if proposals",
    "Settings",
)


class Tool(NamedTuple):
    label: str
    group: str
    effect: str = READS
    # Only for tools that change settings: called without changes (or a reset)
    # they just read the current settings.
    reads_without_changes: bool = False
    data_source: str | None = None
    timestamp: bool = False
    network: bool = False
    load_pf: bool = False
    over_time: bool = False
    long_series: bool = False
    evidence: str = INPUT


_STUDY = dict(network=True, load_pf=True)                 # every study of a network at an operating point
_AT_A_MOMENT = dict(_STUDY, data_source=POINT, timestamp=True)
_OVER_A_WINDOW = dict(_STUDY, data_source=SERIES, over_time=True)

REGISTRY: dict[str, Tool] = {
    # Reading the grid and the data
    "locate_network_element": Tool("Element lookup", GROUPS[0], network=True),
    "get_current_timestamp": Tool("Reading timestamp", GROUPS[0]),
    "advance_timestamp": Tool("Advancing timestamp", GROUPS[0], effect=MOVES_CLOCK),
    "get_current_conditions": Tool("Current conditions", GROUPS[0], **_AT_A_MOMENT),
    "get_element_timeseries": Tool("Element time series", GROUPS[0], long_series=True, **_OVER_A_WINDOW),
    "get_data_quality": Tool("Data quality report", GROUPS[0]),
    # Security at one moment
    "run_rsa": Tool("Security assessment (RSA)", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "simulate_contingency": Tool("Single N-1 contingency", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "simulate_all_contingencies": Tool("N-1 contingency screen", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "compute_violation_attribution": Tool("Violation attribution", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "investigate": Tool("Investigation (why)", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "test_prevention": Tool("Prevention check", GROUPS[1], evidence=POWER_FLOW, **_AT_A_MOMENT),
    # The forecast's next hours: what is coming and what would prevent it.
    "look_ahead": Tool("Look-ahead", GROUPS[1], evidence=POWER_FLOW, **_OVER_A_WINDOW),
    # The agent's own record of an investigation: it runs no study.
    "record_investigation": Tool("Investigation record", GROUPS[1]),
    "compare_results": Tool("Result comparison", GROUPS[1], evidence=POWER_FLOW),
    # Over time, scenarios and risk
    # The scan steps the live clock forward as it goes: measurements only.
    "scan_rsa_over_time": Tool("Security scan over time", GROUPS[2], effect=MOVES_CLOCK, over_time=True,
                               evidence=POWER_FLOW,
                               **_STUDY),
    "find_worst_case_timestamp": Tool("Worst-case scan", GROUPS[2], evidence=POWER_FLOW, long_series=True, **_OVER_A_WINDOW),
    "scan_scenarios": Tool("Renewable scenario scan", GROUPS[2], evidence=POWER_FLOW, long_series=True, **_OVER_A_WINDOW),
    "run_probabilistic_rsa": Tool("Probabilistic security assessment", GROUPS[2], evidence=ESTIMATED, over_time=True,
                                  **_AT_A_MOMENT),
    "compute_historical_risk": Tool("Historical risk", GROUPS[2], evidence=POWER_FLOW, **_OVER_A_WINDOW),
    # Optimisation and flexibility
    "optimize_flexibility": Tool("Flexibility OPF (deterministic)", GROUPS[3], evidence=OPTIMISED, **_AT_A_MOMENT),
    "optimize_contingency": Tool("Post-contingency OPF", GROUPS[3], evidence=OPTIMISED, **_AT_A_MOMENT),
    "optimize_robust_flexibility": Tool("Robust flexibility OPF", GROUPS[3], evidence=OPTIMISED, **_AT_A_MOMENT),
    "compute_flexibility_envelope": Tool("Flexibility envelope", GROUPS[3], evidence=POWER_FLOW, **_AT_A_MOMENT),
    "compute_hosting_capacity": Tool("Hosting capacity", GROUPS[3], evidence=POWER_FLOW, **_AT_A_MOMENT),
    # Its window is its own (start, end); a held option brings its own moment.
    "flexibility_over_time": Tool("Flexibility over time", GROUPS[3], evidence=OPTIMISED, **_OVER_A_WINDOW),
    # Key performance indicators
    "evaluate_kpis": Tool("KPI evaluation", GROUPS[4], evidence=OPTIMISED, **_AT_A_MOMENT),
    # Its window is its own (start, end); a held option brings its own moment.
    "kpis_over_time": Tool("KPIs over time", GROUPS[4], evidence=OPTIMISED, **_OVER_A_WINDOW),
    # What-if proposals
    "edit_proposal": Tool("Network proposal", GROUPS[5], effect=CHANGES_PROPOSAL),
    # Names its proposals itself, so it takes no `network`.
    "compare_networks": Tool("Base vs proposal", GROUPS[5], data_source=POINT, timestamp=True, load_pf=True,
                             long_series=True, evidence=POWER_FLOW),
    # Settings
    "set_load_power_factor": Tool("Load power factor setting", GROUPS[6], effect=CHANGES_SETTINGS),
    "set_generator_model": Tool("Generator model", GROUPS[6], effect=CHANGES_SETTINGS,
                                reads_without_changes=True),
    "set_unit_costs": Tool("Unit costs", GROUPS[6], effect=CHANGES_SETTINGS, reads_without_changes=True),
    "rename_elements": Tool("Renaming", GROUPS[6], effect=CHANGES_SETTINGS, reads_without_changes=True),
}

OVER_TIME = frozenset(n for n, t in REGISTRY.items() if t.over_time)
# Studies of one moment: they take a `timestamp` (and run at the clock without one).
POINT_IN_TIME = frozenset(n for n, t in REGISTRY.items() if t.timestamp)
LONG_SERIES = frozenset(n for n, t in REGISTRY.items() if t.long_series)


def evidence(name: str) -> str:
    """What backs a figure this tool returns (before what is inside the result)."""
    tool = REGISTRY.get(name)
    return tool.evidence if tool else INPUT


def label(name: str) -> str:
    """The name operators see; an unknown tool's own name, readable."""
    tool = REGISTRY.get(name)
    return tool.label if tool else name.replace("_", " ").title()


def changes_state(name: str, kwargs: dict | None = None) -> bool:
    """Whether this call changes what later calls would return."""
    tool = REGISTRY.get(name)
    if tool is None or tool.effect == READS:
        return False
    if tool.reads_without_changes:
        kwargs = kwargs or {}
        return bool(kwargs.get("changes")) or bool(kwargs.get("reset"))
    return True


def tools_in(group: str) -> list[str]:
    return [n for n, t in REGISTRY.items() if t.group == group]
