"""
tool_schemas.py — Gemini FunctionDeclaration schemas for the digital twin tools.

Every schema here is sent on every request, so description length is a running
cost: these twenty-two tools are ~11.9k tokens of each prompt (~13.8k before
the October 2026 trim of repeated OPF parameter advice, "Returns:" field lists
and example-question lists). This is the compacted form, kept after a trial
against a longer version — descriptions were tightened and grid-specific facts
(substation names included) removed, deferring to the system prompt,
which is rebuilt from live backend constants and therefore describes the
network actually loaded. Schemas are built once at import and cannot follow an
upload.

Exports:
  TOOLS: list           — passable directly to genai.GenerativeModel(tools=TOOLS)
  TOOL_DISPATCH: dict   — maps schema name → implementation in tools.py
"""

from __future__ import annotations

from typing import Callable

from google import genai

from . import tools
from .tool_registry import POINT, REGISTRY

# Tool schemas are built once at import, so anything grid-specific baked in here
# is frozen to whatever network happened to be the default — it does NOT follow
# the network the user actually loads. Substation names and the slack-bus name
# therefore live only in the system prompt, which is rebuilt from live backend
# constants on every turn (see system_prompt.build_system_prompt). Descriptions
# below point at it rather than repeating values that would go stale and
# contradict it.
_NAMES_FROM_PROMPT = (
    "Use the exact substation names listed under 'Substation names' in the "
    "system prompt; they describe the network currently loaded."
)

_SLACK_FROM_PROMPT = (
    "the external-grid slack, named under 'Grid domain knowledge' in the system prompt"
)

_LOAD_SCALING_DESC = "Multiplier on all active and reactive loads; 1.0 = as measured."

_SLACK_MAX_MW_DESC = (
    "External-grid exchange cap, MW, both directions. Omit for the network's "
    "own limit; 0.0 = islanded. For islanding or derating questions."
)

_SLACK_Q_MAX_MVAR_DESC = (
    "External-grid reactive exchange cap, Mvar. Omit to use slack_max_mw. "
    "A cap makes local generators supply more Q."
)

_ELEMENT_TYPE_DESC = (
    "Network element type: 'line' for power lines or 'trafo' for transformers. "
    "Element indices depend on the currently loaded network."
)

_DISABLED_GEN_DESC = (
    "Substations whose generators are forced offline (generator loss, "
    f"islanding). {_NAMES_FROM_PROMPT}"
)

_OPF_VM_UPPER_DESC = (
    "OPF voltage upper bound, p.u. Default: the network's own upper limit. "
    "A tighter bound forces more redispatch."
)

_OPF_VM_LOWER_DESC = (
    "OPF voltage lower bound, p.u. Default: the network's own lower limit. "
    "Lowering it widens the feasible region."
)

_OPF_LAMBDA_P_DESC = (
    "OPF weight on squared active-power deviation. Default 0.01; higher keeps "
    "P closer to its base value."
)

_OPF_LAMBDA_Q_DESC = (
    "OPF weight on squared reactive-power deviation. Default 0.001; higher "
    "keeps Q closer to its base value."
)

_FIXED_SETPOINTS_DESC = (
    "Pins units to exact MW: {name: MW}; the rest redispatch freely. Keys are "
    f"substation names or {_SLACK_FROM_PROMPT} (0.0 isolates the external "
    f"interface). {_NAMES_FROM_PROMPT}"
)

_PG_MAX_OVERRIDES_DESC = (
    "Per-substation maximum active power, MW: {name: MW}. Others keep their "
    "capacity."
)

_BRANCH_LIMITS_DESC = "Branches held below a share of their rating: {name or tag: %}, e.g. {\"L30\": 80}. 10–100."

_PG_MIN_OVERRIDES_DESC = (
    "Per-substation minimum active power, MW: {name: MW}; positive = must-run. "
    "Others keep their minimum."
)

_MIN_PF_DESC = (
    "Minimum generator power factor in the OPF. Default 0.95; lower allows "
    "more reactive redispatch."
)

_CURRENT_SAFETY_MARGIN_DESC = (
    "Fraction of rated branch current the OPF may use (0–1). Default 0.9; "
    "1.0 = full rating."
)

# Shared by most tools: kept short, the instructions explain them once.
_VM_UPPER_DESC = "Upper voltage limit, p.u.; default the network's (Grid domain knowledge)."
_VM_LOWER_DESC = "Lower voltage limit, p.u.; default the network's."
_MAX_LINE_LOADING_DESC = "Line overload threshold, %."
_MAX_TRAFO_LOADING_DESC = "Transformer overload threshold, %."



def _str_prop(description: str) -> dict:
    return {"type": "string", "description": description}


def _num_prop(description: str) -> dict:
    return {"type": "number", "description": description}


def _int_prop(description: str) -> dict:
    return {"type": "integer", "description": description}


def _bool_prop(description: str) -> dict:
    return {"type": "boolean", "description": description}


def _arr_str_prop(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# ---------------------------------------------------------------------------
# FunctionDeclaration builders
# ---------------------------------------------------------------------------

_locate_network_element = genai.types.FunctionDeclaration(
    name="locate_network_element",
    description=(
        "Look up one bus, line or transformer by name: what it is, what it "
        "connects to, and the element_index each connection has. Use it "
        "BEFORE any tool that takes element_type/element_index — that index "
        "cannot be guessed from a name. Also answers what a substation is "
        "joined to. Topology only; does not depend on the timestamp."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "element": _str_prop(
                "Name of a bus, line or transformer, e.g. 'Golf' or 'OLS-ØST'."
            ),
        },
        required=["element"],
    ),
)

_get_current_timestamp = genai.types.FunctionDeclaration(
    name="get_current_timestamp",
    description=(
        "Retrieve the current simulation timestamp from the digital twin. "
        "Call this before any time-dependent analysis."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
        },
    ),
)

_advance_timestamp = genai.types.FunctionDeclaration(
    name="advance_timestamp",
    description=(
        "Advance the simulation clock by a given number of 15-minute ticks, "
        "or jump directly to a specific timestamp. "
        "Returns the list of timestamps traversed and the new current timestamp."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "steps": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Number of 15-minute ticks to advance. Default 1. "
                    "Ignored if target_timestamp is provided."
                ),
            ),
            "target_timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Jump directly to this timestamp instead of stepping forward. "
                    "Accepts full timestamp strings (e.g. '2022-03-15 08:00' or "
                    "'2022-03-15 08:00:00'). Prefix matching is supported so a "
                    "partial string like '2022-03-15 08' resolves to the first "
                    "matching tick. If provided, `steps` is ignored."
                ),
            ),
        },
    ),
)

_run_rsa = genai.types.FunctionDeclaration(
    name="run_rsa",
    description=(
        "Run a Real-time Security Assessment (RSA) on the active grid model. "
        "Returns voltage profiles for all buses, thermal loading for all lines "
        "and transformers, and a list of security violations. "
        "Use this to check the current operating point or a hypothetical load scenario."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)

_simulate_contingency = genai.types.FunctionDeclaration(
    name="simulate_contingency",
    description=(
        "Simulate a single N-1 contingency (one line or transformer outage). "
        "Returns whether the system remains secure and any resulting violations."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "element_type": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=_ELEMENT_TYPE_DESC,
            ),
            "element_index": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Zero-based index of the element to take out of service."
                    " Get it from locate_network_element; never guess it."
                ),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
        required=["element_type", "element_index"],
    ),
)

_simulate_all_contingencies = genai.types.FunctionDeclaration(
    name="simulate_all_contingencies",
    description=(
        "Run a full N-1 security sweep across all lines and transformers. "
        "Returns whether the system is N-1 secure, total outages tested, "
        "number causing violations, and the full violation table. "
        "Use this when the user asks 'which contingency is worst' or "
        "'is the grid N-1 secure'. With option, it screens a corrective option instead of the schedule."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "option": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=("A corrective option's `ref` from an options result (e.g. 'S2:cheapest'): the "
                             "sweep runs at that option's moment, network and load level with its dispatch "
                             "applied — after its outage too, if it had one (N-1-1)."),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)

_optimize_contingency = genai.types.FunctionDeclaration(
    name="optimize_contingency",
    description=(
        "Compute the corrective flexibility dispatch after a single contingency "
        "using AC OPF (SMFAE). Returns the generator setpoints before and after "
        "optimization. Use this when the user asks how to fix a specific outage. "
        "options, hold_units, near_optimal(_pct), branch_limits: as in optimize_flexibility, after the outage."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "element_type": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=_ELEMENT_TYPE_DESC,
            ),
            "element_index": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Zero-based index of the contingency element."
                    " Get it from locate_network_element; never guess it."
                ),
            ),
            "options": genai.types.Schema(
                type=genai.types.Type.BOOLEAN,
                description="Several corrective options.",
            ),
            "hold_units": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
                description="Units ruled out: in service, no instruction.",
            ),
            "near_optimal": genai.types.Schema(
                type=genai.types.Type.BOOLEAN,
                description="Near-optimal alternatives instead.",
            ),
            "near_optimal_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="How far above the cheapest an alternative may cost, %. Default 10.",
            ),
            "branch_limits": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_BRANCH_LIMITS_DESC,
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_LOWER_DESC,
            ),
            "opf_lambda_p": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_P_DESC,
            ),
            "opf_lambda_q": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_Q_DESC,
            ),
            "pg_max_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MAX_OVERRIDES_DESC,
            ),
            "pg_min_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MIN_OVERRIDES_DESC,
            ),
            "fixed_setpoints": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_FIXED_SETPOINTS_DESC,
            ),
            "opf_min_power_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MIN_PF_DESC,
            ),
            "opf_current_safety_margin": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_CURRENT_SAFETY_MARGIN_DESC,
            ),
        },
        required=["element_type", "element_index"],
    ),
)

_optimize_flexibility = genai.types.FunctionDeclaration(
    name="optimize_flexibility",
    description=(
        "Optimize local generator dispatch to resolve voltage or thermal violations "
        "in normal (N-0) operation using AC OPF (SMFAE). "
        "Returns generator setpoints and the optimization status. "
        "Use after run_rsa detects violations, or when the user asks about "
        "corrective actions, generator outages, or cable derating. options=true: several "
        "corrective dispatches instead (named strategies, or near_optimal alternatives), each "
        "meeting every limit, scored the same way and checked by a power flow."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "options": genai.types.Schema(
                type=genai.types.Type.BOOLEAN,
                description=("True: several corrective options to choose from, instead of the one "
                             "least-disturbance dispatch."),
            ),
            "hold_units": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
                description=("Units whose active power stays at its schedule — the operator's "
                             "conditions ('without Gen 2', 'don't call the battery'). In service, "
                             "unlike disabled_generators. With options=true, every option respects them."),
            ),
            "near_optimal": genai.types.Schema(
                type=genai.types.Type.BOOLEAN,
                description=("With options=true: instead of the named strategies, the cheapest option and up "
                             "to three near-optimal alternatives, each costing at most near_optimal_pct above "
                             "it and changing other units."),
            ),
            "near_optimal_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="How far above the cheapest an alternative may cost, %. Default 10.",
            ),
            "branch_limits": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_BRANCH_LIMITS_DESC,
            ),
            "n1": genai.types.Schema(
                type=genai.types.Type.STRING,
                enum=["preventive", "corrective", "both"],
                description=("Options secure after the critical outages. preventive: nothing to do when one "
                             "happens; corrective: a quick action after it (plan returned); both."),
            ),
            "n1_outages": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
                description="Outages to secure (names or tags). Default: the critical ones.",
            ),
            "n1_emergency_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Loading allowed after an outage, % of rating. Default: the network's limit.",
            ),
            "n1_reach_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Corrective: MW a unit may move after an outage, % of capacity. Default 20.",
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "disabled_generators": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
                description=_DISABLED_GEN_DESC,
            ),
            "slack_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_MAX_MW_DESC,
            ),
            "slack_q_max_mvar": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_Q_MAX_MVAR_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_LOWER_DESC,
            ),
            "opf_lambda_p": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_P_DESC,
            ),
            "opf_lambda_q": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_Q_DESC,
            ),
            "pg_max_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MAX_OVERRIDES_DESC,
            ),
            "pg_min_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MIN_OVERRIDES_DESC,
            ),
            "fixed_setpoints": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_FIXED_SETPOINTS_DESC,
            ),
            "opf_min_power_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MIN_PF_DESC,
            ),
            "opf_current_safety_margin": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_CURRENT_SAFETY_MARGIN_DESC,
            ),
        },
    ),
)

_evaluate_kpis = genai.types.FunctionDeclaration(
    name="evaluate_kpis",
    description=(
        "Evaluate the three system KPIs for the current timestamp. "
        "KPI-1: available upward flexibility as fraction of demand. "
        "KPI-2: fraction of required flexibility actually activated. "
        "KPI-3: fraction of violations prevented by optimization. "
        "When slack_max_mw is set below the network's physical cap, also runs a constrained-scenario solve."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "slack_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_MAX_MW_DESC,
            ),
            "slack_q_max_mvar": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_Q_MAX_MVAR_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_LOWER_DESC,
            ),
            "opf_lambda_p": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_P_DESC,
            ),
            "opf_lambda_q": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_LAMBDA_Q_DESC,
            ),
            "pg_max_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MAX_OVERRIDES_DESC,
            ),
            "pg_min_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_PG_MIN_OVERRIDES_DESC,
            ),
            "fixed_setpoints": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=_FIXED_SETPOINTS_DESC,
            ),
            "opf_min_power_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MIN_PF_DESC,
            ),
            "opf_current_safety_margin": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_CURRENT_SAFETY_MARGIN_DESC,
            ),
        },
    ),
)

_forecast_kpis = genai.types.FunctionDeclaration(
    name="forecast_kpis",
    description=(
        "Generate a 24-hour ahead forecast of KPI-1 (target demand flexibility %) "
        "at 15-minute resolution (96 data points). "
        "WARNING: This call takes approximately 2 minutes to complete. "
        "Only invoke when the user explicitly asks for a forecast or 'next 24 hours'."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "slack_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_MAX_MW_DESC,
            ),
            "slack_q_max_mvar": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_Q_MAX_MVAR_DESC,
            ),
        },
    ),
)

_get_element_timeseries = genai.types.FunctionDeclaration(
    name="get_element_timeseries",
    description=(
        "Focus tool: scan a time range and plot the evolution of a specific bus voltage (vm_pu), "
        "line loading (%), or transformer loading (%) over time. "
        "Does NOT advance the simulation clock. "
        "Use when the user asks: "
        "'show me the voltage at Alpha over the last 24 hours', "
        "'plot the loading of line X between Monday and Tuesday', "
        "'how did transformer 5 behave this week', "
        "'show me the bus voltage for [substation] in a time range'. "
        "Returns a time-series chart of the primary metric (vm_pu for buses, "
        "loading_percent for lines/trafos) plus the active power flow."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "element_type": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Type of network element to focus on. "
                    "'bus' → bus voltage profile, "
                    "'line' → line thermal loading, "
                    "'trafo' → transformer thermal loading."
                ),
            ),
            "element_name": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Name (or partial name) of the element to focus on. "
                    f"For buses use substation name fragments. {_NAMES_FROM_PROMPT} "
                    "Partial match is supported — a fragment such as 'Alpha' "
                    "matches both 'Alpha 10.5 kV' and 'Alpha 60 kV'. "
                    "For lines and trafos use the element name from the network "
                    "(e.g. the name shown in the contingency results)."
                ),
            ),
            "start_timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Start of the scan window. ISO prefix matching: "
                    "'2022-01-03' covers the whole day; '2022-01-03 08' covers 08:00–08:45. "
                    "If omitted, starts at the current simulation timestamp."
                ),
            ),
            "end_timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "End of the scan window (inclusive). ISO prefix matching. "
                    "If omitted (with n_steps also omitted), scans to the end of the dataset."
                ),
            ),
            "n_steps": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Maximum number of 15-minute ticks to scan. "
                    "96 = 1 day, 672 = 1 week. "
                    "Use with start_timestamp to define the window size precisely."
                ),
            ),
            "step_size": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Sample every Nth tick for speed. Default 1 (every 15 min). "
                    "Use 4 for hourly resolution, 96 for daily summary."
                ),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
        required=["element_type", "element_name"],
    ),
)

_get_current_conditions = genai.types.FunctionDeclaration(
    name="get_current_conditions",
    description=(
        "Return the raw network state at the current timestamp: "
        "per-generator active power (MW), installed maximum capacity (Pg_max_mw), "
        "reactive power (MVAR), per-load consumption (MW/MVAR), "
        "external-grid import (MW), and network totals (generation, load, losses). "
        "Use this whenever the user asks about current generation levels, "
        "maximum generator capacity, load, losses, or external-grid import. "
        "Does NOT run an OPF — just a fast power flow snapshot."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={},
    ),
)

_compare_results = genai.types.FunctionDeclaration(
    name="compare_results",
    description=(
        "Compare the last two comparable results from this turn and compute "
        "per-generator delta dispatch (ΔPg, ΔQg), per-bus voltage delta (ΔVm), and KPI delta. "
        "No backend call — pure client-side computation. "
        "Accepts two optimization results OR one run_rsa result followed by one optimize_flexibility result. "
        "RSA vs OPF usage: call run_rsa() first (captures the raw/violating state), "
        "then optimize_flexibility(), then compare_results() — this gives a true before/after diff. "
        "Two-optimizer usage: run two optimizer tools back-to-back, then compare_results(). "
        "Do NOT pass result_a / result_b — the tool retrieves the last two results automatically. "
        "Only supply label_a and label_b to name the two scenarios in the charts. "
        "Use when the user asks: 'what changed when...', 'compare X vs Y', "
        "'show the difference between the violation state and the fix', "
        "'effect of disabling a generator', 'difference between cable derated to 20 MW vs 70 MW', etc."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "label_a": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Human-readable label for the first (baseline) run, e.g. 'Cable 70 MW' or 'Baseline'. Default: 'Baseline'.",
            ),
            "label_b": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Human-readable label for the second (scenario) run, e.g. 'Cable 1 MW'. Default: 'Scenario'.",
            ),
        },
    ),
)

_scan_rsa_over_time = genai.types.FunctionDeclaration(
    name="scan_rsa_over_time",
    description=(
        "Advance the simulation clock n_steps × 15 minutes, running an RSA at each "
        "tick, and return a time-series of key grid security metrics: voltage envelope "
        "(min/max), max thermal loading (lines and trafos), and violation counts. "
        "Use when the user asks about trends, evolution, or 'next N timesteps'."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "n_steps": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Number of 15-minute steps to scan. Default 4 (1 hour). "
                    "1 day = 96 steps, 7 days = 672 steps, full dataset = use "
                    "get_current_timestamp to check how many ticks remain. "
                    "Each step takes ~2-5 seconds; warn the user for large scans."
                ),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            # Declared so a limit the operator set reaches the scan too — the
            # wrapper always accepted them, but without a schema entry the
            # model could not send them and the turn could not inherit them.
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)

_scan_scenarios = genai.types.FunctionDeclaration(
    name="scan_scenarios",
    description=(
        "Run RSA across a time range for multiple renewable output scaling factors "
        "simultaneously and compare the results. Server-side parallel execution — "
        "does NOT advance the simulation clock. "
        "Use when the user asks: "
        "'what happens to violations if wind drops 30%?', "
        "'how sensitive is the grid to renewable uncertainty?', "
        "'compare low / baseline / high wind over the next day', "
        "'worst case if renewables are at 70%'. "
        "Returns per-scenario time-series of violations, Sweden import, and voltage envelope, "
        "plus a summary comparison table."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "sgen_scales": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.NUMBER),
                description=(
                    "List of renewable output scaling factors to compare. "
                    "Each value multiplies ALL sgen (wind/solar) p_mw and q_mvar before each power flow. "
                    "1.0 = measured baseline output. "
                    "Choose one value below 1.0 (low renewables) and one above 1.0 (high renewables) "
                    "plus 1.0 for the baseline, e.g. [0.7, 1.0, 1.3]. "
                    "Keep the list short (2–4 values) for speed."
                ),
            ),
            "start_timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Start of the scan window. ISO prefix matching "
                    "(e.g. '2022-01-03' = full day). "
                    "If omitted, starts at the current simulation timestamp."
                ),
            ),
            "end_timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "End of the scan window (inclusive). ISO prefix matching. "
                    "If omitted together with n_steps, scans all remaining timestamps."
                ),
            ),
            "n_steps": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Maximum ticks per scenario. 96 = 1 day, 672 = 1 week. "
                    "Total server work = n_steps × len(sgen_scales) power flows, "
                    "but scenarios run in parallel so wall time ≈ one scenario."
                ),
            ),
            "step_size": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Sample every Nth tick. Default 1 (every 15 min). "
                    "Use 4 for hourly, 96 for daily summary."
                ),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)

_find_worst_case_timestamp = genai.types.FunctionDeclaration(
    name="find_worst_case_timestamp",
    description=(
        "Scan forward through all (or n_steps) timestamps WITHOUT advancing the "
        "simulation clock, and find the worst-case operating point for a chosen stress metric. "
        "Returns the worst timestamp + value, per-metric worst summary, and a full time-series "
        "for charting. The simulation clock is unchanged after the call. "
        "Use when the user asks: 'when is the grid most stressed?', 'find the worst case', "
        "'which timestamp has the most violations?', 'peak import moment', "
        "'when is voltage lowest?', 'scan the whole dataset for the worst point'."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "metric": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "The stress metric to find the worst timestamp for. "
                    "One of: 'violations' (default — most security violations), "
                    "'slack_import' (highest external-grid import MW), "
                    "'max_voltage' (highest bus voltage p.u.), "
                    "'min_voltage' (lowest bus voltage p.u. — worst undervoltage), "
                    "'max_loading' (highest line loading %). "
                    "All five are always returned in worst_per_metric regardless of choice."
                ),
            ),
            "n_steps": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Number of ticks to scan from the current position. "
                    "Omit to scan all remaining timestamps. "
                    "1 day = 96 ticks, 7 days = 672. Each tick ~0.1s; "
                    "warn the user before scans > 200 steps."
                ),
            ),
            "step_size": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Scan every Nth tick. Default 1 (every tick, full resolution). "
                    "Use 3–6 for a fast preview (~3–6× faster, lower resolution). "
                    "Recommend step_size=3 for scans over 200 steps."
                ),
            ),
            "data_source": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Which dataset to scan. 'measurements' (default) = historical "
                    "actuals, scanned forward from the current simulation time — use "
                    "for 'worst case in the last week', past-event and historical "
                    "questions. 'forecasts' = the look-ahead planning series, scanned "
                    "from its start — use for 'worst case next week', planning and "
                    "rescheduling questions. If forecasts are not loaded the call "
                    "returns an error; do not assume they exist."
                ),
            ),
            # The scan counts violations, so it needs the same limits as the
            # assessment that follows it. They were absent from this schema
            # while every other tool had them, so a turn that set a ceiling
            # searched at the default and then assessed at the ceiling — two
            # studies that cannot be presented as one. The scan reports what
            # it used in `thresholds_used`; passing them makes that agree.
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Upper voltage limit in p.u. used to count violations while "
                    "scanning. Pass the same value as the assessment that follows, "
                    "or the scan searches against a different definition of "
                    "'worst' than the one you then report. Only affects the "
                    "'violations' metric; the voltage and loading extrema are "
                    "threshold-independent."
                ),
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Lower voltage limit in p.u. for violation counting during the scan.",
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Line loading limit in % for violation counting during the scan.",
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Transformer loading limit in % for violation counting during the scan.",
            ),
        },
    ),
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

_run_probabilistic_rsa = genai.types.FunctionDeclaration(
    name="run_probabilistic_rsa",
    description=(
        "Run a Monte Carlo probabilistic security assessment at the current timestamp. "
        "Draws n_samples Latin Hypercube samples of generation and load uncertainty, runs "
        "a power flow for each "
        "sample in parallel, and returns statistically grounded risk metrics. "
        "Interpretation: current sgen values are treated as forecast estimates of renewable "
        "availability; uncertainty is on available resource around that estimate (with "
        "capacity clipping and setpoint ceiling), while load uncertainty remains multiplicative. "
        "Use for the probability or expected number of violations under "
        "uncertainty, or P5/P50/P95 voltage envelopes. Does NOT advance the clock."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "n_samples": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Number of Monte Carlo samples. Default 200. "
                    "Use 50–100 for a quick preview, 500 for publication-quality results. "
                    "Capped at 1000. Wall time ≈ n_samples/8 × single power flow time."
                ),
            ),
            "sgen_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Relative standard deviation for renewable forecast error around the current "
                    "sgen estimate. Default 0.00 (disabled unless explicitly requested). "
                    "This represents uncertainty in available renewable resource, not uncertainty "
                    "in the commanded setpoint itself. "
                    "Use 0.10 for a tighter uncertainty band, 0.25 for high uncertainty."
                ),
            ),
            "load_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Relative standard deviation for the load multiplier. "
                    "Default 0.05 (±5%). Load is generally less uncertain than wind."
                ),
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)

_optimize_robust_flexibility = genai.types.FunctionDeclaration(
    name="optimize_robust_flexibility",
    description=(
        "Robust OPF with two methods: heuristic constraint tightening (back-off) and scenario-based chance constraints. "
        "Combines probabilistic security assessment with optimal power flow to find "
        "the minimum-cost dispatch that is secure with a user-specified confidence "
        "(e.g. 95% or 99%) under modeled uncertainty. "
        "Step 1: runs probabilistic RSA to compute per-bus voltage percentile envelopes. "
        "Step 2: computes symmetric per-bus back-offs for upper and lower voltage tails. "
        "Step 3: solves OPF with tightened per-bus vm_upper and vm_lower bounds. "
        "Step 4: iterates tighten-and-solve using target_p_any, max_iter, and min_improvement. "
        "Step 5: reports independent validation risk after OPF. "
        "Use for a dispatch that is secure with a stated confidence under "
        "uncertainty (robust OPF, back-offs for forecast error). Scenario mode "
        "also reports whether its scenario count met the nominal bound (guarantee_met; "
        "not a certificate for AC OPF) — the out-of-sample validation risk is the evidence."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "opf_min_power_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MIN_PF_DESC,
            ),
            "risk_target": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Canonical target violation probability used across modes. "
                    "Default 0.05. In heuristic mode maps to confidence=1-risk_target and "
                    "target_p_any unless overridden; in scenario mode maps to alpha."
                ),
            ),
            "robust_method": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Robust method: 'heuristic' (default) or 'scenario' for scenario-based chance-constrained AC-OPF.",
            ),
            "sgen_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Relative standard deviation for the sgen (wind/solar) multiplier. "
                    "Default 0.00 (disabled unless explicitly requested). "
                    "Set this only when the user explicitly asks for generator uncertainty."
                ),
            ),
            "load_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Relative standard deviation for the load multiplier. Default 0.05 (±5%).",
            ),
            "confidence": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Deprecated alias for risk_target in heuristic interpretation: risk_target = 1 - confidence. "
                    "Only consulted when risk_target is absent."
                ),
            ),
            "n_samples": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Monte Carlo sample count for the pre-OPF RSA. "
                    "Default 200. Use 50 for speed, 500 for accuracy."
                ),
            ),
            "validation_samples": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Independent Monte Carlo sample count for post-OPF validation/certification. "
                    "If omitted, defaults to n_samples."
                ),
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "slack_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Maximum external-grid import/export in MW. Default: loaded profile limit.",
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Load stress multiplier for the OPF. Default 1.0 (current load).",
            ),
            "target_p_any": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Heuristic-only override for iterative stop target. "
                    "If omitted, target_p_any is derived from risk_target."
                ),
            ),
            "max_iter": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Maximum number of robust tighten-and-solve iterations. Default 3.",
            ),
            "min_improvement": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Minimum required improvement in p_any between iterations; loop stops when below this threshold. "
                    "Default 0.005."
                ),
            ),
            "alpha": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Deprecated alias for risk_target in scenario interpretation. Only consulted when risk_target is absent.",
            ),
            "beta": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Confidence level parameter for scenario method K sizing (probability guarantee holds). Default 1e-3.",
            ),
            "n_scenarios": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Optional explicit K for scenario method. If omitted, computed from (alpha, beta, d).",
            ),
            "scenario_k_cap": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Maximum scenario count for tractability. Computed K is clamped to this cap. Default 120.",
            ),
            "allowed_violation_fraction": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Optional discard budget fraction in scenario method. Default 0.0 (enforce all scenarios).",
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
        },
    ),
)


_compute_flexibility_envelope = genai.types.FunctionDeclaration(
    name="compute_flexibility_envelope",
    description=(
        "Computes the secure (P, Q) operating region for a single generator at the "
        "current network conditions. Holds all other generators at their current SCADA "
        "setpoints and sweeps the target generator over a rectangular (P, Q) grid, "
        "running a full AC power flow at each point in parallel. "
        "Returns a feasibility map (green = secure, red = constraint violated) and "
        "the safe reactive power range at the generator's current active output. "
        "Use for a generator's safe P/Q range or secure operating region, how "
        "much Q it can inject or absorb, or why the OPF chose its Q."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "gen_name": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Substation name of the generator to sweep. "
                    f"Required. {_NAMES_FROM_PROMPT}"
                ),
            ),
            "p_min_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Minimum P for the sweep in MW. Default: 0.0.",
            ),
            "p_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Maximum P for the sweep in MW. Default: PG_MAX_DATA[gen_name].",
            ),
            "q_min_mvar": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Minimum Q for the sweep in MVAr (absorbing). "
                    "Default: −Pmax × tan(acos(0.9)) (PF=0.9 inductive boundary)."
                ),
            ),
            "q_max_mvar": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=(
                    "Maximum Q for the sweep in MVAr (injecting). "
                    "Default: +Pmax × tan(acos(0.9)) (PF=0.9 capacitive boundary)."
                ),
            ),
            "resolution": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "Grid resolution: resolution × resolution power-flow runs. "
                    "Default 20 (~400 runs, ~8 s). Max 25 (~625 runs, ~20 s). "
                    "Use 25 for maximum detail if time permits."
                ),
            ),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_UPPER_DESC,
            ),
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_VM_LOWER_DESC,
            ),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_LINE_LOADING_DESC,
            ),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_MAX_TRAFO_LOADING_DESC,
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Load stress multiplier for the sweep. Default 1.0 (current load).",
            ),
            "reference_state": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Reference operating point for the envelope sweep: "
                    "'scada' (default), 'post_opf' (latest cached optimization dispatch), "
                    "or 'custom' (apply dispatch_overrides)."
                ),
            ),
            "dispatch_overrides": genai.types.Schema(
                type=genai.types.Type.OBJECT,
                description=(
                    "Custom dispatch setpoints used when reference_state='custom'. "
                    "Format: {element_name: p_mw} or {element_name: {p_mw, q_mvar}}."
                ),
            ),
        },
        required=["gen_name"],
    ),
)


_compute_hosting_capacity = genai.types.FunctionDeclaration(
    name="compute_hosting_capacity",
    description=(
        "Computes hosting capacity for an incremental injection at one bus in deterministic or probabilistic mode. "
        "Returns hosting_capacity_mw for one or more reactive-power strategies "
        "(unity, fixed_pf, reactive_proxy) plus binding constraints. "
        "Use when the user asks: 'how much more can bus X host?', 'hosting capacity at bus X', "
        "or 'compare hosting by reactive strategy'."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "bus": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Target bus identifier (index or label). Use 'all' for deterministic all-bus scan.",
            ),
            "q_mode": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Reactive strategy: all (default), unity, fixed_pf, or reactive_proxy. "
                    "unity => Q≈0; fixed_pf => Q follows selected power_factor and pf_sign; "
                    "reactive_proxy => damped counteracting current-bus-Q clipped to a PF=0.9 capability envelope; "
                    "it is not PF-constrained like fixed_pf."
                ),
            ),
            "power_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Power factor used in fixed_pf mode. Default 0.95.",
            ),
            "pf_sign": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="fixed_pf sign convention: absorbing (default) or injecting.",
            ),
            "mode": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Run mode: deterministic (default) or probabilistic.",
            ),
            "risk_threshold": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Only for probabilistic mode: maximum allowed probability of any violation. Default 0.05.",
            ),
            "n_samples": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Only for probabilistic mode: number of Monte Carlo samples. Default 200, capped at 1000.",
            ),
            "added_gen_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Only for probabilistic mode: sigma for bounded availability uncertainty on the added generator. Default 0.10.",
            ),
            "load_sigma": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Only for probabilistic mode: multiplicative load uncertainty sigma around the active operating point. Default 0.0.",
            ),
            "uncertainty_scope": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Only for probabilistic mode: 'added_generation_only' (default) or 'added_generation_plus_load'. Use the latter only when load uncertainty should also be sampled.",
            ),
            "timestamp": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Optional timestamp or unambiguous prefix for the operating point.",
            ),
            "p_max_search_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Optional upper search bound in MW. If omitted, backend chooses a safe heuristic.",
            ),
            "tolerance_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Bisection tolerance in MW. Default 0.1.",
            ),
            "vm_upper_pu": genai.types.Schema(type=genai.types.Type.NUMBER, description=_VM_UPPER_DESC),
            "vm_lower_pu": genai.types.Schema(type=genai.types.Type.NUMBER, description=_VM_LOWER_DESC),
            "max_line_loading_pct": genai.types.Schema(type=genai.types.Type.NUMBER, description=_MAX_LINE_LOADING_DESC),
            "max_trafo_loading_pct": genai.types.Schema(type=genai.types.Type.NUMBER, description=_MAX_TRAFO_LOADING_DESC),
        },
        required=["bus"],
    ),
)


_compute_historical_risk = genai.types.FunctionDeclaration(
    name="compute_historical_risk",
    description=(
        "Computes empirical historical violation and near-miss statistics over a real "
        "SCADA window. Uses repeated RSA snapshots over a timestamp range and returns "
        "exceedance frequency/count, worst episodes, and a duration-curve-style summary. "
        "Supports conditional risk bins (hour/load/slack_import). "
        "The tool wrapper auto-expands date-only windows (YYYY-MM-DD -> YYYY-MM-DD 00:00:00), "
        "and when target names are misspelled it attempts a safe auto-correction from live RSA labels; "
        "if confidence is low it returns target suggestions so you can ask the user to confirm. "
        "Use when the user asks: 'how often does this violate?', 'what is the historical risk?', "
        "'show empirical violation statistics', or 'how bad was this over the year?'."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "target": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Target to analyze: 'all' or one of 'bus:<name>', 'line:<name>', 'trafo:<name>'. Use exact labels when possible; wrapper may auto-correct close misspellings and/or return target_suggestions.",
            ),
            "window_start": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Start timestamp or unambiguous prefix. Prefer full timestamps ('YYYY-MM-DD HH:MM:SS'). Date-only values are auto-expanded to midnight.",
            ),
            "window_end": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="End timestamp or unambiguous prefix. Prefer full timestamps ('YYYY-MM-DD HH:MM:SS'). Date-only values are auto-expanded to midnight.",
            ),
            "condition": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="Optional conditioning variable for risk bins: hour, load (aliases: load_mw, total_load), slack_import (aliases: slack, slack_import_mw, cable_flow).",
            ),
            "n_bins": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Number of bins for continuous conditions (load/slack_import). Default 4.",
            ),
            "near_miss_band": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Near-miss margin band. Default 0.005 p.u. for buses; thermal branch near-miss uses a fixed 5%% buffer in the first slice.",
            ),
            "worst_n": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Number of worst episodes to return. Default 5.",
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description="Optional load stress multiplier applied to each snapshot. Default 1.0.",
            ),
            "vm_upper_pu": genai.types.Schema(type=genai.types.Type.NUMBER, description=_VM_UPPER_DESC),
            "vm_lower_pu": genai.types.Schema(type=genai.types.Type.NUMBER, description=_VM_LOWER_DESC),
            "max_line_loading_pct": genai.types.Schema(type=genai.types.Type.NUMBER, description=_MAX_LINE_LOADING_DESC),
            "max_trafo_loading_pct": genai.types.Schema(type=genai.types.Type.NUMBER, description=_MAX_TRAFO_LOADING_DESC),
            "parallel": genai.types.Schema(
                type=genai.types.Type.BOOLEAN,
                description="Enable parallel snapshot evaluation across timestamps. Default false.",
            ),
            "max_workers": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Optional worker cap when parallel=true. If omitted, backend picks a safe default.",
            ),
        },
    ),
)
_compute_violation_attribution = genai.types.FunctionDeclaration(
    name="compute_violation_attribution",
    description=(
        "Explain WHY elements are violating their limits at an operating point, "
        "by ranking which generators, renewable units, and loads are driving each "
        "violation. Use this after a security assessment reports violations and the "
        "user asks why, what is causing them, which unit is responsible, or what to "
        "move to fix them. Returns, per violated element, a ranked list of sources "
        "with their measured sensitivity (change in bus voltage or branch loading per "
        "MW or MVAr of that source) and the movement each would need to clear the "
        "violation on its own. Does NOT advance the simulation clock. Prefer this "
        "over inferring causes from a violation table."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "vm_lower_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_VM_LOWER_DESC),
            "vm_upper_pu": genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_VM_UPPER_DESC),
            "max_line_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_MAX_LINE_LOADING_DESC),
            "max_trafo_loading_pct": genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_MAX_TRAFO_LOADING_DESC),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_LOAD_SCALING_DESC),
            "top_n_sources": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description=(
                    "How many injections to test, largest first. Default 20. Each "
                    "costs two power flows, so raise it only when the network is "
                    "large and the expected driver may be a small unit."
                )),
            "top_n_drivers": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Ranked drivers returned per violated element. Default 5."),
        },
    ),
)


_S = genai.types.Schema
_T = genai.types.Type

_get_data_quality = genai.types.FunctionDeclaration(
    name="get_data_quality",
    description=(
        "The quality report of the loaded data (measurements and forecasts): which columns they "
        "carry (measured reactive demand / generation), coverage, gaps and how each was filled "
        "(active power interpolated in time; missing reactive demand from the load power factor), "
        "implausible values (power factor below 0.7, values that look like kvar, all-zero columns), "
        "leading loads, and measured reactive generation that voltage-controlled units do not follow. "
        "Use it for any question about the data themselves. Reads the data; runs no study."
    ),
    parameters=_S(type=_T.OBJECT, properties={}),
)

_set_load_power_factor = genai.types.FunctionDeclaration(
    name="set_load_power_factor",
    description=(
        "Change the session's load power factor: the measurements carry active power only, so "
        "every study gives each measured load the reactive power Q = P·tan(acos pf) (lagging). "
        "Default 0.95. The change applies to every later study and is kept for this network; "
        "the operator sees it in the System tab. Use it only when the operator asks to change the "
        "assumption itself — to rerun one study under another power factor, pass "
        "`load_power_factor` to that study instead. reset=true goes back to 0.95."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "value": _S(type=_T.NUMBER, description="Power factor, 0.5–1.0 (lagging), e.g. 0.92."),
            "reset": _S(type=_T.BOOLEAN, description="True: back to the default 0.95."),
        },
    ),
)

_set_generator_model = genai.types.FunctionDeclaration(
    name="set_generator_model",
    description=(
        "Read or change how each generating unit is modelled. Call with no arguments to list "
        "every unit (name, bus, control, set-point, power factor, reactive limits, dispatchable). "
        "control='voltage': holds a voltage set-point, reactive power free within q_min/q_max "
        "(the OPF keeps those limits); control='power_factor': fixed power factor, injecting or "
        "absorbing (pf 1.0 = no reactive power); control='fixed_q': a set reactive output q_mvar. "
        "dispatchable: whether the OPF may move its active "
        "power. Changes apply to the base network (proposals are re-checked) and are kept for this "
        "network; the operator sees and edits them in the System tab. Change the model only when "
        "the operator asks; reset=true takes every unit back to how it was loaded."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "changes": _S(
                type=_T.ARRAY,
                description=("One item per setting changed: the unit, which setting, and its new value in "
                             "the slot that setting uses. Every other setting keeps its value. Several "
                             "settings or units: several items (voltage control at 1.02 p.u. is control "
                             "and vm_pu)."),
                items=_S(
                    type=_T.OBJECT,
                    properties={
                        "unit": _S(type=_T.STRING, description="Unit name as listed, e.g. 'Gen_bus1'."),
                        "setting": _S(type=_T.STRING, enum=list(tools.GENERATOR_SETTINGS), description=(
                            "Which setting. control: in `control`. vm_pu (set-point, 0.8–1.2), pf "
                            "(0.5–1.0), q_mvar (fixed_q output, positive injects), q_min_mvar, "
                            "q_max_mvar: a number in `value`. absorbing (true: absorbs reactive power), "
                            "dispatchable (false: the OPF keeps its output): in `on`.")),
                        "control": _S(type=_T.STRING, enum=list(tools.GENERATOR_CONTROLS),
                                      description="For setting=control."),
                        "value": _S(type=_T.NUMBER, description="For a numeric setting."),
                        "on": _S(type=_T.BOOLEAN, description="For absorbing or dispatchable."),
                    },
                    required=["unit", "setting"],
                ),
            ),
            "reset": _S(type=_T.BOOLEAN, description="True: every unit back to how it was loaded."),
        },
    ),
)

_set_unit_costs = genai.types.FunctionDeclaration(
    name="set_unit_costs",
    description=(
        "Read or change what each unit costs. No arguments: every unit and the external grid with its "
        "production cost (c0 €/h + c1 €/MWh·P + c2 €/MW²h·P²; the grid's c1 is the import price) and "
        "redispatch costs up/down (€/MWh; down for a renewable is curtailment), a reactive price per "
        "unit (€/Mvarh) and the grid's reactive band (tan φ) and penalty beyond it, each marked 'case', "
        "'synthetic' or 'operator'. Kept for this network and shown in the System tab. Used to cost "
        "the flexibility options (optimize_flexibility with options) and to find the cheapest one; the "
        "default dispatch ignores them."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "changes": _S(
                type=_T.ARRAY,
                description=("One item per price changed: the unit, which price, the new value. Every other "
                             "price keeps its value. Several prices or units: several items."),
                items=_S(
                    type=_T.OBJECT,
                    properties={
                        "unit": _S(type=_T.STRING, description="Unit name as listed, or 'External grid'."),
                        "field": _S(type=_T.STRING, enum=list(tools.COST_FIELDS), description=(
                            "Which price. cost_eur_per_mwh: c1, €/MWh (the grid's: import price); "
                            "cost_eur_per_mw2h: c2; cost_eur_per_h: c0; up_eur_per_mwh / down_eur_per_mwh: "
                            "redispatch; q_eur_per_mvarh: a unit's reactive power, €/Mvarh; "
                            "q_band_tan_phi / q_penalty_eur_per_mvarh: external grid only, the free "
                            "reactive band (tan φ) and €/Mvarh beyond it.")),
                        "value": _S(type=_T.NUMBER, description="The new value."),
                        "to_default": _S(type=_T.BOOLEAN, description="True: back to its default (no value)."),
                    },
                    required=["unit", "field"],
                ),
            ),
            "reset": _S(type=_T.BOOLEAN, description="True: every cost back to its default."),
        },
    ),
)

_rename_elements = genai.types.FunctionDeclaration(
    name="rename_elements",
    description=(
        "Rename buses, lines, transformers, loads or units of the loaded network, when the operator "
        "asks. Every result uses the new names from then on (a bus's lines and transformers are "
        "relabelled with it); measurements stay matched. Use the new names in later calls. No "
        "arguments: every renamed element."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "changes": _S(
                type=_T.ARRAY,
                items=_S(
                    type=_T.OBJECT,
                    properties={
                        "element_type": _S(type=_T.STRING, enum=["bus", "line", "trafo", "load", "gen", "sgen"]),
                        "element": _S(type=_T.STRING, description="Its name now, a label results show, or index."),
                        "name": _S(type=_T.STRING, description="The new name (unique); empty restores the original."),
                    },
                    required=["element_type", "element", "name"],
                ),
            ),
            "reset": _S(type=_T.BOOLEAN, description="True: every name back as loaded."),
        },
    ),
)

_edit_proposal = genai.types.FunctionDeclaration(
    name="edit_proposal",
    description=(
        "Create a network proposal — the base network plus edits — or add edits to one, for "
        "what-if questions about changing the grid (the ops are described in the instructions: "
        "new loads, units, shunts, lines, transformers and buses; reinforcements; switching; any "
        "setting of an existing element via set_parameter; moves; scaling; include_proposal for "
        "'A + B'). The base is never modified. Study it by passing its id as `network` to any "
        "analysis tool, or with compare_networks. Returns the stored edits, a summary and `assumed` "
        "(values filled in that you did not give — state them). `needs_operator` means a decision "
        "is missing: ask the operator its question with its options; do not choose. "
        "Give each edit only the fields its op uses."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "proposal": _S(type=_T.STRING, description=(
                "Proposal id: letters, digits, '_' or '-', e.g. 'dc_bus9'. Created if new.")),
            "edits": _S(
                type=_T.ARRAY,
                description="Edits applied in order after any the proposal already has.",
                items=_S(
                    type=_T.OBJECT,
                    properties={
                        "op": _S(type=_T.STRING, enum=[
                            "add_load", "add_generator", "add_shunt", "reinforce_line", "upgrade_trafo",
                            "add_bus_with_connection", "add_bus_with_transformer", "set_in_service",
                            "set_parameter", "add_line", "add_trafo", "move_element", "scale",
                            "include_proposal"]),
                        "name": _S(type=_T.STRING, description=(
                            "add_load / add_generator / add_shunt: a unique name, e.g. 'Data centre'. "
                            "add_line / add_trafo: optional name of the new branch. "
                            "add_bus_with_connection / add_bus_with_transformer: the new bus's name, "
                            "usable as `bus` in later edits of the same proposal.")),
                        "bus": _S(type=_T.STRING, description=(
                            "add_load / add_generator / add_shunt: bus name as results show it, or index.")),
                        "p_mw": _S(type=_T.NUMBER, description=(
                            "add_load: size in MW (its peak). add_generator (dispatchable): "
                            "output set-point, default p_min_mw.")),
                        "pf": _S(type=_T.NUMBER, description=(
                            "add_load: power factor, default 0.95 (lagging unless `leading`). "
                            "add_generator with control='power_factor': its power factor, default 1.0.")),
                        "leading": _S(type=_T.BOOLEAN, description=(
                            "add_load: true = leading power factor (the load injects reactive power).")),
                        "profile": _S(type=_T.STRING, enum=["flat", "follow_bus", "shift"], description=(
                            "add_load: 'flat' (constant, default), 'follow_bus' (a bus's measured "
                            "demand shape, peaking at p_mw — its own bus, or `follow`), 'shift' (p_mw "
                            "from start_hour to end_hour, else 0).")),
                        "follow": _S(type=_T.STRING, description=(
                            "follow_bus: the bus whose demand shape to follow, if not the load's own "
                            "(a new bus has no measured demand).")),
                        "start_hour": _S(type=_T.INTEGER, description="shift: first hour on (0–23)."),
                        "end_hour": _S(type=_T.INTEGER, description="shift: hour it stops (1–24)."),
                        "p_max_mw": _S(type=_T.NUMBER, description="add_generator: capacity in MW."),
                        "p_min_mw": _S(type=_T.NUMBER, description=(
                            "add_generator (dispatchable): minimum output in MW, default 0.")),
                        "kind": _S(type=_T.STRING, enum=["dispatchable", "pv", "wind", "capacitor", "reactor"],
                                   description=(
                            "add_generator: pv/wind follow the measured production shape; dispatchable "
                            "holds p_mw, redispatchable within p_min_mw–p_max_mw. add_shunt: capacitor or "
                            "reactor.")),
                        "control": _S(type=_T.STRING, enum=["voltage", "power_factor", "fixed_q"], description=(
                            "add_generator: 'voltage' (synchronous, holds vm_pu, Q within limits), "
                            "'power_factor' (inverter at fixed pf; pv/wind default) or 'fixed_q' (set "
                            "q_mvar). Dispatchable unit: ask the operator if not said.")),
                        "vm_pu": _S(type=_T.NUMBER, description=(
                            "add_generator (voltage): set-point in p.u.; default the bus's existing one, "
                            "else 1.0.")),
                        "q_min_mvar": _S(type=_T.NUMBER, description=(
                            "add_generator (voltage): lowest reactive output, Mvar.")),
                        "q_max_mvar": _S(type=_T.NUMBER, description=(
                            "add_generator (voltage): highest reactive output, Mvar (default for both: "
                            "± its rating at pf 0.85).")),
                        "absorbing": _S(type=_T.BOOLEAN, description=(
                            "add_generator (power_factor): true = absorbs reactive power.")),
                        "dispatchable": _S(type=_T.BOOLEAN, description=(
                            "add_generator: false = the OPF keeps its output (default true).")),
                        "q_mvar": _S(type=_T.NUMBER, description=(
                            "add_shunt: size in Mvar. add_generator (fixed_q): reactive output in Mvar, "
                            "positive injects, negative absorbs.")),
                        "line": _S(type=_T.STRING, description="reinforce_line: line name or index."),
                        "factor": _S(type=_T.NUMBER, description=(
                            "reinforce_line: capacity multiplier, e.g. 2 = a second identical circuit. "
                            "scale: multiplier of P and Q at every tick, 0–10 (1.2 = +20 %, 0.5 = half).")),
                        "trafo": _S(type=_T.STRING, description="upgrade_trafo: transformer name or index."),
                        "new_sn_mva": _S(type=_T.NUMBER, description=(
                            "upgrade_trafo: new rating in MVA, above the present one (same design: "
                            "impedance falls in proportion).")),
                        "connect_to": _S(type=_T.STRING, description=(
                            "add_bus_with_connection: existing bus the new line starts from (same "
                            "voltage). add_bus_with_transformer: existing bus the new transformer (or "
                            "the line before it) starts from; either voltage side.")),
                        "vn_kv": _S(type=_T.NUMBER, description=(
                            "add_bus_with_transformer: the new bus's voltage in kV, different from "
                            "connect_to's. May be left out with trafo_like or trafo_std_type (the "
                            "unit's other side).")),
                        "length_km": _S(type=_T.NUMBER, description=(
                            "add_line / add_bus_with_connection: line length in km. add_bus_with_transformer: "
                            "optional — a line of this length first, the transformer at its far end "
                            "(a remote substation); leave out for a transformer at connect_to.")),
                        "line_like": _S(type=_T.STRING, description=(
                            "add_line / add_bus_with_connection / add_bus_with_transformer (with length_km): "
                            "existing line whose conductor to copy (or give std_type / line_std_type).")),
                        "std_type": _S(type=_T.STRING, description=(
                            "add_line / add_bus_with_connection: pandapower line type, e.g. "
                            "'NA2XS2Y 1x240 RM/25 12/20 kV' (or give line_like).")),
                        "line_std_type": _S(type=_T.STRING, description=(
                            "add_bus_with_transformer (with length_km): pandapower line type of the "
                            "line before the transformer (or give line_like).")),
                        # New transformer fields (add_bus_with_transformer, add_trafo).
                        "trafo_like": _S(type=_T.STRING, description=(
                            "New transformer: existing one whose design to copy (with sn_mva, resized); "
                            "values given override the copy. Or trafo_std_type, or sn_mva + vk_percent "
                            "+ vkr_percent.")),
                        "trafo_std_type": _S(type=_T.STRING, description=(
                            "New transformer: pandapower type, e.g. '25 MVA 110/20 kV'; rated voltages "
                            "within 20% of the buses'. Fixes the design.")),
                        "sn_mva": _S(type=_T.NUMBER, description="New transformer: MVA of one unit."),
                        "vk_percent": _S(type=_T.NUMBER, description=(
                            "New transformer: short-circuit voltage % (4–6 MV/LV, 10–14 HV/MV).")),
                        "vkr_percent": _S(type=_T.NUMBER, description=(
                            "New transformer: resistive part %, below vk_percent (0.2–1.5).")),
                        "pfe_kw": _S(type=_T.NUMBER, description="New transformer: iron losses kW, default 0."),
                        "i0_percent": _S(type=_T.NUMBER, description="New transformer: magnetising current %."),
                        "shift_degree": _S(type=_T.NUMBER, description="New transformer: phase shift °, default 0."),
                        "vn_hv_kv": _S(type=_T.NUMBER, description="New transformer: rated HV kV if not the bus's."),
                        "vn_lv_kv": _S(type=_T.NUMBER, description="New transformer: rated LV kV if not the bus's."),
                        "tap_side": _S(type=_T.STRING, enum=["hv", "lv"], description="New transformer: tap side."),
                        "tap_step_percent": _S(type=_T.NUMBER, description="New transformer: % per tap step."),
                        "tap_min": _S(type=_T.INTEGER, description="New transformer: lowest tap."),
                        "tap_max": _S(type=_T.INTEGER, description="New transformer: highest tap."),
                        "tap_pos": _S(type=_T.INTEGER, description="New transformer: tap, default neutral."),
                        "parallel": _S(type=_T.INTEGER, description=(
                            "New transformer or add_line: identical units in parallel, 1–4.")),
                        "max_loading_percent": _S(type=_T.NUMBER, description=(
                            "New transformer: loading limit %, if not 100.")),
                        "min_vm_pu": _S(type=_T.NUMBER, description=(
                            "add_bus_with_connection / add_bus_with_transformer: the new bus's lower "
                            "voltage limit, if not that of connect_to.")),
                        "max_vm_pu": _S(type=_T.NUMBER, description=(
                            "add_bus_with_connection / add_bus_with_transformer: its upper limit.")),
                        "element_type": _S(type=_T.STRING, enum=[
                            "ext_grid", "bus", "line", "trafo", "trafo3w", "load", "sgen", "gen", "shunt",
                            "storage", "impedance"],
                                           description=(
                            "set_in_service / set_parameter / move_element / scale: element type. 'ext_grid' "
                            "is the external grid (slack) connection. move_element: load, sgen, gen, shunt, "
                            "storage, ext_grid or line. scale: load, sgen or gen.")),
                        "element": _S(type=_T.STRING, description=(
                            "set_in_service / set_parameter / move_element / scale: element name, label or "
                            "index as results show it (may be omitted for the only external grid).")),
                        "elements": _S(type=_T.ARRAY, items=_S(type=_T.STRING), description=(
                            "scale: several elements of element_type at once.")),
                        "all": _S(type=_T.BOOLEAN, description=(
                            "scale: true = every element of element_type (system-wide growth).")),
                        "from_bus": _S(type=_T.STRING, description="add_line: one end, an existing bus."),
                        "to_bus": _S(type=_T.STRING, description=(
                            "add_line: the other end, an existing bus at the same voltage. move_element: "
                            "the bus the element (or line end) moves to.")),
                        "hv_bus": _S(type=_T.STRING, description=(
                            "add_trafo: one existing bus (either side; the higher voltage is taken as HV).")),
                        "lv_bus": _S(type=_T.STRING, description=(
                            "add_trafo: the other existing bus, at another voltage.")),
                        "end": _S(type=_T.STRING, enum=["from", "to"], description=(
                            "move_element of a line: which end moves.")),
                        "proposal": _S(type=_T.STRING, description=(
                            "include_proposal: an existing proposal whose edits are copied in here, in order "
                            "(an independent copy).")),
                        "in_service": _S(type=_T.BOOLEAN, description="set_in_service: true or false."),
                        "field": _S(type=_T.STRING, description=(
                            "set_parameter: a column of that element, e.g. vm_pu (ext_grid, gen), tap_pos, "
                            "max_i_ka, max_loading_percent, p_mw, max_vm_pu. An unknown one is refused with "
                            "the editable list.")),
                        "value": _S(type=_T.NUMBER, description="set_parameter: the new numeric value."),
                        "enabled": _S(type=_T.BOOLEAN, description=(
                            "set_parameter: the new value of a true/false field (e.g. controllable).")),
                    },
                    required=["op"],
                ),
            ),
            "replace": _S(type=_T.BOOLEAN, description=(
                "True replaces the proposal's edits instead of appending. Default false.")),
        },
        required=["proposal", "edits"],
    ),
)

_compare_networks = genai.types.FunctionDeclaration(
    name="compare_networks",
    description=(
        "Run one study on the base network and on a proposal at the SAME operating point "
        "(one timestamp, dataset, limits and load scaling) and return both results and the "
        "difference: new and resolved violations, the largest voltage and loading changes, and "
        "`changes` (proposal minus base) — quote those rather than subtracting. study='scan' compares them at every tick of a window instead — how often "
        "and where the proposal violates when the base does not, and its worst moments. "
        "Prefer this to separate calls whenever the question compares a proposal with "
        "today's grid. `against` compares with another proposal instead of the base (is B better "
        "than A?); `proposals` compares several at once against the same reference, at one tick, "
        "and returns a ranking `table` plus each comparison."
    ),
    parameters=_S(
        type=_T.OBJECT,
        properties={
            "proposal": _S(type=_T.STRING, description=(
                "The proposal id to compare with the base (or with `against`).")),
            "proposals": _S(type=_T.ARRAY, items=_S(type=_T.STRING), description=(
                "Instead of `proposal`: up to 6 proposal ids, each compared with the same reference.")),
            "against": _S(type=_T.STRING, description=(
                "A proposal to compare with instead of the base. In the result, `base` is then that "
                "proposal and `compared_with` names it.")),
            "study": _S(type=_T.STRING, enum=["rsa", "n1", "opf", "scan"], description=(
                "'rsa' (security assessment at one moment, default), 'n1' (every single line "
                "and transformer outage at one moment), 'opf' (the deterministic flexibility "
                "OPF at one moment: whether a corrective dispatch exists on each network and "
                "what it takes), or 'scan' (the security assessment at every tick of a window).")),
            "start_timestamp": _S(type=_T.STRING, description=(
                "scan: window start (timestamp prefix); default the first tick.")),
            "end_timestamp": _S(type=_T.STRING, description=(
                "scan: window end (timestamp prefix); default the last tick.")),
            "step_size": _S(type=_T.INTEGER, description=(
                "scan: every Nth tick; default every tick (long windows are thinned and say so).")),
            "load_scaling_factor": _S(type=_T.NUMBER, description=_LOAD_SCALING_DESC),
            "vm_lower_pu": _S(type=_T.NUMBER, description=_VM_LOWER_DESC),
            "vm_upper_pu": _S(type=_T.NUMBER, description=_VM_UPPER_DESC),
            "max_line_loading_pct": _S(type=_T.NUMBER, description=_MAX_LINE_LOADING_DESC),
            "max_trafo_loading_pct": _S(type=_T.NUMBER, description=_MAX_TRAFO_LOADING_DESC),
        },
    ),
)


TOOLS: list = [
    genai.types.Tool(
        function_declarations=[
            _locate_network_element,
            _get_current_timestamp,
            _advance_timestamp,
            _get_current_conditions,
            _get_element_timeseries,
            _run_rsa,
            _simulate_contingency,
            _simulate_all_contingencies,
            _optimize_contingency,
            _optimize_flexibility,
            _evaluate_kpis,
            _forecast_kpis,
            _scan_rsa_over_time,
            _find_worst_case_timestamp,
            _scan_scenarios,
            _compare_results,
            _run_probabilistic_rsa,
            _optimize_robust_flexibility,
            _compute_flexibility_envelope,
            _compute_hosting_capacity,
            _compute_historical_risk,
            _compute_violation_attribution,
            _edit_proposal,
            _get_data_quality,
            _set_load_power_factor,
            _set_generator_model,
            _set_unit_costs,
            _rename_elements,
            _compare_networks,
        ]
    )
]

# ---------------------------------------------------------------------------
# Uniformly expose data_source / timestamp on every timeseries-reading tool.
# The LLM decides which dataset a query needs; tools default to measurements.
# Injected programmatically so the set stays consistent as tools are added.
# ---------------------------------------------------------------------------
_DATA_SOURCE_SCHEMA_DESC = "'measurements' (default, actuals) or 'forecasts' (look-ahead)."
_LOAD_PF_SCHEMA_DESC = (
    "Measured loads' power factor (0.5–1, lagging) for THIS study only; omit for the session setting.")
# Point-in-time tools also study the network's own operating point as loaded.
_POINT_DATA_SOURCE_SCHEMA_DESC = (
    "'measurements' (default), 'forecasts' or 'nominal' (the case's own point; no timestamp).")
_TIMESTAMP_SCHEMA_DESC = "Timestamp prefix, e.g. '2000-01-09 18'; omit for the current clock."
_NETWORK_SCHEMA_DESC = "'base' (default) or a proposal id."

# Which shared arguments each tool takes is part of its row in `tool_registry`.
for _tool in TOOLS:
    for _fd in _tool.function_declarations or []:
        _row = REGISTRY[_fd.name]
        if not (_row.network or _row.load_pf or _row.data_source or _row.timestamp):
            continue
        if _fd.parameters is None:
            _fd.parameters = genai.types.Schema(type=genai.types.Type.OBJECT, properties={})
        if _fd.parameters.properties is None:
            _fd.parameters.properties = {}
        props = _fd.parameters.properties
        if _row.network and "network" not in props:
            props["network"] = genai.types.Schema(
                type=genai.types.Type.STRING, description=_NETWORK_SCHEMA_DESC
            )
        if _row.load_pf:
            # Every study that assigns a tick: its load power factor.
            props.setdefault("load_power_factor", genai.types.Schema(
                type=genai.types.Type.NUMBER, description=_LOAD_PF_SCHEMA_DESC))
        if _row.data_source and "data_source" not in props:
            props["data_source"] = genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(_POINT_DATA_SOURCE_SCHEMA_DESC if _row.data_source == POINT
                             else _DATA_SOURCE_SCHEMA_DESC),
            )
        if _row.timestamp and "timestamp" not in props:
            props["timestamp"] = genai.types.Schema(
                type=genai.types.Type.STRING, description=_TIMESTAMP_SCHEMA_DESC
            )


TOOL_DISPATCH: dict[str, Callable] = {
    "locate_network_element": tools.locate_network_element,
    "get_current_timestamp": tools.get_current_timestamp,
    "advance_timestamp": tools.advance_timestamp,
    "get_current_conditions": tools.get_current_conditions,
    "get_element_timeseries": tools.get_element_timeseries,
    "run_rsa": tools.run_rsa,
    "simulate_contingency": tools.simulate_contingency,
    "simulate_all_contingencies": tools.simulate_all_contingencies,
    "optimize_contingency": tools.optimize_contingency,
    "optimize_flexibility": tools.optimize_flexibility,
    "evaluate_kpis": tools.evaluate_kpis,
    "forecast_kpis": tools.forecast_kpis,
    "scan_rsa_over_time": tools.scan_rsa_over_time,
    "find_worst_case_timestamp": tools.find_worst_case_timestamp,
    "scan_scenarios": tools.scan_scenarios,
    "compare_results": tools.compare_results,
    "run_probabilistic_rsa": tools.run_probabilistic_rsa,
    "optimize_robust_flexibility": tools.optimize_robust_flexibility,
    "compute_flexibility_envelope": tools.compute_flexibility_envelope,
    "compute_hosting_capacity": tools.compute_hosting_capacity,
    "compute_historical_risk": tools.compute_historical_risk,
    "compute_violation_attribution": tools.compute_violation_attribution,
    "edit_proposal": tools.edit_proposal,
    "get_data_quality": tools.get_data_quality,
    "set_load_power_factor": tools.set_load_power_factor,
    "set_generator_model": tools.set_generator_model,
    "set_unit_costs": tools.set_unit_costs,
    "rename_elements": tools.rename_elements,
    "compare_networks": tools.compare_networks,
}
