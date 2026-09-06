"""
tool_schemas.py — Gemini FunctionDeclaration schemas for the digital twin tools.

Every schema here is sent on every request, so description length is a running
cost: these twenty-one tools are ~13.6k tokens of each prompt. This is the
compacted form, kept after a trial against a longer version — descriptions were
tightened and grid-specific facts removed, deferring to the system prompt,
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

_LOAD_SCALING_DESC = (
    "Uniform multiplier on all active and reactive loads. 1.0 = base case. "
    "See 'Grid domain knowledge' in the system prompt."
)

_SLACK_MAX_MW_DESC = (
    "External grid / cable capacity cap in MW. Omit (null/None) to use the "
    "network's physical limit from the loaded network file (default). "
    "0.0 = fully islanded. Positive values cap the bidirectional power exchange "
    "to that level; use this when the user asks about islanding or derating "
    "scenarios."
)

_SLACK_Q_MAX_MVAR_DESC = (
    "Independent reactive power limit on the external-grid interface in Mvar. Default None "
    "(Q bound equals slack_max_mw). Set e.g. to 20 to cap reactive exchange "
    "with the external grid to 20 Mvar regardless of the active power setting. "
    "Forces local generators to supply more Q, shifting Pg_new in the dispatch chart."
)

_ELEMENT_TYPE_DESC = (
    "Network element type: 'line' for power lines or 'trafo' for transformers. "
    "Element indices depend on the currently loaded network."
)

_DISABLED_GEN_DESC = (
    "List of substation names whose generators are forced offline. "
    f"{_NAMES_FROM_PROMPT} "
    "Use when user asks about generator loss or islanding of a substation."
)

_OPF_VM_UPPER_DESC = (
    "Voltage upper bound enforced inside the AC OPF (p.u.). Default 1.05. "
    "Use 1.04 to tighten the optimizer's voltage envelope and force more conservative dispatch. "
    "A tighter bound will result in more redispatch visible in the dispatch chart."
)

_OPF_VM_LOWER_DESC = (
    "Voltage lower bound enforced inside the AC OPF (p.u.). Default 0.95. "
    "Use 0.93 to relax the lower bound and widen the feasible region when the "
    "optimizer would otherwise be infeasible at low load."
)

_OPF_LAMBDA_P_DESC = (
    "Weight on active power deviation squared in the OPF objective. Default 0.01. "
    "Increase (e.g. 0.1) to penalise active redispatch more heavily and keep Pg_new "
    "close to Pg_base; decrease (e.g. 0.001) to allow larger P changes when needed."
)

_OPF_LAMBDA_Q_DESC = (
    "Weight on reactive power deviation squared in the OPF objective. Default 0.001. "
    "Increase to penalise reactive redispatch; the default is 10x smaller than "
    "lambda_p because reactive power is typically cheaper to redispatch than active power."
)

_FIXED_SETPOINTS_DESC = (
    "Optional dict that pins specific generators or the external-grid slack to an exact MW value, "
    "preventing the OPF from changing their output. "
    f"Keys are substation names (generators) or {_SLACK_FROM_PROMPT}. "
    f"{_NAMES_FROM_PROMPT} "
    'Setting the slack entry to 5.0 keeps external-grid import fixed at 5 MW; '
    'setting it to 0.0 fully isolates the external interface. '
    "Unspecified elements are free to redispatch normally."
)

_PG_MAX_OVERRIDES_DESC = (
    "Optional dict overriding the maximum active power capacity (MW) for specific substations. "
    'Example: {"Echo": 5.0, "\u00c5kirkeby": 3.0} caps those generators. '
    "Keys are substation names; values are the new upper bound in MW. "
    "Unspecified substations keep their default capacity."
)

_PG_MIN_OVERRIDES_DESC = (
    "Optional dict overriding the minimum active power (MW) for specific substations. "
    "Use a positive value to set a must-run constraint (e.g. {\"Echo\": 1.0} forces "
    "Echo to produce at least 1 MW). Use a less-negative value to reduce curtailment headroom. "
    "Unspecified substations keep their default minimum."
)

_MIN_PF_DESC = (
    "Minimum generator power factor enforced in the OPF. Default 0.95. "
    "Relax to 0.90 to allow more reactive redispatch and increase feasibility; "
    "tighten to 0.98 to keep generators near unity power factor. "
    "Affects how much Q each generator can provide relative to its P output."
)

_CURRENT_SAFETY_MARGIN_DESC = (
    "Safety factor on branch current limits inside the OPF (0–1). Default 0.9 (90% of rated). "
    "Raise to 1.0 for a stress test that uses 100% of rated current; "
    "lower to 0.85 to enforce a more conservative thermal margin. "
    "Affects how much dispatch freedom the optimizer has before branch overloads are penalised."
)

_VM_UPPER_DESC = (
    "Upper voltage limit, p.u. Use this network's value from 'Grid domain "
    "knowledge' in the system prompt."
)

_VM_LOWER_DESC = (
    "Lower voltage limit, p.u. Use this network's value from 'Grid domain "
    "knowledge' in the system prompt."
)

_MAX_LINE_LOADING_DESC = (
    "Line thermal overload threshold, percent. See 'Grid domain knowledge' in "
    "the system prompt."
)

_MAX_TRAFO_LOADING_DESC = (
    "Transformer thermal overload threshold, percent. Same semantics as "
    "max_line_loading_pct."
)



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
                    "Zero-based index of the element to take out of service. "
                    "Lines: 0–22. Transformers: 0–15."
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
        "'is the grid N-1 secure'."
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

_optimize_contingency = genai.types.FunctionDeclaration(
    name="optimize_contingency",
    description=(
        "Compute the corrective flexibility dispatch after a single contingency "
        "using AC OPF (SMFAE). Returns the generator setpoints before and after "
        "optimization. Use this when the user asks how to fix a specific outage."
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
                    "Zero-based index of the contingency element. "
                    "Lines: 0–22. Transformers: 0–15."
                ),
            ),
            "load_scaling_factor": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_LOAD_SCALING_DESC,
            ),
            "slack_max_mw": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_SLACK_MAX_MW_DESC,
            ),
            "opf_vm_upper": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "opf_vm_lower": genai.types.Schema(
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
        "corrective actions, generator outages, or cable derating."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
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
            "opf_vm_upper": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "opf_vm_lower": genai.types.Schema(
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
            "opf_vm_upper": genai.types.Schema(
                type=genai.types.Type.NUMBER,
                description=_OPF_VM_UPPER_DESC,
            ),
            "opf_vm_lower": genai.types.Schema(
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
        "'effect of disabling Echo', 'difference between cable derated to 20 MW vs 70 MW', etc."
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
                description="Human-readable label for the second (scenario) run, e.g. 'Cable 1 MW' or 'No Echo'. Default: 'Scenario'.",
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
        "'when is voltage lowest?', 'scan the whole dataset for the worst point'. "
        "After calling this, offer to advance_timestamp to the worst timestamp and run "
        "run_rsa() or optimize_flexibility() for a full analysis."
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
        "Use when the user asks: "
        "'what is the probability of a voltage violation at this operating point?', "
        "'how risky is the current state under uncertainty?', "
        "'what is the chance Echo overvoltages with load uncertainty?', "
        "'give me a probabilistic assessment', "
        "'P5/P50/P95 voltage envelope', "
        "'expected number of violations'. "
        "Does NOT advance the clock. "
        "Returns: p_any_violation (probability any element violates), "
        "expected_violations (mean violation count), "
        "bus_violation_probability (per-bus probability dict), "
        "voltage_percentiles (P5/P50/P95 vm_pu per bus), "
        "violation_count_histogram (distribution of total violations per sample)."
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
        "Use when the user asks: "
        "'apply a robust dispatch', "
        "'secure the grid with 95% confidence under load uncertainty', "
        "'guarantee security under uncertainty', "
        "'apply back-off constraints for wind uncertainty', "
        "'tighten voltage bounds to account for forecast error', "
        "'robust OPF'. "
        "Returns: activated_resources, bus_voltages_post_opf, upper/lower back-offs and tightened bounds, "
        "p_any_violation_before, p_any_violation_after, "
        "p_any_violation_after_validation, robust_loop_iterations, robust_loop_stop_reason, "
        "sgen_sigma, risk_target, n_samples, validation_samples. "
        "Scenario mode also returns guarantee diagnostics including guarantee_met, "
        "effective_alpha_upper_bound, and guarantee_interpretation."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
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
        "Use when the user asks: "
        "'what is the safe dispatch range for [generator]?', "
        "'how much Q can [generator] inject or absorb?', "
        "'show the flexibility region for [generator]', "
        "'show the secure operating region for [generator]', "
        "'what Q is safe for [generator] right now?', "
        "'explain why the OPF chose Q=[X] for [generator]', "
        "'what is the flexibility envelope for [generator]?'. "
        "Returns: gen_name, base_point, envelope (list of PQ points with feasibility), "
        "safe_q_range_at_base_p, n_feasible, n_total, and capability_curve_pf metadata."
    ),
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "gen_name": genai.types.Schema(
                type=genai.types.Type.STRING,
                description=(
                    "Substation name of the generator to sweep. "
                    "Must match a 'substation_name' in the network sgen table "
                    "(e.g. 'Echo', 'Alpha', 'Foxtrot'). Required."
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
        ]
    )
]

# ---------------------------------------------------------------------------
# Uniformly expose data_source / timestamp on every timeseries-reading tool.
# The LLM decides which dataset a query needs; tools default to measurements.
# Injected programmatically so the set stays consistent as tools are added.
# ---------------------------------------------------------------------------
_DATA_SOURCE_SCHEMA_DESC = (
    "'measurements' (default, historical actuals) or 'forecasts' (look-ahead). "
    "See 'Data sources' in the system prompt."
)
_TIMESTAMP_SCHEMA_DESC = (
    "Optional ISO timestamp prefix, e.g. '2000-01-09 18'. Omit to use the "
    "current clock. See 'Data sources' in the system prompt."
)

# Clock-control and pure post-processing tools never read a selectable dataset.
_NO_DATA_SOURCE_TOOLS = {
    "locate_network_element",  # topology, not a time series
    "get_current_timestamp",
    "advance_timestamp",
    "compare_results",
    "scan_rsa_over_time",  # advances the live clock — measurements only by definition
}
# Point-in-time tools (no existing time-window params) that benefit from `timestamp`.
_POINT_IN_TIME_TOOLS = {
    "compute_violation_attribution",
    "get_current_conditions",
    "run_rsa",
    "simulate_contingency",
    "simulate_all_contingencies",
    "optimize_contingency",
    "optimize_flexibility",
    "evaluate_kpis",
    "forecast_kpis",
    "run_probabilistic_rsa",
    "optimize_robust_flexibility",
    "compute_flexibility_envelope",
    "compute_hosting_capacity",
}

for _tool in TOOLS:
    for _fd in _tool.function_declarations or []:
        if _fd.name in _NO_DATA_SOURCE_TOOLS:
            continue
        if _fd.parameters is None:
            _fd.parameters = genai.types.Schema(type=genai.types.Type.OBJECT, properties={})
        if _fd.parameters.properties is None:
            _fd.parameters.properties = {}
        props = _fd.parameters.properties
        if "data_source" not in props:
            props["data_source"] = genai.types.Schema(
                type=genai.types.Type.STRING, description=_DATA_SOURCE_SCHEMA_DESC
            )
        if _fd.name in _POINT_IN_TIME_TOOLS and "timestamp" not in props:
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
}
