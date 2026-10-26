"""
reporting_notes.py — how to report a study's result, sent with that result.

The system prompt is paid for on every question; how to narrate a hosting
capacity or a robust dispatch is needed only when one ran, and is needed then —
when the model reads the result. So the system prompt keeps what decides the
call (which tool, which arguments), and each of these studies' guidance on what
to report travels in the result the model is sent, under `how_to_report`.

Only the model's copy carries it (`loop._prepare_tool_result_for_model`): the
logged result, which the answer checks read, stays as the tool returned it, so
a figure in a note is never a source for the answer.
"""

from __future__ import annotations

KEY = "how_to_report"

NOTES: dict[str, str] = {
    "scan_scenarios": (
        "Each scenario carries `violation_summary.buses`, `.lines` and `.trafos` — {element_name: tick_count}, "
        "most frequent first. Narrate: (a) compare total violation counts across scenarios; (b) how much the "
        "import from the external grid swings; (c) flag if the ×1.0 baseline already violates — the risk then "
        "exists regardless of renewable level; (d) for 'which buses/lines/trafos violated?' read those "
        "`violation_summary` fields directly — never infer element names from earlier results or general "
        "knowledge — and report the top elements with tick counts; (e) for 'when did bus X violate?' read "
        "`violations_per_tick` (only ticks with at least one violation); (f) note the worst timestamp per "
        "scenario. Limitation to state: a uniform sensitivity — all substations scale together."),
    "run_probabilistic_rsa": (
        "Key fields: `p_any_violation` (probability that at least one element violates in a sample; a sample "
        "with no converged power flow counts as violating, `n_non_converged`); `expected_violations` (mean per "
        "converged sample); `bus_violation_probability` ({bus: probability}, descending); "
        "`voltage_percentiles` (P5/P50/P95 vm_pu per bus); `violation_count_histogram`. Narrate: (a) lead with "
        "`p_any_violation` and `expected_violations`; (b) name the top-violating buses with exact "
        "probabilities; (c) describe the voltage spread from `voltage_percentiles`; (d) contrast with "
        "`scan_scenarios` (deterministic sensitivity) when both were asked for; (e) if `p_any_violation` < "
        "0.01, declare the point statistically secure; (f) explain the uncertainty: current sgen values are "
        "forecast estimates of renewable availability and `sgen_sigma` perturbs the available resource around "
        "them (not the commanded OPF setpoint), while `load_sigma` is multiplicative demand uncertainty."),
    "optimize_robust_flexibility": (
        "Say, in this order: (1) whether the risk target is met — lead with "
        "`p_any_violation_after_validation` (out of sample) as the risk that remains; (2) if it is not met, "
        "why, from `why_not_met` (the back-off tightens voltage limits only: a thermal risk is beyond it); if "
        "it is met, how — which buses' limits were tightened and by how much, from `back_off_summary`; (3) "
        "the dispatch (`activated_resources`). The method: a probabilistic RSA gives per-bus voltage percentiles, each "
        "bus's limits are tightened by how far its percentile strays, the OPF solves within them, and a fresh "
        "probabilistic RSA validates the result. For the scenario method, `effective_alpha_upper_bound` comes "
        "from scenario theory for convex problems; the AC OPF is not convex, so it is a nominal figure, not a "
        "certificate — never call it certified or guaranteed. `guarantee_met` means only that the scenario "
        "count reached the nominal bound; if K was clamped by `scenario_k_cap` it is false with "
        "`reason_code='k_required_exceeds_cap'` even when the solve is feasible. If `scenario_curtailment` "
        "reports curtailed scenarios, say so: the plan relies on holding back renewable output there, and the "
        "dispatch shown is the setpoint, not what is injected."),
    "compute_flexibility_envelope": (
        "Lead with `message`: the safe reactive range at the unit's present output. Key fields: `base_point` "
        "(the current output); `safe_q_range_at_base_p` ([q_min, q_max] Mvar); `unit_q_limits_mvar` (a "
        "voltage-controlled unit's own reactive limits, studied as an injection: `unit_studied_as_injection` "
        "— say so); `violated_everywhere` (elements outside their limits at every point: this unit alone "
        "cannot clear them — say so) with `safe_q_range_apart_from_them`; `n_feasible` / `n_total`; per point "
        "`binding_constraint`. Name the dominant binding constraint, and say the envelope is valid at this "
        "operating point only."),
    "compute_hosting_capacity": (
        "Always include this q-mode glossary before any conclusion or ranking — even when only one mode ran, "
        "in which case also say which one: `unity` — injected reactive power held near zero (Q≈0); `fixed_pf` "
        "— reactive power follows a fixed power factor (`pf_sign='absorbing'` = inductive absorption, negative "
        "Q; 'injecting' = capacitive injection, positive Q); `reactive_proxy` — a first-slice approximation "
        "using a damped counteracting version of the bus's current reactive operating point, clipped to a "
        "PF=0.9 capability envelope: a proxy, not a true controller, and it does not enforce a power-factor "
        "constraint (with bus Q near zero it can resemble `unity`). When comparing, say that `fixed_pf` is "
        "PF-constrained by definition while `reactive_proxy` is not; never describe `reactive_proxy` as a "
        "fixed-power-factor strategy. Deterministic runs: each mode's `hosting_capacity_mw` and "
        "`binding_constraint` (for bus='all': the ranked buses and typical limiting constraints). "
        "Probabilistic runs: also `risk_threshold` and `p_any_violation_at_best`, and the uncertainty scope in "
        "plain words — `added_generation_only` (only the new injection is uncertain; the realised added P lies "
        "between zero and the candidate hosting value) or `added_generation_plus_load` (load is also sampled "
        "multiplicatively around the current operating point). If `synthetic_uncertainty` is true, say the "
        "result uses assumed uncertainty because the system's operating data are synthetic or uploaded rather "
        "than a measured history."),
    "compute_historical_risk": (
        "Narrate: (a) `exceedance_frequency` and `near_miss_frequency`; (b) `worst_episodes`, by target — for "
        "a bus, line or transformer, the top 3 by `peak_severity` and the top 3 by `duration_steps`, "
        "deduplicated by (start, end), told chronologically; for target_kind 'all', not as local events but "
        "as system stress periods, the highest simultaneous severity first and the longest sustained period "
        "too; with 3 episodes or fewer, all of them; (c) the duration-curve crossing against the exceedance "
        "percent; (d) with `conditional_bins`, compare high- and low-risk bins with their sample counts, so "
        "sparse bins are not over-read; (e) if `used_margin_fallback` is true, say hard violations were rare "
        "and the reading is margin-based."),
}


def with_note(tool_name: str, view):
    """The model's copy of a result, with how to report it when there is a note
    for that study and the study did not fail."""
    note = NOTES.get(tool_name)
    if not note or not isinstance(view, dict) or view.get("error"):
        return view
    # First: read before the data it is about, not after a long result.
    return {KEY: note, **view}
