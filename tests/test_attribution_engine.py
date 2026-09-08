"""Violation attribution — measuring what drives a violation, not narrating it.

The security engines say *which* elements violate. Answering *why* from that
table alone requires the reader to supply the causal story, and a language model
asked to do so produces something plausible but ungrounded. This engine measures
it on the same validated power flow: perturb an injection, re-solve, observe.

These tests run against a real pandapower network, so the sensitivities are
checked against actual power-flow behaviour rather than a mock.
"""

import sys
import os

import pandapower.networks as pn
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from attribution_engine import attribute_violations  # noqa: E402


@pytest.fixture(scope="module")
def violating_case14():
    """case14 with a ceiling tight enough that the base case violates."""
    return pn.case14()


@pytest.fixture(scope="module")
def report(violating_case14):
    return attribute_violations(violating_case14, "2022-01-01 00:00", vm_upper=1.02)


class TestBasicContract:
    def test_reports_convergence_and_violations(self, report):
        assert report["converged"] is True
        assert report["violations"]

    def test_counts_its_own_work(self, report):
        """One base solve, then one per perturbation.

        Two per source where both P and Q can be perturbed, but only one for
        `gen` elements: pandapower's gen table has no q_mvar column, because a
        PV generator sets voltage and its reactive output is solved for.
        """
        sources = report["n_sources_tested"]
        assert sources + 1 <= report["n_power_flows"] <= 2 * sources + 1

    def test_states_its_method_and_limits(self, report):
        assert "perturbation" in report["method"]
        assert any("local" in n for n in report["notes"])

    def test_secure_operating_point_says_so(self, violating_case14):
        result = attribute_violations(violating_case14, "t", vm_lower=0.5, vm_upper=1.5)
        assert result["violations"] == []
        assert "No violations" in result["message"]

    def test_network_is_left_solved_and_unperturbed(self, violating_case14):
        before = violating_case14.sgen["p_mw"].copy() if not violating_case14.sgen.empty else None
        attribute_violations(violating_case14, "t", vm_upper=1.02)
        assert not violating_case14.res_bus.empty
        if before is not None:
            assert violating_case14.sgen["p_mw"].equals(before)


class TestSensitivitiesAreReal:
    def test_pq_buses_get_ranked_drivers(self, report):
        with_drivers = [v for v in report["violations"] if v["drivers"]]
        assert with_drivers, "no violation received any driver"

    def test_drivers_are_sorted_by_influence(self, report):
        for violation in report["violations"]:
            influences = [
                max(abs(d.get("d_per_mw") or 0), abs(d.get("d_per_mvar") or 0))
                for d in violation["drivers"]
            ]
            assert influences == sorted(influences, reverse=True)

    def test_reactive_injection_lowers_voltage_at_load_buses(self, report):
        """Physical sanity: more load MVAr pulls voltage down."""
        checked = 0
        for violation in report["violations"]:
            if violation["quantity"] != "vm_pu":
                continue
            for d in violation["drivers"]:
                if d["kind"] == "load" and d["d_per_mvar"] is not None:
                    assert d["d_per_mvar"] < 0
                    checked += 1
        assert checked, "no load sensitivities were produced"

    def test_relief_moves_the_quantity_exactly_to_its_limit(self, report):
        """value + sensitivity x relief == limit, by construction."""
        for violation in report["violations"]:
            for d in violation["drivers"]:
                if d.get("relief_mvar") is None or d.get("d_per_mvar") is None:
                    continue
                predicted = violation["value"] + d["d_per_mvar"] * d["relief_mvar"]
                assert predicted == pytest.approx(violation["limit"], abs=1e-3)
                return

    def test_loads_are_marked_uncontrollable(self, report):
        for violation in report["violations"]:
            for d in violation["drivers"]:
                if d["kind"] == "load":
                    assert d["controllable"] is False


class TestVoltageControlledBuses:
    """A PV or slack bus is held at setpoint, so no injection can move it."""

    def test_controlled_buses_are_identified(self, report, violating_case14):
        controlled = set(violating_case14.gen.bus) | set(violating_case14.ext_grid.bus)
        flagged = {v["index"] for v in report["violations"] if v.get("voltage_controlled_by")}
        assert flagged <= controlled
        assert flagged, "no controlled bus was identified"

    def test_zero_sensitivity_is_explained_rather_than_left_blank(self, report):
        for violation in report["violations"]:
            if violation.get("voltage_controlled_by"):
                assert violation["drivers"] == []
                assert "setpoint" in violation["explanation"]

    def test_uncontrolled_buses_are_not_given_that_excuse(self, report):
        for violation in report["violations"]:
            if violation["drivers"]:
                assert "voltage_controlled_by" not in violation


class TestBudgetAndRobustness:
    def test_source_cap_bounds_the_power_flow_count(self, violating_case14):
        capped = attribute_violations(violating_case14, "t", vm_upper=1.02, top_n_sources=3)
        assert capped["n_sources_tested"] <= 3
        assert capped["n_power_flows"] <= 7

    def test_driver_cap_is_respected(self, violating_case14):
        """The cap bounds the ranked list, plus any controllable source added
        so the answer is actionable (see TestActionability)."""
        capped = attribute_violations(violating_case14, "t", vm_upper=1.02,
                                      top_n_drivers=2, min_controllable=2)
        assert all(len(v["drivers"]) <= 4 for v in capped["violations"])

    def test_cap_alone_applies_when_controllables_already_rank(self, violating_case14):
        capped = attribute_violations(violating_case14, "t", vm_upper=1.02,
                                      top_n_drivers=2, min_controllable=0)
        assert all(len(v["drivers"]) <= 2 for v in capped["violations"])

    def test_largest_sources_are_tested_first(self, violating_case14):
        capped = attribute_violations(violating_case14, "t", vm_upper=1.02, top_n_sources=2)
        named = {d["source"] for v in capped["violations"] for d in v["drivers"]}
        assert len(named) <= 2

    def test_thermal_violations_are_attributed_too(self, violating_case14):
        report = attribute_violations(violating_case14, "t", vm_upper=2.0, vm_lower=0.1,
                                      max_line_loading_pct=1.0)
        overloads = [v for v in report["violations"] if v["quantity"] == "loading_percent"]
        assert overloads
        assert any(v["drivers"] for v in overloads)

    def test_non_convergent_base_case_is_reported_not_raised(self):
        net = pn.case14()
        net.load["p_mw"] *= 500  # far beyond what the network can serve
        result = attribute_violations(net, "t")
        assert result["converged"] is False
        assert "did not converge" in result["error"]


class TestNoContributionShares:
    """Shares require an arbitrary baseline, so the tool must not invent them."""

    def test_payload_reports_sensitivities_not_percentages(self, report):
        for violation in report["violations"]:
            for d in violation["drivers"]:
                assert "share" not in d
                assert "contribution_pct" not in d
                assert {"d_per_mw", "d_per_mvar"} <= set(d)


class TestActionability:
    """A driver list an operator cannot act on is a diagnosis, not an answer."""

    def test_every_violation_offers_at_least_one_lever(self, violating_case14):
        # The largest driver is often a load, which explains the regime without
        # offering a control action; the controllable sources that do can sit
        # well down the ranking and would otherwise be cut by the cap.
        report = attribute_violations(violating_case14, "t", vm_upper=1.02,
                                      top_n_drivers=3, min_controllable=2)
        for violation in report["violations"]:
            if not violation["drivers"]:
                continue
            assert any(d["controllable"] for d in violation["drivers"]), (
                f"{violation['element']} lists only uncontrollable drivers"
            )

    def test_ranking_order_is_preserved_within_the_ranked_head(self, violating_case14):
        report = attribute_violations(violating_case14, "t", vm_upper=1.02, top_n_drivers=3)
        for violation in report["violations"]:
            head = violation["drivers"][:3]
            influences = [max(abs(d.get("d_per_mw") or 0), abs(d.get("d_per_mvar") or 0))
                          for d in head]
            assert influences == sorted(influences, reverse=True)


class TestReliefFeasibility:
    """A sensitivity will happily recommend the impossible if left unchecked."""

    def test_reduction_beyond_current_output_is_marked_impossible(self):
        from attribution_engine import _relief_feasible
        # A 3.54 MW generator cannot reduce by 6.24 MW.
        assert _relief_feasible(-6.24, 3.54) is False

    def test_reduction_within_output_is_feasible(self):
        from attribution_engine import _relief_feasible
        assert _relief_feasible(-1.0, 3.54) is True

    def test_increase_headroom_is_not_claimed(self):
        from attribution_engine import _relief_feasible
        # Upper capability is unknown here, so the answer is "unknown", not "yes".
        assert _relief_feasible(2.0, 3.54) is None

    def test_absent_relief_is_unknown(self):
        from attribution_engine import _relief_feasible
        assert _relief_feasible(None, 1.0) is None

    def test_every_driver_carries_a_feasibility_verdict(self, report):
        for violation in report["violations"]:
            for d in violation["drivers"]:
                assert "relief_mw_feasible" in d
                assert "relief_mvar_feasible" in d
                assert d["relief_mw_feasible"] in (True, False, None)

    def test_no_reported_relief_is_impossible(self, report):
        """The payload must not contain a number it would be wrong to state.

        Flagging was not enough. With `relief_mw_feasible: false` beside it and
        a resolved action saying no single source could clear the bus, the
        model still wrote "Reduce 05 ALP Sgen by 6.24 MW" about a 3.54 MW unit.
        An impossible movement is now withdrawn, not annotated.
        """
        for violation in report["violations"]:
            for d in violation["drivers"]:
                for relief_key, current_key in (("relief_mw", "current_p_mw"),
                                                ("relief_mvar", "current_q_mvar")):
                    relief = d.get(relief_key)
                    if relief is None:
                        continue
                    assert d[current_key] + relief >= 0, (
                        f"{d['source']} still reports {relief_key}={relief} against "
                        f"{d[current_key]} available"
                    )

    def test_a_withdrawn_movement_says_what_the_source_could_do(self, report):
        """Being told only what cannot be done is what made an earlier version
        of the prompt rule answer with no numbers at all.

        Scoped to controllable drivers. A `False` flag now has two meanings:
        the source cannot deliver the movement, or nobody can command it at
        all. Only the first has a deliverable capacity to report — a load has
        no lever, so `max_deliverable_*` would be a lever that does not exist.
        """
        withdrawn = 0
        for violation in report["violations"]:
            for d in violation["drivers"]:
                if d.get("actionable") is False:
                    continue
                for flag, relief_key, deliverable_key, current_key in (
                    ("relief_mw_feasible", "relief_mw", "max_deliverable_mw", "current_p_mw"),
                    ("relief_mvar_feasible", "relief_mvar", "max_deliverable_mvar", "current_q_mvar"),
                ):
                    if d[flag] is not False:
                        assert deliverable_key not in d
                        continue
                    withdrawn += 1
                    assert d[relief_key] is None
                    assert d[deliverable_key] == pytest.approx(-d[current_key], abs=1e-3)
        assert withdrawn, "expected at least one impossible movement in this scenario"

    def test_an_uncontrollable_driver_offers_no_action(self, report):
        """A load explains the regime and offers no lever. Its movement must
        not survive in a field an answer would read as a recommendation."""
        seen = 0
        for violation in report["violations"]:
            for d in violation["drivers"]:
                if d.get("controllable") is not False:
                    continue
                seen += 1
                assert d["actionable"] is False
                assert d["relief_mw"] is None and d["relief_mvar"] is None
                assert "max_deliverable_mw" not in d
                assert "max_deliverable_mvar" not in d
                # The magnitude survives, renamed: the diagnosis needs it.
                assert d.get("would_require_mw") is not None or d.get("would_require_mvar") is not None
        assert seen, "expected at least one load among the drivers"

    def test_unknown_headroom_is_left_alone(self, report):
        """Only the provably impossible is withdrawn.

        The same severity split the parameter guards use: an increase whose
        headroom is not known here is unusual, not impossible, and removing it
        would delete the diagnostic content that explains the regime.
        """
        for violation in report["violations"]:
            for d in violation["drivers"]:
                if d["relief_mw_feasible"] is None and d["d_per_mw"]:
                    assert "max_deliverable_mw" not in d

    def test_notes_point_the_reader_at_the_resolved_action(self, report):
        assert any("withdrawn" in n.lower() for n in report["notes"])
        assert any("rather than assembling an action" in n.lower() for n in report["notes"])


class TestRecommendedAction:
    """The feasibility decision must be made here, not by the model.

    Deciding whether a movement is deliverable means reading a flag across every
    driver and branching correctly on all of them. In live testing the model got
    that right ten times out of eleven and, on the eleventh, reported an
    infeasible curtailment as feasible — a confident, specific, impossible
    recommendation. The rule is deterministic, so it belongs in code.
    """

    def test_every_violation_carries_a_resolved_action(self, report):
        for violation in report["violations"]:
            action = violation["recommended_action"]
            assert action["text"]
            assert action["kind"] in {
                "single_source", "none_sufficient", "setpoint",
                "no_controllable_source",
            }

    def test_voltage_controlled_buses_point_at_the_setpoint(self, report):
        for violation in report["violations"]:
            if violation.get("voltage_controlled_by"):
                action = violation["recommended_action"]
                assert action["kind"] == "setpoint"
                assert violation["voltage_controlled_by"] in action["text"]

    def test_an_infeasible_movement_is_never_recommended(self, report):
        """The failure this exists to prevent."""
        for violation in report["violations"]:
            action = violation["recommended_action"]
            if action["kind"] != "single_source":
                continue
            driver = next(d for d in violation["drivers"] if d["source"] == action["source"])
            flag = ("relief_mw_feasible" if action["unit"] == "MW"
                    else "relief_mvar_feasible")
            assert driver[flag] is True

    def test_recommended_movement_matches_a_real_driver_value(self, report):
        for violation in report["violations"]:
            action = violation["recommended_action"]
            if action["kind"] != "single_source":
                continue
            driver = next(d for d in violation["drivers"] if d["source"] == action["source"])
            key = "relief_mw" if action["unit"] == "MW" else "relief_mvar"
            assert action["movement"] == pytest.approx(driver[key], abs=1e-3)

    def test_the_smallest_deliverable_movement_is_chosen(self, report):
        for violation in report["violations"]:
            action = violation["recommended_action"]
            if action["kind"] != "single_source":
                continue
            deliverable = [
                abs(d[k]) for d in violation["drivers"] if d["controllable"]
                for k, f in (("relief_mw", "relief_mw_feasible"),
                             ("relief_mvar", "relief_mvar_feasible"))
                if d[k] is not None and d[f] is True
            ]
            assert abs(action["movement"]) == pytest.approx(min(deliverable), abs=1e-3)

    def test_when_nothing_is_deliverable_it_says_so_with_the_shortfall(self, report):
        exhausted = [v for v in report["violations"]
                     if v["recommended_action"]["kind"] == "none_sufficient"]
        assert exhausted, "expected at least one violation no single source can clear"
        for violation in exhausted:
            text = violation["recommended_action"]["text"]
            assert "No single source can clear" in text
            assert "exceeds its available" in text
            assert "several sources acting together" in text

    def test_no_deliverable_movement_is_hidden_behind_none_sufficient(self, report):
        """If a feasible option exists, it must be offered rather than declined."""
        for violation in report["violations"]:
            if violation["recommended_action"]["kind"] != "none_sufficient":
                continue
            for d in violation["drivers"]:
                if not d["controllable"]:
                    continue
                assert d["relief_mw_feasible"] is not True
                assert d["relief_mvar_feasible"] is not True

    def test_notes_tell_the_reader_to_use_the_resolved_action(self, report):
        assert any("recommended_action" in n for n in report["notes"])
