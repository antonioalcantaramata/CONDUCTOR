"""Detecting that the backend is serving a different network.

Networks are uploaded by client-side JavaScript straight to the backend, so no
Python code runs on a swap and nothing clears the chat. Left alone, the model
receives a system prompt describing the new grid alongside history describing
the old one, and blends them.

The response to a detected change is destructive — the conversation is
discarded — so these tests care as much about *not* firing spuriously as about
firing when they should.
"""

import pytest

from llm_agent.agent.config import (
    DEFAULT_GRID_CONSTANTS,
    GridConstants,
    is_network_change,
    network_fingerprint,
)

GRID_A = {
    "name": "Bornholm Distribution Network",
    "n_substations": 2,
    "n_lines": 3,
    "n_trafos": 1,
    "substation_names": ["SUB_A", "SUB_B"],
    "vm_lower": 0.90,
    "vm_upper": 1.10,
}
GRID_B = {
    "name": "Some Other Grid",
    "n_substations": 5,
    "n_lines": 7,
    "n_trafos": 2,
    "substation_names": ["X", "Y", "Z", "W", "V"],
}


def live(values):
    return GridConstants(values=values, live=True)


def degraded():
    return GridConstants(
        values=DEFAULT_GRID_CONSTANTS, live=False, error="connection refused"
    )


class TestFingerprint:
    def test_same_network_yields_the_same_fingerprint(self):
        assert network_fingerprint(GRID_A) == network_fingerprint(dict(GRID_A))

    def test_different_networks_differ(self):
        assert network_fingerprint(GRID_A) != network_fingerprint(GRID_B)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("name", "Renamed"),
            ("n_substations", 99),
            ("n_lines", 99),
            ("n_trafos", 99),
            ("substation_names", ["DIFFERENT"]),
        ],
    )
    def test_identity_fields_change_the_fingerprint(self, field, value):
        assert network_fingerprint({**GRID_A, field: value}) != network_fingerprint(GRID_A)

    @pytest.mark.parametrize("field,value", [("vm_lower", 0.85), ("vm_upper", 1.15),
                                             ("max_loading_pct", 120.0)])
    def test_retuning_limits_is_not_a_new_network(self, field, value):
        # An operator adjusting thresholds on the same grid must not lose their
        # conversation.
        assert network_fingerprint({**GRID_A, field: value}) == network_fingerprint(GRID_A)

    def test_handles_missing_and_null_fields(self):
        assert isinstance(network_fingerprint({}), str)
        assert isinstance(network_fingerprint({"substation_names": None}), str)

    def test_bus_rename_is_detected_even_at_equal_counts(self):
        renamed = {**GRID_A, "substation_names": ["SUB_A", "RENAMED"]}
        assert network_fingerprint(renamed) != network_fingerprint(GRID_A)


class TestChangeDetection:
    def test_detects_a_genuine_swap(self):
        assert is_network_change(network_fingerprint(GRID_A), live(GRID_B))

    def test_same_network_is_not_a_change(self):
        assert not is_network_change(network_fingerprint(GRID_A), live(GRID_A))

    def test_first_observation_establishes_a_baseline(self):
        # Nothing to compare against; resetting here would clear the chat on
        # every fresh session.
        assert not is_network_change(None, live(GRID_A))

    def test_degraded_fetch_is_never_treated_as_a_swap(self):
        """A backend blip must not wipe the user's conversation."""
        assert not is_network_change(network_fingerprint(GRID_A), degraded())

    def test_recovery_after_an_outage_is_not_a_swap(self):
        previous = network_fingerprint(GRID_A)
        assert not is_network_change(previous, degraded())   # during outage
        assert not is_network_change(previous, live(GRID_A))  # after recovery

    def test_swap_that_happens_during_an_outage_is_caught_on_recovery(self):
        previous = network_fingerprint(GRID_A)
        assert not is_network_change(previous, degraded())
        assert is_network_change(previous, live(GRID_B))

    def test_repeated_checks_on_a_stable_network_stay_quiet(self):
        fp = network_fingerprint(GRID_A)
        assert not any(is_network_change(fp, live(GRID_A)) for _ in range(5))


class TestSwapSequence:
    def test_full_lifecycle(self):
        """Baseline, stable use, a swap, then stability on the new network."""
        fingerprint = None
        events = []

        for status in [live(GRID_A), live(GRID_A), live(GRID_B), live(GRID_B)]:
            events.append(is_network_change(fingerprint, status))
            if status.live:
                fingerprint = network_fingerprint(status.values)

        # Only the transition into GRID_B counts.
        assert events == [False, False, True, False]

    def test_outage_in_the_middle_does_not_produce_a_false_positive(self):
        fingerprint = None
        events = []

        for status in [live(GRID_A), degraded(), degraded(), live(GRID_A)]:
            events.append(is_network_change(fingerprint, status))
            if status.live:
                fingerprint = network_fingerprint(status.values)

        assert events == [False, False, False, False]
