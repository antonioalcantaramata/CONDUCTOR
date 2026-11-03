"""
A renewable scaling applies to the units stated, named or declared wind or
solar — static or on voltage control — and says which (main_backend
`_renewable_rows`, `_scale_renewables`).
"""

import pandapower.networks as pn
import pytest


@pytest.fixture
def mb(monkeypatch):
    import main_backend
    monkeypatch.setitem(main_backend.app_data, "unit_sources", {})
    monkeypatch.setitem(main_backend.app_data, "grid_profile", {})
    return main_backend


def test_a_declared_voltage_controlled_unit_is_scaled(mb):
    net = pn.case30()
    assert len(net.sgen) == 0
    mb.app_data["unit_sources"] = {"Gen_bus1": "wind"}
    rows, note = mb._renewable_rows(net)
    assert rows == [("gen", 0)] and note["units"] == ["Gen_bus1"]
    before = float(net.gen.at[0, "p_mw"])
    mb._scale_renewables(net, 0.5, rows)
    assert net.gen.at[0, "p_mw"] == pytest.approx(0.5 * before)


def test_with_none_the_note_still_says_so(mb):
    net = pn.case30()
    mb._study.renewables = None
    try:
        mb._scale_renewables(net, 0.7)
        assert mb._study.renewables["units"] == [] and "changes nothing" in mb._study.renewables["note"]
    finally:
        mb._study.renewables = "off"


def test_only_wind_and_solar_static_units_carry_renewable_uncertainty(mb):
    # Observed: the robust OPF curtailed a conventional unit converted to a
    # static generator "as renewable output".
    net = pn.create_cigre_network_mv(with_der="all")
    mask, note = mb._renewable_mask(net)
    assert dict(zip(net.sgen["name"], mask)) == {
        name: t in ("PV", "WP") for name, t in zip(net.sgen["name"], net.sgen["type"])}
    assert "WKA 7" in note["units"] and "CHP diesel 1" not in note["units"]
    mask, note = mb._renewable_mask(pn.case30())
    assert len(mask) == 0 and note["units"] == []


def test_the_third_arm_restores_the_behaviour_before_the_layer(mb, monkeypatch):
    # CONDUCTOR_ONTOLOGY=none (the ablation's third arm): every static generator is
    # renewable, declarations change nothing.
    net = pn.create_cigre_network_mv(with_der="all")
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "none")
    assert not mb.studies_use_semantic_layer()
    rows, note = mb._renewable_rows(net)
    assert rows == [("sgen", int(i)) for i in net.sgen.index] and "CHP diesel 1" in note["units"]
    mb.app_data["unit_sources"] = {"CHP diesel 1": "thermal"}
    assert len(mb._renewable_rows(net)[0]) == len(net.sgen)
    monkeypatch.setenv("CONDUCTOR_ONTOLOGY", "off")                      # the agent's arm: studies unchanged
    assert mb.studies_use_semantic_layer() and "CHP diesel 1" not in mb._renewable_rows(net)[1]["units"]
