"""
Every network has a CIM model (backend/cgmes_writer.py writes one, as CGMES, for a network that did not come as CIM)
and the computation is a copy of it (backend/cim_adapter.py): pandapower → CGMES → pandapower must give the same
power flow, element by element through the mRIDs, on the bundled networks and on the elements that need care —
parallel circuits, derating, negative reactances, phase shifts that are not a multiple of 30°, 60 Hz, three-winding
transformers, equivalents, storage, costs. What the CGMES profiles alone cannot carry is listed, never dropped.
"""

import copy
import io
import pathlib
import re
import zipfile

import pandapower as pp
import pandapower.networks as pn
import pytest
from fastapi.testclient import TestClient

import cgmes_writer as cw
import cim_adapter as ca
import cim_model as cm
import main_backend as mb
import network_loader as nl

pytest.importorskip("lxml")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _case30():
    return nl.read_matpower(str(ROOT / "systems" / "pglib_opf_case30_as.m"))


def _cigre():
    return pn.create_cigre_network_mv(with_der="all")


def _special():
    """case14 with what needs care: parallel circuits, a derated line, a parallel transformer, a negative reactance,
    a phase shift of 5°, 60 Hz."""
    n = pn.case14()
    n.line.loc[3, "parallel"] = 2
    n.line.loc[5, "df"] = 0.8
    n.line.loc[7, "x_ohm_per_km"] = -abs(n.line.loc[7, "x_ohm_per_km"]) * 0.3
    n.line["c_nf_per_km"] = n.line["c_nf_per_km"].where(n.line["c_nf_per_km"] > 0, 10.0)
    n.trafo.loc[0, "parallel"] = 2
    n.trafo.loc[1, "shift_degree"] = 5.0
    n.f_hz = 60.0
    return n


@pytest.mark.parametrize("make", [_case30, _cigre, pn.mv_oberrhein, pn.case118, pn.example_multivoltage, _special],
                         ids=["case30", "cigre_mv", "mv_oberrhein", "case118", "example_multivoltage", "special"])
def test_the_round_trip_gives_the_same_power_flow(make):
    r = ca.round_trip(make(), "test")
    assert r["matches"], (r["max_diff"], r["worst"])
    assert r["n_missing"] == 0 and r["compared"] > 10


@pytest.mark.parametrize("make", [_case30, _cigre, pn.mv_oberrhein], ids=["case30", "cigre_mv", "mv_oberrhein"])
def test_the_profiles_alone_carry_the_bundled_networks(make):
    """What a tool other than CONDUCTOR reads (no extras): the same power flow for the bundled networks."""
    r = ca.round_trip(make(), "test", extras=False)
    assert r["matches"], (r["max_diff"], r["worst"])


def test_what_the_profiles_cannot_carry_is_listed():
    r = ca.round_trip(_special(), "test", extras=False)
    assert not r["matches"]                                   # pandapower's converter reads a negative r as |r|
    assert any("negative" in loss for loss in r["losses"])


def test_the_copy_is_read_from_the_cgmes_not_copied():
    net = _cigre()
    package = cw.package(copy.deepcopy(net), "cigre")
    copy_net, emap, _ = ca.PandapowerAdapter().project(package, "cigre")
    assert len(copy_net.bus) > len(net.bus)                  # line switches get a node of their own in CIM
    assert len(copy_net.storage) == len(net.storage)          # an energy source in CGMES, a storage unit again
    tripled = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(package)) as zin, zipfile.ZipFile(tripled, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if "_EQ_" in name:
                data = re.sub(rb"(<cim:ACLineSegment.r>)([0-9.eE+-]+)",
                              lambda m: m.group(1) + str(float(m.group(2)) * 3).encode(), data, count=1)
            zout.writestr(name, data)
    changed, emap2, _ = ca.PandapowerAdapter().project(tripled.getvalue(), "cigre")
    pp.runpp(copy_net)
    pp.runpp(changed)
    a, b = ca.PandapowerAdapter().results(copy_net, emap), ca.PandapowerAdapter().results(changed, emap2)
    assert max(abs(a[m]["vm_pu"] - b[m]["vm_pu"]) for m in a if "vm_pu" in a[m] and m in b) > 1e-3


def test_elements_keep_the_layer_s_mrids_and_costs_come_back():
    import semantics as sm
    net = _case30()
    package = cw.package(copy.deepcopy(net), "case30")
    model = cm.CimModel.read(io.BytesIO(package))
    layer = sm.build(net, network_id="case30")
    assert all(model.get(b["mrid"]) is not None for b in layer["buses"])
    assert all(model.get(br["mrid"]) is not None for br in layer["branches"])
    copy_net, emap, _ = ca.PandapowerAdapter().project(package, "case30")
    assert len(copy_net.poly_cost) == len(net.poly_cost)
    assert sorted(copy_net.poly_cost.cp1_eur_per_mw) == pytest.approx(sorted(net.poly_cost.cp1_eur_per_mw))
    assert len(emap) >= len(net.bus) + len(net.line)


# -- the backend: a network that did not come as CIM gets one written --------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    systems, data = tmp_path / "systems", tmp_path / "data_files"
    systems.mkdir()
    data.mkdir()
    monkeypatch.setenv("GRID_PROFILE", "pglib_case30")
    monkeypatch.setattr(mb, "SYSTEMS_DIR", str(systems))
    monkeypatch.setattr(mb, "DATA_FILES_DIR", str(data))
    for name, folder in (("_LAST_UPLOAD_SENTINEL", systems), ("_LAST_TIMESERIES_SENTINEL", data),
                         ("_LAST_FORECAST_SENTINEL", data), ("_PROPOSALS_FILE", systems),
                         ("_ASSUMPTIONS_FILE", systems), ("_SCENARIO_FILE", tmp_path)):
        monkeypatch.setattr(mb, name, str(folder / f"{name}.json"))
    saved = dict(mb.app_data)
    with TestClient(mb.app) as c:
        yield c
    mb.app_data.clear()
    mb.app_data.update(saved)


def test_a_network_not_loaded_as_cim_is_written_as_cgmes(client):
    kept = mb._cim_kept()
    assert kept and kept["derived"] and kept["model"].of("ACLineSegment")
    assert mb._cim_model() is None                            # the layer never takes it for a model's statement
    assert "cim_model" not in client.get("/api/network/semantics").json()
    view = client.get("/api/network/semantics", params={"about": "model"}).json()
    assert view["derived"] and view["round_trip"]["matches"] is True
    r = client.get("/api/network/cim/package")
    assert r.status_code == 200 and r.headers["x-conductor-model"] == "written"
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        assert {n.rsplit("_", 2)[-2] for n in names if n.endswith("_.xml")} >= {"EQ", "SSH", "TP"}
        assert "model/conductor_extras.json" in names and "conductor_additions.xml" in names
        eq = next(n for n in names if "_EQ_" in n)
        model_zip = io.BytesIO()
        with zipfile.ZipFile(model_zip, "w") as only:
            for n in names:
                if n.startswith("model/"):
                    only.writestr(n.split("/", 1)[1], z.read(n))
    assert eq
    # The package reads back as the network itself.
    copy_net, _emap, _ = ca.PandapowerAdapter().project(model_zip.getvalue(), str(mb.app_data["loaded_fingerprint"]))
    assert len(copy_net.bus) == len(mb.app_data["net_loaded"].bus)


# -- phase shifts, motors and static var compensators -------------------------------------------------------------------

def _shifted(ratio_side=None):
    n = pn.case14()
    t = n.trafo.index[0]
    n.trafo.loc[t, "shift_degree"] = 5.0
    if ratio_side:
        n.trafo.loc[t, ["tap_side", "tap_pos", "tap_neutral", "tap_min", "tap_max", "tap_step_percent",
                        "tap_changer_type"]] = [ratio_side, 2, 0, -5, 5, 1.25, "Ratio"]
    else:
        n.trafo.loc[t, ["tap_pos", "tap_step_percent"]] = [float("nan"), float("nan")]
    return n


def test_a_phase_shift_is_carried_by_a_phase_tap_changer():
    """Not a multiple of 30°: the clock carries the multiple, a linear phase tap changer the rest — exact from the
    profiles alone, as another tool reads them."""
    package = cw.package(_shifted(), "s")
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        eq = z.read(next(n for n in z.namelist() if "_EQ_" in n)).decode("utf-8")
    assert "PhaseTapChangerLinear" in eq
    r = ca.round_trip(_shifted(), "s", extras=False)
    assert r["matches"] and not r["losses"]


@pytest.mark.parametrize("side", ["hv", "lv"])
def test_beside_a_ratio_tap_the_shift_is_carried_too(side):
    assert ca.round_trip(_shifted(side), "s")["matches"]
    alone = ca.round_trip(_shifted(side), "s", extras=False)
    assert alone["matches"] and not any("beside a ratio tap" in loss for loss in alone["losses"])


def test_pegase_s_phase_shifters_reach_other_tools_exactly():
    assert ca.round_trip(pn.case1354pegase(), "p", extras=False)["matches"]


def _with(kind):
    n = pn.case14()
    if kind == "motor":
        pp.create_motor(n, bus=8, pn_mech_mw=5.0, cos_phi=0.85, efficiency_percent=92.0, loading_percent=80.0,
                        name="Pump")
    else:
        pp.create_svc(n, bus=9, x_l_ohm=1.0, x_cvar_ohm=-10.0, set_vm_pu=1.02, thyristor_firing_angle_degree=140.0,
                      name="SVC 9", controllable=True)
    return n


@pytest.mark.parametrize("kind, cim_class, words", [("motor", "AsynchronousMachine", "motor"),
                                                     ("svc", "StaticVarCompensator", "static var compensator")])
def test_motors_and_svcs_come_back(kind, cim_class, words):
    r = ca.round_trip(_with(kind), kind)
    assert r["matches"], (r["max_diff"], r["worst"])
    copy_net, _emap, _ = ca.PandapowerAdapter().project(cw.package(_with(kind), kind), kind)
    assert len(copy_net[kind]) == 1
    with zipfile.ZipFile(io.BytesIO(cw.package(_with(kind), kind))) as z:
        assert cim_class in z.read(next(x for x in z.namelist() if "_EQ_" in x)).decode("utf-8")
    assert any(words in loss for loss in ca.round_trip(_with(kind), kind, extras=False)["losses"])


# -- CGMES 3.0 (the default) and 2.4.15 ---------------------------------------------------------------------------------

def test_the_model_is_written_as_cgmes_3_by_default():
    with zipfile.ZipFile(io.BytesIO(cw.package(_cigre(), "c"))) as z:
        names = z.namelist()
        eq = z.read(next(n for n in names if "_EQ_" in n)).decode("utf-8")
        sc = z.read(next(n for n in names if "_SC_" in n)).decode("utf-8")
    assert "http://iec.ch/TC57/CIM100#" in eq and "CoreEquipment-EU/3.0" in eq
    assert "IdentifiedObject.mRID" in eq and "PowerElectronicsConnection" in eq and "BatteryUnit" in eq
    assert "PhotoVoltaicUnit" in eq and "PowerElectronicsWindUnit" in eq
    # CGMES 3.0 states a transformer's vector group in its ShortCircuit profile, not in Equipment.
    assert "phaseAngleClock" in sc and "phaseAngleClock" not in eq


@pytest.mark.parametrize("version", ["3.0", "2.4.15"])
def test_both_versions_read_back_the_same(version, monkeypatch):
    monkeypatch.setattr(cw, "DEFAULT_VERSION", version)
    original_package = cw.package
    monkeypatch.setattr(cw, "package", lambda net, nid="local", name=None, created="2000-01-01T00:00:00Z",
                        version=version: original_package(net, nid, name, created, version))
    for extras in (True, False):
        r = ca.round_trip(_cigre(), "c", extras=extras)
        assert r["matches"], (version, extras, r["max_diff"], r["worst"])


def test_a_3_0_model_s_inverters_and_vector_groups_survive_pandapower_s_reader():
    """pandapower 3.5's CGMES 3.0 reader drops a line's in-service state and every inverter (PV, wind, battery):
    CONDUCTOR's importer puts both back, so the profiles alone read back the same."""
    net = pn.mv_oberrhein()
    copy_net, _emap, _ = ca.PandapowerAdapter().project(cw.package(copy.deepcopy(net), "o"), "o", extras=False)
    assert len(copy_net.sgen) == len(net.sgen)
    assert ca.round_trip(net, "o", extras=False)["matches"]


def test_a_copy_that_does_not_solve_is_reported_not_raised():
    r = ca.round_trip(_with("motor"), "m", extras=False)
    assert r["matches"] is False and any("motor" in loss for loss in r["losses"])



def test_inverters_are_never_converted_twice(monkeypatch):
    """Should a later pandapower convert power-electronics connections for CGMES 3.0 itself, CONDUCTOR's own step
    would convert them again: the importer keeps each once."""
    from pandapower.converter.cim.cim2pp.converter_classes.generators.powerElectronicsConnection import (
        PowerElectronicsConnection)
    base_class = nl._energy_sources_and_inverters()

    class Twice(base_class):                    # as if pandapower did it too
        def convert_energy_sources_cim16(self):
            super().convert_energy_sources_cim16()
            PowerElectronicsConnection(cimConverter=self.cimConverter).convert_power_electronics_connection()
    monkeypatch.setattr(nl, "_energy_sources_and_inverters", lambda: Twice)
    net = pn.mv_oberrhein()
    copy_net, _emap, notes = ca.PandapowerAdapter().project(cw.package(copy.deepcopy(net), "o"), "o")
    assert len(copy_net.sgen) == len(net.sgen)
    assert any("converted twice" in n for n in notes)


# -- DC lines, and a phase shift beside a ratio tap ---------------------------------------------------------------------

@pytest.mark.parametrize("version", ["3.0", "2.4.15"])
def test_a_dc_line_reads_back_the_same(version, monkeypatch):
    """Two voltage-source converters and a DC line segment between their DC nodes: pandapower's converter reads it
    back as the same DC line (the profiles alone too)."""
    import pandapower as pp

    import cgmes_writer as cw
    monkeypatch.setattr(cw, "DEFAULT_VERSION", version)
    net = pn.case14()
    free = [b for b in net.bus.index if b not in set(net.gen.bus) | set(net.ext_grid.bus)]
    pp.create_dcline(net, free[0], free[3], p_mw=20, loss_percent=1.2, loss_mw=0.5, vm_from_pu=1.02,
                     vm_to_pu=1.01, name="HVDC")
    for extras in (True, False):
        r = ca.round_trip(net, "dc", extras=extras)
        assert r["matches"] and r["n_missing"] == 0, (extras, r)


@pytest.mark.parametrize("side", ["hv", "lv"])
@pytest.mark.parametrize("shift", [5.0, 157.5, -12.0])
def test_a_phase_shift_beside_a_ratio_tap_reads_back_the_same(side, shift):
    """The rest of the shift as a linear phase tap changer on the other end: exact, the profiles alone too (the
    importer undoes pandapower's converter copying each tap's step to the other)."""
    net = pn.case14()
    net.trafo.at[0, "tap_pos"] = 2
    net.trafo.at[0, "tap_step_percent"] = 1.5
    net.trafo.at[0, "tap_neutral"] = 0
    net.trafo.at[0, "tap_min"], net.trafo.at[0, "tap_max"] = -5, 5
    net.trafo.at[0, "tap_side"] = side
    net.trafo.at[0, "shift_degree"] = shift
    for extras in (True, False):
        r = ca.round_trip(net, "pt", extras=extras)
        assert r["matches"], (side, shift, extras, r["max_diff"], r["notes"][-1:])


def test_a_cim_model_s_units_keep_their_names_and_mrids_when_converted():
    """network_loader.gen_to_sgen: a machine the model names keeps its name (the second of two alike numbered) and
    its identity (origin_id, class, terminal); an unnamed unit is named by its bus, as before."""
    import network_loader as nl
    fixture = pathlib.Path(__file__).parent / "fixtures" / "cgmes" / "pandapower_example_multivoltage.zip"
    net = nl.read_cgmes(str(fixture))
    named = net.gen[net.gen["name"].astype(str).str.strip() != ""]
    out, created, _kept = nl.gen_to_sgen(copy.deepcopy(net))
    assert set(named["name"]) & set(out.sgen["name"])
    assert out.sgen["origin_id"].notna().all()
    plain = pn.case14()
    _out, created_plain, _k = nl.gen_to_sgen(plain)
    assert all(n.startswith("Gen_bus") for n in created_plain)
