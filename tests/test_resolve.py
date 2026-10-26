"""
One lookup for every name (backend/resolve.py).

Eight lookups had grown, one per endpoint, each accepting different spellings:
"Bus_7" found a unit in one study and nothing in another, "line Bus_8 -> Bus_9"
found nothing because the label carries "[L12]", "external grid" was "not a
bus, line or transformer". These pin down what the one lookup accepts, and
that it never guesses.
"""

import pandapower as pp
import pandapower.networks as pn
import pytest

import resolve as rv


@pytest.fixture(scope="module")
def ieee14():
    net = pn.case14()
    net.bus["name"] = None                         # as the PGLib profile loads it: Bus_<i>
    return net


@pytest.fixture(scope="module")
def named():
    """Real names: two buses share one, two lines start alike, a bus name
    is the start of another's."""
    net = pp.create_empty_network()
    for i, (name, kv) in enumerate((("Golf", 63), ("Hotel", 63), ("Bravo", 63), ("Golf 10 kV", 10),
                                     ("Aux", 63), ("Aux", 63))):
        pp.create_bus(net, vn_kv=kv, name=name, index=i)
    for idx, name, a, b in ((7, "OLS-ØST", 0, 1), (6, "OLS-ALL", 0, 2), (11, "BOD-NEX", 1, 2)):
        pp.create_line(net, a, b, 10, "NA2XS2Y 1x240 RM/25 12/20 kV", name=name, index=idx)
    pp.create_transformer(net, 0, 3, "25 MVA 110/20 kV", name="Golf Trf", index=3)
    pp.create_ext_grid(net, 0)
    pp.create_gen(net, 2, p_mw=10, name="Bravo G1")
    return net


@pytest.mark.parametrize("asked, etype, idx", [
    ("Bus_7", "bus", 7), ("bus 7", "bus", 7), ("bus7", "bus", 7), ("BUS_7", "bus", 7), (7, "bus", 7),
    ("7", "bus", 7), ("the bus Bus_3", "bus", 3),
    ("Bus_0 -> Bus_1 [L0]", "line", 0), ("Bus_0 -> Bus_1", "line", 0), ("Bus_1 -> Bus_0", "line", 0),
    ("line Bus_8 -> Bus_9", "line", 10), ("Bus_9 - Bus_8", "line", 10), ("between Bus_8 and Bus_9", "line", 10),
    ("Bus_8 to Bus_9", "line", 10), ("Bus_8 → Bus_9", "line", 10),
    ("L12", "line", 12), ("line 12", "line", 12), ("T1", "trafo", 1), ("transformer 1", "trafo", 1),
    ("Bus_4На", "bus", 4),                          # a slip of the model's output (observed live)
])
def test_every_way_of_naming_an_element(ieee14, asked, etype, idx):
    found = rv.find(ieee14, asked)
    assert (found.etype, found.idx) == (etype, idx)


def test_real_names_case_and_prefixes(named):
    assert rv.find(named, "golf").idx == 0
    assert rv.find(named, "Golf 10 kV").idx == 3            # not swept into "Golf"
    assert rv.find(named, "OLS-ØST").idx == 7 and rv.find(named, "OLS-ØST").etype == "line"
    assert rv.find(named, "Golf -> Hotel").idx == 7
    assert rv.find(named, "Golf Trf").etype == "trafo"


def test_it_never_guesses(named):
    with pytest.raises(rv.ResolveError) as two:
        rv.find(named, "Aux")
    assert two.value.reason == "ambiguous" and "index 4" in str(two.value) and "index 5" in str(two.value)
    with pytest.raises(rv.ResolveError) as unknown:
        rv.find(named, "Hotell")
    assert unknown.value.reason == "not_found" and "Hotel" in str(unknown.value)
    with pytest.raises(rv.ResolveError) as nothing_close:
        rv.find(named, "Nowhere")
    assert "Golf" in str(nothing_close.value)                  # real names, spelled as they are


def test_the_external_grid_is_said_to_be_a_connection(named):
    with pytest.raises(rv.ResolveError) as said:
        rv.find(named, "external grid")
    assert said.value.reason == "not_an_element" and "at Golf" in str(said.value)
    assert "OLS-ØST" in str(said.value)                        # the branches that reach it


def test_a_unit_by_its_name_or_its_bus(ieee14, named):
    assert rv.find_unit(ieee14, "Gen_bus1").idx == 0
    assert rv.find_unit(ieee14, "Bus_1").label == "Gen_bus1"
    assert rv.find_unit(ieee14, "bus 2").label == "Gen_bus2"
    for said in ("Gen 2", "generator 2", "the unit at Bus_2", "generator at bus 2"):
        assert rv.find_unit(ieee14, said).label == "Gen_bus2", said
    assert rv.find_unit(named, "bravo g1").label == "Bravo G1"
    with pytest.raises(rv.ResolveError) as unknown:
        rv.find_unit(ieee14, "Nope")
    assert "Gen_bus1" in str(unknown.value)


def test_the_studies_use_it(ieee14):
    import main_backend as mb
    import sandbox as sb
    assert sb.resolve_bus(ieee14, "bus7") == 7
    assert sb.resolve_element(ieee14, "line", "Bus_9 - Bus_8") == 10
    assert mb._investigated_element(ieee14, "line Bus_8 -> Bus_9", None)[:2] == ("line", 10)
    assert mb._unit_at(ieee14, "Bus_1") == ("gen", 0)
    with pytest.raises(mb.HTTPException) as ext:
        mb._investigated_element(ieee14, "external grid", None)
    assert ext.value.status_code == 400 and "connection at Bus_0" in ext.value.detail
