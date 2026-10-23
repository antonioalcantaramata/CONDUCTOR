"""The System tab's cost and name panels: rows and form diffs, without Streamlit."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "llm_agent"))

from agent import system_tables as t  # noqa: E402

UNIT = {"unit": "Gen_bus1", "kind": "unit", "bus": 1, "bus_name": "Bus_1", "p_max_mw": 59.0,
        "cost_eur_per_h": 0.0, "cost_eur_per_mwh": 23.27, "cost_eur_per_mw2h": 0.0,
        "up_eur_per_mwh": 27.92, "down_eur_per_mwh": 11.63, "q_eur_per_mvarh": 3.0,
        "source": {"cost_eur_per_h": "case", "cost_eur_per_mwh": "case", "cost_eur_per_mw2h": "case",
                   "up_eur_per_mwh": "operator", "down_eur_per_mwh": "synthetic", "q_eur_per_mvarh": "synthetic"},
        "changed": True}
GRID = {"unit": "External grid", "kind": "external_grid", "bus": 0, "bus_name": "Bus_0", "p_max_mw": None,
        "cost_eur_per_h": 0.0, "cost_eur_per_mwh": 7.92, "cost_eur_per_mw2h": 0.0,
        "up_eur_per_mwh": None, "down_eur_per_mwh": None, "q_eur_per_mvarh": None,
        "q_band_tan_phi": 0.4, "q_penalty_eur_per_mvarh": 10.0,
        "source": {"cost_eur_per_mwh": "case", "q_band_tan_phi": "synthetic", "q_penalty_eur_per_mvarh": "synthetic"},
        "changed": False}


def test_rows_split_units_and_the_grid():
    rows = t.cost_rows([GRID, UNIT])
    assert [r["unit"] for r in rows["units"]] == ["Gen_bus1"] and rows["units"][0]["changed"]
    assert len(rows["units"][0]["cells"]) == len(t.GROUPS["units"]["columns"])
    assert len(rows["grid"][0]["cells"]) == len(t.GROUPS["grid"]["columns"])
    assert rows["grid"][0]["cells"][3] == "0.40"


def test_missing_figures_show_a_dash():
    row = t.cost_rows([{**UNIT, "up_eur_per_mwh": None}])["units"][0]
    assert "—" in row["cells"]


def test_form_changes_only_what_differs():
    assert t.cost_changes(UNIT, {"up_eur_per_mwh": 27.92, "q_eur_per_mvarh": 3.0}) is None
    assert t.cost_changes(UNIT, {"up_eur_per_mwh": 90, "q_eur_per_mvarh": 3.0}) == {"unit": "Gen_bus1",
                                                                                   "up_eur_per_mwh": 90.0}
    assert t.cost_changes(UNIT, {"down_eur_per_mwh": None}) == {"unit": "Gen_bus1", "down_eur_per_mwh": None}


def test_grid_form_shows_the_band_not_unit_prices():
    fields = {k for section in t.fields_of(GRID).values() for k in section}
    assert fields == {"cost_eur_per_mwh", "q_band_tan_phi", "q_penalty_eur_per_mvarh"}
    assert "q_band_tan_phi" not in {k for section in t.fields_of(UNIT).values() for k in section}


def test_back_to_defaults_nulls_the_operator_figures():
    assert t.defaults_change(UNIT) == {"unit": "Gen_bus1", "up_eur_per_mwh": None}
    assert t.defaults_change(GRID) is None


def test_name_rows():
    rows = t.name_rows("line", [{"index": 0, "name": "Tie | Bus_0 -> Bus_1 [L0]", "label": "Tie",
                                 "ends": "Bus_0 – Bus_1", "renamed": True},
                                {"index": 1, "name": "Bus_0 -> Bus_4 [L1]", "label": "", "ends": "Bus_0 – Bus_4"}])
    assert rows[0]["cells"] == ["Tie", "Bus_0 – Bus_1", "0"] and rows[0]["renamed"]
    assert rows[1]["cells"][0] == "(no name)" and rows[1]["name"] == ""
    assert len(rows[0]["cells"]) == len(t.NAME_COLUMNS["line"])
    bus = t.name_rows("bus", [{"index": 3, "name": "North", "label": "North", "renamed": True}])
    assert bus[0]["cells"] == ["North", "3"]


def test_the_data_tab_label_shows_both_names():
    from agent import data_view
    assert data_view.label("Bus_3", {"Bus_3": "North"}) == "North (data: Bus_3)"
    assert data_view.label("Bus_4", {"Bus_3": "North"}) == "Bus_4"
    assert data_view.label("Bus_4", None) == "Bus_4"


def test_the_kept_line():
    assert t.kept_line({"renames": 1, "costs": 2, "proposals": 1}) == \
        "Your changes are kept: 1 rename · 2 cost changes · 1 proposal"
    assert t.kept_line({"load_power_factor": 1}) == "Your changes are kept: load power factor"
    assert t.kept_line({}) == "" and t.kept_line(None) == ""


def test_loads_get_their_own_demand_response_table_and_form():
    load = {"unit": "Load 20", "kind": "load", "bus_name": "Bus_29", "dr_share_pct": 10.0, "dr_eur_per_mwh": 150.0,
            "relief_eur_per_mwh": 3000.0, "source": {"dr_share_pct": "synthetic", "dr_eur_per_mwh": "operator",
                                                     "relief_eur_per_mwh": "synthetic"}, "changed": True}
    rows = t.cost_rows([load])
    assert list(rows) == ["loads"] and rows["loads"][0]["cells"] == ["Load 20", "Bus_29", "10", "150.00", "3000",
                                                                     "changed + synth."]
    assert len(t.GROUPS["loads"]["columns"]) == len(rows["loads"][0]["cells"])
    form = t.fields_of(load)
    assert list(form) == ["Demand response", "Load relief"] and "relief_eur_per_mwh" in form["Load relief"]
    assert t.cost_changes(load, {"dr_eur_per_mwh": 120.0}) == {"unit": "Load 20", "dr_eur_per_mwh": 120.0}
