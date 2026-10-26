import pytest

import flex_engine as fe
import main_backend as mb


def _fake_db_full(net, drop_pair=None, extra_bus=None):
    """A minimal db_full with every entry the OPF reads for *net*: each
    in-service branch in both directions, at tap 1 (line) or the neutral tap
    (transformer), in all four tables.

    Only the keys matter for topology validation; values are dummies.
    """
    keys, to_trafo = [], {}
    for _, row in net.line[net.line["in_service"]].iterrows():
        pair = (int(row["from_bus"]), int(row["to_bus"]))
        keys += [(pair, 1), ((pair[1], pair[0]), 1)]
    for idx, row in net.trafo[net.trafo["in_service"]].iterrows():
        pair = (int(row["hv_bus"]), int(row["lv_bus"]))
        tap = fe._neutral_tap(net, idx)
        keys += [(pair, tap), ((pair[1], pair[0]), tap)]
        to_trafo[pair] = to_trafo[(pair[1], pair[0])] = idx

    if drop_pair is not None:
        rev = (drop_pair[1], drop_pair[0])
        keys = [k for k in keys if k[0] not in (drop_pair, rev)]
    if extra_bus is not None:
        keys.append(((0, extra_bus), 1))

    db = {table: {k: 1.0 for k in keys} for table in ("Yff_r", "Yff_i", "Yft_r", "Yft_i")}
    db["branch_to_trafo"] = to_trafo
    return db


class TestValidateUploadedAdmittance:
    def test_matching_topology_passes(self, case14_net):
        db_full = _fake_db_full(case14_net)
        mb._validate_uploaded_admittance(db_full, case14_net)  # no exception

    def test_unknown_bus_rejected(self, case14_net):
        db_full = _fake_db_full(case14_net, extra_bus=99)
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._validate_uploaded_admittance(db_full, case14_net)
        assert exc_info.value.status_code == 400
        assert "99" in exc_info.value.detail

    def test_missing_branch_coverage_rejected(self, case14_net):
        net = case14_net
        first_line = net.line[net.line["in_service"]].iloc[0]
        pair = (int(first_line["from_bus"]), int(first_line["to_bus"]))
        db_full = _fake_db_full(net, drop_pair=pair)
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._validate_uploaded_admittance(db_full, net)
        assert exc_info.value.status_code == 400
        assert f"{pair[0]}-{pair[1]}" in exc_info.value.detail

    def test_out_of_service_branches_not_required(self, case14_net):
        net = case14_net
        line_idx = net.line.index[0]
        pair = (int(net.line.at[line_idx, "from_bus"]), int(net.line.at[line_idx, "to_bus"]))
        db_full = _fake_db_full(net, drop_pair=pair)
        # Same missing branch, but now it's out of service — must pass.
        net.line.at[line_idx, "in_service"] = False
        mb._validate_uploaded_admittance(db_full, net)

    def test_one_direction_only_is_rejected(self, case14_net):
        """The optimisation reads both directions of every branch: a file with
        one used to pass and fail later with a KeyError (HTTP 500)."""
        net = case14_net
        db_full = _fake_db_full(net)
        first_line = net.line[net.line["in_service"]].iloc[0]
        pair = (int(first_line["from_bus"]), int(first_line["to_bus"]))
        for table in ("Yff_r", "Yff_i", "Yft_r", "Yft_i"):
            db_full[table].pop((pair, 1), None)
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._validate_uploaded_admittance(db_full, net)
        assert exc_info.value.status_code == 400
        assert f"{pair[0]}→{pair[1]} (tap 1)" in exc_info.value.detail

    def test_one_table_short_is_rejected(self, case14_net):
        db_full = _fake_db_full(case14_net)
        db_full["Yft_i"].pop(next(iter(db_full["Yft_i"])))
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._validate_uploaded_admittance(db_full, case14_net)
        assert "the optimisation reads" in exc_info.value.detail

    def test_a_generated_database_passes(self, case14_net):
        mb._validate_uploaded_admittance(mb.nl.build_ybus_full(case14_net), case14_net)
