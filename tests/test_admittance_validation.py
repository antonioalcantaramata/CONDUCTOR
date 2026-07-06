import pytest

import main_backend as mb


def _fake_db_full(net, drop_pair=None, extra_bus=None):
    """Build a minimal db_full whose Yff_r covers every in-service branch of *net*.

    Only the keys matter for topology validation; values are dummies.
    """
    yff_r = {}
    for _, row in net.line[net.line["in_service"]].iterrows():
        pair = (int(row["from_bus"]), int(row["to_bus"]))
        yff_r[pair, 1] = 1.0
        yff_r[(pair[1], pair[0]), 1] = 1.0
    for _, row in net.trafo[net.trafo["in_service"]].iterrows():
        pair = (int(row["hv_bus"]), int(row["lv_bus"]))
        yff_r[pair, 0] = 1.0
        yff_r[(pair[1], pair[0]), 0] = 1.0

    if drop_pair is not None:
        yff_r.pop((drop_pair, 0), None)
        yff_r.pop((drop_pair, 1), None)
        rev = (drop_pair[1], drop_pair[0])
        yff_r.pop((rev, 0), None)
        yff_r.pop((rev, 1), None)
    if extra_bus is not None:
        yff_r[(0, extra_bus), 1] = 1.0

    return {"Yff_r": yff_r}


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

    def test_reverse_direction_coverage_accepted(self, case14_net):
        net = case14_net
        db_full = _fake_db_full(net)
        # Keep only reversed keys for one branch — still counts as covered.
        first_line = net.line[net.line["in_service"]].iloc[0]
        pair = (int(first_line["from_bus"]), int(first_line["to_bus"]))
        db_full["Yff_r"].pop((pair, 1), None)
        mb._validate_uploaded_admittance(db_full, net)
