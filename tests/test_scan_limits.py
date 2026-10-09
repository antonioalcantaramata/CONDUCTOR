"""Envelope and hosting scans judge lines and transformers against their own
limits (code review 2.12).

Both used to be compared with the larger of the two limits: with lines
allowed 50 % and transformers 100 %, a line at 60 % passed.
"""

import pandapower as pp
import pandapower.networks as pn
import pytest

import main_backend as mb


@pytest.fixture
def point_net():
    net = pn.case14()
    pp.create_sgen(net, bus=3, p_mw=0.0)
    pp.runpp(net)
    return net


def _max_line(net):
    pp.runpp(net)
    return float(net.res_line.loading_percent.max()), float(net.res_trafo.loading_percent.max())


def test_a_line_over_its_own_limit_fails(point_net):
    line, trafo = _max_line(point_net)
    line_limit = line * 0.8                      # the lines are over theirs
    trafo_limit = max(trafo, line) * 1.5         # transformers allowed far more
    r = mb._run_envelope_point(0.0, 0.0, point_net, int(point_net.sgen.index[-1]),
                               vm_upper=2.0, vm_lower=0.0, max_line=line_limit, max_trafo=trafo_limit)
    assert r["feasible"] is False and r["binding_constraint"] == "thermal"
    assert r["thermal"]["branch_type"] == "line"
    assert r["thermal"]["limit"] == pytest.approx(line_limit)
    binding = mb._build_hosting_binding(r, 2.0, 0.0, max(line_limit, trafo_limit))
    assert binding["branch_type"] == "line" and binding["limit"] == pytest.approx(line_limit)


def test_within_both_limits_passes(point_net):
    line, trafo = _max_line(point_net)
    r = mb._run_envelope_point(0.0, 0.0, point_net, int(point_net.sgen.index[-1]),
                               vm_upper=2.0, vm_lower=0.0, max_line=line * 1.1, max_trafo=trafo * 1.1)
    assert r["feasible"] is True and r["thermal"] is None


def test_equal_limits_report_as_before(point_net):
    line, trafo = _max_line(point_net)
    limit = min(line, trafo) * 0.9
    r = mb._run_envelope_point(0.0, 0.0, point_net, int(point_net.sgen.index[-1]),
                               vm_upper=2.0, vm_lower=0.0, max_line=limit, max_trafo=limit)
    binding = mb._build_hosting_binding(r, 2.0, 0.0, limit)
    assert binding["value"] == pytest.approx(r["max_loading_pct"])
    assert binding["limit"] == pytest.approx(limit)
