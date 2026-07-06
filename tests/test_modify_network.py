import pytest

import modify_network as mn


class TestUpdateBusMapping:
    def test_decrements_references_above_removed_bus(self, case14_net):
        net = case14_net
        line_before = net.line[["from_bus", "to_bus"]].copy()

        net = mn.update_bus_mapping(net, bus_num=3)

        for col in ("from_bus", "to_bus"):
            expected = line_before[col].apply(lambda x: x - 1 if x > 3 else x)
            assert (net.line[col] == expected).all()

    def test_leaves_references_at_or_below_removed_bus_untouched(self, case14_net):
        net = case14_net
        net.line.loc[0, "from_bus"] = 2
        net.line.loc[0, "to_bus"] = 3

        net = mn.update_bus_mapping(net, bus_num=3)

        assert net.line.loc[0, "from_bus"] == 2  # <= bus_num: untouched
        assert net.line.loc[0, "to_bus"] == 3  # == bus_num: untouched


class TestRemoveTrafoAndConnectedElements:
    def test_unknown_trafo_index_raises(self, case14_net):
        with pytest.raises(ValueError, match="Transformer index"):
            mn.remove_trafo_and_connected_elements(case14_net, trafo_index=9999)

    def test_removes_trafo_and_its_lv_bus(self, case14_net):
        net = case14_net
        trafo_idx = net.trafo.index[0]
        lv_bus_before = net.trafo.loc[trafo_idx, "lv_bus"]
        n_trafos_before = len(net.trafo)
        n_buses_before = len(net.bus)

        net, hv_bus, lv_bus = mn.remove_trafo_and_connected_elements(net, trafo_idx)

        assert lv_bus == lv_bus_before
        assert len(net.trafo) == n_trafos_before - 1
        assert len(net.bus) == n_buses_before - 1
        # Indices are reset after the drop, so no row should keep the old (hv, lv) pair.
        assert not ((net.trafo["hv_bus"] == hv_bus) & (net.trafo["lv_bus"] == lv_bus)).any()


class TestRemoveInactiveBuses:
    def test_removes_out_of_service_buses(self, case14_net):
        net = case14_net
        net.bus.loc[13, "in_service"] = False
        n_buses_before = len(net.bus)

        net = mn.remove_inactive_buses(net)

        assert len(net.bus) == n_buses_before - 1
        assert (net.bus["in_service"] == True).all()  # noqa: E712

    def test_noop_when_all_buses_active(self, case14_net):
        net = case14_net
        n_buses_before = len(net.bus)

        net = mn.remove_inactive_buses(net)

        assert len(net.bus) == n_buses_before


class TestRemoveIsolatedLineBusesWithTrafo:
    def test_unknown_line_index_raises(self, case14_net):
        with pytest.raises(ValueError, match="Line index"):
            mn.remove_isolated_line_buses_with_trafo(case14_net, line_index=9999)

    def test_removes_line_without_isolating_buses(self, case14_net):
        # Bus14 has more than one line in case14, so removing just one line
        # should not isolate either endpoint.
        net = case14_net
        line_idx = net.line.index[0]
        from_bus_before = net.line.loc[line_idx, "from_bus"]
        to_bus_before = net.line.loc[line_idx, "to_bus"]
        n_lines_before = len(net.line)

        net, from_bus, to_bus, lv_bus_values = mn.remove_isolated_line_buses_with_trafo(net, line_idx)

        assert from_bus == from_bus_before
        assert to_bus == to_bus_before
        assert len(net.line) == n_lines_before - 1
