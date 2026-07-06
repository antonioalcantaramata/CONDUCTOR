import pytest

import rsa_engine as rs


class TestMapBusNamesToLines:
    def test_raises_when_bus_has_no_name_column(self, case14_net):
        net = case14_net
        net.bus = net.bus.drop(columns=["name"])
        with pytest.raises(ValueError, match="name"):
            rs.map_bus_names_to_lines(net)

    def test_adds_from_and_to_bus_name_columns(self, case14_net):
        net = case14_net
        net.bus["name"] = [f"Bus_{i}" for i in net.bus.index]

        net = rs.map_bus_names_to_lines(net)

        assert "from_bus_name" in net.line.columns
        assert "to_bus_name" in net.line.columns
        for idx, row in net.line.iterrows():
            assert row["from_bus_name"] == f"Bus_{row['from_bus']}"
            assert row["to_bus_name"] == f"Bus_{row['to_bus']}"
