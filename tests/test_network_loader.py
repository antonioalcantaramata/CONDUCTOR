import pytest

import network_loader as nl


class TestLoadProfile:
    def test_missing_profile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            nl.load_profile("does_not_exist", backend_dir=str(tmp_path))

    def test_reads_yaml(self, tmp_path):
        profiles_dir = tmp_path / "network_profiles"
        profiles_dir.mkdir()
        (profiles_dir / "my_profile.yaml").write_text("source: ieee\ncase: case14\n")

        profile = nl.load_profile("my_profile", backend_dir=str(tmp_path))

        assert profile == {"source": "ieee", "case": "case14"}


class TestLoadNetwork:
    def test_unsupported_source_raises(self):
        with pytest.raises(ValueError, match="Unsupported network source"):
            nl.load_network({"source": "carrier_pigeon"})

    def test_ieee_case_dispatches_to_pandapower_networks(self):
        net = nl.load_network({"source": "ieee", "case": "case14"})
        assert len(net.bus) == 14

    def test_unknown_ieee_case_raises(self):
        with pytest.raises(ValueError, match="Unknown pp.networks case"):
            nl.load_network({"source": "ieee", "case": "not_a_real_case"})


class TestGenToSgen:
    def test_converts_all_pv_generators_in_case14(self, case14_net):
        net = case14_net
        n_gen_before = len(net.gen)  # 4 PV generators, none on the slack bus

        net, created_names, kept_condenser_names = nl.gen_to_sgen(net)

        assert len(net.gen) == 0  # all PV gens converted, no condensers in this case
        assert len(net.sgen) == n_gen_before
        assert created_names == ["Gen_bus1", "Gen_bus2", "Gen_bus5", "Gen_bus7"]
        assert kept_condenser_names == []
        assert net["_gen_to_sgen_applied"] is True

    def test_slack_bus_generator_never_duplicated_into_sgen(self, case14_net):
        net = case14_net
        slack_buses = set(net.ext_grid["bus"].tolist())

        net, _, _ = nl.gen_to_sgen(net)

        assert not slack_buses & set(net.sgen["bus"].tolist())
