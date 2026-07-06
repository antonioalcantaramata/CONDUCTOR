import flex_engine as fe


class TestExtractPgMax:
    def test_uses_max_p_mw_and_falls_back_to_gen_bus_name(self, case14_net):
        result = fe._extract_pg_max(case14_net)

        assert result == {
            "Gen_bus1": 140.0,
            "Gen_bus2": 100.0,
            "Gen_bus5": 100.0,
            "Gen_bus7": 100.0,
        }

    def test_zero_max_p_mw_falls_back_to_1_5x_p_mw(self, case14_net):
        net = case14_net
        net.gen.loc[0, "max_p_mw"] = 0.0
        net.gen.loc[0, "p_mw"] = 40.0

        result = fe._extract_pg_max(net)

        assert result["Gen_bus1"] == 60.0  # 1.5 * 40.0

    def test_sgen_uses_substation_name_column_when_present(self, case14_net):
        net = case14_net
        import pandapower as pp
        pp.create_sgen(net, bus=3, p_mw=10.0, max_p_mw=25.0, name="ignored")
        net.sgen["substation_name"] = "Custom_Sgen"

        result = fe._extract_pg_max(net)

        assert result["Custom_Sgen"] == 25.0

    def test_empty_gen_and_sgen_returns_empty_dict(self, case14_net):
        net = case14_net
        net.gen = net.gen.iloc[0:0]
        net.sgen = net.sgen.iloc[0:0]

        assert fe._extract_pg_max(net) == {}


class TestGetAdmittanceParameters:
    def test_unpacks_all_fields_from_database(self):
        database_admittance = {
            0: {
                "Yff_r": {(0, 1): 1.0},
                "Yff_i": {(0, 1): -1.0},
                "Yft_r": {(0, 1): 0.5},
                "Yft_i": {(0, 1): -0.5},
                "TAPS": [1],
                "trafo_ranges": {},
                "trafo_defaults": {},
                "branch_to_trafo": {},
            }
        }

        result = fe.get_admittance_parameters(database_admittance, 0)

        assert result == (
            {(0, 1): 1.0},
            {(0, 1): -1.0},
            {(0, 1): 0.5},
            {(0, 1): -0.5},
            [1],
            {},
            {},
            {},
        )
