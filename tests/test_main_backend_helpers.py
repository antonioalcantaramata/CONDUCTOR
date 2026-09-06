
import pytest

import main_backend as mb


class TestSanitizeUploadName:
    def test_replaces_unsafe_characters(self):
        assert mb._sanitize_upload_name("some file!.XLSX", "network", ".xlsx") == "some_file.xlsx"

    def test_falls_back_to_stem_when_name_empty(self):
        assert mb._sanitize_upload_name(None, "network", ".xlsx") == "network.xlsx"

    def test_forces_expected_suffix(self):
        assert mb._sanitize_upload_name("data.csv", "network", ".xlsx") == "data.xlsx"

    def test_strips_directory_components(self):
        assert mb._sanitize_upload_name("../../etc/passwd", "network", ".xlsx") == "passwd.xlsx"


class TestSafeFloat:
    def test_rounds_to_requested_precision(self):
        assert mb._safe_float(1.23456, ndigits=2) == 1.23

    def test_nan_returns_none(self):
        assert mb._safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert mb._safe_float(float("inf")) is None

    def test_non_numeric_returns_none(self):
        assert mb._safe_float("not a number") is None


class TestCleanLabel:
    @pytest.mark.parametrize("raw", [None, "", "  ", "nan", "NaN", "none", "None"])
    def test_treats_missing_markers_as_empty(self, raw):
        assert mb._clean_label(raw) == ""

    def test_strips_whitespace_from_real_label(self):
        assert mb._clean_label("  Bus 1  ") == "Bus 1"


class TestBusDisplayName:
    def test_uses_clean_name_when_present(self, case14_net):
        net = case14_net
        net.bus.at[0, "name"] = "Foxtrot"
        assert mb._bus_display_name(net, 0) == "Foxtrot"

    def test_falls_back_to_bus_index_when_blank(self, case14_net):
        net = case14_net
        net.bus.at[0, "name"] = "nan"
        assert mb._bus_display_name(net, 0) == "Bus_0"


class TestLineDisplayName:
    def test_includes_endpoints_and_index(self, case14_net):
        net = case14_net
        line_idx = net.line.index[0]
        from_bus = int(net.line.at[line_idx, "from_bus"])
        to_bus = int(net.line.at[line_idx, "to_bus"])
        net.bus["name"] = [f"Bus_{i}" for i in net.bus.index]
        net.line.at[line_idx, "name"] = "nan"  # treated as missing

        label = mb._line_display_name(net, line_idx)

        assert label == f"Bus_{from_bus} -> Bus_{to_bus} [L{line_idx}]"

    def test_prefixes_raw_name_when_present(self, case14_net):
        net = case14_net
        line_idx = net.line.index[0]
        net.bus["name"] = [f"Bus_{i}" for i in net.bus.index]
        net.line.at[line_idx, "name"] = "Feeder 1"

        label = mb._line_display_name(net, line_idx)

        assert label.startswith("Feeder 1 | ")


class TestTrafoDisplayName:
    def test_includes_endpoints_and_index(self, case14_net):
        net = case14_net
        trafo_idx = net.trafo.index[0]
        hv = int(net.trafo.at[trafo_idx, "hv_bus"])
        lv = int(net.trafo.at[trafo_idx, "lv_bus"])
        net.bus["name"] = [f"Bus_{i}" for i in net.bus.index]
        net.trafo.at[trafo_idx, "name"] = ""  # PGLib cases name no transformer

        label = mb._trafo_display_name(net, trafo_idx)

        assert label == f"Bus_{hv} -> Bus_{lv} [T{trafo_idx}]"

    def test_prefixes_raw_name_when_present(self, case14_net):
        net = case14_net
        trafo_idx = net.trafo.index[0]
        net.bus["name"] = [f"Bus_{i}" for i in net.bus.index]
        net.trafo.at[trafo_idx, "name"] = "T1 63/10.5"

        assert mb._trafo_display_name(net, trafo_idx).startswith("T1 63/10.5 | ")


class TestElementNamesAgreeAcrossEndpoints:
    """The system map joins topology to results by name, so producers must agree.

    They did not: `/api/network/topology` said `Trafo_3-6_0` where the RSA
    snapshot said `Trafo_3-6`, and every transformer in a network that names
    none of them — every PGLib case — drew as "not measured".
    """

    def _topology_names(self, net):
        return ([mb._line_display_name(net, int(i)) for i in net.line.index],
                [mb._trafo_display_name(net, int(i)) for i in net.trafo.index])

    def test_unnamed_branches_get_the_same_name_everywhere(self, case14_net):
        import rsa_engine as rs

        net = case14_net
        net.line["name"] = ""
        net.trafo["name"] = ""

        lines, trafos = self._topology_names(net)
        assert all(n.endswith(f"[L{i}]") for i, n in zip(net.line.index, lines))
        assert all(n.endswith(f"[T{i}]") for i, n in zip(net.trafo.index, trafos))

        # rsa_engine names violations through the same helpers, so a violation
        # can be matched back to the branch the topology describes.
        assert rs.line_display_name(net, int(net.line.index[0])) == lines[0]
        assert rs.trafo_display_name(net, int(net.trafo.index[0])) == trafos[0]

    def test_names_are_unique_even_for_parallel_branches(self, case14_net):
        net = case14_net
        net.trafo["name"] = ""
        # Two transformers between the same buses is ordinary; the endpoints
        # alone would name them identically.
        second = int(net.trafo.index[1])
        net.trafo.at[second, "hv_bus"] = net.trafo.at[net.trafo.index[0], "hv_bus"]
        net.trafo.at[second, "lv_bus"] = net.trafo.at[net.trafo.index[0], "lv_bus"]

        _, trafos = self._topology_names(net)
        assert len(set(trafos)) == len(trafos)


class TestNormalizeTapPolicy:
    @pytest.mark.parametrize("raw", ["current", "preserve", "as_is", "as-is", "uploaded", None])
    def test_normalizes_to_current(self, raw):
        assert mb._normalize_tap_policy(raw) == "current"

    @pytest.mark.parametrize("raw", ["neutral", "tap_neutral", "tap-neutral"])
    def test_normalizes_to_neutral(self, raw):
        assert mb._normalize_tap_policy(raw) == "neutral"

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown tap_policy"):
            mb._normalize_tap_policy("some_bogus_policy")


class TestParseHistoricalTarget:
    def test_defaults_to_all(self):
        assert mb._parse_historical_target(None) == ("all", None)
        assert mb._parse_historical_target("all") == ("all", None)

    def test_parses_kind_and_identifier(self):
        assert mb._parse_historical_target("bus:Foxtrot") == ("bus", "Foxtrot")
        assert mb._parse_historical_target("line:Feeder 1") == ("line", "Feeder 1")
        assert mb._parse_historical_target("trafo:T1") == ("trafo", "T1")

    def test_unknown_kind_raises_http_exception(self):
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._parse_historical_target("substation:Foxtrot")
        assert exc_info.value.status_code == 400

    def test_missing_colon_raises_http_exception(self):
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._parse_historical_target("Foxtrot")
        assert exc_info.value.status_code == 400


class TestRequestModels:
    def test_grid_request_defaults(self):
        req = mb.GridRequest()
        assert req.load_scaling_factor == 1.0
        assert req.vm_upper_pu == 1.05
        assert req.vm_lower_pu == 0.95
        assert req.data_source == "measurements"

    def test_contingency_request_requires_element_fields(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            mb.ContingencyRequest()

    def test_contingency_request_accepts_required_fields(self):
        req = mb.ContingencyRequest(element_type="line", element_index=3)
        assert req.element_type == "line"
        assert req.element_index == 3
        assert req.load_scaling_factor == 1.0

    def test_worst_case_request_defaults(self):
        req = mb.WorstCaseRequest()
        assert req.metric == "violations"
        assert req.step_size == 1
        assert req.n_steps is None


class TestSupplyLost:
    """What a contingency leaves without a source.

    The violation table cannot answer this: an unsupplied bus has no voltage
    to violate, so an islanded substation and a healthy one both report zero
    violations. Asked which substations would lose supply, the agent read
    "0 violations" and answered "none" — and on a later run, having no field
    to read, spent its whole turn budget trying to trace the network by hand
    instead of answering at all.
    """

    @staticmethod
    def _radial():
        """source — b0 — b1 — b2, with the load at the far end."""
        import pandapower as pp

        net = pp.create_empty_network()
        buses = [pp.create_bus(net, vn_kv=20.0, name=f"B{i}") for i in range(3)]
        pp.create_ext_grid(net, buses[0], name="Source")
        pp.create_line(net, buses[0], buses[1], length_km=1.0,
                       std_type="NAYY 4x50 SE", name="L0")
        pp.create_line(net, buses[1], buses[2], length_km=1.0,
                       std_type="NAYY 4x50 SE", name="L1")
        pp.create_load(net, buses[2], p_mw=0.5, name="Far load")
        return net

    def test_an_intact_network_islands_nothing(self):
        result = mb._supply_lost(self._radial(), "line", 0)
        assert result["unsupplied_bus_count"] == 0
        assert result["unsupplied_load_mw"] == 0.0
        assert "nothing is islanded" in result["supply_analysis"]

    def test_cutting_a_radial_feeder_names_what_goes_dark(self):
        net = self._radial()
        net.line.at[1, "in_service"] = False
        result = mb._supply_lost(net, "line", 1)
        assert result["unsupplied_buses"] == ["B2"]
        assert result["unsupplied_load_mw"] == pytest.approx(0.5)

    def test_the_load_left_unsupplied_is_reported_not_just_the_buses(self):
        # A de-energised bus with nothing on it is not an outage anyone
        # notices; the megawatts are what an operator means by "lost supply".
        net = self._radial()
        net.load.at[0, "p_mw"] = 1.25
        net.line.at[1, "in_service"] = False
        assert mb._supply_lost(net, "line", 1)["unsupplied_load_mw"] == pytest.approx(1.25)

    def test_a_network_it_cannot_analyse_says_so_rather_than_raising(self):
        result = mb._supply_lost(object(), "line", 0)
        assert result == {"supply_analysis": "unavailable for this network"}
