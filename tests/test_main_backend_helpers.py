
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
        net.bus.at[0, "name"] = "Nexo"
        assert mb._bus_display_name(net, 0) == "Nexo"

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
        assert mb._parse_historical_target("bus:Nexo") == ("bus", "Nexo")
        assert mb._parse_historical_target("line:Feeder 1") == ("line", "Feeder 1")
        assert mb._parse_historical_target("trafo:T1") == ("trafo", "T1")

    def test_unknown_kind_raises_http_exception(self):
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._parse_historical_target("substation:Nexo")
        assert exc_info.value.status_code == 400

    def test_missing_colon_raises_http_exception(self):
        with pytest.raises(mb.HTTPException) as exc_info:
            mb._parse_historical_target("Nexo")
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
