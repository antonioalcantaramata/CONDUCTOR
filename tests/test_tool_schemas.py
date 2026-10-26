from llm_agent.agent import tool_schemas as ts


class TestPropBuilders:
    def test_str_prop(self):
        assert ts._str_prop("a description") == {"type": "string", "description": "a description"}

    def test_num_prop(self):
        assert ts._num_prop("a number") == {"type": "number", "description": "a number"}

    def test_int_prop(self):
        assert ts._int_prop("an int") == {"type": "integer", "description": "an int"}

    def test_bool_prop(self):
        assert ts._bool_prop("a bool") == {"type": "boolean", "description": "a bool"}

    def test_arr_str_prop(self):
        assert ts._arr_str_prop("a list") == {
            "type": "array",
            "items": {"type": "string"},
            "description": "a list",
        }

class TestSharedParameterRegistration:
    """Point-in-time tools must declare the parameters that select an operating
    point.

    A tool registered in the dispatch table but not marked `timestamp` in the
    tool registry never has `timestamp` / `data_source` injected into
    its schema. The model then cannot reliably target an operating point, and
    the tool silently analyses the simulation clock instead — observed live as
    answers about the wrong day in 4 of 10 runs, with every figure correct.
    """

    def _properties(self, tools):
        from llm_agent.agent.providers.schema import to_json_schema_tools
        return {
            t["function"]["name"]: set((t["function"].get("parameters") or {})
                                       .get("properties") or {})
            for t in to_json_schema_tools(tools)
        }

    def test_point_in_time_tools_declare_timestamp_and_data_source(self):
        from llm_agent.agent import tool_registry, tool_schemas

        props = self._properties(tool_schemas.TOOLS)
        for name in tool_registry.POINT_IN_TIME:
            assert "timestamp" in props[name], f"{name} lacks timestamp"
            assert "data_source" in props[name], f"{name} lacks data_source"

    def test_every_analysis_tool_can_select_a_dataset(self):
        """Only clock control and pure post-processing may omit data_source."""
        from llm_agent.agent import tool_registry, tool_schemas

        props = self._properties(tool_schemas.TOOLS)
        for name in props:
            if not tool_registry.REGISTRY[name].data_source:
                continue
            assert "data_source" in props[name], f"{name} lacks data_source"


def test_point_tools_offer_the_nominal_point_and_window_tools_do_not():
    def desc(name):
        fd = next(fd for t in ts.TOOLS for fd in t.function_declarations if fd.name == name)
        return fd.parameters.properties["data_source"].description or ""
    for name in ("run_rsa", "get_current_conditions", "optimize_flexibility", "compare_networks"):
        assert "'nominal'" in desc(name)
    for name in ("find_worst_case_timestamp", "kpis_over_time", "compute_historical_risk"):
        assert "'nominal'" not in desc(name)


def test_the_shared_arguments_say_what_an_omission_means():
    """Within one question an omitted network, dataset, moment or power factor
    is the one an earlier call used, not the default the text used to claim."""
    def desc(name, arg):
        fd = next(fd for t in ts.TOOLS for fd in t.function_declarations if fd.name == name)
        return fd.parameters.properties[arg].description or ""
    for arg in ("network", "data_source", "timestamp", "load_power_factor"):
        assert "earlier in this question" in desc("run_rsa", arg), arg
    assert "(default)" not in desc("run_rsa", "network")
    assert "earlier in this question" in desc("kpis_over_time", "data_source")


def test_the_data_quality_tool_reads_the_data_and_studies_nothing():
    decl = next(fd for t in ts.TOOLS for fd in t.function_declarations if fd.name == "get_data_quality")
    assert not (decl.parameters.properties or {})          # no network, data source or power factor
    assert "get_data_quality" in ts.TOOL_DISPATCH


class TestFlexibilityOverTime:
    """One tool, three studies: each to its endpoint, with only its arguments."""

    def calls(self, monkeypatch):
        from llm_agent.agent import tools
        sent = []
        monkeypatch.setattr(tools, "_post", lambda path, body: sent.append((path, body, tools._timeout_for(path))) or {
            "mode": "x"})
        return tools, sent

    def test_each_study_goes_to_its_endpoint(self, monkeypatch):
        tools, sent = self.calls(monkeypatch)
        tools.flexibility_over_time(study="hold", option="S3:cheapest", hours=4, screen_n1=True, n1="both")
        tools.flexibility_over_time(study="plan", start="2000-01-06 17", end="2000-01-06 22", steadiness=2,
                                    objective="cheapest", option="S3:cheapest")
        tools.flexibility_over_time(study="security", hours=6, n1="both", steadiness=3)
        (hold, body1, t1), (plan, body2, _), (sec, body3, _) = sent
        assert hold == "/api/flexibility/hold" and body1 == {"option": "S3:cheapest", "hours": 4, "n1": True}
        assert plan == "/api/flexibility/plan" and "option" not in body2 and body2["steadiness"] == 2
        job = body3.pop("progress_id")                                           # its progress is followed
        assert sec == "/api/flexibility/security_over_time" and body3 == {"hours": 6, "n1": "both"}
        assert isinstance(job, str) and len(job) <= 64
        assert t1 == tools.LONG_STUDY_TIMEOUT_S and t1 > 120                    # the slow study is waited for

    def test_the_study_name_is_read_however_it_is_written(self, monkeypatch):
        """The endpoint lookup normalised it and the branches did not: "Hold"
        reached the hold endpoint without its option."""
        tools, sent = self.calls(monkeypatch)
        tools.flexibility_over_time(study=" Hold", option="S3:cheapest", hours=4, screen_n1=True)
        tools.flexibility_over_time(study="PLAN", hours=6, objective="cheapest", steadiness=2)
        (hold, body1, _), (plan, body2, _) = sent
        assert hold == "/api/flexibility/hold" and body1 == {"option": "S3:cheapest", "hours": 4, "n1": True}
        assert plan == "/api/flexibility/plan" and body2 == {"hours": 6, "objective": "cheapest", "steadiness": 2}
        assert "needs option" in tools.flexibility_over_time(study="Hold")["error"]

    def test_what_is_refused_before_any_call(self, monkeypatch):
        tools, sent = self.calls(monkeypatch)
        assert "study must be one of" in tools.flexibility_over_time(study="forecast")["error"]
        assert "needs option" in tools.flexibility_over_time(study="hold")["error"]
        assert tools.flexibility_over_time(study="plan", slack_max_mw=10).get("refusal_required")
        assert sent == []


class TestFollowingALongStudy:
    def test_progress_is_passed_on_while_the_study_runs(self, monkeypatch):
        import time as _time

        from llm_agent.agent import tools
        monkeypatch.setattr(tools, "_PROGRESS_POLL_S", 0.05)
        monkeypatch.setattr(tools, "_post", lambda path, body: (_time.sleep(0.6), {"mode": "security_over_time",
                                                                                    "progress_id": body["progress_id"]})[1])
        reads = iter([{"known": False}, {"known": True, "done": 0, "total": 9, "workers": 4, "elapsed_s": 0.1},
                      {"known": True, "done": 3, "total": 9, "workers": 4, "elapsed_s": 6.0}])

        class Client:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def get(self, url):
                class R:
                    def json(self_inner):
                        return next(reads, {"known": True, "done": 3, "total": 9, "workers": 4, "elapsed_s": 6.0})
                return R()
        monkeypatch.setattr(tools.httpx, "Client", Client)
        said = []
        monkeypatch.setattr(tools, "progress_hook", said.append)
        out = tools.flexibility_over_time(study="security", hours=8)
        assert out["mode"] == "security_over_time" and len(out["progress_id"]) == 32
        assert said[0] == "0 of 9 moments done · 4 in parallel"
        assert said[1] == "3 of 9 moments done · 4 in parallel · about 12 s left"
        assert len(said) == len(set(said))                 # each change once, not every poll

