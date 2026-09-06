"""
Does a tool that advertises an operating point actually send one?

Found live. `optimize_robust_flexibility` declared `timestamp` in its schema,
accepted it without error, and dropped it on the floor: it built its request
body explicitly and never merged `**kwargs`. The robust OPF therefore ran at
the simulation clock while the rest of the answer described the requested
point, and the agent presented the two as one analysis. Every figure was
correct and the composition was not.

Two earlier tests pass on a tool with this defect. The schema test checks the
parameter is *declared*; the dispatch test checks the call *succeeds*. Nothing
checked the value reaches the backend, which is the only thing that matters.

So this goes through the wrapper and inspects the body that would have been
sent. It is deliberately behavioural rather than a source-level check: the
three working idioms in `tools.py` are `**kwargs` inside the dict literal,
`body.update(kwargs)`, and an explicit `body["timestamp"] = ...`, and a test
that enumerated those would pass a fourth idiom that happened to be broken.
"""

import inspect

import pytest

from llm_agent.agent import tools
from llm_agent.agent.tool_schemas import _POINT_IN_TIME_TOOLS

TIMESTAMP = "2022-01-02 21:45"
DATA_SOURCE = "forecasts"

# Arguments a tool cannot be called without. Values are never used: the
# request never leaves the process.
REQUIRED_ARGS = {
    "compute_flexibility_envelope": {"gen_name": "Alpha"},
    "compute_hosting_capacity": {"bus": 23},
    "optimize_contingency": {"element_type": "line", "element_index": 0},
    "simulate_contingency": {"element_type": "line", "element_index": 0},
}


@pytest.fixture
def captured(monkeypatch):
    """Intercept the outgoing request instead of making it."""
    sent = {}

    def _capture(endpoint, body):
        sent["endpoint"] = endpoint
        sent["body"] = body
        return {"ok": True}

    monkeypatch.setattr(tools, "_post", _capture)
    monkeypatch.setattr(tools, "_get", lambda endpoint: {"ok": True})
    return sent


@pytest.mark.parametrize("tool_name", sorted(_POINT_IN_TIME_TOOLS))
class TestOperatingPointReachesTheBackend:
    def test_timestamp_is_forwarded(self, tool_name, captured):
        fn = getattr(tools, tool_name)
        fn(timestamp=TIMESTAMP, **REQUIRED_ARGS.get(tool_name, {}))

        assert captured["body"].get("timestamp") == TIMESTAMP, (
            f"{tool_name} declares `timestamp` in its schema and accepts it without "
            f"error, but never sends it. The backend will run at the simulation "
            f"clock and stamp the result with a different operating point than the "
            f"one the answer claims."
        )

    def test_data_source_is_forwarded(self, tool_name, captured):
        fn = getattr(tools, tool_name)
        fn(data_source=DATA_SOURCE, **REQUIRED_ARGS.get(tool_name, {}))

        assert captured["body"].get("data_source") == DATA_SOURCE, (
            f"{tool_name} declares `data_source` but never sends it — measurements "
            f"would be read where forecasts were asked for."
        )


class TestTheHarnessItself:
    """A test that silently stops exercising anything is worse than no test."""

    def test_every_point_in_time_tool_is_callable(self):
        missing = [name for name in _POINT_IN_TIME_TOOLS if not hasattr(tools, name)]
        assert not missing, f"declared point-in-time tools with no implementation: {missing}"

    def test_required_arguments_are_still_accurate(self):
        # A tool that gains a required argument would otherwise fail with a
        # TypeError that reads like the forwarding defect this file exists for.
        for name in _POINT_IN_TIME_TOOLS:
            signature = inspect.signature(getattr(tools, name))
            required = {
                p.name for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
            }
            supplied = set(REQUIRED_ARGS.get(name, {}))
            assert required == supplied, (
                f"{name} requires {sorted(required)}; REQUIRED_ARGS supplies "
                f"{sorted(supplied)}. Update the table above."
            )


class TestBackendSubCalls:
    """The same defect one layer down.

    `/api/flexibility/robust` builds a `FlexibilityRequest` and posts it to
    `/api/flexibility/optimize` over HTTP. It passed the voltage limits, the
    slack cap and the back-off bounds — and not the timestamp. The OPF
    therefore resolved its own tick from the simulation clock and solved a
    different day than the back-off calibration in the same request, then
    stamped the whole robust result with that day. Not a mis-echoed field: the
    dispatch really was computed at the wrong operating point.

    Any request model inheriting `_TimeseriesRequest` carries `timestamp` and
    `data_source`; constructing one internally without them silently means
    "wherever the clock happens to be".
    """

    @staticmethod
    def _backend_tree():
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent
        return ast.parse((root / "backend" / "main_backend.py").read_text(encoding="utf-8"))

    # Sanctioned exceptions: (enclosing function, request model).
    #
    # `compute_historical_risk` builds one GridRequest as a template and
    # replays it across every tick of a window, passing each timestamp to
    # `_run_rsa_snapshot` explicitly. There is no single operating point to
    # carry, and `_lookup_measurement_df` resolves the dataset from the
    # timestamp itself — measurement and forecast ranges are disjoint by
    # construction, which that helper documents and depends on.
    SANCTIONED = {("compute_historical_risk", "GridRequest")}

    def test_internal_requests_carry_the_operating_point(self):
        import ast

        tree = self._backend_tree()

        timeseries_models = {
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(isinstance(b, ast.Name) and b.id == "_TimeseriesRequest" for b in node.bases)
        }
        assert timeseries_models, "no request models inherit _TimeseriesRequest — has the mixin been renamed?"

        offenders = []
        for holder in ast.walk(tree):
            if not isinstance(holder, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(holder):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id not in timeseries_models:
                    continue
                if (holder.name, node.func.id) in self.SANCTIONED:
                    continue
                missing = {"timestamp", "data_source"} - {kw.arg for kw in node.keywords}
                if missing:
                    offenders.append(
                        f"{holder.name}() builds {node.func.id} at line {node.lineno} "
                        f"without {sorted(missing)}"
                    )

        assert not offenders, (
            "Internal sub-calls that drop the operating point — the sub-analysis runs at "
            "the simulation clock while the caller reports the requested point:\n  "
            + "\n  ".join(offenders)
        )
