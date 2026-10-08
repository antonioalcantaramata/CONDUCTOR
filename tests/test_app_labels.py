"""The UI's tool labels match the tools the agent can actually call.

Read from the source rather than imported: importing app.py runs the whole
Streamlit page.
"""

import ast
import pathlib

from llm_agent.agent.tool_schemas import TOOL_DISPATCH

APP = pathlib.Path(__file__).resolve().parents[1] / "llm_agent" / "app.py"


def _labels() -> dict:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_TOOL_LABELS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("_TOOL_LABELS not found in app.py")


def test_every_tool_has_a_label():
    assert set(TOOL_DISPATCH) <= set(_labels())


def test_no_label_for_a_tool_that_does_not_exist():
    assert set(_labels()) <= set(TOOL_DISPATCH)


def test_the_deterministic_opf_is_not_called_robust():
    assert "robust" not in _labels()["optimize_flexibility"].lower()
