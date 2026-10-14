"""
The generated reference pages (docs/reference/) match their sources: the tool
schemas and the provider catalog. When this fails, run

    python scripts/generate_docs_reference.py
"""

import pathlib
import runpy

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEN = runpy.run_path(str(ROOT / "scripts" / "generate_docs_reference.py"))


@pytest.mark.parametrize("name", sorted(GEN["PAGES"]))
def test_the_page_is_up_to_date(name):
    page = ROOT / "docs" / "reference" / name
    assert page.is_file(), f"docs/reference/{name} is missing: run python scripts/generate_docs_reference.py"
    assert page.read_text(encoding="utf-8") == GEN["PAGES"][name](), (
        f"docs/reference/{name} is out of date: run python scripts/generate_docs_reference.py")


def test_every_tool_has_a_group():
    from llm_agent.agent.providers.schema import to_json_schema_tools
    from llm_agent.agent.tool_schemas import TOOLS

    names = {t["function"]["name"] for t in to_json_schema_tools(TOOLS)}
    placed = {n for group in GEN["TOOL_GROUPS"].values() for n in group}
    assert names - placed == set(), "place the new tool in TOOL_GROUPS (scripts/generate_docs_reference.py)"
    assert placed - names == set(), "TOOL_GROUPS names a tool that no longer exists"
