"""The task catalogue (evaluation/catalogue.py) is generated: a copy in the repository must be what the code writes."""

import pytest

from evaluation import catalogue, suites


def test_the_written_catalogue_is_what_the_code_writes():
    # papers/ is kept out of git until the paper is public: a fresh clone has no copy to compare.
    if not catalogue.DEFAULT_OUT.exists():
        pytest.skip("no catalogue written yet (python -m evaluation.catalogue)")
    assert catalogue.DEFAULT_OUT.read_text(encoding="utf-8") == catalogue.render(), \
        f"{catalogue.DEFAULT_OUT} is stale: run `python -m evaluation.catalogue`"


def test_every_task_and_check_is_in_it():
    text = catalogue.render()
    for item in suites.items(suites.names()):
        assert f"`{item.task.id}`" in text
        for check in item.task.checks:
            assert catalogue._check(check) in text
