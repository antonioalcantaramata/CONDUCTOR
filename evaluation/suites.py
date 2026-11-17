"""
Every agent suite behind one interface, for ablations (`evaluation.ablation`).

A suite here is a list of `Item`s: a task (the questions and their checks),
the scenario it is written for, what to set up before it and undo after, and
how its records are graded. The suites themselves stay where they are
(`whatif`, `investigation`, `options`, `clarify`, `narration`, `semantics`,
`reasoning`); this module only lists them, so an ablation can run any of them
under any arm and grade them all the same way.

    python -m evaluation.ablation list      # the suites, with their task counts
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from .whatif import Task

# The scenario each network's tasks are written for (`/api/scenario`).
SCENARIOS = {"cigre": "cigre_mv", "ieee30": "ieee30"}


class Item(NamedTuple):
    suite: str
    task: Task
    network: str                                   # a key of SCENARIOS
    evaluate: Callable[[Task, list], dict]         # records -> {check label: passed}
    family: str = ""                               # a group within the suite (reasoning's families)
    prepare: Callable[[], None] | None = None      # before the run (after the scenario is loaded)
    cleanup: Callable[[], None] | None = None      # after the run, whatever happened
    always: tuple = ()                             # checks every task of the suite gets (for the catalogue)
    setup: dict = {}                               # what is set up before it, as data (for the catalogue)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.task.id}"


def _whatif_suite(name: str, suite, network: str) -> list[Item]:
    from .whatif import evaluate
    return [Item(name, t, network, lambda task, records, _a=suite.always: evaluate(task, records, _a),
                 always=tuple(suite.always), setup={"proposals": list(t.setup)} if t.setup else {})
            for t in suite.tasks]


def _semantics(network: str) -> list[Item]:
    from .semantics import SUITES
    from .whatif import evaluate
    return [Item(f"semantics-{network}", t, network, lambda task, records: evaluate(task, records, ()))
            for t in SUITES[network].tasks]


def _reasoning() -> list[Item]:
    from . import reasoning
    return [Item("reasoning", i.task, i.network, reasoning.evaluate, family=i.family,
                 prepare=lambda _i=i: reasoning._prepare(_i), cleanup=reasoning._reset, setup=dict(i.prepare))
            for i in reasoning.ITEMS]


def _registry() -> dict[str, Callable[[], list[Item]]]:
    from .clarify import CLARIFY
    from .investigation import INVESTIGATION
    from .narration import NARRATION
    from .options import OPTIONS
    from .whatif import WHATIF
    return {
        "whatif": lambda: _whatif_suite("whatif", WHATIF, "cigre"),
        "investigation": lambda: _whatif_suite("investigation", INVESTIGATION, "ieee30"),
        "options": lambda: _whatif_suite("options", OPTIONS, "ieee30"),
        "clarify": lambda: _whatif_suite("clarify", CLARIFY, "ieee30"),
        "narration": lambda: _whatif_suite("narration", NARRATION, "ieee30"),
        "semantics-cigre": lambda: _semantics("cigre"),
        "semantics-ieee30": lambda: _semantics("ieee30"),
        "reasoning": _reasoning,
    }


def names() -> list[str]:
    return list(_registry())


def items(suites: list[str], tasks: list[str] | None = None) -> list[Item]:
    """The items of these suites, in order; `tasks` narrows them to these task
    ids (or `suite/task` keys). Unknown names raise."""
    reg = _registry()
    unknown = [s for s in suites if s not in reg]
    if unknown:
        raise ValueError(f"Unknown suite(s) {unknown}; known: {list(reg)}")
    out = [i for s in suites for i in reg[s]()]
    if tasks:
        wanted = set(tasks)
        missing = wanted - {i.task.id for i in out} - {i.key for i in out}
        if missing:
            raise ValueError(f"Unknown task(s) {sorted(missing)} in suites {suites}")
        out = [i for i in out if i.task.id in wanted or i.key in wanted]
    return out


def by_key(suites: list[str] | None = None) -> dict[str, Item]:
    return {i.key: i for i in items(suites or names())}
