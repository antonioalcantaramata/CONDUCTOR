"""
withdrawal.py — never ship a number that would be wrong to state.

The rule, learned the hard way in the attribution engine and generalised here.

The engine had already decided correctly: `relief_mw_feasible` was ``False``,
and `recommended_action` said no single source could clear the bus. The model
read that sentence, reached past it into the driver beside it, took the raw
``relief_mw`` of -6.2415, and wrote "Reduce 05 ÅKI Sgen by 6.24 MW" — about a
3.54 MW unit.

Two lessons, and the second is the one that generalises:

  1. Moving a decision into the engine is not enough while the ingredients
     stay in the payload. A number that would be wrong to state should not be
     shipped at all.

  2. Reporting only what *cannot* be done is worse. An earlier version of the
     prompt rule, told to withhold infeasible movements, answered with no
     numbers at all. Every withdrawal must leave a correct figure in its place.

So: the impossible value is replaced by ``None``, and a ``max_*`` field states
the most the source could actually contribute. There is always something true
left to quote.

Severity split, matching the parameter guards: **refuse the impossible, allow
the merely unusual.** A value whose feasibility is unknown (``None``) stays
where it is. Deleting those would strip the diagnostic content that explains
the regime, and the guard would start hiding facts rather than errors.

This is prevention, not detection. `provenance.py` and `characterisation.py`
observe what the model wrote; this stops the wrong figure reaching it. Where
an engine knows the answer, prevention is the stronger form — the model cannot
misquote what it never received.
"""

from __future__ import annotations

from typing import Any


def withdraw(
    record: dict,
    value_key: str,
    flag_key: str,
    capacity_key: str | None = None,
    replacement_key: str | None = None,
) -> bool:
    """Withdraw one provably impossible value from a record, in place.

    Args:
        record:          The object holding the value and its feasibility flag.
        value_key:       The figure to withdraw when it cannot be delivered.
        flag_key:        A boolean that is ``False`` when the value is
                         impossible. ``None`` means unknown and is left alone.
        capacity_key:    Where to read the most the source could contribute.
                         Optional: some records have no such quantity.
        replacement_key: Where to write that maximum. Defaults to
                         ``max_<value_key>``.

    Returns:
        Whether anything was withdrawn.
    """
    if record.get(flag_key) is not False:
        return False
    if record.get(value_key) is None:
        return False

    record[value_key] = None
    if capacity_key is not None:
        target = replacement_key or f"max_{value_key}"
        capacity = record.get(capacity_key)
        record[target] = abs(float(capacity)) if capacity is not None else None
    return True


def withdraw_all(
    records: Any,
    value_key: str,
    flag_key: str,
    capacity_key: str | None = None,
    replacement_key: str | None = None,
) -> int:
    """Apply `withdraw` across a list of records. Returns how many were withdrawn."""
    if not isinstance(records, list):
        return 0
    return sum(
        withdraw(r, value_key, flag_key, capacity_key, replacement_key)
        for r in records
        if isinstance(r, dict)
    )


def mark_unusable(result: dict, reason: str) -> dict:
    """Flag a whole result as one whose figures must not be quoted as outcomes.

    For studies that ran but did not produce a usable answer — a bisection that
    never converged, an OPF that terminated non-optimally. The numbers are left
    in place, because they explain what happened and an operator may want them;
    what is added is the verdict that stops them being read as a result.

    Deliberately not a deletion. A non-converged study's iterates are diagnostic
    and the same reasoning that keeps unknown headroom applies: state the
    verdict, keep the evidence.
    """
    result["feasible"] = False
    result["usable"] = False
    result["unusable_reason"] = reason
    return result
