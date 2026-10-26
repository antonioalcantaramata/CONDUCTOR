"""
plausibility.py — is a cause physical, or a fault in the data?

An investigation names what changed. When what changed is a measured load or
unit, the change may not have happened at all: a spike from a faulty meter, a
value frozen when the telemetry stopped updating. Acting on it would be acting
on a ghost. Before such a change is reported as a cause, it is compared with
the element's own history in the same data:

- **a jump**: the change between the two moments against every change of that
  element over the same number of steps; one larger than nearly all of them
  (the 99.5th percentile) and well beyond its usual largest is unusual;
- **a level**: the value now against everything the element shows elsewhere in
  the data; well outside that range is unusual;
- **frozen**: the same value for two hours or more in a row, from an element
  that otherwise varies.

A flag is a reason to check the data, not a verdict that it is wrong: a real
event can be unusual too. Nothing here changes a share or a verdict.
"""

from __future__ import annotations

import numpy as np

RAMP_PERCENTILE = 99.5      # a jump beyond this share of the element's own jumps…
RAMP_BEYOND = 1.5           # …and this many times its usual largest (the 99th percentile)
LEVEL_MARGIN = 0.10         # beyond the element's range elsewhere by this share of that range
FROZEN_MOMENTS = 8          # the same value this many moments in a row (two hours of 15-minute data)
MIN_HISTORY = 24


def check(values, i_ref: int, i_cur: int, unit: str = "MW") -> dict:
    """The plausibility of an element's change from moment `i_ref` to
    `i_cur`, against its own series `values` (one value per moment)."""
    x = np.asarray(values, dtype=float)
    out = {"flags": []}
    if len(x) < MIN_HISTORY or not (0 <= i_ref < len(x)) or not (0 <= i_cur < len(x)):
        out["note"] = "too little history to judge"
        return {**out, "plausible": None}
    rest = np.delete(x, i_cur)
    if np.nanstd(rest) < 1e-9:
        return {**out, "plausible": True, "note": "the element is constant in the data"}
    k = abs(i_cur - i_ref)
    if 0 < k < len(x) - MIN_HISTORY:
        jumps = np.abs(x[k:] - x[:-k])
        seen = abs(x[i_cur] - x[i_ref])
        mine = min(i_ref, i_cur)
        others = np.delete(jumps, mine) if mine < len(jumps) else jumps
        usual = float(np.percentile(others, 99))
        share = float(100 * np.mean(others < seen))
        out.update(jump=round(seen, 3), usual_largest_jump=round(usual, 3), jump_percentile=round(share, 1))
        if share >= RAMP_PERCENTILE and seen > RAMP_BEYOND * max(usual, 1e-9):
            out["flags"].append(f"a jump of {seen:.2f} {unit} in {k} moment(s), where its largest are usually "
                                f"{usual:.2f} {unit}")
    lo, hi = float(np.nanmin(rest)), float(np.nanmax(rest))
    span = (hi - lo) or abs(hi) or 1.0
    out.update(range_elsewhere=[round(lo, 3), round(hi, 3)])
    if x[i_cur] > hi + LEVEL_MARGIN * span or x[i_cur] < lo - LEVEL_MARGIN * span:
        out["flags"].append(f"a value of {x[i_cur]:.2f} {unit}, outside everything it shows elsewhere in the data "
                            f"({lo:.2f} to {hi:.2f} {unit})")
    run, j = 1, i_cur
    while j > 0 and x[j - 1] == x[j]:
        run, j = run + 1, j - 1
    if run >= FROZEN_MOMENTS:
        out["flags"].append(f"the same value for {run} moments in a row — telemetry that stopped updating?")
    out["frozen_moments"] = run
    out["plausible"] = not out["flags"]
    return out
