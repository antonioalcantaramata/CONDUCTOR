"""
The backend for an evaluation run, isolated: its saved state (uploads,
proposals, assumptions, the scenario chosen) lives in a folder of the run, so
an evaluation never reads or changes the operator's `systems/` and
`data_files/`, and two runs never see each other's declarations or renames.

    python -m evaluation.isolated_backend --port 8791 --store <run>/store [--profile pglib_case30]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_STATE = ("_LAST_UPLOAD_SENTINEL", "_LAST_TIMESERIES_SENTINEL", "_LAST_FORECAST_SENTINEL", "_PROPOSALS_FILE",
          "_ASSUMPTIONS_FILE", "_SCENARIO_FILE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.isolated_backend", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--profile", default="pglib_case30")
    args = parser.parse_args(argv)

    store = pathlib.Path(args.store).resolve()
    for sub in ("systems", "data_files"):
        (store / sub).mkdir(parents=True, exist_ok=True)
    os.environ["GRID_PROFILE"] = args.profile
    backend = _ROOT / "backend"
    sys.path.insert(0, str(backend))
    os.chdir(backend)                                   # the profiles and systems resolve from here
    import main_backend as mb
    import uvicorn

    mb.SYSTEMS_DIR, mb.DATA_FILES_DIR = str(store / "systems"), str(store / "data_files")
    for name in _STATE:
        setattr(mb, name, str(store / f"{name}.json"))
    uvicorn.run(mb.app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
