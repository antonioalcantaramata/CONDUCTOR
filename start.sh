#!/usr/bin/env bash
# start.sh — thin wrapper around the cross-platform launcher (start.py).
#
# Usage:
#   ./start.sh                            # bundled PGLib IEEE 14-bus profile (default)
#   GRID_PROFILE=ieee14 ./start.sh        # pandapower IEEE 14-bus synthetic profile
#   BACKEND_PORT=8010 ./start.sh          # override ports if the defaults are taken
#
# All logic lives in start.py so macOS, Linux, and Windows share one launcher.
# Press Ctrl+C once to stop both services cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v python3 >/dev/null 2>&1; then
    exec python3 "$SCRIPT_DIR/start.py" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$SCRIPT_DIR/start.py" "$@"
elif command -v conda >/dev/null 2>&1; then
    exec conda run -n base --no-capture-output python "$SCRIPT_DIR/start.py" "$@"
else
    echo "[start.sh] ERROR: no Python interpreter found (tried python3, python, conda base)." >&2
    echo "  Install Miniconda first: https://www.anaconda.com/docs/getting-started/miniconda/install" >&2
    exit 1
fi
