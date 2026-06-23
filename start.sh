#!/usr/bin/env bash
# start.sh — Start the AIDT FastAPI backend + Streamlit chat app
#
# Usage:
#   chmod +x start.sh   (once)
#   ./start.sh                            # bundled PGLib IEEE 14-bus profile (default)
#   GRID_PROFILE=pglib_case14 ./start.sh # explicit bundled MATPOWER profile
#   GRID_PROFILE=ieee14 ./start.sh       # pandapower IEEE 14-bus synthetic profile
#
# Press Ctrl+C once to stop both services cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
AGENT_DIR="$SCRIPT_DIR/llm_agent"

# Network profile — override with e.g. GRID_PROFILE=ieee14 ./start.sh
export GRID_PROFILE="${GRID_PROFILE:-pglib_case14}"

BACKEND_PORT=8000
STREAMLIT_PORT=8501

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[start.sh]${NC} $*"; }
warn() { echo -e "${YELLOW}[start.sh]${NC} $*"; }
die()  { echo -e "${RED}[start.sh] ERROR:${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo -e "${RED}[start.sh] ERROR:${NC} conda not found." >&2
    echo "" >&2
    echo "  Miniconda (free, ~100 MB) is the only prerequisite." >&2
    echo "  Install it from: https://www.anaconda.com/docs/getting-started/miniconda/install" >&2
    echo "" >&2
    echo "  Quick install on macOS/Linux:" >&2
    echo "    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname -s)-$(uname -m).sh -o /tmp/miniconda.sh" >&2
    echo "    bash /tmp/miniconda.sh -b -p \$HOME/miniconda3" >&2
    echo "    source \$HOME/miniconda3/etc/profile.d/conda.sh" >&2
    echo "" >&2
    echo "  Then close and reopen your terminal, and run ./start.sh again." >&2
    exit 1
fi

# Auto-create pyopt env from environment.yml if it doesn't exist yet
if ! conda env list | grep -q "^pyopt "; then
    if [[ -f "$SCRIPT_DIR/environment.yml" ]]; then
        warn "Conda env 'pyopt' not found — creating it from environment.yml…"
        warn "This takes a few minutes on first run."
        conda env create -f "$SCRIPT_DIR/environment.yml" \
            || die "Failed to create conda env. Check the output above."
        log "Conda env 'pyopt' created ✓"
    else
        die "Conda env 'pyopt' not found and environment.yml is missing.\nRun: conda env create -f environment.yml"
    fi
fi

# Always use conda base's Python for the agent — avoids picking up the
# macOS system Python 3.9 when conda is not explicitly activated.
BASE_PYTHON="$(conda run -n base which python)"
[[ -x "$BASE_PYTHON" ]] || die "Could not locate Python in conda base env"
log "Using Python: $BASE_PYTHON"

# --- Check backend (pyopt env) requirements ---
log "Checking pyopt environment packages…"
BACKEND_PACKAGES=(
    fastapi
    uvicorn
    pandapower
    pydantic
    pyomo
    pandas
    requests
    dotenv      # python-dotenv
)
MISSING_BACKEND=()
for pkg in "${BACKEND_PACKAGES[@]}"; do
    # Map package names that differ between pip name and import name
    import_name="$pkg"
    [[ "$pkg" == "dotenv" ]] && import_name="dotenv"
    if ! conda run -n pyopt python -c "import $import_name" >/dev/null 2>&1; then
        MISSING_BACKEND+=("$pkg")
    fi
done
if [[ ${#MISSING_BACKEND[@]} -gt 0 ]]; then
    warn "The following packages are missing from the 'pyopt' environment:"
    for p in "${MISSING_BACKEND[@]}"; do echo "    - $p"; done
    die "Install them with: conda run -n pyopt pip install ${MISSING_BACKEND[*]}"
fi
log "pyopt environment OK ✓"

# --- Check Streamlit / agent (conda base env) requirements ---
# Use pip show against the explicit conda base Python — never the system Python.
log "Checking Streamlit agent packages…"
AGENT_PACKAGES=(streamlit plotly httpx google-genai python-dotenv)
MISSING_AGENT=()
for pip_name in "${AGENT_PACKAGES[@]}"; do
    if ! "$BASE_PYTHON" -m pip show "$pip_name" >/dev/null 2>&1; then
        MISSING_AGENT+=("$pip_name")
    fi
done
if [[ ${#MISSING_AGENT[@]} -gt 0 ]]; then
    warn "The following packages are missing from the conda base environment:"
    for p in "${MISSING_AGENT[@]}"; do echo "    - $p"; done
    warn "Installing from llm_agent/requirements.txt…"
    "$BASE_PYTHON" -m pip install -r "$AGENT_DIR/requirements.txt" \
        || die "pip install failed — check your internet connection or Python environment."
    log "Agent packages installed ✓"
else
    log "Agent environment OK ✓"
fi

# API key check — a soft warning only; the Streamlit app shows a setup screen
# on first run so users can enter their key without editing files manually.
if [[ ! -f "$AGENT_DIR/.env" ]] || ! grep -q "GEMINI_API_KEY=." "$AGENT_DIR/.env" 2>/dev/null; then
    warn "No GEMINI_API_KEY found in $AGENT_DIR/.env"
    warn "The app will guide you through setup on first open."
fi

# ---------------------------------------------------------------------------
# Cleanup on exit
# ---------------------------------------------------------------------------
BACKEND_PID=""
STREAMLIT_PID=""

cleanup() {
    echo ""
    warn "Shutting down…"
    [[ -n "$BACKEND_PID" ]]   && kill "$BACKEND_PID"   2>/dev/null && log "Backend stopped   (PID $BACKEND_PID)"
    [[ -n "$STREAMLIT_PID" ]] && kill "$STREAMLIT_PID" 2>/dev/null && log "Streamlit stopped (PID $STREAMLIT_PID)"
    exit 0
}
trap cleanup INT TERM

# ---------------------------------------------------------------------------
# Start FastAPI backend
# ---------------------------------------------------------------------------
log "Starting FastAPI backend on http://localhost:$BACKEND_PORT (conda env: pyopt)…"
log "Grid profile: $GRID_PROFILE"
cd "$BACKEND_DIR"
conda run -n pyopt --no-capture-output \
    uvicorn main_backend:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    > "$SCRIPT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
log "Backend PID: $BACKEND_PID  (logs → backend.log)"

# Wait until the backend is accepting connections (up to 30 s)
log "Waiting for backend to be ready (this can take 60–120 s for pandapower load)…"
for i in $(seq 1 120); do
    # Abort early if the backend process already died
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        warn "Backend process exited unexpectedly. Check backend.log:"
        tail -20 "$SCRIPT_DIR/backend.log" >&2
        die "Backend failed to start."
    fi
    if curl -sf "http://localhost:$BACKEND_PORT/docs" >/dev/null 2>&1; then
        log "Backend is up ✓  (took ${i}s)"
        break
    fi
    printf '.'
    sleep 1
    if [[ $i -eq 120 ]]; then
        echo ""
        warn "Backend did not respond after 120 s — Streamlit will start anyway."
        warn "Check backend.log for errors."
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Start Streamlit chat app
# ---------------------------------------------------------------------------
log "Starting Streamlit chat app on http://localhost:$STREAMLIT_PORT …"
cd "$AGENT_DIR"
"$BASE_PYTHON" -m streamlit run app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.headless true \
    > "$SCRIPT_DIR/streamlit.log" 2>&1 &
STREAMLIT_PID=$!
log "Streamlit PID: $STREAMLIT_PID  (logs → streamlit.log)"

# Open browser on macOS after a short delay
sleep 2
if command -v open >/dev/null 2>&1; then
    open "http://localhost:$STREAMLIT_PORT"
fi

# ---------------------------------------------------------------------------
# Keep running
# ---------------------------------------------------------------------------
echo ""
log "Both services running. Press Ctrl+C to stop."
echo -e "  Backend:   ${GREEN}http://localhost:$BACKEND_PORT/docs${NC}"
echo -e "  Chat app:  ${GREEN}http://localhost:$STREAMLIT_PORT${NC}"
echo ""

wait
