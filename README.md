# CONDUCTOR

An LLM-orchestrated digital twin for uncertainty-aware power-system operations.
CONDUCTOR pairs a power-systems analysis backend (security assessment, N-1
contingencies, probabilistic risk, robust corrective dispatch, flexibility and
hosting-capacity studies, KPIs) with a natural-language agent that drives those
tools from a chat interface.

> ## ⚠️ Work in progress — please don't use this yet
>
> This repository is being prepared for open release. We are still reviewing the
> code, results, and documentation for correctness. **It is not ready for use or
> citation**, interfaces may change without notice, and outputs should not be
> relied upon. A tagged, documented release will follow once the review is
> complete. Until then, treat everything here as a preview.

## Overview

- **`backend/`** — FastAPI service with the power-system engines (pandapower +
  Pyomo/IPOPT) exposed as tools.
- **`llm_agent/`** — Streamlit chat app and the LLM agent that orchestrates the
  backend tools.
- **`systems/`**, **`data_files/`** — networks and time-series data (a small
  bundled IEEE case and an example CSV are included; everything else is generated
  or user-supplied at runtime).

## Requirements

- [git](https://git-scm.com/downloads)
- Conda — either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  (lightweight, recommended) or full Anaconda. The only heavy prerequisite;
  the launcher builds the environment for you.
- A Google Gemini API key (free tier works) — https://aistudio.google.com/apikey

| Platform | Launcher | Notes |
| --- | --- | --- |
| macOS | `./start.sh` or `python start.py` | |
| Linux | `./start.sh` or `python start.py` | |
| Windows (native) | `python start.py` | Run from **Anaconda Prompt** (or any shell where `conda` is on PATH) |
| Windows (WSL) | `./start.sh` | Follows the Linux path |

## Quick start

```Shell
# 1. Download the code
git clone https://github.com/antonioalcantaramata/CONDUCTOR.git
cd CONDUCTOR

# 2. Launch (first run builds the conda env: ~3–5 min, ~2 GB download)
./start.sh          # macOS / Linux
python start.py     # any OS, including Windows
```

When it's ready, the chat app opens at **http://localhost:8501** and asks for
your Gemini API key on first open — paste it there and you're done. (You can
also pre-create `llm_agent/.env` from `llm_agent/.env.example` if you prefer.)

Press **Ctrl+C** in the terminal once to stop both services.

To update later, pull the latest code and relaunch:

```bash
git pull
./start.sh
```

**Tips**

- Ports taken? Override them: `BACKEND_PORT=8010 STREAMLIT_PORT=8502 python start.py`.
- The services bind to `127.0.0.1` by default. Set `BACKEND_HOST=0.0.0.0` only
  if you need LAN access (expect an OS firewall prompt).
- Slow env solve on an older conda? Enable the fast solver once:
  `conda config --set solver libmamba`.
- If the first env build was interrupted, remove the broken env before
  retrying: `conda env remove -n conductor_env`.

## Configuration

All secrets live in `llm_agent/.env` (git-ignored — never commit your API key).
See `llm_agent/.env.example` for the expected variables. You can also pick a
different model there via `GEMINI_MODEL` (default: `gemma-4-31b-it`).

## Development

Unit tests cover pure-logic code in `backend/` and `llm_agent/agent/` (network
loading/editing helpers, error classification, request validation, etc.) and
don't require IPOPT or a conda environment:

```Shell
pip install -r requirements-dev.txt
pytest tests/
ruff check .
```

Both run automatically on every push/PR via GitHub Actions
(`.github/workflows/ci.yml`), on Linux and Windows.

A separate, manually-triggered workflow
(`.github/workflows/windows-first-run.yml`) simulates a brand-new Windows
user end-to-end: bare miniconda → `python start.py` builds the conda env
(including IPOPT) → both services boot → a real OPF solve succeeds. Run it
from the Actions tab (or `gh workflow run windows-first-run.yml`) before
cutting a release.

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
