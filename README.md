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
  `start.sh` builds the environment for you.
- A Google Gemini API key (free tier works) — https://aistudio.google.com/apikey
- macOS or Linux. On Windows, use [WSL](https://learn.microsoft.com/windows/wsl/)
  or Git Bash (the launcher is a bash script).

## Quick start

```bash
# 1. Download the code
git clone https://github.com/antonioalcantaramata/CONDUCTOR.git
cd CONDUCTOR

# 2. Add your API key
cp llm_agent/.env.example llm_agent/.env
#    then open llm_agent/.env and paste your key after GEMINI_API_KEY=

# 3. Launch (first run is slow: it builds the conda env + installs deps)
./start.sh
```

On the **first run**, `start.sh` creates the `pyopt` conda environment from
`environment.yml`, installs the agent's pip requirements
(`llm_agent/requirements.txt`), then starts the backend and the chat app.
Subsequent runs reuse the environment and start in seconds.

When it's ready, the chat app opens at **http://localhost:8501**.
Press **Ctrl+C** in the terminal once to stop both services.

To update later, pull the latest code and relaunch:

```bash
git pull
./start.sh
```

## Configuration

All secrets live in `llm_agent/.env` (git-ignored — never commit your API key).
See `llm_agent/.env.example` for the expected variables. You can also pick a
different model there via `GEMINI_MODEL`.

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
