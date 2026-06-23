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

- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
- A Google Gemini API key (free tier works) — https://aistudio.google.com/apikey

## Quick start

```bash
# 1. Backend environment (conda)
conda env create -f environment.yml      # creates the `pyopt` env

# 2. Agent API key
cp llm_agent/.env.example llm_agent/.env
#   then edit llm_agent/.env and paste your GEMINI_API_KEY

# 3. Launch backend + chat app
./start.sh
```

`start.sh` creates the conda env if needed, installs the agent's pip
requirements (`llm_agent/requirements.txt`), and starts both services. The chat
app opens at http://localhost:8501.

## Configuration

All secrets live in `llm_agent/.env` (git-ignored). Never commit your API key.
See `llm_agent/.env.example` for the expected variables.

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
