# CONDUCTOR

An LLM-orchestrated digital twin for uncertainty-aware power-system operations.
CONDUCTOR pairs a power-systems analysis backend (security assessment, N-1
contingencies, probabilistic risk, robust corrective dispatch, flexibility and
hosting-capacity studies, KPIs) with a natural-language agent that drives those
tools from a chat interface.

The agent runs on a **hosted model** — Google Gemini or OpenAI — or a **local
model via Ollama**. You pick which each time you start it, and no grid data
leaves your machine in local mode.

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
- **One LLM backend**, any of:
  - a Google Gemini API key (free tier works) — https://aistudio.google.com/apikey,
  - an OpenAI API key (paid per token) — https://platform.openai.com/api-keys, or
  - [Ollama](https://ollama.com/download) with a tool-capable model pulled
    (see [Running a local model](#running-a-local-model)).

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

When it's ready, the chat app opens at **http://localhost:8501** and shows a
launch screen where you choose the backend:

- **Google Gemini API** — paste a key once; it's saved to `llm_agent/.env`.
- **OpenAI API** — paste a key and name a model; the key and model are checked
  before the app starts rather than at your first question.
- **Local model (Ollama)** — pick from the tool-capable models you have pulled.

The screen appears on every start, so switching backends is a restart and one
click. Settings are also editable mid-session under **Model settings** in the
sidebar. (You can pre-create `llm_agent/.env` from `llm_agent/.env.example` if
you prefer to skip the screen.)

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

## Running a local model

CONDUCTOR can drive a model running entirely on your own machine through
[Ollama](https://ollama.com/download) — no API key, no quota, and no grid data
sent anywhere.

```bash
ollama pull gemma4:12b-mlx     # on Apple Silicon, -mlx tags are faster
```

**The model must support tool calling.** CONDUCTOR drives every study through
tools, so a model without that capability can only chat. Check with
`ollama show <model>` and look for `tools` in Capabilities; the launch screen
also filters to models that report it.

### What to expect

The agent sends a large prompt — the system instructions plus 20 tool schemas
come to roughly **25,000 tokens before any conversation**. Hosted models absorb
that easily; local models process it at a few hundred tokens per second, so the
first request after a model loads is slow. Measured on an Apple M3 Pro (18 GB)
with a 12B model:

| | time |
| --- | --- |
| First request after the model loads | ~3 min (processing the 25k-token prompt) |
| Every request after that | ~15 s |

Ollama caches the processed prompt, which is why only the first one is
expensive. CONDUCTOR pays that cost up front — after you click Start it loads
the model and primes the cache behind a progress message, so your first question
isn't the thing that waits. Keeping the model resident (`OLLAMA_KEEP_ALIVE`,
default 60 min) is what keeps subsequent questions fast.

A smaller model processes the prompt proportionally faster, at some cost to how
reliably it picks the right tool among twenty.

### Context window

Ollama defaults `num_ctx` to about 4096 **regardless of what the model
supports**, and silently discards anything beyond the window — starting from the
front, which is where the system prompt lives. That produces a model that looks
functional but has lost its instructions.

CONDUCTOR sets the window explicitly and refuses over-long prompts rather than
letting them be truncated. The launch screen suggests a size based on your
machine's detected memory and warns if the value you choose is too small for the
prompt or larger than your RAM comfortably allows.

## Configuration

All secrets live in `llm_agent/.env` (git-ignored — never commit your API key).
See `llm_agent/.env.example` for every variable and what it does; the app writes
this file for you when you use the launch screen or the sidebar settings panel.

Key choices:

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `google`, `openai`, or `ollama` |
| `GEMINI_MODEL` | default `gemini-3.5-flash-lite` |
| `GEMINI_THINKING_LEVEL` | `minimal`, `low`, `medium`, `high`; empty (default) = model's own budget |
| `OPENAI_MODEL` | default `gpt-5.6-luna` — must support tool calling |
| `OPENAI_REASONING_EFFORT` | `none`, `low`, `medium` (default), `high`, `xhigh` |
| `OPENAI_API` | `auto` (default), `chat`, or `responses` — see below |
| `OPENAI_BASE_URL` | any OpenAI-compatible endpoint (Azure, OpenRouter, vLLM) |
| `OLLAMA_MODEL` | any tool-capable model you have pulled |
| `OLLAMA_NUM_CTX` | context window — must exceed the ~25k-token prompt |
| `OLLAMA_KEEP_ALIVE` | how long the model and its prompt cache stay resident |

The OpenAI provider drives two endpoints. Newer reasoning models **refuse
function tools combined with any reasoning effort** on `/v1/chat/completions`,
and CONDUCTOR calls tools on every turn, so reasoning is only reachable through
`/v1/responses`. With `OPENAI_API=auto` the provider picks `/v1/responses` when
a reasoning level is set and the base URL is OpenAI's own, and
`/v1/chat/completions` otherwise. The launch screen names whichever applies.

That fallback is also why `OPENAI_BASE_URL` still points the provider at
anything speaking chat-completions — Azure OpenAI, OpenRouter, Groq, a local
vLLM or llama.cpp server. Those mostly do not implement `/v1/responses`, so a
custom base URL stays on chat-completions unless you set `OPENAI_API=responses`.

Avoid free-tier **Gemma models on the Gemini API**: their 16K
input-tokens-per-minute cap is below this agent's per-call overhead, so requests
fail immediately. This does not apply to Gemma run locally through Ollama, where
no such quota exists.

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
