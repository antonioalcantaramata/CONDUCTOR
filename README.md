<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="llm_agent/assets/conductor-mark-reversed.png">
    <img src="llm_agent/assets/conductor-mark.png" alt="CONDUCTOR logo" width="120">
  </picture>
</p>

# CONDUCTOR

**Ask, test and decide: a solver-grounded AI assistant for distribution grids.**
CONDUCTOR pairs a power-systems analysis backend (security assessment, N-1
contingencies, probabilistic risk, robust corrective dispatch, flexibility and
hosting-capacity studies, KPIs) with a natural-language agent that drives those
tools from a chat interface. Changes to the grid can be tested as proposals
without touching the loaded network, every figure in an answer is traced to the
solver that produced it, and an advisor — a second agent that runs no studies —
suggests next steps and discusses results with the operator.

Corrective actions come as **options to choose from**, not one answer: the
least disturbance, the cheapest, the fewest units, the most headroom,
near-optimal alternatives that use other units, branches held below a chosen
share of their rating, and dispatches that stay secure after an outage
(preventive, or with a corrective plan per outage — a security-constrained OPF).
Each is re-checked by an ordinary AC power flow, and the operator chooses: the
advisor helps weigh what the solver cannot see. When redispatching the units is
not enough, demand response, then load relief, then switching are offered as a
last resort. A chosen option can be carried forward — its N-1 screen, how long it
holds, the next action from it — and a whole peak planned at once, or the cost of
N-1 security priced hour by hour. When a question leaves open something that
changes the answer, the agent asks before running. A session can be exported as
a raw record or as a report written by the model, and saved as context to
continue from later.

The agent runs on a **hosted model** or a **local model via Ollama**, picked
on a launch screen grouped into tiers:

- **Free** — Gemini API (free tier), NVIDIA Build (free endpoints), OpenRouter
  (free models);
- **Paid** — OpenAI, Anthropic (Claude), Gemini paid, OpenRouter paid, DeepSeek
  (a reported one-time signup credit, then pay-as-you-go), or any
  OpenAI-compatible endpoint (Azure, vLLM, …) via `OPENAI_BASE_URL`;
- **Local** — Ollama: no grid data leaves your machine.

Each option shows an info card before you start: context window, what the tier
actually allows per request and per day, rate limits, sign-up needs and a data
notice — see [Choosing a model](#choosing-a-model-tiers-limits-and-privacy).
The only hard requirement is **tool calling**, since CONDUCTOR drives the grid
entirely through tools.

## Documentation

The **[user guide](docs/README.md)** — also published at
<https://antonioalcantaramata.github.io/CONDUCTOR/> — covers everything you can
do once it is running: [getting started](docs/guide/getting-started.md),
[choosing a model](docs/guide/choosing-a-model.md),
[data and the operating point](docs/guide/data-and-operating-point.md),
[asking questions](docs/guide/asking-questions.md),
[reading an answer](docs/guide/reading-an-answer.md),
[what-if proposals](docs/guide/what-if-proposals.md),
[the System tab](docs/guide/system-tab.md), [the advisor](docs/guide/advisor.md),
[sessions and scenarios](docs/guide/sessions-and-scenarios.md) and
[troubleshooting](docs/guide/troubleshooting.md). Researchers: see
[using CONDUCTOR in research](docs/research/using-conductor-in-research.md). The
[tool reference](docs/reference/tools.md) and
[model options](docs/reference/providers.md) are generated from the code.

## Overview

- **`backend/`** — FastAPI service with the power-system engines (pandapower +
  Pyomo/IPOPT) exposed as tools.
- **`llm_agent/`** — Streamlit chat app and the LLM agent that orchestrates the
  backend tools.
- **`systems/`**, **`data_files/`** — networks and time-series data (the bundled
  demos — PGLib's IEEE 30-bus case, the default, and IEEE 14 — and an example
  CSV are included; everything else is generated or user-supplied at runtime).

## Requirements

- [git](https://git-scm.com/downloads)
- Conda — either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
  (lightweight, recommended) or full Anaconda. The only heavy prerequisite;
  the launcher builds the environment for you.
- **One LLM backend**, any of:
  - free: a Gemini API key — https://aistudio.google.com/apikey; an NVIDIA
    Build key (`nvapi-…`, phone verification) — https://build.nvidia.com/settings/api-keys;
    or an OpenRouter key — https://openrouter.ai/settings/keys,
  - a DeepSeek key (signup credit, then paid) — https://platform.deepseek.com/api_keys,
  - paid: an OpenAI, Anthropic, Gemini (billing) or OpenRouter key, or any
    OpenAI-compatible endpoint, or
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
launch screen in two columns: on the left what to load (Resume or a demo), on
the right how to run the assistant — a tier (Free / Paid / Local),
an option, a model, and the option's info card. Paste a key once; it's saved to
`llm_agent/.env`, one variable per provider, so several keys can be stored at
once. For OpenAI and Anthropic the key and model are checked before the app
starts; for Ollama, pick from the tool-capable models you have pulled.

The screen appears on every start, so switching backends is a restart and one
click. Settings are also editable mid-session under **Model settings** in the
sidebar. (You can pre-create `llm_agent/.env` from `llm_agent/.env.example` if
you prefer to skip the screen.)

Hosted providers retire models over time, and a saved `.env` keeps naming the
model that was current when it was written. For Gemini, OpenAI and Anthropic
the launch screen therefore asks the provider whether the configured model is
still offered: a retired model blocks **Start** and lists the alternatives,
and a newer model of the same family is mentioned but never switched to
automatically — session logs record which model answered, so a change of model
is always your decision.

Press **Ctrl+C** in the terminal once to stop both services.

To update later, pull the latest code and relaunch:

```bash
git pull
./start.sh
```

**Tips**

- At start the launcher asks GitHub (read-only, a few seconds at most) whether
  a newer version exists, and says so in the console and in the app's sidebar.
  Turn it off with `CONDUCTOR_UPDATE_CHECK=0`.
- Ports taken? Override them: `BACKEND_PORT=8010 STREAMLIT_PORT=8502 python start.py`.
- The services bind to the local machine by default. Set `BACKEND_HOST=0.0.0.0`
  and `STREAMLIT_HOST=0.0.0.0` only if you need LAN access (expect an OS
  firewall prompt).
- Running on a remote machine over SSH (e.g. VS Code Remote-SSH)? The browser
  is on your own computer, so forward ports **8501** and **8000** to it — in
  VS Code: Ports panel → Forward a Port — then open http://localhost:8501. The
  launcher prints these steps when it detects an SSH session.
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

The agent sends a large prompt — the system instructions plus 29 tool schemas
come to roughly **28,000 tokens before any conversation** (measured with
Gemini's token counter; CONDUCTOR estimates it at startup and refines it from
the token counts providers report). Hosted models absorb
that easily; local models process it at a few hundred tokens per second, so the
first request after a model loads is slow. Measured on an Apple M3 Pro (18 GB)
with a 12B model:

| | time |
| --- | --- |
| First request after the model loads | ~3 min (processing the ~28k-token prompt) |
| Every request after that | ~15 s |

Ollama caches the processed prompt, which is why only the first one is
expensive. CONDUCTOR pays that cost up front — after you click Start it loads
the model and primes the cache behind a progress message, so your first question
isn't the thing that waits. Keeping the model resident (`OLLAMA_KEEP_ALIVE`,
default 60 min) is what keeps subsequent questions fast.

A smaller model processes the prompt proportionally faster, at some cost to how
reliably it picks the right tool among twenty-nine.

### Context window

Ollama defaults `num_ctx` to about 4096 **regardless of what the model
supports**, and silently discards anything beyond the window — starting from the
front, which is where the system prompt lives. That produces a model that looks
functional but has lost its instructions.

CONDUCTOR sets the window explicitly and refuses over-long prompts rather than
letting them be truncated. The default is **65,536 tokens**: the prompt takes
~28k before any conversation, and one study's results can add several thousand
more (a set of corrective options is ~6k), so a 32k window fills within a
question or two. The launch screen suggests a size based on your machine's
detected memory and warns if the value you choose is too small for the prompt or
larger than your RAM comfortably allows; on a small machine, set
`OLLAMA_NUM_CTX` lower and expect shorter conversations.

## Configuration

All secrets live in `llm_agent/.env` (git-ignored — never commit your API key).
See `llm_agent/.env.example` for every variable and what it does; the app writes
this file for you when you use the launch screen or the sidebar settings panel.

Key choices:

| Variable | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `google`, `nvidia`, `openrouter`, `deepseek`, `openai`, `anthropic`, or `ollama` |
| `GEMINI_TIER` / `OPENROUTER_TIER` | `free` by default; `paid` |
| `NVIDIA_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` | one key variable per preset |
| `NVIDIA_MODEL`, `OPENROUTER_MODEL`, `DEEPSEEK_MODEL` | chosen on the launch screen from the live lists |
| `GEMINI_MODEL` | default `gemini-3.5-flash-lite` |
| `GEMINI_THINKING_LEVEL` | `minimal`, `low`, `medium`, `high`; empty (default) = model's own budget |
| `OPENAI_MODEL` | default `gpt-6-luna` (cheapest current tier) — must support tool calling |
| `OPENAI_REASONING_EFFORT` | `none`, `low`, `medium` (default), `high`, `xhigh` |
| `OPENAI_API` | `auto` (default), `chat`, or `responses` — see below |
| `OPENAI_BASE_URL` | any OpenAI-compatible endpoint (Azure, OpenRouter, vLLM) |
| `ANTHROPIC_MODEL` | default `claude-haiku-5-5` (cheapest) — must support tool calling |
| `ANTHROPIC_EFFORT` | `low`, `medium` (default), `high`, `xhigh`, `max` |
| `ANTHROPIC_THINKING` | `1` (default) for adaptive thinking; `0` to disable |
| `OLLAMA_MODEL` | any tool-capable model you have pulled |
| `OLLAMA_NUM_CTX` | context window — must exceed the ~28k-token prompt (default 65536) |
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
fail immediately (the launch screen does not offer them). This does not apply to
Gemma run locally through Ollama, where no such quota exists.

## Choosing a model: tiers, limits and privacy

Everything the launch screen says about an option — limits, links, data
notices — comes from **`llm_agent/provider_catalog.yaml`**, the single source of
truth. Every number there carries its source URL, whether the provider states
it (`official`), someone else reports it (`third_party`) or it could not be
found in the provider's docs (`unverified`), and the date it was last checked.
Limits change often: check your provider console, and update the catalog.

**Why free tiers are tight.** Every model call carries the ~28k-token prompt,
and one question takes several calls (CONDUCTOR measures its own average and
keeps it locally; 4 until it has data). On free tiers the binding limit is
usually requests per day or tokens per minute, not the advertised context
window — so each card shows the *effective* size per request, an estimate of
questions per day (or time per question, or questions left from a balance), and
disables an option whose per-minute token or per-request cap is below one
request.

| Option | Setup | Limits (see the catalog for sources) |
| --- | --- | --- |
| **Gemini API · free** (recommended) | key at https://aistudio.google.com/apikey | per model; daily reset at midnight Pacific. Default `gemini-3.5-flash-lite`, falling back to `gemini-3.1-flash-lite` when its daily quota is spent. Gemini 3.x Flash models are offered as an "advanced, few questions a day" choice |
| **NVIDIA Build · free endpoints** | a free NVIDIA Developer account **with phone verification** (not available in every country), then a key (`nvapi-…`) at https://build.nvidia.com/settings/api-keys | "Up to 40 rpm" and "10,000 requests per day" per the model cards. Allowlist, fastest first as measured on a 4-tool question (October 2026): `nvidia/nemotron-3-super-120b-a12b` (default, 56 s), `nvidia/nemotron-3-ultra-550b-a55b` (88 s), `nvidia/nemotron-3.5-lightning-30b-a3b` (83 s), `z-ai/glm-5.3` (657 s), `z-ai/glm-5.3-flash` (757 s), `deepseek-ai/deepseek-v4.1-flash` (did not finish within 30 min). Free endpoints can be slow: `NVIDIA_TIMEOUT_S` (default 180 s) |
| **OpenRouter · free models** | key at https://openrouter.ai/settings/keys | 20 requests/min; 50 requests/day, or 1,000/day once $10 of credit has ever been bought |
| **DeepSeek API** (paid) | key at https://platform.deepseek.com/api_keys | a one-time credit (reportedly ~5M tokens, ~30 days; not on the official pricing page), then pay-as-you-go. The card estimates questions left from your balance |
| Paid options | your provider's console | limited by your plan |
| Local (Ollama) | see [Running a local model](#running-a-local-model) | limited by your hardware |

**Discovery, not hard-coded model names.** Free models are renamed and retired
often. OpenRouter's list is read live from its public `/api/v1/models` and
filtered to models that are free, call tools, take and return text, have at
least `OPENROUTER_MIN_CONTEXT` tokens (default 131,072) and do not retire within
`FREE_MODEL_EXPIRY_GRACE_DAYS` (default 14; the rest show a "retires on" badge).
NVIDIA's model list carries no pricing or capability data, so the catalog keeps
an allowlist and only models NVIDIA currently lists are offered. Lists are
cached for `DISCOVERY_CACHE_TTL_H` hours (default 24); **Refresh** fetches them
again. A "Test tool calling" button (opt-in, uses one request) checks a model
actually calls tools.

**Thinking.** OpenRouter and NVIDIA presets have a *Thinking* setting. On
OpenRouter it is the documented `reasoning.effort` (none / low / medium / high),
sent only to models that list support for it. On NVIDIA each model documents its
own control on its card, and the catalog carries it: `enable_thinking` for the
Nemotron models (Ultra also has a medium level), `reasoning_effort` low / high /
max for GLM 5.3 (default `max`; `low` halved its response time when tested).
*Model default* sends nothing (`NVIDIA_THINKING`, `OPENROUTER_THINKING`).

**Newer model versions.** Providers rename and replace models often, so the
launch screen also looks for versions newer than the catalog knows: an NVIDIA
model of the same family with a higher version (e.g. `z-ai/glm-5.4` next to
`z-ai/glm-5.3`), or a newer Gemini Flash / Flash-Lite in Google's model list
(read with your key; it generates nothing). These are offered with a "newer ·
unconfirmed" badge — never switched to — and can be chosen once a tool-calling
test passes, since their free status and limits are unknown. Cards also warn
when the catalog's figures for an option were last verified more than
`CATALOG_STALE_DAYS` days ago (default 60).

**When a limit is hit.** A per-minute limit backs off with a visible "waiting
for the rate limit" status. A daily limit or spent credit stops, says which
limit was hit and when it resets if known, and suggests the other free options
or Local. A model that is gone (retired on Gemini, delisted on OpenRouter or
NVIDIA) is replaced by the next one — Gemini's fallback chain, or the next
qualifying model after refreshing the list — the answer says so, and the
request is retried once. A timeout is retried once and reported as a timeout.
On OpenRouter's free tier the next two qualifying models are also sent as
server-side fallbacks (`OPENROUTER_FALLBACKS`). The sidebar shows calls made
today against the daily limit (OpenRouter's own count when available), or the
credit left on DeepSeek. On free tiers, older questions' tool results
are shortened in what is sent, so long conversations fit per-minute token
budgets (`CONDUCTOR_COMPACT_HISTORY`).

**Privacy.** Each card states the option's data terms before Start, with the
long form and sources under "Details and sources". In short:

- *Gemini free:* Google may use free-tier prompts and responses to improve its
  products, and human reviewers may read them; in the EEA, Switzerland and the
  UK the paid-tier terms apply even on the free tier.
- *NVIDIA Build:* for development and testing only; NVIDIA records inputs and
  outputs for the trial and to improve its products, including AI models.
- *OpenRouter free:* OpenRouter does not store prompts unless you opt in, but
  each model's provider has its own policy, and some free-model hosts may train
  on prompts. CONDUCTOR allows them by default and says so on the card:
  asking for providers that do not collect data
  (`OPENROUTER_DENY_DATA_COLLECTION=1`) currently leaves no free model working
  (tested October 2026). Also review your
  [privacy settings](https://openrouter.ai/settings/privacy), and consider
  [Guardrails](https://openrouter.ai/docs/guides/features/guardrails) to restrict
  a key to chosen models and block endpoints that train on or retain data. The
  data-use badge on a model is a keyword heuristic, never a guarantee.
- *DeepSeek:* data is stored in the People's Republic of China.
- *All free options* are meant for testing with sample networks such
  as the bundled IEEE case. For real or confidential grid data, use Local or a
  paid tier with suitable terms; CONDUCTOR reminds you once if you load your own
  network on a free tier.

**Other models for free.** NVIDIA's own Nemotron models, GLM 5.3 and DeepSeek
V4.1 Flash are free on NVIDIA Build (GLM 5.x on Z.ai's own API is paid), and many open models are reachable
through OpenRouter's free list — so there are no separate Z.ai or Mistral
presets.

**Providers that are not presets, and why** (checked October 2026 against
their own docs):

- **Groq** — free models allow 8,000 tokens per minute, below one ~28k-token
  request ([rate limits](https://console.groq.com/docs/rate-limits)).
- **Hugging Face Inference Providers** — free users get no monthly credit
  ([pricing](https://huggingface.co/docs/inference-providers/pricing)).
- **GitHub Models** — retired on 30 July 2026.
- **Cloudflare Workers AI** — its free 10,000 "neurons" a day do fit a request,
  but only about 10–60 a day on its tool-calling models (a handful of
  questions), so it is not offered as a preset
  ([pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)).
- Third-party "free LLM" gateways and resellers are deliberately not offered:
  who operates them and what they log is unknown.

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

## Citation

If you use CONDUCTOR in your work, please cite the paper
[CONDUCTOR: An LLM-Orchestrated Digital Twin for Uncertainty-Aware Distribution Grid Operations](https://arxiv.org/abs/2606.24609)
(arXiv:2606.24609, June 2026):

```bibtex
@misc{alcantara2026conductor,
  title         = {{CONDUCTOR}: An {LLM}-Orchestrated Digital Twin for Uncertainty-Aware Distribution Grid Operations},
  author        = {Alc{\'a}ntara, Antonio and Kahraman, Ayseg{\"u}l and Sundhu, Anosh Arshad and Chatzivasileiadis, Spyros},
  year          = {2026},
  eprint        = {2606.24609},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  doi           = {10.48550/arXiv.2606.24609},
  url           = {https://arxiv.org/abs/2606.24609}
}
```

The paper describes an earlier version of CONDUCTOR. It has changed a great deal
since — network proposals and their comparison, the advisor, corrective options
and N-1 secure dispatches, free, paid and local model options, measured reactive
power and data quality, generator models, unit costs, renaming and more — so
please also name the version you used (the git
commit) and its settings, as described in
[Using CONDUCTOR in research](docs/research/using-conductor-in-research.md#citing).

## License

Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
