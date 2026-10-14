# CONDUCTOR — LLM chat agent (`llm_agent/`)

A Streamlit chat app plus the LLM agent that drives the
[backend](../backend/) power-system tools from natural language. The agent
calls an LLM — hosted Gemini or OpenAI, or a local model via Ollama — which
decides which backend endpoints to invoke, then summarizes the results in text while the app
renders the matching charts.

> See the [root README](../README.md) for setup and how to launch (`./start.sh`
> from the project root starts both this app and the backend). This document
> covers the agent internals.

## Layout

```
llm_agent/
├── app.py                 ← Streamlit UI: sidebar operating point, chat, charts
├── launch_ui.py           ← launch screen: tiers, model picker, info cards, preset settings
├── provider_catalog.yaml  ← every LLM option: tier, limits (with sources), links, data notes
├── agent/
│   ├── catalog.py         ← loads and validates provider_catalog.yaml
│   ├── usage.py           ← prompt overhead, calls per question, estimates, compatibility
│   ├── ratelimit.py       ← per-minute vs per-day limits, models gone, backoff
│   ├── smoke_test.py      ← opt-in "Test tool calling"
│   ├── discovery/         ← live model lists: OpenRouter (filtered), NVIDIA (allowlist), cache
│   ├── config.py          ← env vars + live grid-constants fetch
│   ├── loop.py            ← the agentic turn loop, provider-agnostic
│   ├── providers/         ← pluggable LLM backends
│   │   ├── base.py        ←   neutral message format + LLMProvider protocol
│   │   ├── gemini.py      ←   Google Generative AI
│   │   ├── openai.py      ←   OpenAI + OpenAI-compatible endpoints
│   │   ├── openai_compat.py ← OpenRouter, NVIDIA Build, DeepSeek presets (chat completions)
│   │   ├── ollama.py      ←   local models, preflight + context guard
│   │   └── schema.py      ←   tool schemas → JSON Schema for non-Gemini backends
│   ├── hardware.py        ← RAM/platform detection for local-model sizing advice
│   ├── tools.py           ← Python functions that call the backend over HTTP
│   ├── tool_schemas.py    ← function declarations (TOOLS) — single source of truth
│   ├── renderers.py       ← Plotly renderers, keyed by tool name (RENDERER_MAP)
│   ├── system_prompt.py   ← system instruction, built from live grid constants
│   ├── recommender.py     ← the advisor: suggestions and side discussions, no tools
│   ├── timeline.py        ← operating point: measured / forecast / compare (and at what time), or nominal
│   ├── data_view.py       ← Data tab: the loaded series, forecast against measured
│   ├── staleness.py       ← is an earlier answer still computed under today's settings?
│   ├── report.py          ← the session as one standalone HTML report
│   └── errors.py          ← error classification / retry helpers
├── assets/                ← logo marks (colour, reversed, black) and app icon
├── requirements.txt       ← pip deps (installed by start.sh)
└── .env.example           ← copy to .env and add your key
```

## Starting scenarios

The launch screen has two columns: on the left **what to load** — *Resume* (your last uploaded
network and data; offered once something has been uploaded), *Demo: IEEE 14 ·
active only*, or *Demo: IEEE 14 · with reactive measurements* — and on the right
how to run the assistant, with Start. A choice there loads at once. Later, the sidebar's
*Scenario* expander switches with **Load**, which starts a new conversation (the
old one described another grid). Demos never replace your uploads; each network
keeps its own proposals and assumptions (names, costs, unit models, load power
factor), restored whenever it is loaded again. An option with saved changes says
so ("Your changes are kept: 1 rename · 2 proposals"; the two demos share one
network, so they share them), and **Start fresh** clears them. They live in the
git-ignored `systems/` files, never in the repository. Backend side:
`backend/scenarios.py`, `GET/POST /api/scenario`, `POST /api/scenario/forget`.

## Working aids

- **Answers under other settings.** Each answer is stamped with what its studies
  ran under — the load power factor and generator models (read from the results'
  own `load_power_factor` / `generator_model` blocks) and the loaded data
  (`data_key`). If one changes later, the answer gets a line such as "⏱ Computed
  under other settings: load power factor 0.95 → 0.90", and **↻ Rerun with the
  current settings** asks the same question again as a new turn (asking for fresh
  studies). The earlier answer stays for comparison. `agent/staleness.py`.
- **Session report.** *Export session report* (sidebar, under New conversation)
  downloads one HTML file that opens offline: scenario, network, data, model,
  assumptions, data quality and proposals, then every question and answer with
  its interactive charts, its evidence (tools and arguments, where each figure
  came from, what to check, what the checks cannot see) and any stale-settings
  note. Built only when clicked. `agent/report.py`.
- **Example queries for the scenario.** The sidebar's examples start with
  questions that fit what is loaded (Resume, either demo), then the general ones.
- **Data template.** The data-upload pop-up links to `GET /api/data/template`: a
  CSV with every substation name the network expects, the loaded data's first
  hour as example values, and the optional reactive columns.

## Environment variables

Set these in `llm_agent/.env` (copy from `.env.example`) or as shell vars.

| Variable                  | Required            | Default                   | Description                                        |
| ------------------------- | ------------------- | ------------------------- | -------------------------------------------------- |
| `LLM_PROVIDER`            | No                  | `google`                  | Backend: `google`, `nvidia`, `openrouter`, `deepseek`, `openai`, `anthropic`, or `ollama` (see the root README's "Choosing a model" and `.env.example`) |
| `GEMINI_API_KEY`          | If provider=google  | —                         | Google Gemini API key                               |
| `GEMINI_MODEL`            | No                  | gemini-3.5-flash-lite     | Model id to use                                     |
| `GEMINI_THINKING_LEVEL`   | No                  | *(empty)*                 | `minimal`/`low`/`medium`/`high`; empty = the model's own budget |
| `OPENAI_API_KEY`          | If provider=openai  | —                         | OpenAI API key                                      |
| `OPENAI_MODEL`            | No                  | gpt-6-luna                | Model id; must support tool calling                 |
| `OPENAI_REASONING_EFFORT` | No                  | `medium`                  | `none`/`low`/`medium`/`high`/`xhigh`; empty omits the parameter |
| `OPENAI_API`              | No                  | `auto`                    | `auto`, `chat`, or `responses` — which endpoint to drive |
| `OPENAI_BASE_URL`         | No                  | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint                    |
| `OPENAI_TIMEOUT_S`        | No                  | `180`                     | Request timeout (s)                                 |
| `OLLAMA_MODEL`            | If provider=ollama  | gemma4:12b-mlx            | Local model; must support tool calling              |
| `OLLAMA_HOST`             | No                  | `http://localhost:11434`  | Ollama server URL                                   |
| `OLLAMA_NUM_CTX`          | No                  | `32768`                   | Context window; must exceed the ~28k-token prompt   |
| `OLLAMA_OUTPUT_RESERVE`   | No                  | `4096`                    | Tokens held back from the window for the reply      |
| `OLLAMA_KEEP_ALIVE`       | No                  | `60m`                     | How long the model + prompt cache stay resident     |
| `OLLAMA_THINK`            | No                  | `true`                    | Let the model reason first (off trades quality for speed) |
| `OLLAMA_TIMEOUT_S`        | No                  | `600`                     | Request timeout; local generation is slow           |
| `DT_BACKEND_URL`          | No                  | `http://localhost:8000`   | FastAPI backend base URL                            |
| `DT_HTTP_TIMEOUT`         | No                  | `120.0`                   | Per-request HTTP timeout (s)                        |
| `DT_MAX_AGENT_TURNS`      | No                  | `16`                      | Max model calls per question (a ceiling against loops; at it, one more call answers from the results so far) |

All of these are editable in the app under **Model settings** in the sidebar,
which writes them back to `.env` and applies them to your next message.

### Reasoning

All three backends reason by default, and each spells the setting its own way
rather than through a shared scale that would match none of them:

| Backend | Setting | Default |
| --- | --- | --- |
| Gemini | `GEMINI_THINKING_LEVEL` — `minimal`…`high`, or empty | empty: the model's own dynamic budget |
| OpenAI | `OPENAI_REASONING_EFFORT` — `none`…`xhigh` | `medium` |
| Ollama | `OLLAMA_THINK` — on/off | on |

Two traps worth knowing. On Gemini, `include_thoughts=False` in the request
suppresses *returning* the thought trace and does not stop the model thinking —
reasoning has always been on here. On OpenAI, newer reasoning models **refuse
function tools combined with any reasoning effort** on `/v1/chat/completions`,
and this agent sends tools every turn, so reasoning is only reachable through
`/v1/responses`; `OPENAI_API=auto` picks that endpoint whenever a reasoning
level is set and the base URL is OpenAI's own.

## How a turn works

1. **Tool call** — the model decides to call e.g. `run_rsa`. The matching
   function in `agent/tools.py` makes an HTTP request to the backend and appends
   `("run_rsa", result_dict)` to the module-level `_last_tool_results`.
2. **Response** — the model reads the JSON result, reasons over it, and writes a
   text summary (the LLM never describes charts in prose).
3. **Charts** — after the text renders, `app.py` walks `_last_tool_results`,
   looks up each tool's renderer in `renderers.py`'s `RENDERER_MAP`, and draws
   the Plotly figures. `_last_tool_results` is cleared at the start of each turn.

Steps 1 and 2 are separate requests, and **each one carries the whole system
prompt and all tool schemas** — the APIs are stateless, so nothing is retained
between calls. That fixed prefix measures ~28k tokens, dwarfing the
conversation itself, which is why `OLLAMA_NUM_CTX` has to be generous and why
keeping the prompt cache warm matters so much locally.

Every turn is written to `session_logs/` as one JSON line, including which
backend answered it (`provider`, `model`, `reasoning`, `endpoint`) — without
that, answers from different backends cannot be compared after the fact.

## Measurements vs forecasts

Every analysis tool accepts a `data_source` parameter — `"measurements"`
(historical actuals, the default; drives the simulation clock) or `"forecasts"`
(read-only look-ahead). Point-in-time tools also accept an optional `timestamp`,
and `"nominal"` — the network's own operating point as loaded, with no time. The
agent picks the source from the user's intent and reports it in its answer.

In the sidebar, the dataset switch (Measured / Forecast / Compare) chooses data
in time; the **Nominal point** switch below it sits apart, because nominal has no
time: when on, the clock and slider give way to "Nominal" and the agent is told to
study that point. The Data tab draws the nominal value as a dotted reference line
beside the series.

**Reactive power.** The data carry active power only; each measured load's
reactive power comes from a power factor — 0.95 lagging by default, shown and
changeable in the System tab: a strip of chips at the top shows the assumptions
in force ("Loads pf 0.95", "Generators · as loaded", "Nominal point · off") and
opens the **Model assumptions** view (beside the **Network map** view), where the
load power factor and the generator models are explained and edited. The map's
subtitle names any assumption that is not the default. Every study tool
takes `load_power_factor` to rerun one study at another value;
`set_load_power_factor` changes the session's value (or resets it) when the
operator asks. Results state the value they used.

**Measured reactive power.** An uploaded dataset may carry
`consumption_mvar` / `production_mvar`; then measured reactive power is used
wherever it is given and the power factor only fills gaps. The Data tab shows the
reactive series and a **Data quality** panel (coverage, implausible power factors,
kvar instead of Mvar, missing-value markers…), which CONDUCTOR reads with the
`get_data_quality` tool; the System tab's chip reads "Loads
Q measured · pf 0.95 for gaps". Missing active power is interpolated in time and
reported; units on voltage control at buses with measured reactive generation can
be switched to follow it with one click (or by CONDUCTOR, on request). The header's
Upload Network / Upload Data / Info open as centred pop-ups (controls left, format
guide right).

**Generators.** Each unit's model — voltage control or a fixed power factor,
set-point, reactive limits, dispatchable — is shown and edited in the System tab
("Generators — how each unit is modelled") and by `set_generator_model` (no
arguments: read). Changes apply to every later study and are kept for the
network; "Reset to as loaded" undoes them. The upload dialog's checkbox only sets
the starting model.

**Costs.** Below the generators, a table (each row with its Edit form, like the
generators) of every unit's production
cost (fixed €/h, €/MWh, quadratic €/MW²h), redispatch costs up and down
(€/MWh) and reactive price (€/Mvarh), plus the external grid's free reactive band
(tan φ) and the charge beyond it, each marked as from the case file, the
synthetic rule or the operator;
`set_unit_costs` reads and changes them. Kept for the flexibility OPF's economic
objective; no study uses them yet, and CONDUCTOR says so.

**Names.** The System tab's 🏷️ Names view lists buses, lines, transformers,
loads and units, each with an Edit form to rename it (`rename_elements` does the same on request). Every later result
uses the new names; the data stay matched to the names the network was loaded
with; the conversation is not reset. Earlier answers keep the names they ran
with and say so in a quiet note (a rename changes no figure, so no rerun is
offered); the Data tab shows a renamed bus as "North (data: Bus_3)".

## Network proposals (sandbox)

When a what-if leaves a real decision open — what kind of unit, whether a
conventional generator holds voltage or runs at a fixed power factor, a
capacitor or a reactor — `edit_proposal` returns `needs_operator` with a
question and options, and CONDUCTOR asks the operator (a numbered list) instead
of choosing. Every value the tool fills in is returned as `assumed` and stated
in the answer, so the operator can change it.

`edit_proposal` builds a proposal — the loaded network plus edits (add a load or
a generator, reinforce a line, upgrade a transformer, add a new bus joined by a
line or, at another voltage, by a new transformer, switch an element in or out of
service, change any setting of an existing element, add a line or transformer
between two existing buses, move an element or a line end to another bus, scale
existing loads or units keeping their profile, or copy in another proposal's
edits to build "A + B"); the base
is never modified. Every analysis tool accepts `network` (`"base"` or a proposal
id), and `compare_networks` runs one study on the base and a proposal at the
same operating point — or, with `study="scan"`, at every tick of a window. With
`against` it compares a proposal with another one, and with `proposals` it ranks
up to six against the same reference in one call. The guards carry `network` through a turn like
`timestamp`, flag a turn that mixes base and proposal unless the question
compares them, and append a note to any answer that uses a proposal without
naming it. The advisor's suggestions flag a proposal judged at one moment or
without N-1 and may propose an alternative proposal.

## The advisor — suggestions and discussion

A second agent, run only when the operator asks (`recommender.py`). It never
runs a tool and never produces a figure of its own: every number it writes is
checked against the session's results and flagged when not found. It has two
modes over the same view of the session and the same rules:

The UI keeps the two roles apart: **CONDUCTOR** (the orchestrator, blue) runs
the studies — its chat box is labelled ⚡ CONDUCTOR and sends with **Run**; the
**advisor** (violet, a tinted panel) discusses and suggests and runs nothing. An
answer leads to it ("💬 Discuss with the advisor"); its studies and suggestions
lead back ("Run in CONDUCTOR →" fills CONDUCTOR's box, the operator sends it).
The welcome message gives an example for each.

Both modes live in the **advisor panel** to the right of the conversation, about
the latest answer, switched with ✨ Suggestions / 💬 Discuss; the conversation
on the left stays the record of questions, studies and answers (it shows only
a one-line hint when the agent has something to flag, and decision notes).
The panel starts closed: a slim strip on the right holds its ✨ button,
highlighted when the latest answer has something to flag (the line under the
answer then points to it). Once opened it stays open for the session until
hidden again with its header button.

- **Suggestions** — "Suggest next steps": a short comment and 3–4 next steps;
  "Use" places one in the chat box.
- **Discussion** — an open side thread to think a result through; it needs no
  suggestions first and stays in the panel while studies it proposed are run. It separates evidence
  from opinion, asks about the operator's criteria, argues the other side when
  asked, and proposes studies as cards that go to the chat box like
  suggestions — the operator sends them, and their answers become evidence in
  the thread. "Close and record a decision" writes a decision note (decision,
  evidence, open risks, studies run) into the session log.
- **Before the first question** the panel is already there, for planning: a
  discussion only, which helps frame the question and proposes the first
  studies without saying anything about the grid's state (there are no results
  yet). It carries on beside the first answer; a decision can be recorded once
  there is one.

The deterministic answer checks (provenance, characterisation, completeness)
feed both modes as "answer check: …" concerns. One model call per suggestion
round, per message in a discussion, and per decision note.

## Extending the agent

Tool functions use a `**kwargs` catch-all and spread it into the request body,
so new optional backend parameters flow through with no plumbing changes:

```python
def run_rsa(load_scaling_factor: float = 1.0, **kwargs) -> dict:
    body = {"load_scaling_factor": load_scaling_factor, **kwargs}
    return _record("run_rsa", _post("/api/grid/rsa", body))
```

To add a **new** tool:

1. **`agent/tools.py`** — add a function (accept `**kwargs`, post to the backend,
   record the result, return `{"error": ...}` on failure).
2. **`agent/tool_schemas.py`** — add a `genai.types.FunctionDeclaration` (every
   parameter needs a description with units/range/default) and append it to `TOOLS`.
3. **`agent/renderers.py`** — add a renderer returning a `go.Figure` (or list);
   map the tool name in `RENDERER_MAP`. Handle empty data gracefully.
4. **`agent/system_prompt.py`** *(optional)* — add a behavioral rule if the
   tool's trigger isn't obvious from its name/description.
