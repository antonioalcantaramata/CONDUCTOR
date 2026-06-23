# CONDUCTOR — LLM chat agent (`llm_agent/`)

A Streamlit chat app plus the LLM agent that drives the
[backend](../backend/) power-system tools from natural language. The agent
calls Gemini, which decides which backend endpoints to invoke, then summarizes
the results in text while the app renders the matching charts.

> See the [root README](../README.md) for the work-in-progress notice, full
> setup, and how to launch (`./start.sh` from the project root starts both this
> app and the backend). This document covers the agent internals.

## Layout

```
llm_agent/
├── app.py                 ← Streamlit UI: sidebar clock/scrubber, chat, charts
├── agent/
│   ├── config.py          ← env vars + live grid-constants fetch
│   ├── loop.py            ← the agentic turn loop (LLM ⇄ tool calls)
│   ├── tools.py           ← Python functions that call the backend over HTTP
│   ├── tool_schemas.py    ← Gemini function declarations (TOOLS)
│   ├── renderers.py       ← Plotly renderers, keyed by tool name (RENDERER_MAP)
│   ├── system_prompt.py   ← system instruction, built from live grid constants
│   └── errors.py          ← error classification / retry helpers
├── assets/                ← logo
├── requirements.txt       ← pip deps (installed by start.sh)
└── .env.example           ← copy to .env and add your key
```

## Environment variables

Set these in `llm_agent/.env` (copy from `.env.example`) or as shell vars.

| Variable               | Required      | Default                   | Description                                  |
| ---------------------- | ------------- | ------------------------- | -------------------------------------------- |
| `GEMINI_API_KEY`     | **Yes** | —                        | Google Gemini API key                        |
| `GEMINI_MODEL`       | No            | gemma-4-31b-it            | Model id to use                              |
| `DT_BACKEND_URL`     | No            | `http://localhost:8000` | FastAPI backend base URL                     |
| `DT_HTTP_TIMEOUT`    | No            | `120.0`                 | Per-request HTTP timeout (s)                 |
| `DT_MAX_AGENT_TURNS` | No            | `20`                    | Max tool-calling iterations per user message |

## How a turn works

1. **Tool call** — Gemini decides to call e.g. `run_rsa`. The matching function
   in `agent/tools.py` makes an HTTP request to the backend and appends
   `("run_rsa", result_dict)` to the module-level `_last_tool_results`.
2. **Response** — Gemini reads the JSON result, reasons over it, and writes a
   text summary (the LLM never describes charts in prose).
3. **Charts** — after the text renders, `app.py` walks `_last_tool_results`,
   looks up each tool's renderer in `renderers.py`'s `RENDERER_MAP`, and draws
   the Plotly figures. `_last_tool_results` is cleared at the start of each turn.

## Measurements vs forecasts

Every analysis tool accepts a `data_source` parameter — `"measurements"`
(historical actuals, the default; drives the simulation clock) or `"forecasts"`
(read-only look-ahead). Point-in-time tools also accept an optional `timestamp`.
The agent picks the source from the user's intent and reports it in its answer.

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
