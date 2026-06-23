# Power-System Digital Twin — LLM Chat Interface

## What this is

A conversational AI assistant layered on top of the existing **power-system
digital twin (DT)**. It lets operators and researchers query the DT
in plain language — running security assessments, contingency analyses, flexibility
optimizations, and KPI evaluations — without touching the Dash frontend or the FastAPI
backend.

The chat interface runs as a **parallel Streamlit app**. Both the Streamlit assistant
and the existing Dash frontend talk to the same FastAPI backend simultaneously.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) | Used to manage the backend Python environment |
| Google AI Studio API key | Free at [aistudio.google.com](https://aistudio.google.com) |

---

## First-time setup (new machine)

```bash
# 1. Clone / download the project, then open a terminal in the project root
cd /path/to/CONDUCTOR

# 2. Create the backend conda environment (takes 3–5 minutes, once only)
conda env create -f environment.yml
#    → creates the 'pyopt' env with FastAPI, pandapower, pyomo, etc.

# 3. Create the .env file with your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > llm_agent/.env

# 4. Make the launcher executable (once)
chmod +x start.sh

# 5. Run everything
./start.sh
```

The `start.sh` script will:
- Automatically detect if the `pyopt` conda env is missing and create it from `environment.yml`
- Install the Streamlit agent packages into your conda base env if needed (streamlit, plotly, httpx, etc.)
- Start the FastAPI backend and the Streamlit chat UI
- Open the browser automatically on macOS

> **Note:** If `./start.sh` creates the conda env for you (step 2 is skipped above), it will take a few extra minutes on the very first run.

---

## Subsequent runs

```bash
./start.sh
```

That's it. Press `Ctrl+C` to stop both services.

---

## Batch evaluation from Excel

For prompt-regression testing, use the headless evaluator at the repo root:

```bash
python evaluate_agent_from_excel.py \
   --agent llm_agent \
   --input prompt_catalog.xlsx \
   --sheet "Prompt Catalog" \
   --output-dir eval_results/catalog_run_01
```

The evaluator can read either:
- a flat machine-friendly sheet with a normal header row
- the richer prompt catalog used in this repository, where the real header row appears below title rows

Recommended normalized columns:
- `prompt` (required)
- `case_id`
- `conversation_id` for explicit multi-turn scenarios
- `start_timestamp` to reset the backend state before a case
- `reset_history` to clear chat history inside a conversation
- `expected_tools`
- `forbidden_tools`
- `must_include`
- `must_not_include`
- `expected_behavior`
- `system_prompt_mode` with `default` or `empty`

For the included prompt catalog, the script also infers follow-up conversations from prompts such as `Re-run that...`, `Same bus...`, or `For the same window...`. The normalized execution plan is written to `planned_cases.csv` before any LLM calls.

Outputs under the chosen result directory:
- `planned_cases.csv` — normalized cases, inferred conversation grouping, and run selection
- `raw_attempts.jsonl` — append-only raw records per attempt, including prompt, assistant output, turn record, and serialized post-turn history for later LLM judging
- `results.csv` — latest result per case
- `results.xlsx` — same rolled-up results when an Excel writer is available
- `summary.json` — aggregate counts, infra-error counts, and output paths
- `turns.jsonl` — lightweight turn-record view for quick inspection

Useful modes:

```bash
# Parse and plan only; do not call the LLM
conda run -n pyopt python evaluate_agent_from_excel.py \
   --agent llm_agent \
   --input prompt_catalog.xlsx \
   --sheet "Prompt Catalog" \
   --dry-run

# Resume an existing run and fill only missing or infra-failed cases
conda run -n pyopt python evaluate_agent_from_excel.py \
   --agent llm_agent \
   --input prompt_catalog.xlsx \
   --sheet "Prompt Catalog" \
   --output-dir eval_results/catalog_run_01 \
   --only-missing

# Re-run one specific case and keep prior conversation context when available
conda run -n pyopt python evaluate_agent_from_excel.py \
   --agent llm_agent \
   --input prompt_catalog.xlsx \
   --sheet "Prompt Catalog" \
   --output-dir eval_results/catalog_run_01 \
   --case-id 40

# Ablation: run with an empty system instruction instead of the normal prompt
conda run -n pyopt python evaluate_agent_from_excel.py \
   --agent llm_agent \
   --input prompt_catalog.xlsx \
   --sheet "Prompt Catalog" \
   --system-prompt-mode empty
```

Infrastructure failures such as provider rate limits, temporary overload, or backend connectivity issues are recorded as `infra_error` and excluded from auto-pass/fail counts. This keeps model-quality metrics separate from API availability. Use deterministic columns like `expected_tools` and `must_include` for fast auto-scoring, and keep `expected_behavior` for later LLM judging or manual review.

---

## Getting a Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with a Google account
3. Click **Get API key** → **Create API key**
4. Copy it into `llm_agent/.env`:
   ```
   GEMINI_API_KEY=AIzaSy...
   ```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | **Yes** | — | Google Generative AI API key |
| `DT_BACKEND_URL` | No | `http://localhost:8000` | FastAPI backend base URL |
| `GEMINI_MODEL` | No | `gemini-2.0-flash` | Gemini model name |
| `DT_HTTP_TIMEOUT` | No | `120.0` | HTTP timeout in seconds |
| `DT_MAX_AGENT_TURNS` | No | `12` | Max agentic loop iterations per turn |

All variables can be set in a `.env` file in the `llm_agent/` directory or as
shell environment variables.

---

## How charts work

The LLM never describes charts — it summarizes key findings in text. Charts are
rendered automatically by Streamlit using this pipeline:

1. **Tool call:** Gemini decides to call e.g. `run_rsa`. The tool function in
   `agent/tools.py` makes an HTTP request to the FastAPI backend and appends
   `("run_rsa", result_dict)` to the module-level `_last_tool_results` list.

2. **Agent response:** Gemini receives the JSON result, reasons over it, and writes
   a text summary (violations detected, which buses, how much redispatch etc.).

3. **Chart injection:** After the text is rendered, `app.py` calls
   `render_charts_for_turn()`, which iterates `_last_tool_results`, looks up the
   matching renderer in `agent/renderers.py`'s `RENDERER_MAP`, and calls
   `st.plotly_chart()`.

4. `_last_tool_results` is cleared at the start of every new agent turn, so each
   response only triggers its own charts.

---

## How to extend tools — the `**kwargs` pattern

All tool functions use a `**kwargs` catch-all and spread it into the HTTP request body:

```python
def run_rsa(load_scaling_factor: float = 1.0, **kwargs) -> dict:
    body = {"load_scaling_factor": load_scaling_factor, **kwargs}
    result = _post("/api/grid/rsa", body)
    ...
```

When the backend gains a new optional parameter (e.g. `tap_override`):
1. Add it to the function signature with a default value in `agent/tools.py`.
2. Add it to the Gemini schema in `agent/tool_schemas.py` so the LLM knows about it.
3. The `{**required, **kwargs}` pattern passes it through automatically — no changes
   needed in `agent/loop.py` or `app.py`.

---

## How to add a new tool

1. **`agent/tools.py`** — Add a new function following the existing pattern:
   - Accept `**kwargs` and pass them into the request body.
   - Append `(function_name, result)` to `_last_tool_results` before returning.
   - Return a structured `{"error": ...}` dict on failure.

2. **`agent/tool_schemas.py`** — Add a `genai.protos.FunctionDeclaration`:
   - All parameters need a `description` with physical meaning, range, unit, default.
   - Include the `options` catch-all object parameter.
   - Add the new declaration to `TOOLS` and map the name to the function in
     `TOOL_DISPATCH`.

3. **`agent/renderers.py`** — Add a renderer function:
   - Accept `result: dict`, return `go.Figure` (or `list[go.Figure]`).
   - Handle empty/missing data gracefully (return `_empty_figure(...)`).
   - Add the tool name → renderer mapping to `RENDERER_MAP`.

4. **`agent/system_prompt.py`** *(optional)* — Add a behavioral rule if the new
   tool's trigger condition is non-obvious from its name and description.

---

## Relationship to the Dash app

Both apps are pure **HTTP consumers** of the same FastAPI backend. Running them
simultaneously is safe — the backend's `app_data` singleton is process-level and
effectively read-only from both frontends' perspective (the only mutation is the
simulation clock, which is shared, so advancing the clock in one UI will be reflected
in the other on the next query).

```
FastAPI backend (localhost:8000)
        ├── Dash app  (localhost:8050)   ← existing frontend, do not modify
        └── Streamlit (localhost:8501)   ← this chat interface
```
