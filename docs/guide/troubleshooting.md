# Troubleshooting

## Limits and model errors

| What you see | What it means | What to do |
| --- | --- | --- |
| "Waiting for the rate limit" | A per-minute limit was reached | Nothing: CONDUCTOR waits and continues |
| A daily-limit or credit message | The day's requests, or the credit, are used up. The message says which limit and, when known, when it resets | Wait for the reset, or switch to another free option, a paid tier or Local (sidebar **Model settings**) |
| "…was retired / is no longer offered; answered with …" | The model was withdrawn by the provider | Nothing: CONDUCTOR moved to the next model and says so. Choose another model on the next start if you prefer |
| A timeout | The provider did not answer in time (common on busy free endpoints) | CONDUCTOR retries once. If it keeps happening, choose a faster model; NVIDIA's timeout can be raised with `NVIDIA_TIMEOUT_S` |
| An authentication error | The key is wrong, revoked or for another product | Paste a new key in **Model settings** |
| The answer stopped after many steps | A question needed more steps than allowed | CONDUCTOR answers from the results it has. Ask a narrower question, or split it |

The sidebar shows the calls made today against the daily limit (or the credit
left, on DeepSeek).

## The app

| Problem | What to do |
| --- | --- |
| A red box says the backend is unreachable | The analysis service is not running or still starting. Wait a few seconds; if it stays, restart CONDUCTOR |
| Ports already in use | Start with other ports: `BACKEND_PORT=8010 STREAMLIT_PORT=8502 python start.py` |
| Running on a remote machine over SSH | Forward ports 8501 and 8000 to your computer and open <http://localhost:8501> (the launcher prints the steps) |
| A sidebar note says a newer version exists | `git pull`, then start again |
| An answer used the wrong moment or data | Name them in the question ("at 18:00", "over the forecast"); the answer always states what it used |

## Studies

| Problem | What it means |
| --- | --- |
| "The power flow did not converge" | The operating point has no solution: often an extreme load scaling, an islanded part of the network, or a proposal that disconnects something. Try a smaller change or check the proposal |
| "Nominal point not available" | The network file's own values are incomplete; the message lists what is missing |
| An optimisation finds no feasible dispatch | No allowed redispatch removes the violations within the limits. Try relaxing a limit, allowing more units to move (generator models), a robust or N-1 variant, or a proposal that adds capacity |
| A load or unit gets no measured values | Its bus name does not match a `substation_name` in the data. Check the data-quality report, or download the data template for this network |
| A proposal edit is refused | The message says why (e.g. a line between buses of different voltages, a name already in use). CONDUCTOR shows it and suggests what would work |

## Local models (Ollama)

| Problem | What to do |
| --- | --- |
| The first question takes minutes | Expected: the model processes CONDUCTOR's instructions once. Later questions are fast while the model stays loaded (`OLLAMA_KEEP_ALIVE`) |
| "Prompt too large for the context window" | Raise `OLLAMA_NUM_CTX` (the model must support it, and larger windows use more memory), or start a new conversation |
| The model chats but never runs a study | It does not support tool calling. Choose a model that lists `tools` under Capabilities (`ollama show <model>`) |
