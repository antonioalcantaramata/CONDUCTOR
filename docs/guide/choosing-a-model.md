# Choosing a model

CONDUCTOR needs a language model that can **call tools**: every study is a tool
call. You choose the model on the launch screen, grouped in three tiers. The
full list, with each option's limits and sources, is in
[Model options](../reference/providers.md).

| Tier | Options | Good for |
| --- | --- | --- |
| **Free** | Gemini API (free tier, recommended), NVIDIA Build (free endpoints), OpenRouter (free models) | Trying CONDUCTOR on the demos and other sample networks |
| **Paid** | OpenAI, Anthropic (Claude), Gemini paid, OpenRouter paid, DeepSeek, any OpenAI-compatible endpoint (Azure, vLLM, …) | Regular use; real data under terms that fit your organisation |
| **Local** | Ollama, on your own machine | Confidential data: nothing leaves your machine |

## What the option card tells you

Before you start, the card for the selected option shows:

- **Questions per day** (or time per question, or questions left from a
  credit balance) — an estimate. One question takes several model calls, and
  each call carries CONDUCTOR's instructions and tool list (about 28,000
  tokens), so on free tiers the daily or per-minute limit usually binds long
  before the advertised context window. CONDUCTOR measures its own average calls
  per question on your machine and refines the estimate as you use it.
- **Context window** and **maximum per request** that the tier actually allows.
- **Latency** measured on your machine, once there is a measurement.
- **What you need** to sign up, and the steps to get a key.
- **Data** — how the provider handles your prompts. Read it before using real
  network data.

An option whose per-minute or per-request limit is below one CONDUCTOR request
is disabled, with the reason.

## Free options in practice

- **Gemini API · free** — the default. Flash-Lite models allow a few hundred
  requests a day; when the daily quota of the first model runs out, CONDUCTOR
  moves to the next one and says so. The larger *Flash* models are offered as
  "advanced, few questions a day".
- **NVIDIA Build** — free endpoints for NVIDIA's Nemotron models, GLM and
  DeepSeek, after phone verification. The Nemotron models are the fastest;
  some others can take minutes per question.
- **OpenRouter** — many free open models, listed live because they change often:
  only models that are free, call tools and have a large enough context are
  shown, and a model about to be retired carries a badge. 50 requests a day
  (1,000 once you have bought $10 of credit).

Free options are meant for **testing with sample networks** such as the bundled
IEEE case. Some free providers may use prompts to train or improve their models.
For real or confidential grid data, use Local or a paid tier whose terms fit
your organisation; CONDUCTOR reminds you once if you load your own network on a
free tier.

## Settings on the card

- **Model** — pick from the list; **Refresh** reloads the live lists.
- **Thinking** — how much the model reasons before answering. More thinking is
  slower and, on paid tiers, costs more; *Model default* leaves it to the model.
  For OpenRouter, NVIDIA and DeepSeek it is under *Advanced* on the card; for
  Gemini, OpenAI, Anthropic and Ollama it is in the option's settings
  (also under **Model settings** in the sidebar).
- **Test tool calling** — an optional single request that checks the model
  really calls tools. Useful for a new or unfamiliar model.
- **Newer versions** — when a provider has a model newer than the ones
  CONDUCTOR knows, it is shown with a "newer · unconfirmed" badge. It is never
  switched to on its own; you can choose it once a tool-calling test passes.

You can change the model during a session under **Model settings** in the
sidebar; the next question uses it.

## Local models (Ollama)

Install [Ollama](https://ollama.com/download), pull a model that supports tools
(`ollama show <model>` lists `tools` under Capabilities), and pick it under
**Local**. The first question after the model loads is slow (a few minutes on a
laptop), because the model processes CONDUCTOR's instructions once; later
questions take seconds while the model stays loaded. The launch screen suggests a
context size for your machine's memory. Details: the main README's
*Running a local model*.

## When a limit is reached

See [Troubleshooting](troubleshooting.md#limits-and-model-errors). In short:
per-minute limits wait and retry on their own; daily limits and spent credit
stop with a message saying which limit was hit and when it resets; a retired
model is replaced by the next one, and the answer says so. The sidebar shows the
calls made today against the daily limit.
