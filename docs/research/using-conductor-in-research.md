# Using CONDUCTOR in research

CONDUCTOR separates the work cleanly: **solvers compute**, the **language model
chooses studies and writes the answer**, and **fixed rules check the answer
against the results**. Everything below follows from that split: you can
reproduce the studies, see exactly what the model did, and measure how faithful
its answers were.

## Reproducible starting points

- **The demos are deterministic.** Both IEEE 14-bus demos (PGLib case 14 with a
  synthetic week; the second adds measured reactive power with deliberate gaps)
  are generated with fixed seeds, so the same scenario gives the same data on
  every machine. Name the scenario in your method section.
- **Your own data.** Upload a network (MATPOWER, pandapower, UCTE) and a CSV of
  measurements and forecasts; see
  [Data and the operating point](../guide/data-and-operating-point.md).
  A *Start fresh* on the launch screen returns a network to its as-loaded state
  between experiments.
- **Settings are recorded.** Each result states the operating point, the data
  source, the limits, the load power factor and any generator models that differ
  from the network as loaded. Each answer is stamped with these, and the session
  report lists them.

## Session logs

Every question and answer is appended to a JSON Lines file in
`llm_agent/session_logs/` (one file per run of the app; the folder can be changed with
`CONDUCTOR_SESSION_LOG_DIR`). Each turn record holds:

| Field | Content |
| --- | --- |
| `user`, `assistant` | The question and the final answer |
| `provider`, `model`, reasoning settings | Which model answered, and how |
| `tool_calls`, `tool_results` | Every study run, with its arguments and full result |
| `grid` | The network facts the model was given that turn |
| `model_calls`, `prompt_tokens`, `duration_s` | Cost and time of the turn |
| `status`, `error_classification` | How the turn ended |
| `reflection` | When answer review was on: the checks' findings and what the model did with them |

Advisor decision notes and suggestions are recorded in the same file, with a
`kind` field. Logs stay on your machine and are not part of the repository.

## Grading sessions

The `evaluation` package grades logs after the fact, with fixed rules — no model
acting as judge, nothing labelled by hand:

```bash
python -m evaluation                 # the newest session log
python -m evaluation path/to/log.jsonl
python -m evaluation --all           # every log in the log folder
python -m evaluation --json          # machine-readable output
```

The exit status is 1 when any turn has a failing finding, so it can gate a run.
Per turn it reports the figures in the answer and how each was grounded:

| Check | Finds |
| --- | --- |
| **Provenance** | Figures that appear in no study result, argument or user text (*ungrounded*); real figures attached to the wrong element (*misattributed*); right figure, wrong precision (*imprecise*); figures the model computed itself with the operands shown (*derived*) |
| **Characterisation** | A correct figure describing an action the result marks as impossible (e.g. a "reduction" flagged as infeasible) |
| **Completeness** | The quantity the question asked for, missing from an answer when it was available |
| **Integrity** | Answers that present results from different operating points (another moment, other data) as one assessment; answers that mix the base and a proposal unasked, or do not name their proposal |
| **Refusal** | Requests a study marks as unsatisfiable (e.g. an unsupported combination of analyses), answered anyway with an improvised result |
| **Proposals** | Edits the backend refused; proposals judged at a single moment or without an N-1 screen (warnings: a quick question may legitimately be answered at one moment) |

## The what-if evaluation

A fixed set of proposal questions for the bundled IEEE 14-bus network, run
repeatedly against the real agent and graded from the records:

```bash
python -m evaluation.whatif list
python -m evaluation.whatif run --repeats 10 [--tasks dc_flat,new_site] [--follow-suggestion]
python -m evaluation.whatif grade <run folder>
```

It measures how often the agent builds the change that was described, names the
proposal, looks beyond one moment, screens N-1 when asked, and passes on
modelling notes. `--follow-suggestion` repeats the run following the advisor's
suggestions, to measure what they change. **`run` uses model credit** and asks
for confirmation; `grade` costs nothing. Runs are written to their own folder,
never to the session logs.

## Comparing models

The launch screen offers free, paid and local models (see
[Model options](../reference/providers.md)). Because every log records which
model answered, the same questions can be graded across models and providers.
For a fair comparison, keep the scenario, the operating point and the settings
fixed, and report the model, its reasoning setting and the date (providers
change models without notice).

## What the checks can and cannot see

State these limits alongside any result produced with the checks:

- A traced figure can still be the wrong figure for the question: provenance
  checks that a number came from a result, not that it was the right number to
  quote.
- A misleading description with no field in the result to contradict it is
  invisible to the characterisation check.
- A missing quantity is often the correct answer (the study failed or does not
  support the question), so completeness findings are warnings only.
- The checks see the answer and the results, not the operator's intent: a
  well-grounded answer to a misunderstood question passes.

## Citing

Please cite the paper describing CONDUCTOR,
[arXiv:2606.24609](https://arxiv.org/abs/2606.24609):

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

The paper describes an **earlier version** (June 2026). CONDUCTOR has gained many
capabilities and changes since — network proposals and their comparison, the
advisor, free, paid and local model options, measured reactive power and data
quality, generator models, unit costs, renaming and more — so results from the
current version may differ from those reported there. Alongside the citation,
name:

- the version (the git commit);
- the scenario or data;
- the model, its provider and its reasoning setting;
- whether answer review was on.
