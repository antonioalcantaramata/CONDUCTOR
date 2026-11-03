# Using CONDUCTOR in research

CONDUCTOR separates the work cleanly: **solvers compute**, the **language model
chooses studies and writes the answer**, and **fixed rules check the answer
against the results**. Everything below follows from that split: you can
reproduce the studies, see exactly what the model did, and measure how faithful
its answers were.

## Reproducible starting points

- **The demos are deterministic.** The IEEE 30-bus demos (PGLib's Alsac–Stott
  case with a synthetic week; the second adds measured reactive power with
  deliberate gaps) and the CIGRE MV distribution demo are generated with fixed
  seeds, so the same scenario gives the same data on every machine. Name the
  scenario in your method section. The 30-bus demo is the one for flexibility
  studies: five dispatchable units with the case's own costs, and congestion at
  the evening peaks.
- **Your own data.** Upload a network (MATPOWER, pandapower, UCTE, CGMES) and a CSV of
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
| `tool_calls`, `tool_results` | Every study run, with its arguments and full result. A call repeated with identical arguments (and nothing changed in between) is not run again: it carries `reused_from` (the earlier call's number) and its result points to the earlier one |
| `grid` | The network facts the model was given that turn |
| `model_calls`, `prompt_tokens`, `duration_s` | Cost and time of the turn |
| `timing` | Where the time went: `model_s` (every model call, including rate-limit waits, flagged by `waited_for_rate_limit`), `studies_s` (tools running), `checks_s` (CONDUCTOR's own checks), `total_s`; each call in `tool_calls` also carries its `duration_s` |
| `status`, `error_classification` | How the turn ended: `completed`; `max_turns_summarised` (the step limit was reached); `repeated_failure_summarised` (stopped after the model repeated a failed call it had been told would fail again); or an error |
| `reflection` | When answer review was on: the checks' findings and what the model did with them |

A turn given an earlier session as background records it in `earlier_context`
(the context's title and save time); saving and loading a context are recorded
as `context_saved` and `context_loaded` events. For experiments, compare runs
with and without a context only as a deliberate condition.

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

A fixed set of proposal questions for the CIGRE MV distribution demo (load the
**Demo: CIGRE MV · distribution with DER** scenario first), run repeatedly against
the real agent and graded from the records. The sizes are distribution sizes, and
two tasks check what only a radial network asks: that a site on one line, or a
feeder with a line opened, is said to lose supply.

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

## The options evaluation

A fixed set of questions about corrective options for the IEEE 30-bus demo
(the default scenario), graded the same way:

```bash
python -m evaluation.options list
python -m evaluation.options run --repeats 10 [--tasks options_peak,rule_out_unit]
python -m evaluation.options grade <run folder>
```

It measures whether the agent asks the solver for options when the operator
wants a choice; keeps a unit the operator rules out in service with no
instruction (`hold_units`, not taken out of service); asks for near-optimal
alternatives — and the margin the operator names — when cost matters but
another way is wanted, and says what each alternative avoids or adds; holds a
line below the share of its rating the operator names (`branch_limits`); asks
for the N-1 security study (`n1`) when the options must stay secure after an
outage — corrective when the operator would act after it — and says what it
could not secure and the corrective plan; presents the last resort (demand
response, load relief) when redispatch alone cannot clear the violations;
holds a chosen option over the next hours; plans a peak as one window; answers
"no action needed" when the schedule meets every limit; and leaves the choice
to the operator. That last check is a narrow phrase check ("I recommend…",
"the best option is…"): it catches a choice stated outright, not every way of
steering. Like the what-if evaluation, **`run` uses model credit** and asks
for confirmation; runs go to their own folder.

## The clarification evaluation

Whether the agent asks before running when it should — and only then:

```bash
python -m evaluation.clarify list
python -m evaluation.clarify run --repeats 10 [--tasks which_line,secure_now]
python -m evaluation.clarify grade <run folder>
```

Two tasks leave something open that changes the answer (an outage of "the line
into Bus_23", which three lines fit; options keeping a line "comfortably" below
its rating): the agent must ask first — a question, numbered choices, nothing
run but lookups — and, given the answer, run the study it chose. Three are
clear enough to run at once ("is the grid secure right now?", options secure
after a line trips, the most critical outage): asking there fails, so an agent
that asks about everything scores as badly as one that guesses. The rule's
examples in the system prompt are kept apart from the tasks' wording. **`run`
uses model credit** and asks for confirmation.

## The investigation evaluation

Two parts. The **bench** needs no model and costs nothing. It injects known
causes into the IEEE 30-bus demo as proposals and asks the
[investigation](../methods/investigation.md) why the element they affect
changed, against the base:

- six causes alone: a load scaled up, a data centre added, a unit's output or
  voltage set-point lowered, a line opened, the external grid's voltage lowered;
- four pairs of causes, two of them interacting;
- two with the agent's own hypotheses: two loads the standard grouping joins,
  and a wrong hypothesis that must be ruled out;
- two other scopes: the whole grid with two causes, and a cause after an outage;
- three causes with background, compared with the base an hour earlier, so the
  demand and the units' schedule moved too.

```bash
python -m evaluation.investigation bench [--json]
```

It runs against the running backend and removes the proposals it creates
(their ids start with `bench_`). Per case it reports:

- whether the main cause is the injected one (for a group, whether its
  drill-down puts the injected member first);
- the recall of the injected causes;
- any other factor blamed;
- the injected causes' share of the change;
- whether the two states are reproduced exactly.

It also scores three baselines on the same cases:

- the attribution's top driver, what the agent could say before the
  investigation existed;
- undoing each factor alone: its top-1, and what its effects add up to as a
  share of the change, which exceeds 100 % when the causes interact;
- the classical linear estimate (sensitivities, LODF for a branch out): its
  top-1, its mean error against the exact shares, and the cases where it has no
  estimate for an injected cause.

The **agent suite** spends model credit:

```bash
python -m evaluation.investigation list
python -m evaluation.investigation run --repeats 10 [--tasks why_bus29,why_method]
python -m evaluation.investigation grade <run folder>
```

Its tasks check that a why about a problem in the grid is investigated (and
recorded), and that a why about a method, a recommendation or physics, or a
what-if, is not. Two tasks test the agent's freedom. In one, it must split two
loads into its own hypotheses and record both as supported. In the other, a
hypothesis no tool can test (the weather) must be recorded as untested, with
the tool it would need, and said to be untested in the answer. Others check
the scopes (an outage, a window, a comfortable reference, the classical views),
that a recurring problem is investigated episode by episode, and that a question
about prevention is answered with the prevention check and its warning time.
One asks what to watch for in the next day of the forecast: it must look ahead,
say when to decide, and say that the episodes are expected, not observed.
Two are asked with Deep Research switched on: a *why*, which must go into depth
(prevention, recurrence, the record), and an options question, which must be
answered as an analysis.

The tools investigations lacked, across sessions or runs, are ranked by:

```bash
python -m evaluation.tool_requests                 # the local requests file
python -m evaluation.tool_requests --logs <folder> # the records of session logs or runs
```

They are listed in two parts: **tools to build** (an analysis of the grid
CONDUCTOR could compute and does not, such as a voltage stability margin) and
**data to integrate** (an input it does not have, such as the weather).

They also check that a symptom that is not there is said rather than
explained, that several violations with none named lead to a question, and
that a cause injected through a proposal is found and named.

## The narration evaluation

How to report six studies is not in the system prompt: it comes with each
study's result (`how_to_report`), so a question pays for it only when the study
ran. The narration suite checks that the answer follows it, with one task per
study:

- the hosting capacity with its glossary of reactive modes;
- the probabilistic assessment leading with the probability of any violation
  and saying what uncertainty was modelled;
- the robust dispatch's out-of-sample risk, and what the back-off did or why it
  fell short;
- the flexibility envelope at this operating point, with its binding
  constraint;
- the historical risk's near misses and worst episodes;
- the renewable scenario scan against its baseline, with the limits of uniform
  scaling.

```bash
python -m evaluation.narration list
python -m evaluation.narration run --repeats 2 [--tasks hosting,robust]
python -m evaluation.narration grade <run folder>
```

## Ablations

An ablation runs the same tasks with a part of CONDUCTOR taken away, and
grades every arm the same way. One command per study:

```bash
python -m evaluation.ablation list                         # the studies, suites, knobs, tool groups
python -m evaluation.ablation plan components --repeats 5  # what it would run and roughly cost (free)
python -m evaluation.ablation run components --repeats 5   # run it (spends model credit; asks first)
python -m evaluation.ablation status <run folder>          # how far a run is
python -m evaluation.ablation run --resume <run folder>    # continue an interrupted run
python -m evaluation.ablation analyse <run folder> [<run folder> …] [--baseline full]   # free
```

### Studies that come ready

Every task these studies run, with its question, set-up and checks, is written
out by `python -m evaluation.catalogue`, from the same definitions the runner
uses.

The first arm of each is its baseline.

| Study | Question | Arms | Suites |
| --- | --- | --- | --- |
| `components` | What each part adds over the rest (leave one out), and against an agent with solvers only | full; without the semantic layer, the reading check, the routing rules, the evidence wording, the ask-first rule, the investigation tools; solver-only (no semantics for the agent or the studies, no reading check) | all |
| `ontology` | What the semantic layer is worth (below) | on, off, none | semantics, reasoning |
| `answer_review` | Do the answer checks, shown to the agent, ground its answers better? | off, inform | narration, options, what-if |
| `reading_check` | What re-running the decisive studies under the question's other readings changes | on, off | clarify, narration, options |
| `prompt_rules` | What each behavioural rule of the prompt is worth | full; without each of six rules | clarify, narration, options, investigation, what-if |
| `tools` | What each capability adds, on the tasks that need it | all; without investigation, proposals, element lookup, studies over time | investigation, what-if, semantics, options |
| `deep_research` | Deep Research as each task asks, and never | task, off | investigation |
| `advisor` | Following the AI Agent Suggestion after a what-if | alone, with the suggestion | what-if |
| `step_budget` | Model calls per question: quality against cost | 16, 8, 4 | investigation, options, what-if |
| `effort` | The configured model's reasoning effort | medium, low, high | all |

### Arms of your own

`--arm` adds an arm to a study, or makes one with `--suites`:

```bash
python -m evaluation.ablation run --suites options,narration --repeats 3 \
    --arm "full:" \
    --arm "lean: drop_rules=routing,evidence_wording hide_tools=group:kpis" \
    --arm "small: model=<another model> effort=low"
```

| Knob | Values | What it changes |
| --- | --- | --- |
| `ontology` | `on`, `off`, `none` | the semantic layer, for the agent (`off`) and also for the studies (`none`) |
| `review` | `off`, `warn`, `inform` | the answer checks shown to the agent before its answer goes out |
| `reading_check` | `on`, `off` | the reading check |
| `deep` | `task`, `on`, `off` | Deep Research on the first question |
| `follow_suggestion` | `off`, `on` | one more turn, following the AI Agent Suggestion |
| `drop_rules` | parts of the system prompt | rules and sections left out; `python -m evaluation.ablation prompt` lists them with their size, `prompt --drop <ids> --show` prints what goes |
| `hide_tools` | tool names, `group:<name>` | tools the model is not given (groups: investigation, proposals, optimisation, over_time, kpis, semantic_layer, lookup); a call to one by name is refused |
| `max_steps` | an integer | model calls per question before it must answer |
| `provider`, `model`, `effort` | the provider's values | the model, for comparing models in one study |
| `env.<NAME>` | any | any other environment variable |

A setting that names nothing (a prompt part that is not in today's prompt, a
tool that does not exist, a misspelt knob) stops the run before it starts:
an ablation that silently removed nothing would read as a null result.
Prompt parts are found by the words they begin with, not by their number, and
`tests/test_ablation_framework.py` checks each against the real prompt.

### How a run is made

- Each arm runs in its own process with its settings in the environment, so
  even the settings read once at start (the model, the provider) differ
  between arms. Every arm uses an isolated backend
  (`evaluation.isolated_backend`; one per kind an arm needs), never the
  operator's. Session logs, the cache and the saved state go to the run
  folder.
- `--order interleaved` (the default) runs repeat 1 of every arm, then repeat
  2, and so on. A model or an API that drifts during a long run then affects
  every arm alike. `blocked` runs one arm after the other.
- The scenario is loaded afresh before each task, and what a task set up is
  undone after it.
- Each run's record names its arm and a hash of the arm's settings.
  `--resume` keeps the runs made under the same settings and runs the rest.
- **When something fails.** A run fails when the infrastructure fails, not
  the agent: the harness itself, or the model's API (quota, rate limit,
  timeout, server error, an answer the model could not write). A failed run
  is left out of the grading, counted, and run again on resume. Stopping at
  the step limit is the agent's own outcome, and is graded. After three
  failed runs in a row, an arm's process stops. If its backend has died, the
  backend is restarted and the arm retried once; otherwise the run stops,
  with the manifest saying why. An exhausted quota stops it at once.
- **Running a part, again or later:**

  ```bash
  python -m evaluation.ablation status <folder> --detail          # every run a resume would run, and why
  python -m evaluation.ablation run --resume <folder>             # the rest of the plan
  python -m evaluation.ablation run --resume <folder> --only full --suites reasoning   # part of it now
  python -m evaluation.ablation run --resume <folder> --repeats 5 --reason "…"         # extend the plan
  python -m evaluation.ablation run --resume <folder> \
      --rerun "arm=full task=names_cigre repeat=2" --reason "…"   # these runs again
  ```

  A re-run moves the selected records aside to `runs/_replaced/<time>/`,
  never deleting them, and runs those runs again. The manifest's change log
  keeps each re-run, extension, backend restart and stop, with its reason.
  The analysis, the error injection and the catalogue are separate commands
  that call no model, and can be run again at any time.
- `plan` (and `run`, before it asks) shows the agent turns per arm and an
  estimate of tokens and time. The estimate comes from CONDUCTOR's usage
  record for the configured model or, more closely, from earlier runs
  (`--estimate-from <folder>`). Only the records of earlier runs say how much
  of the input came from the provider's prompt cache. The share is large
  (97 % on OpenAI: the system prompt and the tools go out unchanged with every
  call), and cached input is billed at a fraction of the price.
- CONDUCTOR knows no prices, so `plan` shows no cost unless you give your
  model's prices in USD per million tokens: `--price-in` (uncached input),
  `--price-cached` and `--price-out`, from the provider's price list or your
  bill. A cost without the cached price, or from records that do not say what
  was cached, is labelled an upper bound.
- `--max-turns` refuses a larger plan, and `--per-suite N --seed S` draws a
  pilot of N tasks from each suite. A pilot run first gives the records a
  close estimate needs.

### What it writes

The run folder holds the records (`runs/<arm>/<suite>/<task>-<repeat>.jsonl`)
and `manifest.json`. The manifest records:

- the date, the commit, and whether the tree had uncommitted changes;
- each arm's settings and hash, with the prompt characters each dropped part
  removed and the tools it hid;
- the suites, the tasks, the repeats and the order;
- any pilot sample, and the command.

The analysis grades every arm with the suites' own checks and every grader of
`python -m evaluation`. It reports:

- **checks passed** and **tasks with every check passed**, each with a 95 %
  Wilson interval;
- the **difference from the baseline**, averaged over the tasks both arms
  ran, with a 95 % bootstrap interval over tasks (runs of one task are not
  independent, so the task is the unit) and a two-sided sign test over tasks;
- the **cost per run**: tool calls, model calls, input tokens (and how many
  of them were cached), output tokens, seconds;
- **the reading check**: how many turns it re-checked, and how often another
  reading moved the figure by more than 10 % (shifts) or changed the
  conclusion (changes);
- **ungrounded figures** and **entity errors** per run;
- a table **by suite** (and by family), and every **check where the arms
  differ**.

It writes `analysis.md`, `summary.csv`, `runs.csv`, `checks.csv` (one row per
run and check, the long form for a plot), `summary.tex` (a booktabs table)
and `results.json`. `analyse` can merge folders, for example an arm added
later. It grades with today's checks, so a grader fix applies to old records
too; say so when you report it.

### The semantic layer

The `ontology` study measures the semantic layer
([What the network is](../methods/what-the-network-is.md)), in three arms
(`CONDUCTOR_ONTOLOGY`):

| Arm | The agent sees the layer | The studies use it |
| --- | --- | --- |
| `on` | yes | yes — CONDUCTOR as it is |
| `off` | no (`describe_network`, `declare_network_facts`, the facts `locate_network_element` adds and the prompt's rule are hidden) | yes |
| `none` | no | no — as before the layer: every static generator is renewable in scaling, the probabilistic and robust studies and the look-ahead; declarations change nothing |

and three comparisons: `on` vs `off` (what the agent gains), `off` vs `none`
(what grounding the studies gains), `on` vs `none` (the whole). Answers are
checked against the layer in every arm.

```bash
python -m evaluation.ablation run ontology --repeats 3 [--only on,off]
```

`none` gets its own backend, started with `CONDUCTOR_ONTOLOGY=none`, and an
arm refuses a backend in the wrong mode. Besides the general tables, its
analysis adds one per network and per reasoning family for each comparison.
It runs two suites:

- **`evaluation.semantics`** — looking up what elements are, on CIGRE MV (the
  network states nearly everything) and IEEE 30 (it states almost nothing, so
  the right answer is often that it does not say): storage, voltage levels,
  renewables, feeders, what an outage cuts off, what carries the external
  grid's power;
- **`evaluation.reasoning`** — reasoning from the facts:
  - *misleading names* (a CHP unit renamed "Solar Farm North": the stated type
    must win over the name);
  - *intervention* (the same question before and after a unit is declared
    wind: the answer, and the study's own scaling, must change with the fact);
  - *composition* (what a transformer outage cuts off, how much of it is solar,
    which tie restores it, and whether that switch is normally open);
  - *switches* (which switches are normally open, and which one ties two
    feeders);
  - *data* (is a value measured, a share of its bus's measurement, or derived
    from the power-factor assumption?);
  - *consistency* (one question in three wordings; a fact stated in one turn
    used in the next).

Every check is a fact of the network or of the study's own record, computed
rather than judged, and `tests/test_ablation_ground_truth.py` checks those
facts against the networks, so a change to a network or its data cannot leave
the ablation grading against a stale answer. Each suite also runs on its own
(`python -m evaluation.semantics run --network cigre --arm off`,
`python -m evaluation.reasoning run --arm on --out <folder>`).

## Error injection: what the checks catch

Labelling the checks' findings by hand measures how often a flag is right. It
cannot measure how many errors the checks miss: an error nobody flagged is
invisible in the records. Error injection measures that. It takes real
answers from any run folder, puts one known error into a copy of each, runs
the same checks the live system runs, and counts an error as caught only when
the check flags *that* error (its figure or its sentence):

```bash
python -m evaluation.injection <run folder> [<run folder> …] [--seed 0] [--out <folder>]
```

| Level | Error put into the answer | Check |
| --- | --- | --- |
| Figures | a figure changed by a few per cent | provenance |
| Figures | a violated element's figure replaced by another violated element's | provenance (misattribution) |
| Figures | an insecure assessment called secure | characterisation |
| Figures | a dispatch "proved" optimal with no certificate; "secure" with no study | evidence |
| Entities | a unit called the wrong source; one whose source is unknown given one | entities |
| Entities | a bus at a voltage the network does not have | entities |
| Entities | a line called a transformer, before and after its name | entities |
| Entities | a value called measured that was derived from an assumption | entities |
| Entities | a real element's name replaced by one the network does not have | entities |

- **True statements are injected too, as controls.** These use the same
  wording, but are true ("PV 3 is a solar PV unit", where it is). A check
  that flags them raises false alarms, and their rate is reported beside the
  recall.
- **Some errors are undetectable by construction**, and are counted apart:
  - a changed figure that another result of the question holds anyway;
  - a swapped figure that the sentence's own element also holds.
- **The unmodified answers are checked as well,** so a change to a check that
  raises new flags on real answers is seen.
- **Each injected error follows what its check reads,** so the table reports
  the reach of each check on real answers.
- **The facts are rebuilt as the live system builds them** for the edited
  answer. That includes the network's full list of names, needed to tell an
  invented name from a real one.

It costs nothing (no model is called), is deterministic for a seed, and writes
`injection.md`, `injection.csv` and `injection.json`.

Run on the pilot runs' records, it found four gaps in the checks, all fixed
and pinned by tests in `tests/test_injection.py`:

1. Names with underscores (`Gen_bus12`, `Bus_25`) were never matched.
2. A sentence closed by markdown after its full stop ran on into the next one.
3. A branch named by its short name ("Line 1-2") was not recognised.
4. Element names written with a space were hidden from the misattribution
   check.

When it is used for a paper, report it on records other than those it was
developed on.

## Comparing models

The launch screen offers free, paid and local models (see
[Model options](../reference/providers.md)). Because every log records which
model answered, the same questions can be graded across models and providers.
For a fair comparison, keep the scenario, the operating point and the settings
fixed, and report the model, its reasoning setting and the date (providers
change models without notice). An ablation with one arm per model
(`--arm "a: model=…" --arm "b: provider=… model=…"`) does all of that, on the
same tasks, interleaved.

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
