# Reading an answer

An answer has three parts: the text, the charts, and a collapsed panel that
shows how it was produced.

## The text

Every figure in the text comes from a study result. CONDUCTOR names the moment,
the data and, when it applies, the proposal the answer is about ("On proposal
*dc_bus9* (base + 20 MW data centre at Bus 9), at 18:00…"). When it filled in a
value you did not give, it says so and you can change it.

## The charts

Each study draws its own charts under the answer: voltage profiles, loadings,
violations, dispatch changes, time series, comparisons. Above each chart a
**caption** names the study, the moment and anything that differs between the
charts of one answer (the power factor, the load scaling, the outage, the
proposal…), so two charts side by side can always be told apart.

Hover over a chart for values; use the toolbar to zoom or download it as an
image.

## How this answer was produced

The panel under each answer, collapsed by default. Its label always shows the
counts (tools run, figures traced), any concern and how long the answer took,
so you see a warning without opening it. Inside:

- **Worth checking** — anything the checks flagged.
- **What ran** — where the time went (for example *14 s — model 9.8 s (4 calls),
  studies 3.2 s, checks <0.1 s*), then each study with its arguments, whether it
  succeeded and how long it took. Most of the wait is usually the model; a long
  study shows up here. A step
  marked **↺ same call as step 2, not run again** repeated an earlier study with
  the same arguments, and was answered from it: nothing had changed in between.
  If the model repeats a call that failed after being told it would fail the
  same way, CONDUCTOR stops the question there and answers from what is known,
  with a note saying so.
- **Where the numbers came from** — each figure in the text, traced to the
  study result it came from.
- **What this does not tell you** — what the checks cannot see.

The checks describe; they do not certify. A traced figure can still be the
wrong figure for the question (a correct number under the wrong label, for
example). The panel is there so you can check.

## Answer checks

Every answer is checked by fixed rules (no second model involved): each figure
is traced to a result, compared with the fields beside it, and the answer is
checked for the kind of quantity the question asked for. What they find is
always shown in the panel.

In the sidebar, **Answer checks** has one switch: **Let the agent review its
answer**. Off (the default), findings are shown to you only. On, CONDUCTOR sees
them before the answer is delivered and may keep, revise or extend it — one
more model call when something is found.

## Answers that changed a setting

Most answers change nothing. When one changes a session setting (the load power
factor, a generator's model, a unit's costs, an element's name), a line under it
says exactly what changed, whatever the answer's text says:

> ⚙ Changed the session's settings: load power factor 0.95 (default) → 0.9 · ↶ Undo

**↶ Undo** puts back exactly what that answer changed and nothing else. If you
have changed one of those settings again since, Undo does nothing and says which,
so a change you made by hand is never overwritten. Every later study uses the
restored settings.

## Answers computed under other settings

An answer remembers what it ran under: the load power factor, the generator
models and the loaded data. If one of them changes later, the answer gets a
line such as:

> ⏱ Computed under other settings: load power factor 0.95 → 0.90

with **↻ Rerun with the current settings**, which asks the same question again
as a new turn. The earlier answer stays for comparison.

A **rename** changes no figure, so it gets a quiet note instead, with no rerun:

> 🏷 Elements renamed since — this answer shows the names it ran with

## When something fails

If a study cannot run (a power flow that does not converge, an unknown element,
a limit reached), the answer says which study failed and why, and the panel
shows the failed step. CONDUCTOR does not invent a result for a study that did
not run.
