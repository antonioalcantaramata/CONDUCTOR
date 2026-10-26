# Reading an answer

An answer has three parts: the text, a card saying what backs it, and the
charts.

## The text

Every figure in the text comes from a study result. CONDUCTOR names the moment,
the data and, when it applies, the proposal the answer is about ("On proposal
*dc_bus9* (base + 20 MW data centre at Bus 9), at 18:00…"). When it filled in a
value you did not give, it says so and you can change it.

## What backs this answer

Right under the text, one card; the **(?)** after each of its headings explains
it. Always in view:

- **What backs this answer** — the answer's claims counted by evidence level,
  strongest first: **Proved** (a certificate — "no dispatch costs less than
  22.48 €/h"), **Power flow** (computed by an AC power flow of the network),
  **Optimised** (found by the optimiser, not proved to be the best),
  **Estimated** (over sampled scenarios), **Input** (your question, the data, or
  a setting the study used), and the agent's own **Interpretation** or
  **Advice**. **Unsupported** counts a figure found in no result, and a claim
  worded beyond what backs it: "proved" or "impossible" with no certificate,
  "secure" with no study. **⚠ to check** counts anything the checks flagged.
- **Reading check** — whether the conclusion depends on how your question was
  read. CONDUCTOR re-ran the answer's studies under what the question left open
  (the load power factor, the loading limit, the voltage band, measured or
  forecast data) and compared the conclusions: ✓ it holds, ↗ a figure moves by
  more than 10 %, ⚠ the conclusion changes: the answer depends on a value you
  did not give. When it changes, the row names the
  reading and what it gives — for example *Conclusion changes with loads at power factor
  1.00 instead of 0.95 (the default) — not secure, 1 violation* — and the
  Readings tab lists
  every other reading. Say which reading you meant, or ask again with it;
  CONDUCTOR keeps the finding with the answer, so it knows in your next
  question.
- **Mark figures** colours each figure in the text by its level; hover over one
  for where it came from.
- **Details**, with what ran, how many figures were traced and how long the
  answer took in its label (for example *Details · 2 studies · 8/8 figures
  traced · 13 s*).

Inside Details, **Worth checking** comes first when the checks flagged
anything — a figure in no result, a claim worded beyond its evidence, a study
that ran with unusual parameters, a tool error. Then four tabs:

- **Claims** — each claim and what backs it: the figures grouped by level and
  study (hover over one for the phrase it is in; what the question gave and what
  was assumed are told apart), then the sentences graded by what they assert.
- **Readings** — each other reading of the question and what it gave.
- **How it ran** — where the time went (for example *13 s — model 9.8 s (4
  calls), studies 1.1 s, checks 0.4 s*), then each study with its arguments,
  whether it succeeded and how long it took. A study marked **↺ same call as
  step 2, not run again** repeated an earlier study with the same arguments and
  was answered from it; **carried forward** shows a value set earlier in the
  question and kept for this study. If the model repeats a call that failed
  after being told it would fail the same way, CONDUCTOR stops the question
  there and answers from what is known, with a note saying so.
- **Limits** — what the checks cannot see.

Nothing on the card is model output: it is computed from the studies' results.
It describes, it does not certify: a level says what kind of evidence stands
behind a claim, not that the claim is true (a correct number under the wrong
label, for example). The card is there so you can check.

## The investigation

An answer to a *why* about a problem in the grid has an **Investigation** card
above *What backs this answer*. It has one row per possible cause CONDUCTOR
considered:

- **Supported** or **Ruled out**, with what the studies showed and which
  studies they were;
- **Not tested**, when no tool can test it yet, with what it would need: an
  analysis CONDUCTOR does not do yet, or data it does not have. A cause that
  was not tested is not a finding, and the answer says so.

The agent chooses what to test, but a verdict must cite studies that ran in
that question and agree with what they found, or the record is refused. What
untested causes would need is collected in the sidebar, under **Tools
investigations needed**, in two lists: tools to build and data to integrate.

## The investigation report

Under an answer that investigated something, **Download the investigation
report** saves the whole investigation as a Markdown file:

- the question and the answer;
- the comparison and each factor's exact share and verdict;
- what is inside a group and the causes that interact;
- whether the main cause holds against other references;
- whether a cause looked like a fault in the data;
- for a look-ahead, each expected episode: when to decide, its cause, the
  action and whether it holds if the forecast errs;
- the hypotheses tested and the tools untested ones would need;
- what backs the answer and every study that ran;
- the method and its limits.

It is built from the studies' own results, not rewritten by the model.

## The charts

Each study draws its own charts under the answer: voltage profiles, loadings,
violations, dispatch changes, time series, comparisons. Above each chart a
**caption** names the study, the moment and anything that differs between the
charts of one answer (the power factor, the load scaling, the outage, the
proposal…), so two charts side by side can always be told apart.

An investigation draws a **waterfall of the change** since its reference
state: each step is a factor's share (teal raises the value, purple lowers it),
ending at the total change, with the limit as a dashed red line where it sits
relative to the reference. A group that is a cause, such as the demand near
the element, gets a second chart that divides its share among its members.

A look-ahead draws a **timeline** of the forecast: one row per expected
episode, the warning in amber (the time to decide) before the episode in red,
and a label saying whether an action was found and whether it holds if the
forecast errs (more demand, less renewable output). Hover over an episode for its cause and the action.

Hover over a chart for values; use the toolbar to zoom or download it as an
image.

## Answer checks

Every answer is checked by fixed rules (no second model involved): each figure
is traced to a result, compared with the fields beside it, and the answer is
checked for the kind of quantity the question asked for. What they find is
always shown in the card.

In the sidebar, **Answer checks** has two switches. **Let the agent review its
answer**: off (the default), findings are shown to you only; on, CONDUCTOR sees
them before the answer is delivered and may keep, revise or extend it — one
more model call when something is found. **Check other readings of the
question** (on by default) runs the reading check above: a few seconds of
studies, no model call.

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
a limit reached), the answer says which study failed and why, and the card
counts it under ⚠ to check and shows the failed step under *How it ran*. CONDUCTOR does not invent a result for a study that did
not run.
