# Sessions and scenarios

## Scenarios

A **scenario** is what CONDUCTOR loads: **Resume** (your last uploaded network
and data) or one of the **demos** (IEEE 30-bus, with or without reactive
measurements, and the smaller IEEE 14-bus). Choose it on the launch screen
(it loads at once), or later in the sidebar under **Settings → Scenario** with
**Load**, which starts a new conversation (the old one was about another grid).

Demos never replace your uploads: Resume always loads what you uploaded last.
Uploading a network or data makes Resume the current scenario.

## What is kept, and where

Each network keeps its own work:

- proposals;
- renamed elements;
- unit costs;
- generator models;
- the load power factor.

Load the network again (or the demo) and they come back. The two demos use the
same network, so they share these.

The launch screen says when a scenario has saved work:

> 🗂 Your changes are kept: 1 rename · 1 cost change · 2 proposals

**Start fresh**, under the scenario list, clears that work for the selected
scenario's network: names, costs, generator models and the power factor go back
to the network as loaded, and its proposals are discarded. It asks for
confirmation and cannot be undone.

All of this is saved on your machine, in CONDUCTOR's `systems/` folder. It is not
part of the code repository: a fresh copy of CONDUCTOR starts with the demos as
shipped. Uploaded data are saved in `data_files/`, also only on your machine.

## Conversations

- **New conversation** (sidebar) clears the chat and the advisor's discussions;
  the loaded network, data, proposals and settings stay.
- Changing the model during a conversation keeps the conversation.
- Renaming elements keeps the conversation; CONDUCTOR uses the new names from
  then on.

## Session context: continuing another day

A conversation ends when the app closes; what a network keeps (above) does not
include what was asked, found and decided. A **session context** does.

**Save** — in the sidebar, **Session context → Save** writes one Markdown file:

- what was loaded and the settings in force;
- every question, the studies that ran with their key arguments, and their
  verdicts, with the start of each answer;
- the decision notes, with the option chosen and its figures **copied from the
  solver result**, what mattered, and the open risks;
- the studies the advisor proposed that were never run.

CONDUCTOR writes all of this from the session's records — nothing in it is
invented. **Add a summary by the model** (on by default; one model call) puts a
short summary on top, labelled as the model's; any figure in it that the
session's results do not contain is flagged. The file is kept for the network
(in `llm_agent/session_contexts/`, only on your machine) and **Download the
file** gives you a copy to keep or share.

**Continue** — on the next start, the launch screen asks: *Continue from an
earlier session on this network?* It lists the contexts saved for the network
loaded, newest first, and starts **without** one unless you choose. You can
also load a context file, here or later under **Session context** in the
sidebar; one written for another network says so.

**What the agents do with it** — both CONDUCTOR and the advisor receive it as
background from an earlier session: they build on it (*"last time you ruled
out Gen_bus1"*) but do not treat it as evidence. Its figures may no longer hold,
so a figure is stated as current only after a study reruns it; the answer
checks treat a figure that only the earlier session contains as untraced. Each
answer's record notes the context it was given. **Stop using it** in the same
menu returns to a plain session.

The context reaching the agents is capped at about 12,000 characters (around
3,000 tokens, so it fits small local models): a long session keeps its newest
turns and every decision.

## The session report

**Export session report** (sidebar) offers two HTML files that open offline in
any browser.

**Raw — the complete record:**

- what was loaded — scenario, network, data, model, assumptions, data quality
  and proposals;
- every question and answer as given, with its interactive charts and the
  evidence ("How this answer was produced");
- notes on answers computed under other settings.

**Tailored by the model — for readers who were not there** (one model call,
**Write it**): a summary, the context, the findings grouped by theme rather
than in the order asked, the decisions and the open questions, each point
citing the turn it rests on (*T3*). What stays CONDUCTOR's:

- the charts are the session's own, placed by the model by reference — never
  redrawn;
- every figure the model writes is checked against the session's results; any
  it cannot trace is listed at the top of the report;
- it says it was written by a model, names which, and points to the raw report
  as the complete record.

Both are built when you ask, and contain your network and results: share them
like any other study document.

## Session logs

Each question and answer is also written to a local log file
(`llm_agent/session_logs/`), including which model answered. These are for
review and research; see
[Using CONDUCTOR in research](../research/using-conductor-in-research.md).
