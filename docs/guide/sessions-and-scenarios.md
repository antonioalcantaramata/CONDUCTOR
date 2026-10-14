# Sessions and scenarios

## Scenarios

A **scenario** is what CONDUCTOR loads: **Resume** (your last uploaded network
and data) or one of the two IEEE 14-bus **demos**. Choose it on the launch screen
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

## The session report

**Export session report** (sidebar) downloads one HTML file that opens offline in
any browser:

- what was loaded — scenario, network, data, model, assumptions, data quality
  and proposals;
- every question and answer, with its interactive charts and the evidence
  ("How this answer was produced");
- notes on answers computed under other settings.

It is built when you click, and contains your network and results: share it
like any other study document.

## Session logs

Each question and answer is also written to a local log file
(`llm_agent/session_logs/`), including which model answered. These are for
review and research; see
[Using CONDUCTOR in research](../research/using-conductor-in-research.md).
