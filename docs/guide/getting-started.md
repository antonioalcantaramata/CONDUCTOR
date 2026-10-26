# Getting started

This page assumes CONDUCTOR is installed and running (see the
[main README](https://github.com/antonioalcantaramata/CONDUCTOR#readme): `./start.sh` on macOS and Linux,
`python start.py` on any system). The app opens at <http://localhost:8501>.

## The launch screen

Every start shows a launch screen in two columns.

**1 · Choose what to load**

- **Resume** — your last uploaded network and data. Greyed out until you have
  uploaded something.
- **Demo: IEEE 30 · active only** (the default) — PGLib's IEEE 30-bus network
  (Alsac–Stott) with a synthetic week of active power. Its five units can all
  move their active power, each with its own cost from the case, and hold their
  voltage as the case describes them. The line from the external grid congests
  at the evening peaks, so the flexibility studies have real choices to show.
  Loads take their reactive power from the load power factor. The network's own
  operating point (the *nominal point*) is included.
- **Demo: IEEE 30 · with reactive measurements** — the same network and week,
  with measured reactive demand and generation. A few values are left out on
  purpose, so the data-quality report has gaps to show.
- **Demo: IEEE 14 · smaller** — PGLib's IEEE 14-bus network and its week: one
  unit can move its active power and three condensers hold voltages. The
  what-if evaluation is written for it.

When a session was saved for the network loaded, the launch screen also asks
whether to **continue from it** — see
[Session context](sessions-and-scenarios.md#session-context-continuing-another-day).

The demos are the same every time, so they are a good place to learn and to
compare results. An option you have worked on before says so ("Your changes are
kept: 1 rename · 2 proposals"); see
[Sessions and scenarios](sessions-and-scenarios.md).

**2 · Choose how to run the assistant**

Pick a tier (**Free**, **Paid** or **Local**), an option and a model. Each option
shows a card with what it allows (questions per day, context, speed), what you
need to sign up, and how your data is handled. If you have no key yet, the card
shows the steps to get one. Paste the key once: it is saved on your machine
(`llm_agent/.env`) and reused. See [Choosing a model](choosing-a-model.md).

Click **Start CONDUCTOR**.

## The app

- **Header:** *Upload Network*, *Upload Data* and *Info*.
- **Sidebar:** the **operating point** (which moment of the data the studies
  use; see [Data and the operating point](data-and-operating-point.md)), your
  **proposals**, **settings** (scenario, model, answer checks, advisor),
  **example queries** and *Export session report*.
- **Tabs:**
  - **Chat** — the conversation with CONDUCTOR, and the advisor panel on the
    right;
  - **System** — the network map and the model assumptions (generators, costs,
    load power factor) and names; see [The System tab](system-tab.md);
  - **Data** — the loaded measurements and forecasts, and the data-quality
    report.

## Your first questions

Type into the box labelled **⚡ CONDUCTOR** and press **Run**. Some questions to
start with (the sidebar's *Example queries* has more, matched to what is
loaded):

- *Is the grid secure at the current timestamp?*
- *Find the worst-case timestamp in this week's measurements.*
- *Which single N-1 outage is the most critical right now?*
- *What is the hosting capacity at the most heavily loaded bus?*
- *What if we connect a 20 MW data centre at Bus 9?*

While it works, a status box shows each study as it runs. The answer comes
with a card, **What backs this answer** — the evidence behind each claim,
anything worth checking, and in its details what ran and where each figure
came from — and its charts; see [Reading an answer](reading-an-answer.md).

## Stopping

Press **Ctrl+C** once in the terminal where you started CONDUCTOR. Your
proposals, settings and renames are kept for the next start.
