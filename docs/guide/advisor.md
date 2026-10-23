# The advisor

CONDUCTOR has two roles, kept apart on screen:

- **CONDUCTOR** (blue) runs the studies. Its box is labelled **⚡ CONDUCTOR**
  and sends with **Run**.
- **The advisor** (violet) discusses results and suggests next steps. It runs
  no studies and produces no figures of its own: every number it writes is
  checked against the session's results and flagged when it is not found
  there.

The advisor lives in a panel to the right of the conversation. It starts closed:
a slim strip holds its button, highlighted when the latest answer has something
worth flagging (for example, a proposal judged at a single moment, or without
an N-1 screen). Once opened, it stays open until you hide it.

## 💬 Discuss

The default mode: an open conversation about the latest answer, to think a
result through.

- It separates **evidence** (what the studies showed) from **opinion**.
- It asks about your **criteria** — cost, margin, number of actions, risk — when
  they matter for the decision.
- It can **argue the other side** when you ask it to.
- It proposes **studies** as cards. **Run in CONDUCTOR →** puts one in
  CONDUCTOR's box; you send it, and its answer becomes evidence in the
  discussion.
- **📝 Close and record a decision** writes a decision note (the decision, the
  evidence, open risks and the studies run) into the session log.

Before your first question, the panel is already there for planning: it helps
frame the question and proposes first studies, without saying anything about
the grid's state (there are no results yet).

You can also start a discussion from an answer with **💬 Discuss with the
advisor**.

### Discussing corrective options

Under an answer with [corrective options](asking-questions.md#corrective-options),
**💬 Discuss these options** opens the advisor on them. The options exist because
the solver cannot see everything you weigh: how far you trust a unit today, how
quickly someone answers, contracts, the cost of each instruction. The advisor
helps with those; the solver made and scored every option.

- It opens with a neutral reading of the trade-offs, in the solver's figures,
  and two or three questions about these options — for example, when several
  lean on the same unit, how far you trust that unit today. It does not
  recommend one yet.
- It never picks an option on its own. Asked for its view, it gives it as its
  view, against what you said matters, quoting each option's figures.
- What you say matters is noted under its reply (*Noted as what matters to you*)
  and goes into the decision note.
- A condition the solver can act on becomes a study at once — "I don't trust
  Gen_bus1 today" becomes **Options without Gen_bus1**: the same question with
  that unit held at its schedule, run by CONDUCTOR and back in the discussion
  as a new set of options.
- Before you settle on one, it can propose the N-1 screen of that option
  (*Run the N-1 screen with option S2:cheapest applied*) — the plain N-1
  screen covers only the schedule.
- When cost matters but a unit is doubted, it can propose the
  [near-optimal alternatives](asking-questions.md#near-optimal-alternatives):
  almost as cheap, changing other units.
- When a loading worries you, it can propose the same options with that line
  [held below a share of its rating](asking-questions.md#keeping-a-line-below-a-share-of-its-rating);
  when security after an outage matters, the
  [N-1 secure options](asking-questions.md#options-secure-after-an-outage-n-1).
  Discussing those, it compares their N-1 screens and what security costs
  against the cheapest, says which outages each secures, and asks whether a
  corrective plan can be carried out in time — a corrective option relies on it.
- An option that reduces load asks customers, not units, to act: it says so, and
  asks whether you can call that demand response (or accept relief).
- Once you lean towards one, it can propose what follows from it:
  [how long it holds, or the next action from it](asking-questions.md#carrying-an-option-forward),
  or a [plan for the whole peak](asking-questions.md#over-a-window-a-plan-for-the-peak-and-the-cost-of-security).
- Options are named by turn and strategy, *T3·cheapest*, so a name keeps its
  meaning after a re-solve adds more.
- When you close with a decision, the note records the option chosen with its
  figures **copied from the solver result**, not written by the advisor; an
  option name the conversation does not have is flagged.

## ✨ Next steps

**✨ Suggest next steps** gives a short comment on the latest answer and three or
four next steps. **Run in CONDUCTOR →** puts one in the box to edit and send.
**✨ More ideas** asks for another round.

By default suggestions are made only when you ask. In the sidebar,
**Advisor suggestions → Suggest after every answer** makes them automatic, at
the cost of one extra model call per answer.

## Cost

One model call per suggestion round, per message in a discussion, and per
decision note. These count towards your model's limits like any other call.
