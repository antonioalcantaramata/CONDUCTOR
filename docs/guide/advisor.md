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
