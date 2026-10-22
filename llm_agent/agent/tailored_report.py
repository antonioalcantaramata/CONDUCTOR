"""
tailored_report.py — the session written up by the model, for readers who were not there.

The raw report (report.py) is the record: every question and answer verbatim,
with charts and evidence. This one is written by the model from that record —
a summary, the context, the findings grouped by theme, the decisions, what
remains open — each point citing the turn it rests on (T3).

Two things stay CONDUCTOR's. The charts are the session's own, placed by
reference (`{{chart T3}}` on its own line), never redrawn. And every figure
the model writes is checked against the session's results; any that cannot be
traced is listed at the top of the report. It says it was written by a model,
and that the raw report is the complete record.

Pure logic apart from the one model call in `write`.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime

from . import usage

logger = logging.getLogger(__name__)

MAX_CHARTS = 6
_MAX_ANSWER_CHARS = 3_000          # per turn, in what the model reads
_MAX_INPUT_CHARS = 60_000          # the newest turns are kept when a session is longer
_CHART_RE = re.compile(r"\{\{\s*chart\s+T(\d+)\s*\}\}", re.IGNORECASE)

SYSTEM_PROMPT = """\
You write the report of a CONDUCTOR session for colleagues and co-authors who \
were not there. CONDUCTOR is a decision-support system for power-system \
operators: validated solvers run every study, and the record below lists each \
turn (T1, T2, …) — the question, the studies with their key results, and the \
answer that was given.

Write the report in Markdown, with these sections:
## Summary — 3 to 5 sentences: what was asked, what was found, what was decided.
## Context — the network, the data and the settings the studies ran under.
## Findings — grouped by theme (### headings), not by turn; summarise repeated \
or exploratory turns into what they established.
## Decisions — what the operator decided, and on what evidence; "None recorded" if none.
## Open questions — what the studies did not settle, and studies proposed but not run.

Rules:
- Every finding cites the turns it rests on, e.g. (T3) or (T2, T5).
- Figures only from the record, copied or rounded as the record shows them; \
never compute, estimate or extrapolate one. Every figure is checked against \
the session and the ones not found are flagged to the reader.
- Keep the record's caveats: a result at one moment is about that moment; a \
dispatch not screened under N-1 is not N-1 secure; a proposal is not the \
loaded network; a refused or failed study is reported as such.
- Recommend nothing the session did not decide; open questions are questions.
- To show a chart from the session, put `{{chart T3}}` on its own line where it \
supports a finding (the turn's charts are inserted there); at most 6, only \
turns marked "charts: yes".
- Plain, precise language; no marketing tone; no emoji.
"""


def turn_records(messages: list[dict]) -> list[tuple[int, dict, dict]]:
    """(turn number, the answer message, its record) for every answer with a
    record — numbered as the digest and the advisor number them (T1, T2, …)."""
    out = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("record"):
            out.append((len(out) + 1, msg, msg["record"]))
    return out


def _args(args) -> str:
    """A call's arguments, compact; empty ones left out."""
    import json
    shown = {k: v for k, v in (args or {}).items() if v not in (None, "", [], {})}
    text = json.dumps(shown, ensure_ascii=False, default=str, separators=(", ", ": "))
    return text if len(text) <= 600 else text[:597] + "..."


def request_text(messages: list[dict], meta: dict, notes: list[dict] = ()) -> str:
    """What the model reads: the session's context, then each turn."""
    from .recommender import turn_digest
    head = ["# Session record",
            f"- Network: {meta.get('network') or 'unknown'}" + (f" ({meta['scenario']})" if meta.get("scenario") else ""),
            f"- Data: {meta.get('data') or 'unknown'}",
            f"- Model that answered: {meta.get('model') or 'unknown'}"]
    for line in meta.get("settings") or []:
        head.append(f"- Setting: {line}")
    for prop in meta.get("proposals") or []:
        head.append(f"- Proposal: {prop}")
    turns = []
    for n, msg, record in turn_records(messages):
        answer = " ".join((msg.get("text") or "").split())
        if len(answer) > _MAX_ANSWER_CHARS:
            answer = answer[:_MAX_ANSWER_CHARS] + " […]"
        turns.append("\n".join([
            f"## T{n}",
            f"Question: {record.get('user') or ''}",
            "Studies: " + turn_digest(record, n).split(" · ", 2)[-1],
            # Every argument, so a difference between two runs of the same
            # question (a safety margin, a load level) can be explained.
            *(f"- {call.get('name')} arguments: {_args(call.get('args'))}" for call in record.get("tool_calls") or []),
            f"Charts: {'yes' if msg.get('charts') else 'no'}",
            f"Answer: {answer}"]))
    tail = []
    if notes:
        tail.append("## Decision notes")
        for note in notes:
            tail.append(f"- Decision: {note.get('decision')}")
            chosen = note.get("chosen_option")
            if chosen:
                tail.append(f"  - Option chosen: {chosen.get('ref')} ({chosen.get('label')}), "
                            f"{chosen.get('cost_eur_per_h')} €/h, units called {chosen.get('units_called')}")
            for key, label in (("criteria", "What mattered"), ("evidence", "Evidence"), ("open_risks", "Open risks")):
                if note.get(key):
                    tail.append(f"  - {label}: " + "; ".join(str(x) for x in note[key]))

    def assemble(kept: list[str], omitted: int) -> str:
        return "\n\n".join(["\n".join(head)] + ([f"({omitted} earlier turns omitted)"] if omitted else [])
                           + kept + (["\n".join(tail)] if tail else []))

    text, omitted = assemble(turns, 0), 0
    while len(text) > _MAX_INPUT_CHARS and omitted < len(turns):
        omitted += 1
        text = assemble(turns[omitted:], omitted)
    return text


def write(text: str, provider=None) -> tuple[str, str]:
    """One tool-less model call. Returns (Markdown, error)."""
    from .providers import get_provider, user_message
    provider = provider or get_provider()
    try:
        response = provider.chat(messages=[user_message(text)], tools=[], system=SYSTEM_PROMPT)
        usage.count_call(provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tailored report call failed.", exc_info=True)
        return "", f"The model could not write the report: {exc}"
    body = (response.text or "").strip()
    body = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", body)        # a fenced reply
    return (body, "") if body else ("", "The model returned an empty report.")


def unverified_figures(markdown_text: str, records: list[dict]) -> tuple[str, ...]:
    """Figures in the report that no result of the session contains."""
    from .recommender import _figure_check
    try:
        return tuple(_figure_check(records)(_CHART_RE.sub("", markdown_text)))
    except Exception:  # noqa: BLE001 — shown unmarked rather than lost
        logger.warning("Could not check the tailored report's figures.", exc_info=True)
        return ()


def render(markdown_text: str, meta: dict, *, charts_by_turn: dict, plotly_js: str = "",
           unverified: tuple = (), questions: list[tuple[int, str]] = ()) -> str:
    """The report as one HTML page, in the raw report's style. `charts_by_turn`:
    turn number -> that turn's charts as HTML."""
    from .report import _CSS, _RESIZE_JS, _meta_rows, markdown

    placed: list[int] = []

    def chart(match: re.Match) -> str:
        turn = int(match.group(1))
        if turn in placed or len(placed) >= MAX_CHARTS:
            return ""
        if not charts_by_turn.get(turn):
            return f"<p class='muted'>(No chart for T{turn}.)</p>"
        placed.append(turn)
        return (f"<figure class='card'>{charts_by_turn[turn]}"
                f"<figcaption>Chart from T{turn}, as the session drew it.</figcaption></figure>")

    body = markdown(markdown_text)
    # The shared converter moves headings two levels down (an answer's headings
    # sit under the raw report's own); here they are the report's sections.
    for low, high in (("h4", "h2"), ("h5", "h3"), ("h6", "h4")):
        body = body.replace(f"<{low}", f"<{high}").replace(f"</{low}>", f"</{high}>")
    body = re.sub(r"<p>\s*" + _CHART_RE.pattern + r"\s*</p>", chart, body, flags=re.IGNORECASE)
    body = _CHART_RE.sub(chart, body)
    title = meta.get("title") or "CONDUCTOR session report"
    flags = ""
    if unverified:
        flags = (f"<div class='stale'>⚠️ Figures in this report that the session's results do not contain: "
                 f"{html.escape(', '.join(unverified))}. Check them against the raw report.</div>")
    asked = "".join(f"<li><b>T{n}</b> {html.escape(q)}</li>" for n, q in questions)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}
figure {{ margin: 1rem 0; }} figcaption {{ color: #6b7280; font-size: 0.85rem; margin-top: 0.3rem; }}
section > h2 {{ margin-top: 28px; }} section > h3 {{ font-size: 1.05rem; margin: 18px 0 6px; }}
.muted {{ color: #6b7280; }}</style>
{f'<script>{plotly_js}</script>' if plotly_js else ''}
</head><body><main>
<header class="cover"><h1>{html.escape(title)}</h1>
<div class="sub">Written by {html.escape(meta.get('model') or 'the model')} from the session's record. Figures are
checked against the session's results and charts are the session's own; the raw report is the complete record.</div>
<dl class="meta">{_meta_rows(meta)}</dl>{flags}</header>
<section>{body}</section>
<section><h2>Questions asked</h2><div class="card"><ul class="asked">{asked or '<li>None.</li>'}</ul></div></section>
<footer>A summary written by an AI model; the session's raw report holds every question, answer, chart and its
evidence. Exported {html.escape(meta.get('generated') or datetime.now().strftime('%Y-%m-%d %H:%M'))}.</footer>
</main>
{_RESIZE_JS}</body></html>
"""
