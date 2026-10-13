"""
report.py — the session as one standalone HTML file.

For a co-author's review, an appendix, or the record of a decision: what was
loaded (scenario, network, data), the assumptions in force, the proposals, and
the conversation — each answer with its charts and its evidence (which tools
ran with what arguments, where its figures came from, what was not verified).

The evidence follows `trace.py`'s rules: it describes, it never certifies, and
a concern is never left out. Charts arrive as ready HTML from the caller (the
app owns the renderers); plotly.js is embedded once, so the file opens offline.

Pure functions: no Streamlit, no HTTP.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# A small Markdown subset — what the answers use: headings, paragraphs, lists,
# tables, bold / italic / code, links. Everything is escaped first.
# ---------------------------------------------------------------------------

_INLINE = [
    (re.compile(r"`([^`]+)`"), r"<code>\1</code>"),
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])"), r"<em>\1</em>"),
    (re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"), r"<em>\1</em>"),
    (re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)"), r'<a href="\2">\1</a>'),
]


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    codes: list[str] = []

    def keep(m):                       # code spans are not formatted further
        codes.append(m.group(0))
        return f"\x00{len(codes) - 1}\x00"
    out = re.sub(r"`[^`]+`", keep, out)
    for pattern, repl in _INLINE[1:]:
        out = pattern.sub(repl, out)
    out = re.sub(r"\x00(\d+)\x00", lambda m: _INLINE[0][0].sub(_INLINE[0][1], codes[int(m.group(1))]), out)
    return out


def _table(lines: list[str]) -> str:
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]
    head, body = cells(lines[0]), [cells(x) for x in lines[2:]]
    out = ["<table><thead><tr>", *(f"<th>{_inline(c)}</th>" for c in head), "</tr></thead><tbody>"]
    for row in body:
        out += ["<tr>", *(f"<td>{_inline(c)}</td>" for c in row), "</tr>"]
    return "".join(out + ["</tbody></table>"])


def markdown(text: str) -> str:
    """Markdown (the subset answers use) as HTML."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []
    i = 0

    def flush():
        if para:
            out.append(f"<p>{'<br>'.join(_inline(p) for p in para)}</p>")
            para.clear()
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(f"<pre>{html.escape(chr(10).join(block))}</pre>")
            i += 1
            continue
        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            flush()
            level = min(len(m.group(1)) + 2, 6)            # the report's own headings come first
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue
        if re.match(r"([-*+]|\d+[.)])\s+", stripped):
            flush()
            ordered = bool(re.match(r"\d", stripped))
            items = []
            while i < len(lines) and re.match(r"\s*([-*+]|\d+[.)])\s+", lines[i]):
                item = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", lines[i])
                nested = len(lines[i]) - len(lines[i].lstrip()) >= 2
                items.append(f"<li{' class=nested' if nested else ''}>{_inline(item)}</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        if stripped in ("---", "***"):
            flush()
            out.append("<hr>")
            i += 1
            continue
        para.append(stripped)
        i += 1
    flush()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

_CSS = """
:root { --ink:#22252f; --muted:#6b7080; --line:#e3e5ec; --blue:#1a73e8; --purple:#6B4FD8; --warn:#8a5a00; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f7fa; color:var(--ink); font:15px/1.55 -apple-system, "Segoe UI", Roboto,
       "Source Sans Pro", sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 20px 64px; }
header.cover { background:#fff; border:1px solid var(--line); border-radius:14px; padding:24px 28px; }
header.cover h1 { margin:0 0 4px; font-size:1.6rem; letter-spacing:.01em; }
header.cover .sub { color:var(--muted); }
dl.meta { display:grid; grid-template-columns: max-content 1fr; gap:4px 18px; margin:16px 0 0; }
dl.meta dt { color:var(--muted); } dl.meta dd { margin:0; }
section { margin-top: 26px; }
section > h2 { font-size:1.05rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 10px; }
.card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px 20px; margin-bottom:14px; }
.turn .who { font-size:.75rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
.turn.user { background:#eef3fd; border-color:#d6e3fb; }
.turn.user .who { color:var(--blue); }
.turn .who.conductor { color:var(--blue); }
.stale { display:inline-block; font-size:.82rem; color:var(--warn); background:rgba(240,170,30,.12);
         border:1px solid rgba(240,170,30,.35); border-radius:999px; padding:2px 10px; margin:6px 0; }
.chart-caption { color:var(--muted); font-size:.85rem; margin:14px 0 4px; }
.status { padding:8px 12px; border-radius:8px; background:#eef6ee; margin:6px 0; }
.status.error { background:#fdecec; } .status.info { background:#eef3fd; }
.charts { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:10px; }
.charts > div { min-width: 0; }
table { border-collapse:collapse; margin:8px 0; font-size:.88rem; width:auto; max-width:100%; display:block; overflow-x:auto; }
th, td { border:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }
th { background:#f3f4f8; }
code { background:#f1f2f6; padding:1px 4px; border-radius:4px; font-size:.88em; }
pre { background:#f1f2f6; padding:10px 12px; border-radius:8px; overflow-x:auto; font-size:.85rem; }
details.evidence { margin-top:12px; border-top:1px dashed var(--line); padding-top:8px; }
details.evidence summary { cursor:pointer; color:var(--muted); font-size:.88rem; }
details.evidence li { font-size:.86rem; }
.concern { color:#a12a2a; }
.limits { color:var(--muted); font-size:.82rem; }
li.nested { margin-left: 1.2rem; }
footer { color:var(--muted); font-size:.8rem; margin-top:36px; }
@media print { body { background:#fff; } .card, header.cover { break-inside: avoid; } details.evidence { display:block; }
               details.evidence > *:not(summary) { display:block; } }
@media (max-width: 640px) { .charts { grid-template-columns: 1fr; } main { padding: 16px; } }
"""


_RESIZE_JS = """<script>
// Plotly sizes a chart when it is drawn, before the grid has placed it: size
// every chart again once the page is laid out.
window.addEventListener("load", function () {
  if (window.Plotly) document.querySelectorAll(".js-plotly-plot").forEach(function (el) { Plotly.Plots.resize(el); });
});
</script>
"""


def _meta_rows(meta: dict) -> str:
    rows = [(k, v) for k, v in (
        ("Generated", meta.get("generated")),
        ("Scenario", meta.get("scenario")),
        ("Network", meta.get("network")),
        ("Data", meta.get("data")),
        ("Assistant model", meta.get("model")),
    ) if v]
    return "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in rows)


def _assumptions_html(assumptions: dict | None, quality: str | None) -> str:
    if not assumptions:
        return "<p>Not available (the backend could not be read).</p>"
    pf = assumptions.get("load_power_factor") or {}
    items = [f"<li><strong>Load power factor</strong>: {pf.get('value')} (lagging, {html.escape(str(pf.get('source')))})"
             "</li>"]
    model = assumptions.get("generator_model") or {}
    if model:
        units = "; ".join(f"{html.escape(u)}: {html.escape(json.dumps(c, default=str))}" for u, c in model.items())
        items.append(f"<li><strong>Generators modelled differently from the network as loaded</strong>: {units}</li>")
    else:
        items.append("<li><strong>Generators</strong>: as loaded</li>")
    measured = (assumptions.get("reactive_measured") or {}).get("measurements") or {}
    if measured.get("consumption") or measured.get("production"):
        which = [w for w, on in (("demand", measured.get("consumption")), ("generation", measured.get("production")))
                 if on]
        items.append(f"<li><strong>Measured reactive power</strong>: {' and '.join(which)} — used wherever given; "
                     "the power factor fills the rest</li>")
    if quality:
        items.append(f"<li><strong>Data quality</strong>: {html.escape(quality)}</li>")
    return f"<ul>{''.join(items)}</ul>"


def _proposals_html(proposals: list | None) -> str:
    if not proposals:
        return "<p>None — every answer is about the network as loaded unless it says otherwise.</p>"
    out = []
    for p in proposals:
        summary = p.get("summary")
        if isinstance(summary, dict):
            summary = summary.get("edits")
        lines = summary if isinstance(summary, list) else [summary] if summary else []
        edits = "".join(f"<li>{html.escape(str(x))}</li>" for x in lines) or \
            "".join(f"<li><code>{html.escape(json.dumps(e, default=str))}</code></li>" for e in p.get("edits") or [])
        out.append(f"<div class='card'><strong>{html.escape(str(p.get('id')))}</strong><ul>{edits}</ul></div>")
    return "".join(out)


def _evidence_html(trace) -> str:
    """The audit panel, as text: describes, never certifies."""
    if trace is None:
        return ""
    steps = "".join(
        f"<li><code>{html.escape(s.name)}</code>"
        f"{' — ' + html.escape(json.dumps(s.args, default=str)) if s.args else ''}"
        f"{'' if s.ok else ' — <span class=concern>failed: ' + html.escape(str(s.error)) + '</span>'}"
        f"{' · carried forward: ' + html.escape(s.inherited_summary) if s.inherited_summary else ''}</li>"
        for s in trace.steps)
    figures = "".join(
        f"<li>{html.escape(f'{f.value:g}')}{(' ' + html.escape(f.unit)) if f.unit else ''} — "
        f"{html.escape(f.status)}{(' (' + html.escape(f.source) + ')') if f.source else ''}</li>"
        for f in trace.figures)
    concerns = "".join(f"<li class='concern'>{html.escape(c)}</li>" for c in trace.concerns)
    limits = "".join(f"<li>{html.escape(x)}</li>" for x in trace.limits)
    parts = []
    if steps:
        parts.append(f"<p><strong>Tools that ran</strong></p><ol>{steps}</ol>")
    if figures:
        parts.append(f"<p><strong>Figures in the answer and where they came from</strong></p><ul>{figures}</ul>")
    if concerns:
        parts.append(f"<p><strong>To check</strong></p><ul>{concerns}</ul>")
    if limits:
        parts.append(f"<p class='limits'><strong>What these checks cannot see</strong></p><ul class='limits'>{limits}</ul>")
    return (f"<details class='evidence'><summary>Evidence · {html.escape(trace.headline())}</summary>"
            f"{''.join(parts)}</details>")


def build(messages: list[dict], meta: dict, *, charts_html=None, plotly_js: str = "",
          stale=None) -> str:
    """The whole report as one HTML document.

    `messages`: the conversation as the app stores it. `meta`: generated,
    scenario, network, data, model, assumptions, quality, proposals.
    `charts_html(tool_results) -> str`: the charts of one answer. `plotly_js`:
    the plotly.js source, embedded once. `stale(msg) -> list[str]`: what has
    changed since an answer (agent/staleness.py).
    """
    turns = []
    for msg in messages:
        if msg.get("role") == "user":
            turns.append(f"<div class='card turn user'><div class='who'>Question</div>{markdown(msg.get('text', ''))}"
                         "</div>")
            continue
        body = ["<div class='who conductor'>CONDUCTOR</div>", markdown(msg.get("text", ""))]
        why = stale(msg) if stale else []
        if why:
            body.append(f"<div class='stale'>⏱ Computed under other settings than the ones above: "
                        f"{html.escape('; '.join(why))}</div>")
        if charts_html and msg.get("charts"):
            body.append(charts_html(msg["charts"]))
        body.append(_evidence_html(msg.get("trace")))
        turns.append(f"<div class='card turn'>{''.join(body)}</div>")
    title = meta.get("title") or "CONDUCTOR session"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
{f'<script>{plotly_js}</script>' if plotly_js else ''}
</head><body><main>
<header class="cover"><h1>{html.escape(title)}</h1>
<div class="sub">Grid studies with CONDUCTOR — every figure from a solver run, with its evidence under each answer.</div>
<dl class="meta">{_meta_rows(meta)}</dl></header>
<section><h2>Assumptions in force</h2><div class="card">{_assumptions_html(meta.get('assumptions'), meta.get('quality'))}</div></section>
<section><h2>Proposals</h2>{_proposals_html(meta.get('proposals'))}</section>
<section><h2>Conversation</h2>{''.join(turns) or '<p>No questions yet.</p>'}</section>
<footer>The evidence describes where each figure came from; it does not certify that an answer is right. Exported
{html.escape(meta.get('generated') or datetime.now().strftime('%Y-%m-%d %H:%M'))}.</footer>
</main>
{_RESIZE_JS}</body></html>
"""
