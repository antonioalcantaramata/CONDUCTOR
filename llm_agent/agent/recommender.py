"""
recommender.py — the AI Agent Suggestion.

A second agent, run only when the operator asks for it. It reads the
conversation, a digest of every earlier turn and the latest turn in full, gives
a short comment — its reading of what happened — and proposes three or four
ways to continue: going deeper, testing robustness, trying a change to the
system, or stepping back to see something the operator has not asked about.

It is deliberately not a tool, and it runs nothing. Tool output in CONDUCTOR is
solver output; this is model reasoning, and the interface labels it as such.
Each suggestion is a self-contained prompt the operator can edit and send to
the orchestrator, which then runs the real analysis. The suggestion agent
therefore never produces an engineering figure of its own — the only numbers
it may write are ones already in the conversation, and those are traced with
the same provenance check the answers get.

Same provider and model as the orchestrator, called with no tools.

Pure logic apart from the single model call in `suggest`: the prompt, the
compact turn summary, parsing, grounding and the attention check are all
testable without a provider or a browser.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, NamedTuple

from .provenance import MISATTRIBUTED, UNGROUNDED, check_answer, collect_sources, subject_names

logger = logging.getLogger(__name__)

ANGLES = ("deeper", "stress", "change", "step_back")
ANGLE_LABELS = {
    "deeper": "Deeper",
    "stress": "Stress",
    "change": "Change",
    "step_back": "Step back",
}
MAX_SUGGESTIONS = 4

# Clock control is left out of what the agent is told it can suggest: a
# suggestion names its timestamp explicitly, so the operator's view of the
# simulation clock is never moved by following one.
_EXCLUDED_CAPABILITIES = frozenset({"get_current_timestamp", "advance_timestamp"})

# How much of the conversation the agent sees. Older turns matter less than
# the cost of sending them on every request.
_MAX_MESSAGES = 10
_MAX_MESSAGE_CHARS = 2500
# Every earlier turn, one line each, built in code from its tool results — so a
# long session is not forgotten beyond the last few messages. Bounded: the
# newest lines are kept when it grows past this.
_MAX_DIGEST_CHARS = 6000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Suggestion(NamedTuple):
    angle: str
    title: str
    prompt: str
    why: str
    # Figures in `why` that appear nowhere in this conversation's evidence,
    # rendered for display ("1.07 p.u."). Empty when everything traced.
    unverified: tuple[str, ...] = ()

    def as_record(self) -> dict:
        return {
            "angle": self.angle,
            "title": self.title,
            "prompt": self.prompt,
            "why": self.why,
            "unverified": list(self.unverified),
        }


class SuggestionResult(NamedTuple):
    suggestions: tuple[Suggestion, ...]
    # The agent's reading of what happened, a few sentences, shown above the
    # suggestions. Opinion, labelled as such; its figures are checked like `why`.
    comment: str = ""
    comment_unverified: tuple[str, ...] = ()
    # The agent's own explanation when it has nothing useful to suggest.
    note: str = ""
    # Set when the call or the parse failed; the UI shows a short message,
    # never the raw model output.
    error: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# The "change" angle describes the levers that exist: operating changes, and
# network proposals (the sandbox) — the base network plus edits.
_CHANGE_ANGLE = (
    "a what-if worth studying with the levers CONDUCTOR has: scaling load, "
    "scaling renewable output, disabling or derating generators, capping the "
    "external-grid import, adding injection at a bus (hosting capacity), "
    "re-dispatching flexibility, or a network proposal — the loaded network "
    "plus edits: add a load (flat; follow_bus — the demand shape of its own bus or, with "
    "`follow`, of another bus, since a new bus has no measured demand; or shift), add a "
    "generator (dispatchable, pv or wind), reinforce a line, upgrade a transformer, "
    "add a new bus joined by a line, add a new bus at another voltage behind a new "
    "transformer (optionally after a new line; parallel units for redundancy), or switch "
    "a branch or unit in or out of service"
)

# One agent, two modes: suggestions (a comment and 3–4 next steps, on click)
# and discussion (a side thread with the operator under an answer). Both see
# the same session and follow the same rules; each adds its task.

_ROLE = """\
You are the advisor of CONDUCTOR, a decision-support system for power system \
operators. A separate orchestrator agent answers the operator's questions by \
running validated power-system tools. You do not run tools and you do not \
compute anything. {task_sentence}
"""

_INPUTS = """\
You receive:
- the most recent messages of the conversation;
- a digest of every earlier turn: the question, the tools run with their key \
arguments, and the verdicts they returned — use it to see patterns across a \
long session;
- a compact record of the latest turn (tools run, their arguments, key \
results, and any concerns the system raised);
- the session context (network, simulation time, data source, proposals);
- the list of capabilities the orchestrator can run;
- {task_input}.
"""

_SHARED_RULES = """\
Always:
- Do not state engineering figures except those that appear in the turn \
records or the conversation, copied exactly. Never estimate, extrapolate or \
calculate. Figures you write are checked against the session, and any that \
are not found are flagged to the operator.
- Never claim the result of a study that has not been run.
- Propose analyses and options to explore; do not instruct operational \
actions. Never write "you should curtail…" or "dispatch…". Write "Study…", \
"Compare…", "Check whether…".
- Everything you propose must be answerable with the listed capabilities, \
written as a complete, self-contained request the orchestrator can execute: \
name the timestamp, the data source, the elements and any proposal by name. \
Never rely on "the same as before" or "that bus".
- Match the operator's language.
- A bus in `voltage_controlled_buses` is held at its setpoint by a unit: its \
voltage is not a weak spot. Its margin is that unit's `q_headroom_up_mvar`.
- Concerns that start with "answer check:" are about how the answer was \
written, found by deterministic checks — not about the grid. Say plainly what \
each means for the operator (e.g. a figure the tools never produced) and, \
where a study would settle it, propose that study; never restate an \
unverified figure as if it were confirmed.
- A result carrying a `network` block was computed on a network proposal, \
not on the loaded network; name the proposal whenever you refer to it.
- To look beyond one moment at a proposal, `compare_networks` with study \
"scan" compares it with the base at every tick of a window — by default the \
whole dataset, so it needs no timestamp (give start_timestamp / end_timestamp \
only to narrow it). Study "rsa" and "n1" each cover one moment; never ask \
them for a window.
"""

_SUGGEST_TASK = """\
First, write a short "comment": 2 to 4 sentences giving your reading of \
what happened in the latest turn and what it means for the operator — \
whether the result answers the question, what is surprising or uncertain, \
and how it fits with earlier turns. It is your opinion, so say what the \
results show and where they stop; never present a guess as a result.

Then propose 3 or 4 next steps for the operator, each from a different angle:
- deeper:    follow the current line of analysis one step further;
- stress:    test how robust the current conclusion is (contingencies, \
uncertainty, other hours, forecasts);
- change:    {change_angle};
- step_back: something the operator has not asked but should consider — a \
pattern across the session, a question behind the question, a different \
framing of the problem.
Use step_back for genuinely new perspectives, not a restatement of the others.

Rules for suggestions:
1. If the latest result shows a problem (violations, insecure or infeasible \
results, contingency failures, concerns raised by the system), at least one \
suggestion must address it directly.
2. Do not repeat analyses already done in this conversation or suggestions \
already given, unless a changed parameter makes them meaningfully new — and \
then say what changed.
3. Keep "title" under 12 words and "why" to one sentence that points to what \
in the conversation motivates the suggestion.
4. When the latest turn studied a network proposal:
   - "stress" should cover what a planner would check before trusting it: other \
hours (compare_networks with study "scan" over the dataset, the forecast), and \
an N-1 screen on the proposal at its worst moment, if the session has not done them;
   - "change" should offer an ALTERNATIVE to the proposal, written as a request \
to create a new proposal with a new id and compare it with the base using the \
same study, timestamp and limits. Ground it in what the session showed — the \
elements that violated, the drivers attribution ranked, a bus with more \
hosting capacity — e.g. the same load at another bus, a different profile or \
size, reinforcing the branch that overloaded, a dispatchable unit near the bus \
that dropped, or opening/closing a branch. Say what motivates it; never claim \
the alternative is better — that is for the study to show.
5. If there is nothing useful to suggest, return an empty list and a one-line \
"note" explaining why; still write the comment.

Respond with JSON only, in this format:
{{"comment": "...", "suggestions": [{{"angle": "deeper|stress|change|step_back", \
"title": "...", "prompt": "...", "why": "..."}}], "note": "optional"}}
"""

_DISCUSS_TASK = """\
You are now in a side discussion with the operator about the results of this \
session — the thread is given below the session. Answer the operator's latest \
message directly, in at most about 200 words.

Rules for the discussion:
1. Separate evidence from opinion. When you say what the results show, say \
which result (the tool, and the turn when it helps). When you give your \
reading, say it is your view.
2. When the operator is deciding something, ask what matters to them — which \
limits, how much risk, which time horizon — before recommending.
3. Asked why a conclusion might be wrong, argue against it using only what the \
session shows, and say which study would test each objection.
4. When the discussion needs evidence the session does not have, propose at \
most 3 studies. The operator sends them to the orchestrator; you will see \
their results in later messages of the thread.
5. Do not repeat a study the session already ran unless a changed parameter \
makes it new; say what changed.
6. Before any question has been asked (the request says so), help the operator \
plan: what exactly to ask, at which moment or period, against which limits, and \
which studies to run first. There are no results yet — say nothing about the \
grid's state (voltages, loadings, security, margins); only the session context \
describes the network.

Respond with JSON only, in this format:
{{"reply": "...", "studies": [{{"title": "...", "prompt": "...", "why": "..."}}]}}
"""

_NOTE_TASK = """\
The operator is closing the side discussion below and asks you to record it. \
Write a short decision note from the thread and the session.

Rules for the note:
1. "decision": what the operator decided, in their words or close to them. If \
no decision was reached, write "No decision recorded" and say what remains open.
2. "evidence": the results the decision rests on — each naming the study and \
the turn, figures copied exactly. Only results in the session; nothing from \
studies that were proposed but not run.
3. "open_risks": what the evidence does not cover, or modelling choices it \
depends on.
4. "studies_run": the studies in this session that bear on the decision.
Keep each item to one sentence.

Respond with JSON only, in this format:
{{"decision": "...", "evidence": ["..."], "open_risks": ["..."], "studies_run": ["..."]}}
"""

def capability_list() -> list[tuple[str, str]]:
    """(tool name, first sentence of its description) for every runnable tool.

    Generated from the same declarations the orchestrator is given, so the two
    agents cannot disagree about what CONDUCTOR can do.
    """
    from .providers.schema import function_declarations
    from .tool_schemas import TOOLS

    out = []
    for fd in function_declarations(TOOLS):
        if fd.name in _EXCLUDED_CAPABILITIES:
            continue
        out.append((fd.name, _first_sentence(fd.description or "")))
    return out


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    match = re.search(r"(?<=[.!?])\s", text)
    return text[: match.start()] if match else text


def _prompt(task: str, task_sentence: str, task_input: str,
            capabilities: list[tuple[str, str]] | None, **fmt) -> str:
    caps = capability_list() if capabilities is None else capabilities
    lines = "\n".join(f"- {name}: {summary}" for name, summary in caps)
    return (_ROLE.format(task_sentence=task_sentence) + "\n"
            + _INPUTS.format(task_input=task_input) + "\n"
            + _SHARED_RULES + "\n"
            + task.format(**fmt)
            + "\nCapabilities the orchestrator can run:\n" + lines + "\n")


def build_system_prompt(capabilities: list[tuple[str, str]] | None = None) -> str:
    """The suggestion mode."""
    return _prompt(_SUGGEST_TASK, "Here your job is to help the operator decide what to look at next.",
                   "suggestions already given in this conversation", capabilities,
                   change_angle=_CHANGE_ANGLE)


def build_discussion_prompt(capabilities: list[tuple[str, str]] | None = None) -> str:
    """The discussion mode: a side thread with the operator."""
    return _prompt(_DISCUSS_TASK, "Here you are the operator's advisor: they think a result "
                   "through with you, and decide what to do and what to study next.",
                   "the side discussion so far", capabilities)


def build_note_prompt(capabilities: list[tuple[str, str]] | None = None) -> str:
    """The decision note that closes a discussion."""
    return _prompt(_NOTE_TASK, "Here you record what a side discussion with the operator decided.",
                   "the side discussion to record", capabilities)


# ---------------------------------------------------------------------------
# What the agent is shown
# ---------------------------------------------------------------------------


def _compact(value: Any, depth: int = 0) -> Any:
    """A bounded view of a tool result.

    Scalars and short strings survive; long lists keep their first few items
    and their length; nesting is cut off. Every number that remains is a
    number from the payload, so anything the agent quotes from this view is
    traceable to the full record.
    """
    if isinstance(value, dict):
        if depth >= 3:
            return f"<{len(value)} fields>"
        out = {}
        for key, item in value.items():
            if str(key).startswith("_") and key != "_integrity_warnings":
                continue
            out[key] = _compact(item, depth + 1)
        return out
    if isinstance(value, list):
        if depth >= 3:
            return f"<{len(value)} items>"
        head = [_compact(item, depth + 1) for item in value[:5]]
        if len(value) > 5:
            head.append(f"<{len(value) - 5} more>")
        return head
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    return value


def compact_turn(record: dict, max_chars_per_tool: int = 3000) -> dict:
    """The latest turn, reduced to what the suggestion agent needs."""
    from .loop import _prepare_tool_result_for_model

    tools = []
    for call, result in zip(record.get("tool_calls") or [], record.get("tool_results") or []):
        payload = _prepare_tool_result_for_model(result.get("name", ""), result.get("result"))
        view = _compact(payload)
        text = json.dumps(view, ensure_ascii=False, default=str)
        if len(text) > max_chars_per_tool:
            # Still bounded on a large network; keep the top-level scalars,
            # which carry the verdicts (secure, feasible, totals).
            # The `network` block stays: without it a proposal's result would
            # read as the loaded network's.
            view = {k: v for k, v in view.items()
                    if not isinstance(v, (dict, list)) or k == "network"} \
                if isinstance(view, dict) else text[:max_chars_per_tool]
        tools.append({"name": call.get("name"), "args": call.get("args") or {}, "result": view})
    return {"question": record.get("user") or "", "tools": tools}


# Studies that look at more than one moment.
_OVER_TIME = frozenset({"find_worst_case_timestamp", "scan_rsa_over_time", "scan_scenarios",
                        "get_element_timeseries", "compute_historical_risk", "run_probabilistic_rsa",
                        "forecast_kpis"})


def proposal_gaps(record: dict | None) -> list[str]:
    """Why a turn's study of a network proposal is thinner than a planner
    would accept: new violations against the base, a verdict resting on one
    moment, or no N-1 screen of the proposal. Decided in code, like the rest
    of the attention check."""
    if not record:
        return []
    studied: dict[str, dict] = {}
    for call, entry in zip(record.get("tool_calls") or [], record.get("tool_results") or []):
        result = entry.get("result") if isinstance(entry, dict) else None
        net = result.get("network") if isinstance(result, dict) else None
        if not isinstance(net, dict) or not net.get("id") or result.get("error"):
            continue
        seen = studied.setdefault(str(net["id"]), {"over_time": False, "n1": False, "new": 0})
        name = call.get("name") or entry.get("name")
        seen["over_time"] |= name in _OVER_TIME or (
            name == "compare_networks" and result.get("study") == "scan")
        seen["n1"] |= name == "simulate_all_contingencies" or (
            name == "compare_networks" and result.get("study") == "n1")
        seen["new"] += len(result.get("new_violations") or [])
    reasons = []
    for pid, seen in studied.items():
        if seen["new"]:
            reasons.append(f"proposal {pid}: {seen['new']} violation(s) the base does not have")
        if not seen["over_time"]:
            reasons.append(f"proposal {pid}: judged at a single moment — no study over time")
        if not seen["n1"]:
            reasons.append(f"proposal {pid}: no N-1 screen of the proposal in this turn")
    return reasons


def answer_concerns(record: dict | None) -> list[str]:
    """What the deterministic answer checks found in the delivered answer, in
    plain words: a figure that traces to no tool result, a figure written about
    the wrong element, a figure described against its own record, or a kind of
    quantity the question asked for and the answer does not contain.

    The same checks as the audit panel and the reflection setting
    (`reflection.check_all`) — no model is involved. The suggestion agent is
    given them so it can say what they mean and propose what would settle
    them; it does not judge the answer itself.
    """
    answer = (record or {}).get("assistant") or ""
    if not answer.strip() or not record.get("tool_results"):
        return []
    from .reflection import check_all
    try:
        findings, _report, conflicts, gaps = check_all(answer, record)
    except Exception:  # noqa: BLE001 — a check that cannot run says nothing
        logger.debug("Answer checks could not run for the suggestion agent.", exc_info=True)
        return []

    def shown(value, unit):
        return f"{value:g}" + (f" {unit}" if unit else "")

    out = []
    for v in findings:
        if v.status == MISATTRIBUTED:
            out.append(f"answer check: {shown(v.claim.value, v.claim.unit)} is written about "
                       f"{v.subject or 'one element'} but comes from {v.source}, which belongs to "
                       f"{v.belongs_to or 'another element'}")
        else:
            out.append(f"answer check: {shown(v.claim.value, v.claim.unit)} appears in no tool result "
                       "(the agent may have computed or estimated it)")
    for c in conflicts:
        lead = "" if c.value == 0 and c.unit is None else f"{shown(c.value, c.unit)}: "
        out.append(f"answer check: {lead}{c.detail}")
    out.extend(f"answer check: {g.render()}" for g in gaps)
    return list(dict.fromkeys(out))


def needs_attention(record: dict | None) -> list[str]:
    """Plain reasons the latest result deserves a second look, or [].

    Decided in code from verdict fields the engines state themselves, so the
    highlight costs nothing and never depends on the model.
    """
    if not record:
        return []
    reasons: list[str] = []
    reasons.extend(answer_concerns(record))

    def scan(node: dict, tool: str) -> None:
        for key in ("secure", "system_secure", "system_n1_secure",
                    "feasible", "converged", "usable"):
            if node.get(key) is False:
                reasons.append(f"{tool}: {key} is false")
        total = node.get("total_violations")
        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
            reasons.append(f"{tool}: {int(total)} violation(s)")
        if node.get("any_violations") is True:
            reasons.append(f"{tool}: violations in the scanned window")
        if node.get("error") and tool not in retried:
            reasons.append(f"{tool}: returned an error")

    reasons.extend(proposal_gaps(record))
    # A call that failed and was then repeated successfully in the same turn
    # is a retry (a lookup of "Bus 8" before "Bus_8"), not a concern.
    retried = set()
    for n, entry in enumerate(record.get("tool_results") or []):
        res = entry.get("result")
        if isinstance(res, dict) and res.get("error") and any(
                later.get("name") == entry.get("name") and isinstance(later.get("result"), dict)
                and not later["result"].get("error")
                for later in (record.get("tool_results") or [])[n + 1:]):
            retried.add(entry.get("name"))
    for entry in record.get("tool_results") or []:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        tool = entry.get("name", "tool")
        # The result's own verdicts, and those of its direct sub-objects (a
        # constrained-slack scenario, a robust stage). Nothing inside lists:
        # those are per-sample or per-scenario, and one infeasible Monte Carlo
        # sample is not a reason to alarm the operator.
        scan(result, tool)
        for item in result.values():
            if isinstance(item, dict):
                scan(item, tool)
    return list(dict.fromkeys(reasons))


def _conversation_text(messages: list[dict]) -> str:
    """The operator-visible conversation: questions and answers, no payloads."""
    lines = []
    for msg in messages[-_MAX_MESSAGES:]:
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if len(text) > _MAX_MESSAGE_CHARS:
            text = text[:_MAX_MESSAGE_CHARS] + " […]"
        role = "Operator" if msg.get("role") == "user" else "CONDUCTOR"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


# Fields that state what a result concluded. Top-level only, as in
# `needs_attention`: per-sample or per-element detail is the latest turn's job.
_VERDICT_KEYS = (
    "status", "secure", "system_secure", "system_n1_secure", "feasible", "converged",
    "usable", "total_violations", "total_outages_tested", "total_outages_causing_violations",
    "p_any_violation", "worst_timestamp", "worst_value", "guarantee_met",
)
_ARG_KEYS = (
    "network", "proposal", "study",
    "timestamp", "data_source", "element_type", "element_index", "gen_name", "bus",
    "metric", "load_scaling_factor", "slack_max_mw", "disabled_generators",
)


def turn_digest(record: dict, number: int) -> str:
    """One line for one turn: question, tools with key arguments, verdicts."""
    question = " ".join(str(record.get("user") or "").split())
    if len(question) > 140:
        question = question[:137] + "..."
    tools = []
    for call, entry in zip(record.get("tool_calls") or [], record.get("tool_results") or []):
        args = call.get("args") or {}
        shown_args = ", ".join(f"{k}={args[k]}" for k in _ARG_KEYS
                               if k in args and args[k] not in (None, "", []))
        result = entry.get("result")
        if isinstance(result, dict) and result.get("error"):
            verdict = "error"
        elif isinstance(result, dict):
            verdict = ", ".join(f"{k}={result[k]}" for k in _VERDICT_KEYS
                                if k in result and isinstance(result[k], (bool, int, float, str))
                                and len(str(result[k])) <= 40)
        else:
            verdict = ""
        tools.append(f"{call.get('name')}({shown_args})" + (f" → {verdict}" if verdict else ""))
    return f"T{number} · Q: {question} · " + ("; ".join(tools) if tools else "no tools")


def session_digest(records: list[dict], latest: dict | None) -> str:
    """Every turn before the latest, newest kept when the digest is too long."""
    earlier = [r for r in records if r is not latest]
    lines = [turn_digest(r, i) for i, r in enumerate(earlier, start=1)]
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if size + len(line) > _MAX_DIGEST_CHARS:
            break
        kept.append(line)
        size += len(line) + 1
    omitted = len(lines) - len(kept)
    head = [f"({omitted} earlier turn(s) omitted)"] if omitted else []
    return "\n".join(head + list(reversed(kept)))


def _session_parts(messages: list[dict], record: dict | None, context: dict,
                   records: list[dict] | None) -> list[str]:
    """What both modes are shown about the session."""
    parts = [
        "## Conversation (most recent messages)\n" + (_conversation_text(messages) or "(empty)"),
        "## Earlier turns (digest)\n" + (session_digest(records or [], record) or "(none)"),
        "## Latest turn record\n" + (
            json.dumps(compact_turn(record), ensure_ascii=False, default=str)
            if record else "(no tools were run in the latest turn)"),
    ]
    attention = needs_attention(record)
    if attention:
        parts.append("## Concerns raised by the system\n" + "\n".join(f"- {r}" for r in attention))
    parts.append("## Session context\n" + json.dumps(context, ensure_ascii=False, default=str))
    return parts


def build_request(messages: list[dict], record: dict | None, context: dict,
                  shown: list[dict], records: list[dict] | None = None) -> str:
    """The single user message sent to the suggestion agent."""
    parts = _session_parts(messages, record, context, records)
    if shown:
        parts.append("## Suggestions already given (do not repeat)\n" + "\n".join(
            f"- [{s.get('angle')}] {s.get('title')}" for s in shown))
    parts.append("Return the JSON now.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing and grounding
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _normalise_angle(raw: Any) -> str | None:
    angle = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    return angle if angle in ANGLES else None


def _key(text: str) -> str:
    return " ".join(str(text).lower().split())


def parse_response(text: str, shown: list[dict] | None = None) -> SuggestionResult:
    """Turn the model's reply into suggestions, or a result carrying an error.

    Lenient about wrapping (code fences, prose around the object), strict about
    content: an entry without a known angle, a title or a prompt is dropped
    rather than repaired, and repeats of suggestions already shown are removed.
    """
    raw = text or ""
    body = _FENCE_RE.sub("", raw.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return SuggestionResult((), error="The suggestion agent did not return JSON.", raw=raw)
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return SuggestionResult((), error="The suggestion agent returned malformed JSON.", raw=raw)
    if not isinstance(data, dict):
        return SuggestionResult((), error="The suggestion agent returned an unexpected shape.", raw=raw)

    seen = {_key(s.get("title", "")) for s in shown or []} | \
           {_key(s.get("prompt", "")) for s in shown or []}
    out: list[Suggestion] = []
    for entry in data.get("suggestions") or []:
        if not isinstance(entry, dict):
            continue
        angle = _normalise_angle(entry.get("angle"))
        title = str(entry.get("title") or "").strip()
        prompt = str(entry.get("prompt") or "").strip()
        why = str(entry.get("why") or "").strip()
        if not (angle and title and prompt):
            continue
        if _key(title) in seen or _key(prompt) in seen:
            continue
        seen.update({_key(title), _key(prompt)})
        out.append(Suggestion(angle=angle, title=title, prompt=prompt, why=why))
        if len(out) == MAX_SUGGESTIONS:
            break

    note = str(data.get("note") or "").strip()
    comment = str(data.get("comment") or "").strip()
    return SuggestionResult(tuple(out), comment=comment, note=note, raw=raw)


def ground(suggestions: tuple[Suggestion, ...], records: list[dict]) -> tuple[Suggestion, ...]:
    """Mark figures in each `why` that no record in the conversation contains.

    Only `why` is checked: it is the one field that asserts something about the
    grid. A prompt carries parameters the operator is invited to try ("at 120 %
    load"), which are proposals, not claims.
    """
    check = _figure_check(records)
    return tuple(s._replace(unverified=check(s.why)) for s in suggestions)


def _figure_check(records: list[dict]):
    """A function returning the figures in a text that no record contains."""
    sources = []
    subjects: set[str] = set()
    for record in records:
        sources.extend(collect_sources(record))
        subjects |= set(subject_names(record))

    def unverified(text: str) -> tuple[str, ...]:
        report = check_answer(text, sources, subjects=subjects)
        return tuple(
            f"{v.claim.value:g}" + (f" {v.claim.unit}" if v.claim.unit else "")
            for v in report.verdicts if v.status in (UNGROUNDED, MISATTRIBUTED)
        )

    return unverified


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def suggest(messages: list[dict], latest: dict | None, records: list[dict],
            context: dict, shown: list[dict] | None = None,
            provider=None) -> SuggestionResult:
    """Ask the suggestion agent how the operator might continue.

    Args:
        messages: the operator-visible conversation, `[{role, text}, ...]`.
        latest:   the turn record of the answer being followed up, or None
                  when that answer has none (no tools ran). Passed explicitly
                  rather than taken from `records`, so an answer without a
                  record is never described with an earlier turn's results.
        records:  every turn record in this conversation — the evidence the
                  `why` of each suggestion is grounded against.
        context:  session facts — network, simulation time, data source.
        shown:    suggestions already displayed, as dicts with angle/title/prompt.
        provider: injected for tests; defaults to the orchestrator's provider.
    """
    from .providers import get_provider, user_message

    shown = list(shown or [])
    provider = provider or get_provider()
    request = build_request(messages, latest, context, shown, records)
    try:
        response = provider.chat(
            messages=[user_message(request)],
            tools=[],
            system=build_system_prompt(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Suggestion agent call failed.", exc_info=True)
        return SuggestionResult((), error=f"The suggestion agent could not be reached: {exc}")

    result = parse_response(response.text or "", shown)
    if result.suggestions or result.comment:
        # Grounding is a check: if it fails, the comment and suggestions are
        # still shown, unmarked, rather than lost.
        try:
            result = result._replace(
                suggestions=ground(result.suggestions, records),
                comment_unverified=_figure_check(records)(result.comment) if result.comment else (),
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not ground suggestion figures.", exc_info=True)
    return result


# ---------------------------------------------------------------------------
# Discussion mode — a side thread with the operator under an answer
# ---------------------------------------------------------------------------

MAX_STUDIES = 3
_MAX_THREAD_MESSAGES = 16
_MAX_THREAD_CHARS = 2000


class Study(NamedTuple):
    """A study the advisor proposes; the operator sends it to the orchestrator."""
    title: str
    prompt: str
    why: str = ""
    unverified: tuple[str, ...] = ()

    def as_record(self) -> dict:
        return {"title": self.title, "prompt": self.prompt, "why": self.why,
                "unverified": list(self.unverified)}


class AdvisorReply(NamedTuple):
    text: str
    studies: tuple[Study, ...] = ()
    # Figures in `text` that appear nowhere in this conversation's evidence.
    unverified: tuple[str, ...] = ()
    error: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class DecisionNote(NamedTuple):
    decision: str
    evidence: tuple[str, ...] = ()
    open_risks: tuple[str, ...] = ()
    studies_run: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    error: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def as_record(self) -> dict:
        return {"decision": self.decision, "evidence": list(self.evidence),
                "open_risks": list(self.open_risks), "studies_run": list(self.studies_run),
                "unverified": list(self.unverified)}


def _thread_text(thread: list[dict]) -> str:
    """The side discussion so far, newest kept when it is long. A study the
    operator sent is shown with the turn it was answered in, when known."""
    lines = []
    for item in thread[-_MAX_THREAD_MESSAGES:]:
        role = item.get("role")
        text = " ".join(str(item.get("text") or "").split())[:_MAX_THREAD_CHARS]
        if role == "operator":
            lines.append(f"Operator: {text}")
        elif role == "advisor":
            studies = "; ".join(s.get("title", "") for s in item.get("studies") or [])
            lines.append(f"You: {text}" + (f" [proposed studies: {studies}]" if studies else ""))
        elif role == "sent":
            lines.append(f"(The operator sent the study \"{text}\" to the orchestrator"
                         + (f"; it was answered in turn {item['answered_turn']}." if item.get("answered_turn")
                            else "; no answer yet.") + ")")
    return "\n".join(lines)


def build_discussion_request(question: str, thread: list[dict], messages: list[dict],
                             record: dict | None, context: dict, records: list[dict]) -> str:
    parts = _session_parts(messages, record, context, records)
    if not records and not any(m.get("role") == "user" for m in messages):
        parts.append("## Before the first question\nThe operator has not asked the orchestrator "
                     "anything yet and no study has run: this is planning (rule 6).")
    parts.append("## Side discussion so far\n" + (_thread_text(thread) or "(this is its first message)"))
    parts.append("## The operator's message\n" + question.strip())
    parts.append("Return the JSON now.")
    return "\n\n".join(parts)


def build_note_request(thread: list[dict], messages: list[dict], record: dict | None,
                       context: dict, records: list[dict]) -> str:
    parts = _session_parts(messages, record, context, records)
    parts.append("## Side discussion to record\n" + (_thread_text(thread) or "(empty)"))
    parts.append("Return the JSON now.")
    return "\n\n".join(parts)


def _json_object(text: str) -> dict | None:
    body = _FENCE_RE.sub("", (text or "").strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_discussion(text: str) -> AdvisorReply:
    data = _json_object(text)
    if data is None:
        return AdvisorReply("", error="The advisor did not return a readable reply.", raw=text or "")
    reply = str(data.get("reply") or "").strip()
    studies = []
    for entry in data.get("studies") or []:
        if not isinstance(entry, dict):
            continue
        title, prompt = str(entry.get("title") or "").strip(), str(entry.get("prompt") or "").strip()
        if title and prompt:
            studies.append(Study(title, prompt, str(entry.get("why") or "").strip()))
        if len(studies) == MAX_STUDIES:
            break
    if not reply and not studies:
        return AdvisorReply("", error="The advisor returned an empty reply.", raw=text or "")
    return AdvisorReply(reply, tuple(studies), raw=text or "")


def parse_note(text: str) -> DecisionNote:
    data = _json_object(text)
    if data is None:
        return DecisionNote("", error="The advisor did not return a readable note.", raw=text or "")

    def items(key):
        return tuple(str(x).strip() for x in (data.get(key) or []) if str(x).strip())

    decision = str(data.get("decision") or "").strip() or "No decision recorded"
    return DecisionNote(decision, items("evidence"), items("open_risks"), items("studies_run"),
                        raw=text or "")


def _call(system: str, request: str, provider):
    from .providers import get_provider, user_message
    provider = provider or get_provider()
    return provider.chat(messages=[user_message(request)], tools=[], system=system)


def discuss(question: str, thread: list[dict], messages: list[dict], latest: dict | None,
            records: list[dict], context: dict, provider=None) -> AdvisorReply:
    """One reply of the advisor in a side discussion.

    `thread` is the discussion so far, `[{role: operator|advisor|sent, text, ...}]`;
    `records` are every turn record of the conversation, including answers to
    studies sent from the thread. Figures in the reply and in each study's
    `why` are checked against those records and flagged when not found.
    """
    try:
        response = _call(build_discussion_prompt(),
                         build_discussion_request(question, thread, messages, latest, context, records),
                         provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Advisor call failed.", exc_info=True)
        return AdvisorReply("", error=f"The advisor could not be reached: {exc}")
    reply = parse_discussion(response.text or "")
    if reply.ok:
        try:
            check = _figure_check(records)
            reply = reply._replace(unverified=check(reply.text),
                                   studies=tuple(st._replace(unverified=check(st.why)) for st in reply.studies))
        except Exception:  # noqa: BLE001 — shown unmarked rather than lost
            logger.warning("Could not ground the advisor's figures.", exc_info=True)
    return reply


def decision_note(thread: list[dict], messages: list[dict], latest: dict | None,
                  records: list[dict], context: dict, provider=None) -> DecisionNote:
    """The note that closes a side discussion: decision, evidence, open risks,
    studies run. Its figures are checked like every other text of the agent."""
    try:
        response = _call(build_note_prompt(),
                         build_note_request(thread, messages, latest, context, records), provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decision note call failed.", exc_info=True)
        return DecisionNote("", error=f"The advisor could not be reached: {exc}")
    note = parse_note(response.text or "")
    if note.ok:
        try:
            check = _figure_check(records)
            text = " ".join([note.decision, *note.evidence, *note.open_risks])
            note = note._replace(unverified=check(text))
        except Exception:  # noqa: BLE001
            logger.warning("Could not ground the decision note's figures.", exc_info=True)
    return note
