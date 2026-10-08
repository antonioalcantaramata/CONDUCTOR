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

# The "change" angle describes the levers that exist today. When network
# proposals (the sandbox) land, this text grows to include them.
_CHANGE_ANGLE = (
    "a what-if worth studying with the levers CONDUCTOR has: scaling load, "
    "scaling renewable output, disabling or derating generators, capping the "
    "external-grid import, adding injection at a bus (hosting capacity), or "
    "re-dispatching flexibility"
)

_BASE_PROMPT = """\
You are the suggestion agent of CONDUCTOR, a decision-support system for power \
system operators. A separate orchestrator agent answers the operator's \
questions by running validated power-system tools. You do not run tools and \
you do not compute anything. Your only job is to help the operator decide what \
to look at next.

You receive:
- the most recent messages of the conversation;
- a digest of every earlier turn: the question, the tools run with their key \
arguments, and the verdicts they returned — use it to see patterns across a \
long session;
- a compact record of the latest turn (tools run, their arguments, key \
results, and any concerns the system raised);
- the session context (network, simulation time, data source);
- the list of capabilities the orchestrator can run;
- suggestions already given in this conversation.

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

Rules:
1. Every suggestion must be answerable with the listed capabilities. If an \
idea needs data or tools CONDUCTOR does not have, do not suggest it.
2. Write each "prompt" as a complete, self-contained request the orchestrator \
can execute: name the timestamp, the data source, and the elements by name. \
Never rely on "the same as before" or "that bus".
3. Do not state engineering figures — in the comment or the suggestions — \
except those that appear in the turn records or the conversation, copied \
exactly. Never estimate, extrapolate or calculate.
4. You suggest analyses and options to explore; you do not instruct \
operational actions. Never write "you should curtail…" or "dispatch…". Write \
"Study…", "Compare…", "Check whether…".
5. If the latest result shows a problem (violations, insecure or infeasible \
results, contingency failures, concerns raised by the system), at least one \
suggestion must address it directly.
6. Do not repeat analyses already done in this conversation or suggestions \
already given, unless a changed parameter makes them meaningfully new — and \
then say what changed.
7. Keep "title" under 12 words and "why" to one sentence that points to what \
in the conversation motivates the suggestion.
8. Match the operator's language.
9. If there is nothing useful to suggest, return an empty list and a one-line \
"note" explaining why; still write the comment.

Respond with JSON only, in this format:
{{"comment": "...", "suggestions": [{{"angle": "deeper|stress|change|step_back", \
"title": "...", "prompt": "...", "why": "..."}}], "note": "optional"}}
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


def build_system_prompt(capabilities: list[tuple[str, str]] | None = None) -> str:
    caps = capability_list() if capabilities is None else capabilities
    lines = "\n".join(f"- {name}: {summary}" for name, summary in caps)
    return (
        _BASE_PROMPT.format(change_angle=_CHANGE_ANGLE)
        + "\nCapabilities the orchestrator can run:\n"
        + lines
        + "\n"
    )


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
            view = {k: v for k, v in view.items() if not isinstance(v, (dict, list))} \
                if isinstance(view, dict) else text[:max_chars_per_tool]
        tools.append({"name": call.get("name"), "args": call.get("args") or {}, "result": view})
    return {"question": record.get("user") or "", "tools": tools}


def needs_attention(record: dict | None) -> list[str]:
    """Plain reasons the latest result deserves a second look, or [].

    Decided in code from verdict fields the engines state themselves, so the
    highlight costs nothing and never depends on the model.
    """
    if not record:
        return []
    reasons: list[str] = []

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
        if node.get("error"):
            reasons.append(f"{tool}: returned an error")

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


def build_request(messages: list[dict], record: dict | None, context: dict,
                  shown: list[dict], records: list[dict] | None = None) -> str:
    """The single user message sent to the suggestion agent."""
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
