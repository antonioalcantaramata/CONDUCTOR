"""
Ablation arms: what is switched in each, and the studies (presets) that come ready.

An arm is a name and a few settings ("knobs"). Written on the command line as

    "<name>: knob=value knob=value"        e.g.  "no-routing: drop_rules=routing"
                                                  "small-model: model=gpt-6-mini effort=low"

with a list value comma-separated ("hide_tools=group:investigation,edit_proposal"). An arm with no
settings ("full:") is CONDUCTOR as it is. The knobs:

| knob | values | what it changes |
|---|---|---|
| ontology | on, off, none, tables | the semantic layer: for the agent (off hides it; tables gives it the raw data tables instead), and for the studies (none; its own backend) |
| review | off, warn, inform | the answer checks shown to the agent before the answer goes out (`CONDUCTOR_REFLECTION`) |
| reading_check | on, off | the decisive studies re-run under the question's other readings (`robustness.py`) |
| deep | task, on, off | Deep Research on the first question: as each task says, always, never |
| follow_suggestion | off, on | after the task, follow the AI Agent Suggestion's stress suggestion (one more turn) |
| drop_rules | ids of `system_prompt.PROMPT_PARTS` | parts of the system prompt left out |
| hide_tools | tool names, `group:<name>` of `tool_registry.ABLATION_GROUPS` | tools the model is not given |
| max_steps | an integer | model calls per question before it must answer (`DT_MAX_AGENT_TURNS`) |
| provider | google, openai, anthropic, ollama, … | the model provider (`LLM_PROVIDER`) |
| model | a model id | the provider's model (`OPENAI_MODEL`, `GEMINI_MODEL`, …) |
| effort | the provider's levels | reasoning effort / thinking level |
| env.<NAME> | any | an environment variable, for anything else |

Each arm runs in its own process with its settings in the environment, so even the settings read
once at start (the model, the provider) differ between arms of one study.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import NamedTuple

# Environment variables of the model's settings, per provider.
_MODEL_ENV = {"google": "GEMINI_MODEL", "openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL",
              "ollama": "OLLAMA_MODEL"}
_EFFORT_ENV = {"google": "GEMINI_THINKING_LEVEL", "openai": "OPENAI_REASONING_EFFORT",
               "anthropic": "ANTHROPIC_EFFORT"}

KNOBS = {
    "ontology": ("on", "off", "none", "tables"),
    "review": ("off", "warn", "inform"),
    "reading_check": ("on", "off"),
    "deep": ("task", "on", "off"),
    "follow_suggestion": ("off", "on"),
    "drop_rules": None, "hide_tools": None, "max_steps": None,
    "provider": None, "model": None, "effort": None,
}
_LISTS = ("drop_rules", "hide_tools")


class Arm(NamedTuple):
    name: str
    settings: dict            # knob -> value (lists for drop_rules / hide_tools)

    @property
    def hash(self) -> str:
        """Identifies the settings: a resumed run refuses records made under other settings."""
        return hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest()[:12]

    def get(self, knob: str, default=None):
        return self.settings.get(knob, default)

    def describe(self) -> str:
        if not self.settings:
            return "as it is"
        return " ".join(f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in self.settings.items())

    # --- what the arm changes -------------------------------------------------------------------------

    def provider(self) -> str:
        return (self.get("provider") or os.environ.get("LLM_PROVIDER") or _configured_provider()).lower()

    def env(self) -> dict[str, str]:
        """The agent process's environment for this arm (on top of the parent's)."""
        env: dict[str, str] = {"CONDUCTOR_ONTOLOGY": self.get("ontology", "on")}
        if self.get("drop_rules"):
            env["CONDUCTOR_DROP_RULES"] = ",".join(self.get("drop_rules"))
        if self.get("hide_tools"):
            env["CONDUCTOR_HIDE_TOOLS"] = ",".join(self.get("hide_tools"))
        if self.get("max_steps"):
            env["DT_MAX_AGENT_TURNS"] = str(self.get("max_steps"))
        if self.get("provider"):
            env["LLM_PROVIDER"] = self.get("provider")
        provider = self.provider()
        if self.get("model"):
            if provider not in _MODEL_ENV and provider not in ("openrouter", "nvidia", "deepseek"):
                raise ValueError(f"Arm {self.name!r}: no model setting known for provider {provider!r}")
            env[_MODEL_ENV.get(provider, f"{provider.upper()}_MODEL")] = self.get("model")
        if self.get("effort"):
            if provider not in _EFFORT_ENV:
                raise ValueError(f"Arm {self.name!r}: no effort setting known for provider {provider!r}")
            env[_EFFORT_ENV[provider]] = self.get("effort")
        for k, v in self.settings.items():
            if k.startswith("env."):
                env[k.removeprefix("env.")] = str(v)
        return env

    def backend_mode(self) -> str:
        """The backend this arm needs: `none` (studies without the layer) or `on`."""
        return "none" if self.get("ontology") == "none" else "on"

    def run_kwargs(self) -> dict:
        """`whatif.run_task`'s settings for this arm."""
        deep = self.get("deep", "task")
        return {"reflection": self.get("review", "off"),
                "reading_check": None if "reading_check" not in self.settings else self.get("reading_check") == "on",
                "deep": None if deep == "task" else deep == "on",
                "follow_suggestion": self.get("follow_suggestion", "off") == "on"}


def _configured_provider() -> str:
    from llm_agent.agent.config import LLM_PROVIDER
    return LLM_PROVIDER


def parse(spec: str) -> Arm:
    """`"<name>: knob=value …"` → an Arm, every value checked (a typo must not ablate nothing)."""
    name, _, rest = spec.partition(":")
    name = name.strip()
    if not name or not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError(f"Arm name {name!r}: letters, digits, '-', '_' or '.' only")
    settings: dict = {}
    for token in rest.split():
        knob, eq, value = token.partition("=")
        if not eq or not value:
            raise ValueError(f"Arm {name!r}: {token!r} is not knob=value")
        if knob not in KNOBS and not knob.startswith("env."):
            raise ValueError(f"Arm {name!r}: unknown knob {knob!r}; known: {sorted(KNOBS)} or env.<NAME>")
        allowed = KNOBS.get(knob)
        if allowed and value not in allowed:
            raise ValueError(f"Arm {name!r}: {knob} must be one of {allowed}, not {value!r}")
        settings[knob] = [v for v in value.split(",") if v] if knob in _LISTS else value
    arm = Arm(name, settings)
    validate(arm)
    return arm


def validate(arm: Arm) -> None:
    """The settings name things that exist: prompt parts found in today's prompt, tools in the registry,
    an integer step budget."""
    from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
    from llm_agent.agent.system_prompt import build_system_prompt, drop_parts
    from llm_agent.agent.tool_registry import expand_hidden
    if arm.get("drop_rules"):
        drop_parts(build_system_prompt(DEFAULT_GRID_CONSTANTS), arm.get("drop_rules"))
    if arm.get("hide_tools"):
        expand_hidden(arm.get("hide_tools"))
    if arm.get("max_steps") and not str(arm.get("max_steps")).isdigit():
        raise ValueError(f"Arm {arm.name!r}: max_steps must be an integer")
    arm.env()


def footprint(arm: Arm) -> dict:
    """What the arm takes from the agent, measured: the prompt characters each dropped part removed, and the
    tools withheld (for the manifest — a reader sees the size of each ablation)."""
    from llm_agent.agent.config import DEFAULT_GRID_CONSTANTS
    from llm_agent.agent.system_prompt import build_system_prompt, drop_parts
    from llm_agent.agent.tool_registry import expand_hidden
    prompt = build_system_prompt(DEFAULT_GRID_CONSTANTS)
    drop = list(arm.get("drop_rules") or [])
    if arm.get("ontology") in ("off", "none") and "element_identity" not in drop:
        drop.append("element_identity")
    _, removed = drop_parts(prompt, drop) if drop else (prompt, {})
    hidden = set(expand_hidden(arm.get("hide_tools") or []))
    if arm.get("ontology") in ("off", "none", "tables"):
        hidden |= {"describe_network", "declare_network_facts"}
    out = {"prompt_chars": len(prompt), "prompt_chars_removed": removed, "tools_hidden": sorted(hidden)}
    if arm.get("ontology") == "tables":
        # The baseline's own tool, and its rule on what an element is in place of the layer's.
        out.update(tools_added=["read_network_tables"], prompt_parts_replaced=["element_identity"])
    return out


# ---------------------------------------------------------------------------
# Presets — the studies that come ready
# ---------------------------------------------------------------------------

class Preset(NamedTuple):
    name: str
    question: str
    arms: tuple[str, ...]          # arm specs; the first is the baseline
    suites: tuple[str, ...]


_ALL = ("whatif", "investigation", "options", "clarify", "narration", "semantics-cigre", "semantics-ieee30",
        "reasoning")

PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset("ontology", "What the semantic layer is worth — to the agent, and to the studies.",
           ("on: ontology=on", "off: ontology=off", "none: ontology=none"),
           ("semantics-cigre", "semantics-ieee30", "reasoning")),
    Preset("raw_tables", "What the layer adds over the data it is built from: the agent given the network's raw "
           "data tables instead of the layer (the studies read the layer in every arm).",
           ("on: ontology=on", "off: ontology=off", "tables: ontology=tables"),
           ("semantics-cigre", "semantics-ieee30", "reasoning")),
    Preset("components", "Leave one out: what each part of CONDUCTOR adds over the rest (the paper's main table).",
           ("full:", "no-semantic-layer: ontology=off", "no-reading-check: reading_check=off",
            "no-routing-rules: drop_rules=routing", "no-evidence-wording: drop_rules=evidence_wording",
            "no-ask-first: drop_rules=ask_first", "no-investigation-tools: hide_tools=group:investigation",
            # What prior LLM-for-grids systems are: the model and the solvers, with no semantics (for the agent or
            # the studies) and no re-check under other readings — the comparison "compared to what?" asks for.
            "solver-only: ontology=none reading_check=off"),
           _ALL),
    Preset("answer_review", "Do the answer checks, shown to the agent, make its answers better grounded?",
           # `warn` (findings recorded, never shown) answers as `off` does: add it to count what would fire.
           ("off: review=off", "inform: review=inform"),
           ("narration", "options", "whatif")),
    Preset("reading_check", "What re-running the decisive studies under the question's other readings changes.",
           ("on: reading_check=on", "off: reading_check=off"),
           ("clarify", "narration", "options")),
    Preset("prompt_rules", "What each behavioural rule of the system prompt is worth.",
           ("full:", "no-routing: drop_rules=routing", "no-ask-first: drop_rules=ask_first",
            "no-evidence-wording: drop_rules=evidence_wording", "no-no-invention: drop_rules=no_invention",
            "no-data-source-check: drop_rules=data_source_check", "no-investigation-rule: drop_rules=investigation"),
           ("clarify", "narration", "options", "investigation", "whatif")),
    Preset("tools", "What each capability adds: the agent without it, on the tasks that need it.",
           ("all:", "no-investigation: hide_tools=group:investigation", "no-proposals: hide_tools=group:proposals",
            "no-lookup: hide_tools=group:lookup", "no-over-time: hide_tools=group:over_time"),
           ("investigation", "whatif", "semantics-cigre", "options")),
    Preset("deep_research", "Deep Research on the questions that ask for it, and on none.",
           ("task: deep=task", "off: deep=off"),
           ("investigation",)),
    Preset("advisor", "What following the AI Agent Suggestion adds after a what-if.",
           ("alone: follow_suggestion=off", "with-suggestion: follow_suggestion=on"),
           ("whatif",)),
    Preset("step_budget", "How many model calls per question the agent needs: quality against cost.",
           ("16: max_steps=16", "8: max_steps=8", "4: max_steps=4"),
           ("investigation", "options", "whatif")),
    Preset("effort", "Reasoning effort of the configured model: quality against cost and time.",
           ("medium: effort=medium", "low: effort=low", "high: effort=high"),
           _ALL),
)}


def preset_arms(name: str) -> list[Arm]:
    return [parse(s) for s in PRESETS[name].arms]
