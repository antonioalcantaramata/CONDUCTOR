"""
A load power factor the question never asked for does not override the
session's (llm_agent/agent/validators.py, TurnOperatingPoint).

Observed live on CIGRE MV, whose profile states its loads' own pf 0.97: the
model filled every optional argument, load_power_factor=0.95 among them, and
each study ran at 0.95 "for this study".
"""

from llm_agent.agent.validators import TurnOperatingPoint, mentions_power_factor

ACCEPTS = {"load_power_factor", "timestamp"}


def test_the_generic_default_is_dropped_when_the_question_says_nothing_of_it():
    op = TurnOperatingPoint(power_factor_asked=mentions_power_factor("If Line 2-3 were out, what loses supply?"))
    kwargs, _ = op.resolve("simulate_contingency", {"load_power_factor": 0.95, "timestamp": "t"}, ACCEPTS)
    assert "load_power_factor" not in kwargs and op.dropped == {"load_power_factor": 0.95}


def test_a_stated_or_deliberate_power_factor_is_kept():
    asked = TurnOperatingPoint(power_factor_asked=mentions_power_factor("Is it secure with loads at pf 0.95?"))
    assert asked.resolve("run_rsa", {"load_power_factor": 0.95}, ACCEPTS)[0]["load_power_factor"] == 0.95
    other = TurnOperatingPoint(power_factor_asked=False)
    assert other.resolve("run_rsa", {"load_power_factor": 0.9}, ACCEPTS)[0]["load_power_factor"] == 0.9
