"""Hardware detection and local-model sizing advice.

Guidance for running a local model is meaningless in the abstract — 32k of
context is trivial on a workstation and impossible on a small laptop. These
tests check the advice actually varies with the machine, and that it degrades
sensibly when the machine can't be detected.
"""

import pytest

from llm_agent.agent import hardware

REQUIRED = 25_000  # the agent's system prompt + tool schemas, in tokens


def machine(ram_gb, system="Linux", arch="x86_64"):
    return hardware.Hardware(
        system=system, machine=arch, ram_gb=ram_gb, cpu_count=8
    )


class TestDetection:
    def test_detects_this_machine(self):
        hw = hardware.detect()
        assert hw.system
        assert hw.ram_gb is None or hw.ram_gb > 0

    def test_describe_is_human_readable(self):
        assert "16 GB RAM" in machine(16).describe()

    def test_describe_survives_unknown_ram(self):
        assert "unknown RAM" in machine(None).describe()

    def test_apple_silicon_detection(self):
        assert machine(16, system="Darwin", arch="arm64").apple_silicon
        assert not machine(16, system="Darwin", arch="x86_64").apple_silicon
        assert not machine(16, system="Linux", arch="aarch64").apple_silicon

    def test_usable_memory_reserves_headroom_for_the_solver(self):
        # pandapower/Pyomo and the browser run alongside the model.
        assert machine(16).usable_gb < 16

    def test_usable_memory_never_goes_negative(self):
        assert machine(2).usable_gb >= 1.0

    def test_usable_memory_is_unknown_when_ram_is(self):
        assert machine(None).usable_gb is None


class TestWeightsFit:
    def test_small_model_fits_a_large_machine(self):
        assert hardware.weights_fit(7.7, machine(64)) is True

    def test_large_model_does_not_fit_a_small_machine(self):
        assert hardware.weights_fit(20.0, machine(8)) is False

    def test_unknown_ram_yields_unknown_answer(self):
        # None, not False — "we don't know" must not read as "it won't fit".
        assert hardware.weights_fit(7.7, machine(None)) is None


class TestContextCeiling:
    def test_ceiling_grows_with_memory(self):
        ceilings = [hardware.context_ceiling(machine(r)) for r in (8, 16, 32, 64)]
        assert ceilings == sorted(ceilings)

    def test_small_machine_gets_a_modest_ceiling(self):
        assert hardware.context_ceiling(machine(8)) <= 32_768

    def test_workstation_gets_a_large_ceiling(self):
        assert hardware.context_ceiling(machine(64)) >= 131_072

    def test_unknown_hardware_assumes_a_modest_machine(self):
        assert hardware.context_ceiling(machine(None)) == 32_768


class TestRecommendedNumCtx:
    @pytest.mark.parametrize("ram", [8, 16, 32, 64, None])
    def test_never_recommends_less_than_the_prompt_needs(self, ram):
        # A window below the prompt guarantees truncation of the system prompt.
        assert hardware.recommended_num_ctx(machine(ram), REQUIRED) >= REQUIRED

    def test_respects_the_model_maximum(self):
        assert hardware.recommended_num_ctx(
            machine(64), REQUIRED, model_max=32_768
        ) <= 32_768

    def test_returns_a_power_of_two(self):
        value = hardware.recommended_num_ctx(machine(32), REQUIRED)
        assert value & (value - 1) == 0

    def test_leaves_headroom_above_the_bare_requirement(self):
        assert hardware.recommended_num_ctx(machine(32), 8_000) > 8_000


class TestNumCtxWarning:
    def test_warns_when_the_window_is_smaller_than_the_prompt(self):
        warning = hardware.num_ctx_warning(8_192, REQUIRED, machine(32))
        assert warning and "smaller than the prompt" in warning

    def test_warns_when_the_window_exceeds_what_the_machine_should_attempt(self):
        warning = hardware.num_ctx_warning(262_144, REQUIRED, machine(8))
        assert warning and "advisable" in warning

    def test_silent_for_a_sensible_choice(self):
        assert hardware.num_ctx_warning(65_536, REQUIRED, machine(32)) is None


class TestAdvice:
    def test_advice_varies_with_available_memory(self):
        small = " ".join(hardware.advice(machine(6)))
        large = " ".join(hardware.advice(machine(64)))
        assert small != large

    def test_small_machine_is_told_to_consider_the_hosted_option(self):
        assert "hosted" in " ".join(hardware.advice(machine(6)))

    def test_apple_silicon_gets_the_mlx_hint(self):
        assert "mlx" in " ".join(hardware.advice(machine(18, "Darwin", "arm64")))

    def test_non_apple_machines_do_not_get_the_mlx_hint(self):
        assert "mlx" not in " ".join(hardware.advice(machine(18, "Linux", "x86_64")))

    def test_unknown_ram_still_produces_usable_guidance(self):
        notes = hardware.advice(machine(None))
        assert notes and "Could not detect" in notes[0]
