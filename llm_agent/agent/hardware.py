"""
hardware.py — detect what this machine can actually run, and size local-model
settings from that.

Sizing advice for local inference is worthless in the abstract: 32k of context is
trivial on a 64 GB workstation and impossible on an 8 GB laptop. Everything here
is derived from the running machine so the setup screen can give advice that
applies to the user in front of it, rather than to whichever machine the
defaults were written on.

No third-party dependencies — this has to work before anything is configured.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Leave room for the OS, the browser, Streamlit, and CONDUCTOR's own pandapower
# and Pyomo work, which run alongside the model.
_HEADROOM_GB = 5.0

# Context ceilings by usable memory. Deliberately a coarse band rather than a
# computed byte figure: KV-cache size depends on layer count, KV-head count, head
# dimension, and cache quantisation, none of which are derivable from a
# parameter count. A tier that is roughly right beats a formula that is
# confidently wrong.
_CTX_CEILING_BY_USABLE_GB = (
    (4, 16_384),
    (8, 32_768),
    (16, 65_536),
    (32, 131_072),
)
_CTX_CEILING_LARGE = 262_144


def total_ram_gb() -> float | None:
    """Physical RAM in GB, or None if it can't be determined."""
    try:  # Linux, macOS, most Unix
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2**30
    except (ValueError, AttributeError, OSError):
        pass

    if platform.system() == "Windows":
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
        except Exception:  # noqa: BLE001
            logger.debug("Windows memory query failed.", exc_info=True)

    return None


@dataclass
class Hardware:
    system: str
    machine: str
    ram_gb: float | None
    cpu_count: int | None

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"

    @property
    def usable_gb(self) -> float | None:
        """RAM plausibly available to a model once everything else has its share."""
        if self.ram_gb is None:
            return None
        return max(self.ram_gb - _HEADROOM_GB, 1.0)

    def describe(self) -> str:
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "unknown RAM"
        chip = "Apple Silicon" if self.apple_silicon else f"{self.system} {self.machine}"
        return f"{chip}, {ram}"


def detect() -> Hardware:
    return Hardware(
        system=platform.system(),
        machine=platform.machine(),
        ram_gb=total_ram_gb(),
        cpu_count=os.cpu_count(),
    )


def weights_fit(weights_gb: float, hw: Hardware) -> bool | None:
    """
    Whether the model's weights alone fit in usable memory. None if RAM unknown.

    Deliberately weights-only: that figure is knowable (it's the download size).
    The KV cache on top is not predictable from the information we have, so it is
    treated as headroom rather than estimated.
    """
    usable = hw.usable_gb
    if usable is None:
        return None
    return weights_gb <= usable


def context_ceiling(hw: Hardware) -> int:
    """Largest context window worth suggesting on this machine."""
    usable = hw.usable_gb
    if usable is None:
        return 32_768  # unknown hardware: assume a modest machine
    for limit_gb, ceiling in _CTX_CEILING_BY_USABLE_GB:
        if usable < limit_gb:
            return ceiling
    return _CTX_CEILING_LARGE


def recommended_num_ctx(
    hw: Hardware,
    required_tokens: int,
    model_max: int | None = None,
) -> int:
    """
    A context window that holds the prompt plus room for conversation, without
    exceeding what this machine should attempt.

    Returns at least `required_tokens` rounded up to a power of two even when
    that exceeds the machine's tier — a window smaller than the prompt would
    truncate the system prompt, so the caller must surface that as a problem
    rather than silently accept it (see `num_ctx_warning`).
    """
    floor = 8192
    while floor < required_tokens:
        floor *= 2

    ceiling = min(context_ceiling(hw), model_max or _CTX_CEILING_LARGE)
    # One doubling above the bare requirement gives room for tool results.
    return max(floor, min(floor * 2, ceiling))


def num_ctx_warning(num_ctx: int, required_tokens: int, hw: Hardware) -> str | None:
    """A caution to show alongside a chosen window, or None if it looks sensible."""
    if num_ctx < required_tokens:
        return (
            f"This window ({num_ctx:,}) is smaller than the prompt itself "
            f"(~{required_tokens:,} tokens). Requests will be refused rather than "
            "silently truncated — choose a larger window or a smaller tool set."
        )
    if num_ctx > context_ceiling(hw):
        return (
            f"This window is larger than typically advisable for the detected "
            f"memory ({hw.describe()}). It may still work — memory use depends on "
            "the model's architecture — but watch for swapping."
        )
    return None


def advice(hw: Hardware) -> list[str]:
    """Plain-language guidance for choosing a local model on *this* machine."""
    usable = hw.usable_gb
    if usable is None:
        return [
            "Could not detect how much memory this machine has — pick a model "
            "whose download size leaves several GB free alongside your other work."
        ]

    notes = [
        f"Detected {hw.describe()}. Roughly {usable:.0f} GB is realistically "
        f"available for a model once the OS, browser, and CONDUCTOR's own solver "
        f"have their share."
    ]

    if usable < 6:
        notes.append(
            "That is tight for tool-calling work. Expect to need a small model, "
            "and consider the hosted Google option instead."
        )
    elif usable < 12:
        notes.append(
            "A model in the 4–8 GB range should fit comfortably. Larger ones may "
            "run but will compete with the grid solver for memory."
        )
    else:
        notes.append(
            "There is room for a larger model; quality of tool selection usually "
            "improves with size."
        )

    if hw.apple_silicon:
        notes.append(
            "On Apple Silicon, Ollama's `-mlx` model tags use the GPU more "
            "effectively and are generally the faster choice."
        )

    return notes
