"""sysstat.py — throttled, non-blocking laptop resource sampling for the status card.

The menu's status card wanted the robot's bus voltage and servo temperatures, but the Pi's
host process owns the serial bus and publishes neither, so there is no laptop-side source
for them. What the laptop CAN answer about itself is CPU, RAM, GPU and VRAM, which is what
this module provides.

Built to the same contract as :mod:`~lekiwi_tui.hostprobe`, for the same reason: ``draw``
runs every frame, so sampling must never happen on the render path. :meth:`SysStat.poll` is
called from ``draw`` but only *starts* a sample when the throttle window has elapsed and no
sample is in flight; the reads happen on a daemon thread and later frames pick up the
result. ``nvidia-smi`` in particular costs tens of milliseconds and would be visible as
stutter if it ran inline.

Everything is best-effort and degrades to ``None`` per field: no ``/proc`` (macOS), no
NVIDIA GPU, or an unparseable line each leave that field empty and the card simply omits
it. A diagnostic must never be able to break the screen it decorates.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Seconds between samples. CPU and VRAM move fast, but a status card is glanced at, not
#: watched — and this bounds how often nvidia-smi is spawned.
INTERVAL = 2.0

#: Window used to measure CPU busy-time. /proc/stat is cumulative, so a percentage needs
#: two reads separated in time; this happens on the sample thread, never on the render path.
CPU_WINDOW = 0.15

#: Per-sample budget for nvidia-smi. Generous: it is on a thread and a slow answer just
#: means the previous sample stays on screen one cycle longer.
GPU_TIMEOUT = 4.0

_GB = 1024 ** 3


@dataclass(frozen=True)
class Sample:
    """One resource reading. Every field is optional: absent means "could not tell"."""

    cpu_pct: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    gpu_pct: int | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None


def _cpu_times() -> tuple[int, int] | None:
    """(total jiffies, idle jiffies) from /proc/stat, or None where /proc is absent."""
    try:
        fields = Path("/proc/stat").read_text().split("\n", 1)[0].split()
    except OSError:
        return None
    if len(fields) < 5 or fields[0] != "cpu":
        return None
    try:
        values = [int(v) for v in fields[1:]]
    except ValueError:
        return None
    # Layout: user nice system idle iowait irq softirq steal ... — idle+iowait is "not busy".
    return sum(values), values[3] + (values[4] if len(values) > 4 else 0)


def _sample_cpu() -> float | None:
    first = _cpu_times()
    if first is None:
        return None
    time.sleep(CPU_WINDOW)
    second = _cpu_times()
    if second is None:
        return None
    d_total, d_idle = second[0] - first[0], second[1] - first[1]
    if d_total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))


def _sample_ram() -> tuple[float, float] | None:
    """(used GB, total GB). Uses MemAvailable, which is the kernel's own estimate of what a
    new workload could claim — a truer "used" than total-minus-free, which counts cache."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    kb: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            try:
                kb[key] = int(rest.split()[0])
            except (IndexError, ValueError):
                return None
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        return None
    total = kb["MemTotal"] * 1024 / _GB
    return total - kb["MemAvailable"] * 1024 / _GB, total


def _sample_gpu() -> tuple[int, float, float] | None:
    """(utilisation %, VRAM used GB, VRAM total GB) via nvidia-smi, or None without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=GPU_TIMEOUT, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().split("\n")[0] if out.strip() else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    try:
        # nvidia-smi reports memory in MiB.
        return int(float(parts[0])), float(parts[1]) * 1024 ** 2 / _GB, float(parts[2]) * 1024 ** 2 / _GB
    except ValueError:
        return None


class SysStat:
    """Background CPU/RAM/GPU sampling. Read :attr:`sample`, call :meth:`poll` from the
    render path to keep it fresh. ``sample`` is an empty :class:`Sample` until the first
    result lands, so callers never have to handle None."""

    def __init__(self) -> None:
        self.sample = Sample()
        self._last_start = 0.0
        self._inflight = False
        self._lock = threading.Lock()

    def poll(self) -> None:
        """Kick a sample if the throttle window elapsed. Returns immediately."""
        now = time.monotonic()
        with self._lock:
            if self._inflight or (now - self._last_start) < INTERVAL:
                return
            self._inflight = True
            self._last_start = now
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        cpu = _sample_cpu()
        ram = _sample_ram()
        gpu = _sample_gpu()
        result = Sample(
            cpu_pct=cpu,
            ram_used_gb=ram[0] if ram else None,
            ram_total_gb=ram[1] if ram else None,
            gpu_pct=gpu[0] if gpu else None,
            vram_used_gb=gpu[1] if gpu else None,
            vram_total_gb=gpu[2] if gpu else None,
        )
        with self._lock:
            self.sample = result
            self._inflight = False


def get_sysstat(ctx: Any) -> SysStat:
    """The shared per-app sampler, kept on the context like the host probe is."""
    stat = ctx.ui_state.get("sysstat")
    if not isinstance(stat, SysStat):
        stat = SysStat()
        ctx.ui_state["sysstat"] = stat
    return stat


__all__ = ["SysStat", "Sample", "get_sysstat", "INTERVAL"]
