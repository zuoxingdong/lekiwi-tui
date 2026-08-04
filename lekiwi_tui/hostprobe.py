"""hostprobe.py — a throttled, non-blocking liveness probe for the Pi host.

Answers the question every screen's chip row wants to answer live: "is the robot up?".
The probe is a plain TCP connect to the host's ZMQ command port (the lekiwi_host binds
it for its whole session; a stray TCP connect+close is harmless to ZMQ). It NEVER runs
on the render path: :meth:`poll` is called from ``draw`` every frame, but it only
*starts* a probe when the throttle window has elapsed and no probe is in flight — the
connect itself happens on a daemon thread and stores its result for later frames to
read. So a dead/unroutable host costs the UI nothing (the thread eats the timeout).

One instance is shared app-wide through ``ctx.ui_state["hostprobe"]`` (see
:func:`get_probe`); it re-targets itself automatically when LEKIWI_HOST changes in
Settings mid-session.
"""
from __future__ import annotations

import socket
import subprocess
import threading
import time
from typing import Any

#: The lekiwi_host command (PULL) port — bound for the whole host session.
DEFAULT_PORT = 5555

#: Seconds between probes; a robot host coming up/down is a human-scale event.
INTERVAL = 3.0

#: Per-probe connect budget. Short: LAN or nothing.
TIMEOUT = 0.8


class HostProbe:
    """Background TCP liveness for one host:port. Read ``alive`` (None until the first
    result lands), call :meth:`poll` from the render path to keep it fresh."""

    def __init__(self, host: str, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self.alive: bool | None = None      # None = not probed yet
        self._addr: str | None = None       # resolved connect target (see _resolve)
        self._last_start = 0.0
        self._inflight = False
        self._lock = threading.Lock()

    def _resolve(self) -> str:
        """The address to actually connect to. LEKIWI_HOST is usually an ~/.ssh/config
        ALIAS (`Host lekiwi` → `HostName 192.168.x.x`) that plain DNS cannot resolve —
        which made the chip report ○ while the host was clearly up. Resolve through
        `ssh -G` (it evaluates the ssh config exactly like ssh itself) and cache; DNS-
        resolvable names short-circuit via gethostbyname. Runs on the probe thread only."""
        if self._addr:
            return self._addr
        addr = self.host
        try:
            socket.gethostbyname(self.host)
        except OSError:
            try:
                out = subprocess.run(
                    ["ssh", "-G", self.host], capture_output=True, text=True, timeout=3,
                ).stdout
                for line in out.splitlines():
                    if line.startswith("hostname "):
                        addr = line.split(None, 1)[1].strip()
                        break
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._addr = addr
        return addr

    def poll(self) -> None:
        """Kick a probe if the throttle window elapsed. Returns immediately."""
        now = time.monotonic()
        with self._lock:
            if self._inflight or (now - self._last_start) < INTERVAL:
                return
            self._inflight = True
            self._last_start = now
        threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self) -> None:
        ok = False
        try:
            with socket.create_connection((self._resolve(), self.port), timeout=TIMEOUT):
                ok = True
        except OSError:
            ok = False
        with self._lock:
            self.alive = ok
            self._inflight = False


def get_probe(ctx: Any) -> "HostProbe | None":
    """The shared per-app probe, (re)built when LEKIWI_HOST changes. Returns None when
    the host value fails basic sanity (empty), so chips degrade to config-only."""
    try:
        host = str(ctx.cfg["LEKIWI_HOST"]).strip()
    except Exception:
        return None
    if not host:
        return None
    probe = ctx.ui_state.get("hostprobe")
    if not isinstance(probe, HostProbe) or probe.host != host:
        probe = HostProbe(host)
        ctx.ui_state["hostprobe"] = probe
    return probe


def session_remaining(ctx: Any) -> int | None:
    """Seconds left in the announced host session (host.py publishes
    ``ctx.ui_state['host_session'] = {'ends_at': monotonic}`` on launch), or None when
    no session is announced / it already expired."""
    info = ctx.ui_state.get("host_session")
    if not isinstance(info, dict):
        return None
    ends_at = info.get("ends_at")
    if not isinstance(ends_at, (int, float)):
        return None
    left = int(ends_at - time.monotonic())
    return left if left > 0 else None


__all__ = ["HostProbe", "get_probe", "session_remaining", "DEFAULT_PORT"]


def host_alive(ctx: Any) -> bool | None:
    """The probe's live verdict for slim headers / warning-as-plan rows: True/False,
    or None while unknown (no probe configured, or first poll still in flight)."""
    probe = get_probe(ctx)
    return probe.alive if probe is not None else None
