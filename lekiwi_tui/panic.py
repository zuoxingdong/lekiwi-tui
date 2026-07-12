"""panic.py — the double-K emergency host stop, reachable from ANY screen.

Safety affordances should not require navigating to the Stop-host screen. A single
capital ``K`` ARMS the panic (a toast says so); a second capital ``K`` within the
window fires the same remote kill bash the HostKillScreen uses (``host.sh emit-kill``
over ``ssh``), in a background thread so the UI never blocks. Capital-K-twice is
deliberate: it cannot be typed by accident while wasd-ing a robot or filling a lowercase
robot-id field, and modals never see it (the app-level hook only runs in the main loop).

Wired through :class:`~.framework.app.App`'s ``global_key`` hook by ``__main__``.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

from . import ROOT
from .remote import RemoteValueError, validate_remote_name, validate_ssh_host

if TYPE_CHECKING:
    from .context import Context
    from .framework.app import App

HOST_SCRIPT = ROOT / "scripts" / "host.sh"

#: Seconds the first K stays armed.
ARM_WINDOW = 2.0

_armed_at = 0.0


def _kill_argv(ctx: "Context") -> list[str]:
    host = validate_ssh_host(ctx.cfg["LEKIWI_HOST"])
    robot_id = validate_remote_name(str(ctx.cfg["ROBOT_ID"]), "robot id")
    remote = subprocess.check_output(
        ["bash", str(HOST_SCRIPT), "emit-kill", "--robot-id", robot_id], text=True)
    return ["ssh", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes", host, remote]


def _fire(app: "App", ctx: "Context") -> None:
    """Run the kill in a daemon thread; toast the outcome from the render side via a
    plain attribute handoff (App.notify is safe to call cross-thread: it only appends
    to a list the render loop reads)."""
    try:
        argv = _kill_argv(ctx)
    except (RemoteValueError, subprocess.SubprocessError, OSError) as exc:
        app.notify(f"✗ panic stop could not start: {exc}", "error")
        return

    def run() -> None:
        try:
            rc = subprocess.run(argv, capture_output=True, text=True, timeout=20).returncode
        except (OSError, subprocess.TimeoutExpired):
            rc = 1
        if rc == 0:
            app.notify("🛑 panic stop sent — host processes killed", "warn")
        else:
            app.notify("✗ panic stop failed — use Stop host / check the Pi", "error")

    ctx.ui_state.pop("host_session", None)   # chip stops counting down immediately
    threading.Thread(target=run, daemon=True).start()
    app.notify("🛑 panic: stopping the host…", "warn")


def make_global_key(ctx: "Context"):
    """Build the App ``global_key`` hook: returns an awaitable when the key was consumed
    (the App awaits it and swallows the key), None to pass the key to the screen."""

    async def _noop() -> None:
        return None

    def hook(app: "App", key: Any) -> Any:
        global _armed_at
        if key.name != "K" or key.ctrl or key.alt:
            return None
        now = time.monotonic()
        if now - _armed_at <= ARM_WINDOW:
            _armed_at = 0.0
            _fire(app, ctx)
        else:
            _armed_at = now
            app.notify("panic armed — press K again within 2s to stop the host", "warn")
        return _noop()

    return hook


__all__ = ["make_global_key", "ARM_WINDOW"]
