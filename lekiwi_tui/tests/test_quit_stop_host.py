"""Quitting the app while the Pi host session is backgrounded.

The host stream is deliberately app-lifetime (q on the host screen backgrounds it), so a
plain quit used to drop the PTY and SIGHUP the remote host past its finally-block: torque
left on, cameras left claimed. These tests pin the three pieces of the fix: the menu gates
quit behind a confirm modal when (and only when) a session is live, the confirm flow quits
only on the explicit stop choice, and StreamController.shutdown_sync stops a child without
the asyncio loop (the state __main__ is in when it runs).
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import types
from unittest.mock import MagicMock

from lekiwi_tui.framework.events import ESC, Key
from lekiwi_tui.framework.screen import Invoke, Quit
from lekiwi_tui.framework.stream import StreamController
from lekiwi_tui.screens.menu import MenuScreen

from conftest import make_ctx


def _screen(ui_state: dict | None = None) -> MenuScreen:
    return MenuScreen(MagicMock(), make_ctx(ui_state=ui_state))


# ── the menu-side gate ─────────────────────────────────────────────────────────
def test_q_quits_directly_when_no_host_session_exists():
    assert isinstance(_screen().handle_key(Key(name="q")), Quit)


def test_q_quits_directly_when_the_session_already_ended():
    """An ended stream must not gate the quit — otherwise every quit after a normal
    host stop would show a pointless confirm."""
    ended = types.SimpleNamespace(running=False)
    act = _screen({"host_stream": ended}).handle_key(Key(name="q"))
    assert isinstance(act, Quit)


def test_q_and_esc_confirm_when_the_host_stream_is_running():
    """A live session must route through the confirm flow, for BOTH quit keys — a
    direct Quit here is the SIGHUP-past-cleanup bug this file exists to prevent."""
    for name in ("q", ESC):
        live = types.SimpleNamespace(running=True)
        act = _screen({"host_stream": live}).handle_key(Key(name=name))
        assert isinstance(act, Invoke), f"key {name!r} bypassed the confirm"


# ── the confirm flow ───────────────────────────────────────────────────────────
def _run_flow(modal_result: str | None) -> MagicMock:
    """Drive _confirm_quit_with_host with a canned modal answer; return the app mock."""
    screen = _screen({"host_stream": types.SimpleNamespace(running=True)})

    async def fake_run_modal(_modal):
        return modal_result

    screen.app.run_modal = fake_run_modal
    asyncio.run(screen._confirm_quit_with_host())
    return screen.app


def test_confirm_stop_choice_quits_the_app():
    app = _run_flow(MenuScreen.STOP_AND_QUIT)
    app.quit.assert_called_once()


def test_confirm_cancel_choice_stays_in_the_app():
    app = _run_flow("Cancel")
    app.quit.assert_not_called()


def test_confirm_dismissed_modal_stays_in_the_app():
    """Esc out of the modal (result None) must behave like Cancel, not like consent."""
    app = _run_flow(None)
    app.quit.assert_not_called()


# ── shutdown_sync: the loop-free stop __main__ relies on ──────────────────────
def _controller_with(proc, master=None, grace: float = 5.0) -> StreamController:
    ctl = StreamController(grace=grace)
    ctl._proc = proc
    ctl._master = master
    ctl.phase = "running"
    return ctl


def test_shutdown_sync_is_a_noop_without_a_child():
    assert StreamController().shutdown_sync() is None


def test_shutdown_sync_is_a_noop_when_the_child_already_exited():
    done = types.SimpleNamespace(pid=99999, returncode=0)
    assert _controller_with(done).shutdown_sync() is None


def test_shutdown_sync_reports_stopped_when_the_child_exits_within_grace():
    """The graceful branch: a child that obeys within the grace window is 'stopped',
    reaped os-level (no asyncio loop involved), and the phase flips to ended."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.3)"],
                            start_new_session=True)
    ctl = _controller_with(proc)
    try:
        assert ctl.shutdown_sync(grace=5.0) == "stopped"
        assert ctl.ended
    finally:
        proc.returncode = 0  # already reaped via os.waitpid; quiet Popen.__del__


def test_shutdown_sync_escalates_to_sigkill_on_a_wedged_child():
    """The escalation branch: a child that ignores the grace window is killed and
    reported as 'killed' — the caller warns that the arm may stay stiff."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            start_new_session=True)
    ctl = _controller_with(proc)
    try:
        assert ctl.shutdown_sync(grace=0.2) == "killed"
        assert ctl.ended
    finally:
        proc.returncode = 0
