"""Detect cameras ON THE ROBOT: the emitter's golden bash, its refusals, and the screen
key that fronts it.

The whole feature exists because the device nodes in lekiwi.yaml are Pi-side and renumber
themselves, so the two things worth pinning are (a) the probe really is list-only, since a
capture run writes files on the Pi for no reason, and (b) it refuses while the host session
holds the cameras, since then it would report failures for the devices that work.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lekiwi_tui import ROOT
from lekiwi_tui.framework.events import Key
from lekiwi_tui.framework.screen import Invoke, Nothing
from lekiwi_tui.screens.robot_config import RobotConfigScreen, build_find_cameras_argv

from conftest import make_ctx

FIND_SH = ROOT / "scripts" / "find_cameras.sh"


def _emit(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(FIND_SH), *args], capture_output=True, text=True)


# ── the emitted remote bash ───────────────────────────────────────────────────


def test_the_default_probe_is_list_only():
    """--record-time 0 keeps lerobot's printout and writes no frames on the Pi."""
    out = _emit("emit-detect", "--conda-env", "lekiwi")
    assert out.returncode == 0
    assert "lerobot-find-cameras opencv" in out.stdout
    assert "--record-time-s 0" in out.stdout
    assert "--output-dir /tmp/lekiwi-find-cameras" in out.stdout, "never the Pi's home"
    assert "mamba activate lekiwi" in out.stdout


def test_frames_are_opt_in_and_the_backend_widens_on_request():
    out = _emit("emit-detect", "--conda-env", "lekiwi", "--backend", "all", "--record-time", "2")
    assert "--record-time-s 2" in out.stdout
    # `all` means no type filter, so realsense is probed too
    assert "lerobot-find-cameras  \\" in out.stdout
    assert "record-time 2s" in out.stdout, "the remote banner states what it is about to do"


def test_realsense_only_is_expressible():
    assert "lerobot-find-cameras realsense" in _emit(
        "emit-detect", "--conda-env", "lekiwi", "--backend", "realsense").stdout


@pytest.mark.parametrize(("args", "message"), [
    (("emit-detect", "--conda-env", "bad env"), "whitespace"),
    (("emit-detect", "--conda-env", "lekiwi", "--backend", "usb"), "--backend must be"),
    (("emit-detect", "--conda-env", "lekiwi", "--record-time", "2.5"), "whole number"),
    (("emit-detect", "--conda-env", "lekiwi", "--warmup", "-1"), "whole number"),
    (("emit-detect", "--conda-env", "lekiwi", "--nope"), "unknown flag"),
    (("wat", "--conda-env", "lekiwi"), "unknown subcommand"),
])
def test_the_emitter_refuses_bad_input(args, message):
    out = _emit(*args)
    assert out.returncode == 2 and message in out.stderr


def test_a_missing_env_is_rejected_rather_than_emitted():
    """An empty conda env would emit `mamba activate` with no argument."""
    out = _emit("emit-detect", "--conda-env", "")
    assert out.returncode == 2 and "must not be empty" in out.stderr


# ── the ssh argv ──────────────────────────────────────────────────────────────


def test_the_argv_is_ssh_plus_the_emitted_payload():
    argv = build_find_cameras_argv(make_ctx(gpu_name=""))
    assert argv[0] == "ssh"
    assert "ConnectTimeout=5" in argv, "a powered-down robot must not hang the suspend"
    assert "-t" not in argv, "nothing here reads keys; -t would only complicate the suspend"
    assert "lerobot-find-cameras" in argv[-1]


# ── the screen key ────────────────────────────────────────────────────────────


class _App:
    def __init__(self) -> None:
        self.suspended: list[str] | None = None
        self.toasts: list[tuple[str, str]] = []

    async def suspend(self, argv, **kwargs):  # noqa: ANN001, ANN003
        self.suspended = list(argv)
        return 0

    def notify(self, msg, level="info", **kwargs):  # noqa: ANN001, ANN003
        self.toasts.append((msg, level))


def _screen(monkeypatch, alive):
    import lekiwi_tui.hostprobe as hostprobe

    monkeypatch.setattr(hostprobe, "host_alive", lambda ctx: alive)
    app = _App()
    return app, RobotConfigScreen(app, make_ctx(gpu_name=""))


def test_f_returns_a_flow_and_the_other_keys_still_work(monkeypatch):
    _, screen = _screen(monkeypatch, False)
    assert isinstance(screen.handle_key(Key(name="f")), Invoke)
    assert screen.handle_key(Key(name="x")) is Nothing


def test_it_refuses_while_the_host_holds_the_cameras(monkeypatch):
    import asyncio

    app, screen = _screen(monkeypatch, True)
    asyncio.run(screen._detect_cameras())
    assert app.suspended is None, "must not probe cameras the host has open"
    assert app.toasts and "stop it first" in app.toasts[0][0] and app.toasts[0][1] == "warn"


def test_it_probes_when_the_host_is_down_or_unknown(monkeypatch):
    import asyncio

    for alive in (False, None):   # None = probe in flight; a maybe must not block
        app, screen = _screen(monkeypatch, alive)
        asyncio.run(screen._detect_cameras())
        assert app.suspended is not None and app.suspended[0] == "ssh"
        assert "lerobot-find-cameras" in app.suspended[-1]
        assert app.toasts == []


def test_the_hint_line_advertises_the_key(monkeypatch):
    _, screen = _screen(monkeypatch, False)
    source = Path(ROOT / "lekiwi_tui" / "screens" / "robot_config.py").read_text()
    assert '("f", "detect cameras")' in source
