from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import lekiwi_tui.screens.teleop as teleop_mod
from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.framework.events import DOWN, ENTER, Key
from lekiwi_tui.framework.screen import Invoke, Nothing
from lekiwi_tui.screens.teleop import TELEOP_SCRIPT, TeleopScreen


ROOT = Path(__file__).resolve().parents[2]


def _public_workspace(tmp_path: Path) -> dict[str, str]:
    """Fresh-checkout workspace: public example config only, no private lekiwi.yaml."""
    shutil.copy2(ROOT / "lekiwi.example.yaml", tmp_path / "lekiwi.example.yaml")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    return {**os.environ, "LEKIWI_ROOT": str(tmp_path)}


def _ctx() -> Context:
    return Context(
        cfg=Config(values={"LEADER_PORT": "/dev/ttyACM0", "LEKIWI_HOST": "lekiwi"}),
        doc={"teleop": {"display_data": True, "fps": 30}},
        gpu_name="",
        is_tty=True,
    )


class _FakeApp:
    def __init__(self) -> None:
        self.suspended = None

    async def suspend(self, argv, **kwargs):  # noqa: ANN001
        self.suspended = (list(argv), kwargs)
        return 0


def test_teleop_form_does_not_consume_wasd_as_robot_control_before_start():
    screen = TeleopScreen(None, _ctx())

    action = screen.handle_key(Key("w"))

    assert action is Nothing
    assert screen.display.value is True


def test_teleop_start_still_suspends_to_script_with_passthrough_args(monkeypatch):
    async def _preflight_ok(*args, **kwargs):  # noqa: ANN001
        return True

    monkeypatch.setattr(teleop_mod, "confirm_preflight", _preflight_ok)
    app = _FakeApp()
    screen = TeleopScreen(None, _ctx(), extra=["--robot.foo=bar"])
    screen.app = app
    for _ in range(3):
        screen.handle_key(Key(DOWN))

    action = screen.handle_key(Key(ENTER))

    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())
    assert app.suspended is not None
    argv, kwargs = app.suspended
    assert kwargs["title"] == "teleop"
    assert argv == [
        "bash",
        str(TELEOP_SCRIPT),
        "--display",
        "on",
        "--fps",
        "30",
        "--duration",
        "0",
        "--robot.foo=bar",
    ]


def test_headless_dry_run_teleop_prints_preview_not_real_run(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "lekiwi_tui", "--dry-run", "teleop"],
        env=_public_workspace(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "lerobot_teleop_kbd.py" in proc.stdout
    assert "--display_data=" in proc.stdout


def test_headless_dry_run_host_launch_does_not_scp_or_ssh(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "lekiwi_tui", "--dry-run", "host-launch"],
        env=_public_workspace(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "[preview] command would run:" in proc.stdout
    assert "ssh" in proc.stdout


def test_headless_dry_run_host_kill_does_not_ssh(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "lekiwi_tui", "--dry-run", "host-kill"],
        env=_public_workspace(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "[preview] command would run:" in proc.stdout
    assert "ssh" in proc.stdout
    assert "emit-kill" not in proc.stderr


def test_unknown_cli_action_suggests_near_matches(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "lekiwi_tui", "hots"],
        env={**os.environ, "LEKIWI_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "unknown action: 'hots'" in proc.stderr
    assert "did you mean:" in proc.stderr
    assert "host" in proc.stderr
