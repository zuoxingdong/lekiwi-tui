from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lekiwi_tui.remote import (
    RemoteValueError,
    validate_positive_int,
    validate_remote_name,
    validate_ssh_host,
)
from lekiwi_tui.screens.host import build_host_ssh_argv


ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def _public_workspace(tmp_path: Path) -> Path:
    """Fresh-checkout workspace: public example config only, no private lekiwi.yaml."""
    shutil.copy2(ROOT / "lekiwi.example.yaml", tmp_path / "lekiwi.example.yaml")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    return tmp_path


def _run_in(
    cwd: Path,
    args: list[str],
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def test_python_remote_validators_accept_simple_values():
    assert validate_ssh_host("pi@lekiwi.local") == "pi@lekiwi.local"
    assert validate_remote_name("lekiwi_env-1", "conda env") == "lekiwi_env-1"
    assert validate_positive_int("0600", "connection time") == "600"


@pytest.mark.parametrize("host", ["", "-bad", "bad host", "bad;host", "bad/host"])
def test_python_remote_validator_rejects_unsafe_hosts(host: str):
    with pytest.raises(RemoteValueError):
        validate_ssh_host(host)


def test_host_launch_builder_rejects_unsafe_host_before_ssh():
    with pytest.raises(RemoteValueError):
        build_host_ssh_argv(
            "-bad",
            "lekiwi",
            "600",
            conda_env="lekiwi",
            cfg_flag="",
            loop_flag="--host.max_loop_freq_hz=30",
        )


def test_sync_dry_run_quotes_remote_repo_with_spaces():
    proc = _run([
        "bash",
        "scripts/sync.sh",
        "--dry-run",
        "--host",
        "pi@lekiwi.local",
        "--repo",
        "le kiwi/lerobot",
    ])

    assert proc.returncode == 0, proc.stderr
    assert "pi@lekiwi.local:le\\ kiwi/lerobot/" in proc.stdout


def test_sync_dry_run_uses_example_config_when_private_config_is_absent(tmp_path):
    workspace = _public_workspace(tmp_path)

    proc = _run_in(workspace, ["bash", "scripts/sync.sh", "--dry-run"])

    assert proc.returncode == 0, proc.stderr
    assert "rsync" in proc.stdout


def test_sync_rejects_host_that_could_be_an_option():
    proc = _run([
        "bash",
        "scripts/sync.sh",
        "--dry-run",
        "--host",
        "-bad",
        "--repo",
        "lekiwi/lerobot",
    ])

    assert proc.returncode == 2
    assert "LEKIWI_HOST must not start with '-'" in proc.stderr


def test_pi_provision_rejects_invalid_env_name_before_dry_run():
    proc = _run(
        ["bash", "scripts/pi_provision.sh", "--dry-run", "conda"],
        env={"PI_ENV": "bad env"},
    )

    assert proc.returncode == 2
    assert "PI_ENV must not contain whitespace" in proc.stderr


def test_pi_provision_accepts_repo_path_with_spaces_in_dry_run():
    proc = _run(
        ["bash", "scripts/pi_provision.sh", "--dry-run", "lerobot"],
        env={"PI_REPO": "le kiwi/lerobot"},
    )

    assert proc.returncode == 0, proc.stderr
    assert "PI_REPO=le kiwi/lerobot" in proc.stdout


def test_host_emitter_rejects_unsafe_robot_id():
    proc = _run([
        "bash",
        "scripts/host.sh",
        "emit-kill",
        "--robot-id",
        "bad id",
    ])

    assert proc.returncode == 2
    assert "robot id must not contain whitespace" in proc.stderr


def test_calibrate_follower_rejects_passthrough_flags():
    proc = _run([
        "bash",
        "scripts/calibrate.sh",
        "--dry-run",
        "--target",
        "follower",
        "--host",
        "lekiwi",
        "--conda-env",
        "lekiwi",
        "--robot-id",
        "lekiwi",
        "--extra",
    ])

    assert proc.returncode == 2
    assert "follower calibration does not accept passthrough flags" in proc.stderr


# ── ROBOT_TYPE: the follower type flows from config into both launchers ────────


def test_host_emit_launch_defaults_to_pincopen_plugin_module():
    proc = _run([
        "bash", "scripts/host.sh", "emit-launch",
        "--conda-env", "lekiwi", "--robot-id", "lekiwi", "--connection-time", "600",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "python -m lerobot_robot_lekiwi_pincopen.lekiwi_host" in proc.stdout


def test_host_emit_launch_robot_type_lekiwi_uses_stock_module():
    proc = _run([
        "bash", "scripts/host.sh", "emit-launch",
        "--conda-env", "lekiwi", "--robot-id", "lekiwi", "--robot-type", "lekiwi",
        "--connection-time", "600",
    ])
    assert proc.returncode == 0, proc.stderr
    assert "python -m lerobot.robots.lekiwi.lekiwi_host" in proc.stdout


def test_host_emit_launch_rejects_unknown_robot_type():
    # A case-map whitelist, not string splicing: an arbitrary type must never reach
    # the remote `python -m` line.
    proc = _run([
        "bash", "scripts/host.sh", "emit-launch",
        "--conda-env", "lekiwi", "--robot-id", "lekiwi", "--robot-type", "evil.module",
        "--connection-time", "600",
    ])
    assert proc.returncode == 2
    assert "unknown robot type" in proc.stderr


def test_calibrate_follower_remote_carries_robot_type():
    default = _run([
        "bash", "scripts/calibrate.sh", "emit-follower-remote",
        "--conda-env", "lekiwi", "--robot-id", "lekiwi",
    ])
    stock = _run([
        "bash", "scripts/calibrate.sh", "emit-follower-remote",
        "--conda-env", "lekiwi", "--robot-id", "lekiwi", "--robot-type", "lekiwi",
    ])
    assert default.returncode == 0 and stock.returncode == 0
    assert "--robot.type=lekiwi_pincopen " in default.stdout
    assert "--robot.type=lekiwi " in stock.stdout


def test_build_host_ssh_argv_rejects_unsafe_robot_type():
    with pytest.raises(RemoteValueError):
        build_host_ssh_argv(
            "lekiwi", "lekiwi", "600",
            conda_env="lekiwi", robot_type="bad type",
        )


# ── LOCAL_REPO/LOCAL_PLUGIN config + provenance + provision env plumbing ────────


def test_sync_dry_run_honors_local_repo_env_override():
    proc = _run(
        ["bash", "scripts/sync.sh", "--dry-run"],
        env={"LOCAL_REPO": "/tmp/elsewhere/lerobot", "LOCAL_PLUGIN": "/tmp/elsewhere/plugin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "/tmp/elsewhere/lerobot/" in proc.stdout
    assert "/tmp/elsewhere/plugin/" in proc.stdout


def test_provision_env_passes_paths_and_python_knobs(monkeypatch):
    from lekiwi_tui.screens.provision import provision_env

    cfg = {
        "LEKIWI_HOST": "lekiwi",
        "CONDA_ENV": "lekiwi",
        "PI_REPO": "lekiwi/lerobot",
        "LOCAL_REPO": "../lerobot",
        "LOCAL_PLUGIN": "",
    }
    env = provision_env(cfg, py_ver="3.13", recreate=True)
    assert env["PY_VER"] == "3.13"
    assert env["RECREATE_ENV"] == "1"
    assert env["LOCAL_REPO"].startswith("/") and env["LOCAL_REPO"].endswith("/lerobot")
    assert "LOCAL_PLUGIN" not in env  # empty = keep the script's sibling default


def test_shipping_summary_reads_version_and_ref():
    from lekiwi_tui.screens.provision import shipping_summary

    line = shipping_summary({"LOCAL_REPO": "", "LOCAL_PLUGIN": ""})
    assert line.startswith("ships lerobot ")
    assert "+ plugin " in line


def test_ship_plugin_rejects_missing_local_dir():
    from lekiwi_tui.screens.host import ship_plugin

    assert ship_plugin("lekiwi", repo="lekiwi/lerobot", local="/nonexistent/plugin") is False


def test_sync_install_flag_is_consumed_not_passed_to_rsync():
    # --install forces the editable installs; it must never leak into the rsync argv
    # (where an unknown flag would abort the transfer).
    proc = _run(["bash", "scripts/sync.sh", "--dry-run", "--install"])
    assert proc.returncode == 0, proc.stderr
    assert "--install" not in proc.stdout
    assert "rsync" in proc.stdout


def test_sync_screen_argv_and_env_plumbing():
    from lekiwi_tui.screens.sync import build_sync_argv, sync_env

    assert build_sync_argv()[-1].endswith("sync.sh")
    assert build_sync_argv(install=True)[-1] == "--install"

    cfg = {
        "LEKIWI_HOST": "lekiwi",
        "PI_REPO": "lekiwi/lerobot",
        "CONDA_ENV": "lekiwi",
        "LOCAL_REPO": "../lerobot",
        "LOCAL_PLUGIN": "",
    }
    env = sync_env(cfg)
    assert env["CONDA_ENV"] == "lekiwi"
    assert env["LOCAL_REPO"].startswith("/") and env["LOCAL_REPO"].endswith("/lerobot")
    assert "LOCAL_PLUGIN" not in env  # empty = script's sibling default
