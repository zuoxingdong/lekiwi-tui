from __future__ import annotations

import os
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
