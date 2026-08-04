"""record.sh keeps the dataset NAME it was given: --dataset.no_stamp, and the feature
probe that decides whether the installed lerobot understands the flag.

lerobot appends a _YYYYmmdd_HHMMSS tag to repo_id at creation, so meta/info.json ends
up naming a dataset nobody else refers to. The flag opts out, but it is newer than
some lerobots we run against, and passing an unknown field makes draccus reject the
whole command — hence a probe, and hence these tests pin BOTH directions of it.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lekiwi_tui import ROOT

RECORD_SH = ROOT / "scripts" / "record.sh"
LIB_SH = ROOT / "scripts" / "lib.sh"


def _fake_lerobot(tmp_path: Path, *, declares: bool, with_file: bool = True) -> Path:
    """A minimal importable `lerobot` package to put on PYTHONPATH."""
    pkg = tmp_path / "site" / "lerobot"
    (pkg / "configs").mkdir(parents=True)
    (pkg / "__init__.py").write_text("__version__ = '0.6.1'\n")
    (pkg / "configs" / "__init__.py").write_text("")
    if with_file:
        body = "class DatasetRecordConfig:\n    fps: int = 30\n"
        if declares:
            body += "    no_stamp: bool = False\n"
        (pkg / "configs" / "dataset.py").write_text(body)
    return tmp_path / "site"


def _dry_record(env_extra: dict) -> list[str]:
    env = dict(os.environ, DRY="1", **env_extra)
    out = subprocess.run(["bash", str(RECORD_SH), "--name", "demo", "--task", "t"],
                         capture_output=True, text=True, env=env, check=True)
    return out.stdout.splitlines()


def _declares(env_extra: dict, name: str = "no_stamp", rel: str = "configs/dataset.py") -> bool:
    """Call lerobot_declares the way record.sh does, in a fresh shell."""
    r = subprocess.run(
        ["bash", "-c", f'source "{LIB_SH}"; lerobot_declares {name} {rel}'],
        capture_output=True, text=True, env=dict(os.environ, **env_extra))
    return r.returncode == 0


# ── the probe ─────────────────────────────────────────────────────────────────


def test_probe_sees_a_declared_field(tmp_path):
    assert _declares({"PYTHONPATH": str(_fake_lerobot(tmp_path, declares=True))})


def test_probe_rejects_a_lerobot_without_the_field(tmp_path):
    assert not _declares({"PYTHONPATH": str(_fake_lerobot(tmp_path, declares=False))})


def test_probe_rejects_a_lerobot_missing_the_module(tmp_path):
    site = _fake_lerobot(tmp_path, declares=True, with_file=False)
    assert not _declares({"PYTHONPATH": str(site)})


def test_probe_says_no_when_lerobot_is_not_importable():
    """The safe direction: no lerobot means no flag, so the launch still runs."""
    assert not _declares({"PYTHON": "/bin/false"})


# ── record.sh ─────────────────────────────────────────────────────────────────


def test_no_stamp_is_emitted_next_to_the_name_when_forced_on():
    argv = _dry_record({"NO_STAMP": "on"})
    assert "--dataset.no_stamp=true" in argv
    # it is about the NAME, so it sits with repo_id/root, before the episode knobs
    assert argv.index("--dataset.no_stamp=true") == argv.index("--dataset.root=../../datasets/demo") + 1
    assert argv.index("--dataset.no_stamp=true") < argv.index("--dataset.num_episodes=5")


def test_no_stamp_is_omitted_when_forced_off():
    assert "--dataset.no_stamp=true" not in _dry_record({"NO_STAMP": "off"})


def test_auto_follows_the_installed_lerobot(tmp_path):
    has = _dry_record({"PYTHONPATH": str(_fake_lerobot(tmp_path / "yes", declares=True))})
    has_not = _dry_record({"PYTHONPATH": str(_fake_lerobot(tmp_path / "no", declares=False))})
    assert "--dataset.no_stamp=true" in has
    assert "--dataset.no_stamp=true" not in has_not
    # ...and nothing else about the command changes with it
    assert [a for a in has if "no_stamp" not in a] == has_not
