"""Checkpoint preflight: the diagnosis itself, its refusal to over-block, and the
eval.sh wiring (a failing preflight must abort BEFORE the rollout is executed).

The unit tests inject the two lerobot-facing seams (`facts`, `load`) so they run in a
bare env — which is also the env CI has, and the reason the real check exits 0 there.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from lekiwi_tui import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import ckpt_preflight as pf  # noqa: E402

EVAL_SH = ROOT / "scripts" / "eval.sh"
FACTS = lambda: ("0.6.1", "/env/lib/python3.12/site-packages/lerobot")  # noqa: E731


def _ckpt(tmp_path: Path, **extra) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"type": "smolvla", **extra}))
    return d


def _no_cache(monkeypatch, tmp_path):
    """Point the verdict cache at a scratch dir so tests never share state."""
    monkeypatch.setattr(pf, "CACHE_DIR", tmp_path / "cache")


# ── the diagnosis ─────────────────────────────────────────────────────────────


def test_unknown_fields_are_named_and_block(tmp_path, monkeypatch):
    _no_cache(monkeypatch, tmp_path)

    def boom(_):
        raise RuntimeError("The fields `attention_backend`, `chunk_drift` are not valid for SmolVLAConfig")

    code, text = pf.diagnose(str(_ckpt(tmp_path)), facts=FACTS, load=boom)
    assert code == pf.INCOMPATIBLE
    assert "attention_backend, chunk_drift" in text
    assert "smolvla" in text and "0.6.1" in text
    assert "Nothing was sent to the robot." in text


def test_other_load_errors_block_with_their_last_line(tmp_path, monkeypatch):
    _no_cache(monkeypatch, tmp_path)

    def boom(_):
        raise ValueError("traceback noise\n`num_steps` must be positive, got -1")

    code, text = pf.diagnose(str(_ckpt(tmp_path)), facts=FACTS, load=boom)
    assert code == pf.INCOMPATIBLE
    assert "must be positive" in text and "missing:" not in text


def test_a_loadable_checkpoint_passes_and_caches(tmp_path, monkeypatch):
    _no_cache(monkeypatch, tmp_path)
    calls = []
    ckpt = _ckpt(tmp_path)

    code, text = pf.diagnose(str(ckpt), facts=FACTS, load=lambda p: calls.append(p))
    assert (code, text) == (pf.OK, "")
    assert calls == [str(ckpt)]

    # second run is a cache hit: the (expensive) load is not repeated
    code, _ = pf.diagnose(str(ckpt), facts=FACTS, load=lambda p: calls.append(p))
    assert code == pf.OK and len(calls) == 1

    # ...but a re-saved config invalidates it
    (ckpt / "config.json").write_text(json.dumps({"type": "smolvla", "n_action_steps": 10}))
    pf.diagnose(str(ckpt), facts=FACTS, load=lambda p: calls.append(p))
    assert len(calls) == 2


def test_a_new_lerobot_version_invalidates_the_verdict(tmp_path, monkeypatch):
    _no_cache(monkeypatch, tmp_path)
    ckpt, calls = _ckpt(tmp_path), []
    pf.diagnose(str(ckpt), facts=FACTS, load=lambda p: calls.append(p))
    pf.diagnose(str(ckpt), facts=lambda: ("0.7.0", "/env"), load=lambda p: calls.append(p))
    assert len(calls) == 2, "a verdict from another lerobot must not be reused"


@pytest.mark.parametrize("policy", ["lerobot/smolvla_base", "does/not/exist"])
def test_non_local_policies_are_not_our_business(policy, tmp_path, monkeypatch):
    """A hub repo id has no local config.json; the launch itself resolves it."""
    _no_cache(monkeypatch, tmp_path)

    def unexpected(_):
        raise AssertionError("must not try to load")

    assert pf.diagnose(policy, facts=FACTS, load=unexpected) == (pf.OK, "")


def test_no_lerobot_means_no_verdict(tmp_path, monkeypatch):
    """Cannot tell is not the same as incompatible: CI has no lerobot at all."""
    _no_cache(monkeypatch, tmp_path)

    def unexpected(_):
        raise AssertionError("must not try to load")

    assert pf.diagnose(str(_ckpt(tmp_path)), facts=lambda: None, load=unexpected) == (pf.OK, "")


def test_main_never_blocks_on_being_called_wrong():
    assert pf.main(["ckpt_preflight.py"]) == pf.OK


# ── eval.sh wiring ────────────────────────────────────────────────────────────


def _fake_python(tmp_path: Path, preflight_exit: int) -> Path:
    """A `python` earlier on PATH: it logs its argv, fails the preflight with
    *preflight_exit*, and stands in for the rollout shim otherwise."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> {tmp_path}/argv.log\n'
        'case "$1" in\n'
        f'  *ckpt_preflight.py) exit {preflight_exit} ;;\n'
        '  *lerobot_rollout_kbd.py) echo ROLLOUT_RAN; exit 0 ;;\n'
        # the cfg_slice / cfg_get helpers also run through `python`
        '  -) exec /usr/bin/env python3 "$@" ;;\n'
        '  *) exec /usr/bin/env python3 "$@" ;;\n'
        'esac\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def _run_eval(tmp_path: Path, bin_dir: Path, policy: Path):
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(EVAL_SH), "--policy", str(policy), "--backend", "sync", "--display", "off"],
        capture_output=True, text=True, env=env)


def test_eval_sh_aborts_when_the_preflight_refuses(tmp_path):
    policy = _ckpt(tmp_path)
    r = _run_eval(tmp_path, _fake_python(tmp_path, preflight_exit=2), policy)
    assert r.returncode == 2
    assert "ROLLOUT_RAN" not in r.stdout
    log = (tmp_path / "argv.log").read_text()
    assert "ckpt_preflight.py" in log and "lerobot_rollout_kbd.py" not in log


def test_eval_sh_runs_the_rollout_when_the_preflight_passes(tmp_path):
    policy = _ckpt(tmp_path)
    r = _run_eval(tmp_path, _fake_python(tmp_path, preflight_exit=0), policy)
    assert r.returncode == 0 and "ROLLOUT_RAN" in r.stdout
    log = (tmp_path / "argv.log").read_text()
    assert log.index("ckpt_preflight.py") < log.index("lerobot_rollout_kbd.py"), "preflight runs FIRST"


def test_dry_run_stays_offline(tmp_path):
    """The parity gate must not import lerobot: dry-run exits before the preflight."""
    policy = _ckpt(tmp_path)
    env = dict(os.environ, PATH=f"{_fake_python(tmp_path, 2)}:{os.environ['PATH']}", DRY="1")
    r = subprocess.run(["bash", str(EVAL_SH), "--policy", str(policy)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "ckpt_preflight.py" not in (tmp_path / "argv.log").read_text()
