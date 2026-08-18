"""scripts/dagger.sh — the sole source of the DAgger lerobot-rollout argv.

Pins the dry-run token stream the way test_eval_cam_slots_and_steps.py pins
eval.sh's: a fresh-checkout workspace (example yaml + scripts) runs the script
with --dry-run and the tests assert the exact tokens, so any argv regression
turns the suite red before it reaches a robot.

Dagger-specific surface under test:
  * --dataset.root is ALWAYS emitted; blank --dataset-root computes a per-session
    stamped dir (rollout stamps repo_id but never the root, so reuse would fail);
  * --task feeds BOTH --task= and --dataset.single_task= (policy conditioning and
    the dataset stamp must agree);
  * --target/--record-autonomous/--input map to strategy.* tokens, each emitted
    only off its default;
  * the sliced `dagger:` block carries teleop + strategy.type=dagger + a
    `rollout_`-prefixed repo_id (lerobot validates the prefix at startup).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _script_workspace(tmp_path: Path) -> Path:
    """Fresh-checkout workspace for running scripts/dagger.sh: example yaml + scripts."""
    shutil.copy2(ROOT / "lekiwi.example.yaml", tmp_path / "lekiwi.example.yaml")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    return tmp_path


def _dry_run(ws: Path, *flags: str, policy: str = "/p") -> list[str]:
    proc = subprocess.run(
        ["bash", str(ws / "scripts" / "dagger.sh"), "--dry-run", "--policy", policy, *flags],
        env={**os.environ, "LEKIWI_ROOT": str(ws)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


# ── the golden argv ───────────────────────────────────────────────────────────


def test_dagger_sh_golden_argv(tmp_path):
    ws = _script_workspace(tmp_path)
    tokens = _dry_run(
        ws,
        "--task", "Pick the cube",
        "--target", "10",
        "--dataset-root", "../../datasets/rollout_x",
    )
    assert tokens == [
        "python",
        str(ws / "scripts" / "lerobot_dagger_kbd.py"),
        "--config_path",
        str(ws / ".lekiwi-cache" / "dagger.yaml"),
        "--policy.path=/p",
        "--inference.type=sync",
        "--display_data=false",
        "--dataset.root=../../datasets/rollout_x",
        "--task=Pick the cube",
        "--dataset.single_task=Pick the cube",
        "--strategy.num_episodes=10",
    ]


def test_blank_dataset_root_computes_a_stamped_session_dir(tmp_path):
    ws = _script_workspace(tmp_path)
    roots = [t for t in _dry_run(ws) if t.startswith("--dataset.root=")]
    assert len(roots) == 1
    # ../../datasets = the my_robot data root (record's convention), stamped so a
    # second session never collides with the first (rollout refuses a reused dir).
    assert re.fullmatch(
        r"--dataset\.root=\.\./\.\./datasets/rollout_dagger_\d{8}_\d{6}", roots[0]
    ), roots[0]


# ── conditional tokens ────────────────────────────────────────────────────────


def test_blank_task_and_zero_target_emit_no_tokens(tmp_path):
    tokens = _dry_run(_script_workspace(tmp_path), "--target", "0")
    assert not [t for t in tokens if t.startswith(("--task=", "--dataset.single_task="))]
    assert not [t for t in tokens if t.startswith("--strategy.num_episodes=")]


def test_strategy_knobs_emit_only_off_their_defaults(tmp_path):
    ws = _script_workspace(tmp_path)
    # defaults (corrections-only, keyboard): no strategy tokens at all
    tokens = _dry_run(ws, "--record-autonomous", "off", "--input", "keyboard")
    assert not [t for t in tokens if t.startswith(("--strategy.record_autonomous", "--strategy.input_device"))]
    # switched on: exactly the two tokens
    tokens = _dry_run(ws, "--record-autonomous", "on", "--input", "pedal")
    assert "--strategy.record_autonomous=true" in tokens
    assert "--strategy.input_device=pedal" in tokens


def test_rtc_backend_emits_exec_horizon_and_sync_does_not(tmp_path):
    ws = _script_workspace(tmp_path)
    tokens = _dry_run(ws, "--backend", "rtc", "--exec-horizon", "22")
    assert "--inference.rtc.execution_horizon=22" in tokens
    tokens = _dry_run(ws, "--backend", "sync")
    assert not [t for t in tokens if t.startswith("--inference.rtc.")]


def test_gpu_duration_and_passthrough(tmp_path):
    ws = _script_workspace(tmp_path)
    tokens = _dry_run(ws, "--gpu", "CUDA", "--duration", "600", "--policy.num_steps=8")
    assert "--device=cuda" in tokens
    assert "--duration=600" in tokens
    assert tokens[-1] == "--policy.num_steps=8"  # passthrough forwarded last (last-wins)
    tokens = _dry_run(ws, "--duration", "0")
    assert not [t for t in tokens if t.startswith(("--device=", "--duration="))]


def test_action_and_flow_steps_emit_only_when_positive(tmp_path):
    ws = _script_workspace(tmp_path)
    tokens = _dry_run(ws, "--action-steps", "10", "--flow-steps", "8")
    assert "--policy.n_action_steps=10" in tokens
    assert "--policy.num_steps=8" in tokens
    tokens = _dry_run(ws, "--action-steps", "0", "--flow-steps", "0")
    assert not [t for t in tokens if t.startswith(("--policy.n_action_steps=",
                                                   "--policy.num_steps="))]


def test_native_cam_slots_emits_the_identity_rename_map(tmp_path):
    tokens = _dry_run(_script_workspace(tmp_path), "--cam-slots", "native")
    identity = [t for t in tokens if t.startswith("--rename_map=")]
    assert len(identity) == 1
    body = identity[0][len("--rename_map={"):-1]
    pairs = [p.strip() for p in body.split(",")]
    # identity over the example yaml's dagger rename_map KEYS (robot-side names)
    assert pairs == [
        "observation.images.front: observation.images.front",
        "observation.images.wrist: observation.images.wrist",
        "observation.images.top: observation.images.top",
    ]


# ── fail-fast validation ──────────────────────────────────────────────────────


def test_rejects_auto_cam_slots_and_unknown_input(tmp_path):
    ws = _script_workspace(tmp_path)
    for flags, needle in ((["--cam-slots", "auto"], "cam-slots"), (["--input", "footpedal"], "--input")):
        proc = subprocess.run(
            ["bash", str(ws / "scripts" / "dagger.sh"), "--dry-run", "--policy", "/p", *flags],
            env={**os.environ, "LEKIWI_ROOT": str(ws)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert needle in proc.stderr


# ── the sliced config block ───────────────────────────────────────────────────


def test_sliced_dagger_block_carries_teleop_strategy_and_rollout_prefixed_repo(tmp_path):
    ws = _script_workspace(tmp_path)
    _dry_run(ws)  # writes .lekiwi-cache/dagger.yaml as a side effect
    block = yaml.safe_load((ws / ".lekiwi-cache" / "dagger.yaml").read_text())
    assert block["strategy"]["type"] == "dagger"
    assert block["strategy"]["record_autonomous"] is False
    assert block["teleop"]["type"] == "lekiwi_pincopen_leader"  # corrections need the leader
    # lerobot validates the rollout_ prefix at startup; catch a rename here instead
    assert block["dataset"]["repo_id"].split("/", 1)[-1].startswith("rollout_")
    assert block["dataset"]["video_encoding_batch_size"] == 1  # the batch>1 path loses data
    assert block["duration"] == 0  # sessions end on num_episodes or ESC, not a timer
