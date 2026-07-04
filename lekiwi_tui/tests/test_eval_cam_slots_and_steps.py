"""Camera-slots auto-detection + action/flow-steps knobs (EvalScreen ↔ scripts/eval.sh).

Covers the knobs added for checkpoints trained on NATIVE robot camera keys
alongside the renamed-slot FT checkpoints (camera1/2/3):
  * detect_cam_slots — config.json input_features → "map"/"native" + note;
  * the form rows (steps sync-only, flow + cameras always — deliberately no
    model-type gating) + the 0-sentinel labels showing checkpoint values + memory;
  * _start argv — the resolved --cam-slots / --action-steps / --flow-steps flags;
  * eval.sh --dry-run — the identity --rename_map token (draccus MERGES dict CLI
    overrides, so native emits key→key over the slice map, never an empty {}) and
    the --policy.n_action_steps / --policy.num_steps tokens, all conditional.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import lekiwi_tui.screens.eval as eval_mod
from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.framework.events import ENTER, LEFT, RIGHT, Key
from lekiwi_tui.framework.screen import Invoke
from lekiwi_tui.screens.eval import EvalScreen, detect_cam_slots

ROOT = Path(__file__).resolve().parents[2]

RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
    "observation.images.top": "observation.images.camera3",
}


def _checkpoint(tmp_path: Path, name: str, cams: list[str], **cfg_extra) -> Path:
    """A minimal loadable checkpoint dir: config.json (with input_features) + weights."""
    ckpt = tmp_path / name / "pretrained_model"
    ckpt.mkdir(parents=True)
    features = {"observation.state": {"type": "STATE", "shape": [9]}}
    features.update({c: {"type": "VISUAL", "shape": [3, 480, 640]} for c in cams})
    (ckpt / "config.json").write_text(
        json.dumps({"type": "smolvla", "input_features": features, **cfg_extra})
    )
    (ckpt / "model.safetensors").write_text("")
    return ckpt


def _ctx(tmp_path: Path, policy: str = "") -> Context:
    root = tmp_path / "models"
    root.mkdir(exist_ok=True)
    return Context(
        cfg=Config(values={
            "POLICY_ROOT": str(root),
            "POLICY_PATH": policy,
            "INFERENCE": "sync",
            "EXECUTION_HORIZON": "20",
            "DISPLAY_DATA": "off",
        }),
        doc={"rollout": {"task": "pick up the cube", "rename_map": dict(RENAME_MAP)}},
        gpu_name="",
        is_tty=True,
    )


# ── detect_cam_slots ──────────────────────────────────────────────────────────


def test_detect_renamed_slots_checkpoint_keeps_the_yaml_map(tmp_path):
    ckpt = _checkpoint(tmp_path, "ft", [f"observation.images.camera{i}" for i in (1, 2, 3)])
    mode, note = detect_cam_slots(str(ckpt), RENAME_MAP)
    assert mode == "map"
    assert note == "camera1/camera2/camera3"


def test_detect_native_keys_checkpoint_neutralizes_the_map(tmp_path):
    ckpt = _checkpoint(tmp_path, "native", [f"observation.images.{c}" for c in ("front", "wrist", "top")])
    mode, note = detect_cam_slots(str(ckpt), RENAME_MAP)
    assert mode == "native"
    assert note == "front/top/wrist"


def test_detect_unknown_keys_falls_back_to_map_with_a_visible_note(tmp_path):
    ckpt = _checkpoint(tmp_path, "pi05", ["observation.images.base_0_rgb"])
    mode, note = detect_cam_slots(str(ckpt), RENAME_MAP)
    assert mode == "map"
    assert "unknown keys" in note and "base_0_rgb" in note


def test_detect_unreadable_checkpoint_or_missing_map_defaults_to_map(tmp_path):
    assert detect_cam_slots("lerobot/smolvla_base", RENAME_MAP) == ("map", "checkpoint config unreadable")
    ckpt = _checkpoint(tmp_path, "any", ["observation.images.front"])
    assert detect_cam_slots(str(ckpt), {}) == ("map", "no yaml rename_map")


# ── form behavior ─────────────────────────────────────────────────────────────


def test_steps_row_is_sync_only_and_flow_and_cameras_rows_are_always_present(tmp_path):
    screen = EvalScreen(None, _ctx(tmp_path))
    assert screen._backend == "sync"
    assert "steps" in screen._fields() and "exec" not in screen._fields()
    assert "cameras" in screen._fields() and "flow" in screen._fields()

    screen._backend = "rtc"
    assert "exec" in screen._fields() and "steps" not in screen._fields()
    assert "cameras" in screen._fields() and "flow" in screen._fields()

    # deliberately NO model-type gating: any checkpoint keeps the flow row
    # (a policy that never reads num_steps ignores the override — it is inert)
    native = _checkpoint(tmp_path, "native-rows", ["observation.images.front"])
    screen._policy = str(native)
    assert "flow" in screen._fields()


def test_zero_sentinel_labels_show_the_checkpoint_own_values(tmp_path):
    ctx = _ctx(tmp_path)
    ft = _checkpoint(tmp_path, "ft-labels", ["observation.images.camera1"],
                     n_action_steps=50, num_steps=10)
    atypical = _checkpoint(tmp_path, "atypical-labels", ["observation.images.front"],
                           n_action_steps=1, num_steps=1)

    screen = EvalScreen(None, ctx)
    screen._policy = str(ft)
    screen._refresh_ckpt_defaults()
    assert screen._steps.zero_label == "checkpoint default (50)"
    assert screen._flow.zero_label == "checkpoint default (10)"

    screen._policy = str(atypical)
    screen._refresh_ckpt_defaults()
    assert screen._steps.zero_label == "checkpoint default (1)"  # atypical default, visible
    assert screen._flow.zero_label == "checkpoint default (1)"

    screen._policy = "lerobot/smolvla_base"  # unreadable -> generic label, no crash
    screen._refresh_ckpt_defaults()
    assert screen._steps.zero_label == "checkpoint default"


def test_cameras_row_cycles_modes_and_the_form_remembers_both_knobs(tmp_path):
    ctx = _ctx(tmp_path)
    screen = EvalScreen(None, ctx)
    assert screen._cam_mode == "auto"

    screen._fpos = screen._fields().index("cameras")
    screen.handle_key(Key(RIGHT))
    assert screen._cam_mode == "map"
    screen.handle_key(Key(ENTER))
    assert screen._cam_mode == "native"
    screen.handle_key(Key(LEFT))
    assert screen._cam_mode == "map"

    screen._steps.set_text("10")
    screen._remember()
    reopened = EvalScreen(None, ctx)
    assert reopened._cam_mode == "map"
    assert reopened._steps.value == 10


def test_start_argv_carries_the_resolved_cam_slots_and_action_steps(tmp_path, monkeypatch):
    async def _preflight_ok(*args, **kwargs):  # noqa: ANN001
        return True

    monkeypatch.setattr(eval_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(eval_mod, "eval_issues", lambda *a, **k: [])

    class _FakeApp:
        def __init__(self) -> None:
            self.suspended = None

        async def suspend(self, argv, **kwargs):  # noqa: ANN001
            self.suspended = list(argv)
            return 0

    ckpt = _checkpoint(tmp_path, "native", [f"observation.images.{c}" for c in ("front", "wrist", "top")])
    ctx = _ctx(tmp_path)
    app = _FakeApp()
    screen = EvalScreen(app, ctx)
    screen._policy = str(ckpt)
    screen._steps.set_text("10")
    screen._flow.set_text("8")

    action = screen.handle_key(Key("s"))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())

    argv = app.suspended
    assert argv is not None
    i = argv.index("--cam-slots")
    assert argv[i + 1] == "native"  # auto mode resolved against the native-key checkpoint
    j = argv.index("--action-steps")
    assert argv[j + 1] == "10"
    k = argv.index("--flow-steps")
    assert argv[k + 1] == "8"


# ── eval.sh argv (dry-run) ────────────────────────────────────────────────────


def _script_workspace(tmp_path: Path) -> Path:
    """Fresh-checkout workspace for running scripts/eval.sh: example yaml + scripts."""
    shutil.copy2(ROOT / "lekiwi.example.yaml", tmp_path / "lekiwi.example.yaml")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    return tmp_path


def _dry_run(ws: Path, *flags: str) -> list[str]:
    proc = subprocess.run(
        ["bash", str(ws / "scripts" / "eval.sh"), "--dry-run", "--policy", "/p", *flags],
        env={**os.environ, "LEKIWI_ROOT": str(ws)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()

def test_eval_sh_flow_steps_emit_num_steps_only_when_positive(tmp_path):
    ws = _script_workspace(tmp_path)
    tokens = _dry_run(ws, "--flow-steps", "10")
    assert "--policy.num_steps=10" in tokens
    tokens = _dry_run(ws, "--flow-steps", "0")
    assert not [t for t in tokens if t.startswith("--policy.num_steps=")]


def test_eval_sh_native_emits_the_identity_rename_map_token(tmp_path):
    tokens = _dry_run(_script_workspace(tmp_path), "--cam-slots", "native", "--action-steps", "10")
    identity = [t for t in tokens if t.startswith("--rename_map=")]
    assert len(identity) == 1
    body = identity[0][len("--rename_map={"):-1]
    pairs = [p.strip() for p in body.split(",")]
    # every robot-side key maps to ITSELF (identity over the example yaml's map keys)
    assert pairs == [
        "observation.images.front: observation.images.front",
        "observation.images.wrist: observation.images.wrist",
        "observation.images.top: observation.images.top",
    ]
    assert "--policy.n_action_steps=10" in tokens


def test_eval_sh_map_and_zero_steps_emit_no_extra_tokens(tmp_path):
    tokens = _dry_run(_script_workspace(tmp_path), "--cam-slots", "map", "--action-steps", "0")
    assert not [t for t in tokens if t.startswith("--rename_map=")]
    assert not [t for t in tokens if t.startswith("--policy.n_action_steps=")]


def test_eval_sh_rejects_auto_cam_slots(tmp_path):
    ws = _script_workspace(tmp_path)
    proc = subprocess.run(
        ["bash", str(ws / "scripts" / "eval.sh"), "--dry-run", "--policy", "/p", "--cam-slots", "auto"],
        env={**os.environ, "LEKIWI_ROOT": str(ws)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "cam-slots" in proc.stderr
