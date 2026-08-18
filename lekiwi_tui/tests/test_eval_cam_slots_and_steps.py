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


def _checkpoint(tmp_path: Path, name: str, cams: list[str],
                trained_map: dict | None = None, **cfg_extra) -> Path:
    """A minimal loadable checkpoint dir: config.json (with input_features) + weights.

    ``trained_map`` writes the train_config.json a REAL checkpoint carries. Most tests
    leave it None on purpose: that is the "records nothing" checkpoint whose slots have
    to be inferred from the yaml map.
    """
    ckpt = tmp_path / name / "pretrained_model"
    ckpt.mkdir(parents=True)
    features = {"observation.state": {"type": "STATE", "shape": [9]}}
    features.update({c: {"type": "VISUAL", "shape": [3, 480, 640]} for c in cams})
    (ckpt / "config.json").write_text(
        json.dumps({"type": "smolvla", "input_features": features, **cfg_extra})
    )
    if trained_map is not None:
        (ckpt / "train_config.json").write_text(json.dumps({"rename_map": trained_map}))
    (ckpt / "model.safetensors").write_text("")
    return ckpt


#: A pi05-style checkpoint: slot names that appear NOWHERE in the yaml map, so the
#: subset checks cannot place them and only the recorded map can.
PI05_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.top": "observation.images.right_wrist_0_rgb",
}


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


def test_detect_prefers_the_map_the_checkpoint_records(tmp_path):
    """The branch that makes unknown slot names routable at all: pi05's base_0_rgb et al
    are in neither the yaml map's keys NOR its values, so every subset check misses."""
    ckpt = _checkpoint(tmp_path, "pi05-trained",
                       [f"observation.images.{c}" for c in
                        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")],
                       trained_map=PI05_MAP)
    assert detect_cam_slots(str(ckpt), RENAME_MAP) == ("trained", "from the checkpoint")
    from lekiwi_tui.screens.eval import detect_cam_detail
    assert detect_cam_detail(str(ckpt), RENAME_MAP)[2] is True  # DERIVED, not guessed


def test_detect_prefers_the_recorded_map_over_a_permuting_yaml_map(tmp_path):
    """A PERMUTATION is what the subset checks structurally cannot see: both sides use
    exactly {camera1,2,3} while two views are swapped. Preferring the recorded map makes
    it impossible to send rather than merely visible in a warning."""
    permuted = {  # trained top->camera2 / wrist->camera3; the yaml sends the reverse
        "observation.images.front": "observation.images.camera1",
        "observation.images.top": "observation.images.camera2",
        "observation.images.wrist": "observation.images.camera3",
    }
    ckpt = _checkpoint(tmp_path, "permuted",
                       [f"observation.images.camera{i}" for i in (1, 2, 3)],
                       trained_map=permuted)
    assert detect_cam_slots(str(ckpt), RENAME_MAP)[0] == "trained"
    # the yaml map alone would have looked like a confident match
    assert set(permuted.values()) == set(RENAME_MAP.values())


def test_headless_resolves_the_same_cam_slots_as_the_form(tmp_path, monkeypatch):
    """THE regression this guards: the preference used to live in the form only, so
    `lekiwi eval` sent the yaml's slots to a checkpoint the TUI routed correctly."""
    ckpt = _checkpoint(tmp_path, "pi05-headless",
                       ["observation.images.base_0_rgb"], trained_map=PI05_MAP)
    ctx = _ctx(tmp_path, policy=str(ckpt))
    captured = {}

    def _capture(argv):  # noqa: ANN001 - stands in for the real subprocess run
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(eval_mod.runner, "headless_run", _capture)

    assert eval_mod.run_headless(ctx, []) == 0
    argv = captured["argv"]
    i = argv.index("--cam-slots")
    assert argv[i + 1] == "trained"

    screen = EvalScreen(None, ctx)
    screen._policy = str(ckpt)
    assert screen._cam_resolved()[0] == argv[i + 1]  # form and headless agree


# ── form behavior ─────────────────────────────────────────────────────────────


def test_no_false_alarm_when_the_mapping_came_from_the_checkpoint(tmp_path):
    """The note is a DERIVATION here, so the "not verified" line must stay silent — it
    fired on exactly the checkpoints the trained branch handles best."""
    ckpt = _checkpoint(tmp_path, "pi05-warn",
                       [f"observation.images.{c}" for c in
                        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")],
                       trained_map=PI05_MAP)
    screen = EvalScreen(None, _ctx(tmp_path))
    screen._policy = str(ckpt)
    assert screen._cam_detail() == ("trained", "from the checkpoint", True)
    assert screen._cam_warning() == ""

    # ...but forcing native over a checkpoint that records its own map still warns
    screen._cam_mode = "native"
    assert "ignoring the mapping this checkpoint records" in screen._cam_warning()

    # and a checkpoint that records NOTHING keeps the guess visible
    blind = _checkpoint(tmp_path, "pi05-blind", ["observation.images.base_0_rgb"])
    screen._cam_mode, screen._policy = "auto", str(blind)
    assert "not verified" in screen._cam_warning()


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

    # TWO modes: auto ⇄ native. "map"/"trained" were dropped from the picker — auto already
    # resolves to the yaml map whenever that is the best answer, and the only case forced
    # "map" behaved DIFFERENTLY was sending a mapping the checkpoint contradicts.
    from lekiwi_tui.screens.eval import CAM_MODES
    assert CAM_MODES == ("auto", "native")

    screen._fpos = screen._fields().index("cameras")
    screen.handle_key(Key(RIGHT))
    assert screen._cam_mode == "native"
    screen.handle_key(Key(ENTER))
    assert screen._cam_mode == "auto"
    screen.handle_key(Key(LEFT))
    assert screen._cam_mode == "native"

    screen._steps.set_text("10")
    screen._remember()
    reopened = EvalScreen(None, ctx)
    assert reopened._cam_mode == "native"
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

        async def run_modal(self, modal):  # noqa: ANN001 - the post-run verdict; skip
            return None

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


# ── compile knob ──────────────────────────────────────────────────────────────


def test_compile_row_defaults_to_auto_and_cycles_and_persists(tmp_path):
    from lekiwi_tui.screens.eval import COMPILE_MODES

    assert COMPILE_MODES == ("auto", "on", "off")
    ctx = _ctx(tmp_path)
    screen = EvalScreen(None, ctx)
    assert screen._compile == "auto"  # untouched form runs the checkpoint as trained
    assert "compile" in screen._fields()

    screen._fpos = screen._fields().index("compile")
    screen.handle_key(Key(RIGHT))
    assert screen._compile == "on"
    screen.handle_key(Key(ENTER))
    assert screen._compile == "off"
    screen.handle_key(Key(LEFT))
    assert screen._compile == "on"

    screen._remember()
    assert EvalScreen(None, ctx)._compile == "on"


def test_compile_note_names_the_checkpoints_own_setting_under_auto(tmp_path):
    """`auto` is only informative if you can see what it resolved to — and
    compile_model=true is the answer that costs minutes at the worst moment."""
    ctx = _ctx(tmp_path)
    screen = EvalScreen(None, ctx)

    compiled = _checkpoint(tmp_path, "compiled", ["observation.images.front"],
                           compile_model=True, compile_mode="max-autotune")
    screen._policy = str(compiled)
    assert screen._compile_note() == "checkpoint default (true) (max-autotune)"

    eager = _checkpoint(tmp_path, "eager", ["observation.images.front"], compile_model=False)
    screen._policy = str(eager)
    assert screen._compile_note() == "checkpoint default (false)"

    silent = _checkpoint(tmp_path, "silent", ["observation.images.front"])
    screen._policy = str(silent)
    assert screen._compile_note() == "checkpoint default"

    screen._compile = "off"  # forced: the note describes the override, not the checkpoint
    assert "eager" in screen._compile_note()


def test_start_argv_carries_the_compile_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(eval_mod, "eval_issues", lambda *a, **k: [])
    ckpt = _checkpoint(tmp_path, "compile-argv", ["observation.images.front"],
                       compile_model=True)
    app = _StartApp()
    screen = EvalScreen(app, _ctx(tmp_path))
    screen._policy = str(ckpt)
    screen._compile = "off"
    _run_start(screen)
    argv = app.suspended
    assert argv[argv.index("--compile") + 1] == "off"


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


def test_eval_sh_compile_emits_the_policy_flag_only_when_forced(tmp_path):
    ws = _script_workspace(tmp_path)
    assert "--policy.compile_model=true" in _dry_run(ws, "--compile", "on")
    assert "--policy.compile_model=false" in _dry_run(ws, "--compile", "off")
    # auto (and the default, when the flag is absent) leaves the checkpoint alone
    for tokens in (_dry_run(ws, "--compile", "auto"), _dry_run(ws)):
        assert not [t for t in tokens if t.startswith("--policy.compile_model=")]


def test_eval_sh_rejects_an_unknown_compile_value(tmp_path):
    ws = _script_workspace(tmp_path)
    proc = subprocess.run(
        ["bash", str(ws / "scripts" / "eval.sh"), "--dry-run", "--policy", "/p",
         "--compile", "yes"],
        env={**os.environ, "LEKIWI_ROOT": str(ws)},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 2
    assert "--compile" in proc.stderr


def test_eval_sh_invokes_lerobot_rollout_directly(tmp_path):
    """No keyboard shim in front of it: since lerobot 0.6 the rollout's own
    init_keyboard_listener falls back to a TTY reader, so there is nothing to patch."""
    tokens = _dry_run(_script_workspace(tmp_path))
    assert tokens[0] == "lerobot-rollout"
    assert tokens[1] == "--config_path"
    assert not [t for t in tokens if t.endswith("_kbd.py")]


def test_eval_sh_trained_emits_the_checkpoint_recorded_map_token(tmp_path):
    """The other half of the fix: what `auto` now resolves to has to survive the
    launcher. Every yaml key must be re-pointed, because draccus MERGES the CLI dict
    into the slice map — a partial override would leave stale slots behind."""
    ws = _script_workspace(tmp_path)
    ckpt = _checkpoint(tmp_path, "pi05-sh", ["observation.images.base_0_rgb"],
                       trained_map=PI05_MAP)
    proc = subprocess.run(
        ["bash", str(ws / "scripts" / "eval.sh"), "--dry-run", "--policy", str(ckpt),
         "--cam-slots", "trained"],
        env={**os.environ, "LEKIWI_ROOT": str(ws)},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    token = [t for t in proc.stdout.splitlines() if t.startswith("--rename_map=")]
    assert len(token) == 1
    body = token[0][len("--rename_map={"):-1]
    assert [p.strip() for p in body.split(",")] == [
        "observation.images.front: observation.images.base_0_rgb",
        "observation.images.wrist: observation.images.left_wrist_0_rgb",
        "observation.images.top: observation.images.right_wrist_0_rgb",
    ]


def test_eval_sh_map_and_zero_steps_emit_no_extra_tokens(tmp_path):
    tokens = _dry_run(_script_workspace(tmp_path), "--cam-slots", "map", "--action-steps", "0")
    assert not [t for t in tokens if t.startswith("--rename_map=")]
    assert not [t for t in tokens if t.startswith("--policy.n_action_steps=")]


# ── extra passthrough flags ─────────────────────────────────────────────────


async def _preflight_ok(*args, **kwargs):  # noqa: ANN002, ANN003
    return True


class _StartApp:
    """Captures the suspended argv; skips the post-run verdict modal."""

    def __init__(self, answer: str | None = None) -> None:
        self.suspended = None
        self._answer = answer

    async def suspend(self, argv, **kwargs):  # noqa: ANN001
        self.suspended = list(argv)
        return 0

    async def run_modal(self, modal):  # noqa: ANN001
        return self._answer


def _run_start(screen) -> None:  # noqa: ANN001
    action = screen.handle_key(Key("s"))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())


def test_extra_flags_row_present_and_persists(tmp_path):
    ctx = _ctx(tmp_path)
    screen = EvalScreen(None, ctx)
    assert "extra" in screen._fields()
    screen._extra_text = "--policy.num_inference_timesteps=10"
    screen._remember()
    reopened = EvalScreen(None, ctx)
    assert reopened._extra_text == "--policy.num_inference_timesteps=10"


def test_extra_flags_prompt_strips_and_updates_session_memory(tmp_path):
    ctx = _ctx(tmp_path)
    screen = EvalScreen(_StartApp(answer="  --policy.vlm_dtype=bfloat16  "), ctx)
    asyncio.run(screen._edit_extra())
    assert screen._extra_text == "--policy.vlm_dtype=bfloat16"  # stripped
    assert EvalScreen(None, ctx)._extra_text == "--policy.vlm_dtype=bfloat16"


def test_start_argv_appends_shlex_split_extra_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(eval_mod, "eval_issues", lambda *a, **k: [])
    ckpt = _checkpoint(tmp_path, "native", ["observation.images.front"])
    app = _StartApp()
    screen = EvalScreen(app, _ctx(tmp_path))
    screen._policy = str(ckpt)
    screen._extra_text = "--policy.num_inference_timesteps=10 --policy.vlm_dtype=bfloat16"
    _run_start(screen)
    argv = app.suspended
    assert "--policy.num_inference_timesteps=10" in argv
    assert "--policy.vlm_dtype=bfloat16" in argv
    # forwarded after the built flags (draccus last-wins); --gpu is the last fixed knob
    assert argv.index("--policy.num_inference_timesteps=10") > argv.index("--gpu")


def test_start_argv_has_no_trailing_tokens_when_extra_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(eval_mod, "eval_issues", lambda *a, **k: [])
    ckpt = _checkpoint(tmp_path, "native", ["observation.images.front"])
    app = _StartApp()
    screen = EvalScreen(app, _ctx(tmp_path))
    screen._policy = str(ckpt)
    _run_start(screen)
    # nothing trails the --gpu <name> pair (empty extra field, no CLI-dispatch extra)
    assert app.suspended[-2] == "--gpu"


def test_start_rejects_malformed_extra_flags_without_launching(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(eval_mod, "eval_issues", lambda *a, **k: [])
    ckpt = _checkpoint(tmp_path, "native", ["observation.images.front"])
    app = _StartApp()
    screen = EvalScreen(app, _ctx(tmp_path))
    screen._policy = str(ckpt)
    screen._extra_text = '--task="unterminated'
    _run_start(screen)
    assert app.suspended is None  # never handed to the TTY
    assert "extra flags" in screen._err



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
