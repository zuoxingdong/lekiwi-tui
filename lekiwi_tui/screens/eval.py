"""eval.py — EvalScreen: configure + launch a policy rollout on the robot (lerobot-rollout).

A form with a DYNAMIC field list (the exec-horizon row appears only for the rtc backend,
the action-steps row only for sync), a PolicyPicker, an editable Task, a flow-steps row
(FM integration steps; a policy that never reads num_steps ignores the override, so
deliberately NO model-type gating), a camera-slots mode (auto-detected from the
checkpoint's config.json input_features — see detect_cam_slots),
Backend/Duration/Display, and a free-text Extra-flags row (shlex-split, forwarded verbatim
to lerobot-rollout — e.g. --policy.num_inference_timesteps=10 for a field with no dedicated
row). The action/flow-steps 0-sentinels display the checkpoint's OWN
values (read once per policy via _ckpt_info) so "checkpoint default" is never a surprise. Start validates the checkpoint then SUSPENDS into the rollout
(pause=True) — it owns the real TTY for its keyboard controls. Fronts scripts/eval.sh (the
sole argv source). NO compile toggle (intentional). Ports resolve_eval_policy /
_device_note / _checkpoint_error verbatim.
"""
from __future__ import annotations

import json
import os
import shlex
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Line, Span

from .. import ROOT
from ..config import cfg_get, collapse_home
from ..hostprobe import host_alive
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key, is_char
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField, wrap_words
from ..preflight import confirm_preflight, eval_issues
from ..policies import discover_policies, is_valid_checkpoint, resolve_policy
from ..scoreboard import append_score, ckpt_label, load_scores, score_tally
from ..widgets.pickers import CUSTOM, PolicyPicker
from .chrome import clip_end as _clip_end
from .chrome import clip_middle as _clip_middle
from .chrome import (
    number_line, setting_line, toggle,
    draw_form_page, padded_line, plan_row, section_line, seg, slim_status_spans,
)

# The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

EVAL_SCRIPT = ROOT / "scripts" / "eval.sh"
STATE_KEY = "eval"
FORM_LABEL_WIDTH = 13
SUMMARY_LABEL_WIDTH = 10

# Camera-slots modes, in ←→ cycle order. TWO choices on purpose.
#
#   auto   — send the mapping the CHECKPOINT recorded (the only authoritative source);
#            fall back to the yaml map, or to raw names, when it records none.
#   native — force NO renaming. The one case auto can get wrong in the other direction:
#            a policy trained on raw front/top/wrist whose config auto cannot read.
#
# "map" and "trained" were dropped from the picker and remain valid `eval.sh --cam-slots`
# values for CLI/headless use. "trained" was identical to auto whenever it worked and a lie
# when it did not (it reported "trained" while eval.sh silently emitted no token). "map"
# only ever DIFFERED from auto when the checkpoint recorded a map and the yaml contradicted
# it, i.e. a button whose sole distinct behaviour is the bug we detect.
CAM_MODES = ("auto", "native")


def _policy_root_path(cfg) -> Path:
    p = Path(cfg["POLICY_ROOT"]).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def resolve_eval_policy(policy_path: str, root: Path) -> tuple[str, str | None]:
    """Resolve POLICY_PATH + the dead-path fallback (port of the Textual eval helper)."""
    policy = resolve_policy(policy_path, root)
    note: str | None = None
    if (policy.startswith("/") or policy.startswith("./")) and not Path(policy).is_dir():
        found = discover_policies(root)
        if found:
            newest = found[0]
            try:
                rel = str(newest.relative_to(root))
            except ValueError:
                rel = str(newest)
            note = f"Configured policy path is missing ({collapse_home(policy)}); using newest checkpoint: {rel}"
            policy = str(newest)
    return policy, note


def _checkpoint_error(policy: str) -> str | None:
    p = Path(policy)
    if p.is_dir():
        if not is_valid_checkpoint(p):
            return f"'{policy}' is not a valid checkpoint; expected config.json and model.safetensors."
        return None
    if p.exists():
        return f"'{policy}' exists, but it is not a directory."
    return None


def _device_note(policy: str, gpu_name: str) -> str:
    if gpu_name:
        return f"CUDA available: {gpu_name}; CPU fallback if needed"
    # No GPU: read the checkpoint config's device so a cpu/mps-trained checkpoint is
    # reported as such (else the generic no-GPU line). Guard the read — an empty/non-dir
    # policy must not raise inside draw() (port of the Textual _device_note 3rd branch).
    ckpt_dev = ""
    p = Path(policy)
    if p.is_dir():
        try:
            ckpt_dev = json.loads((p / "config.json").read_text()).get("device", "") or ""
        except Exception:
            ckpt_dev = ""
    if ckpt_dev and ckpt_dev != "cuda":
        return f"{ckpt_dev} (from checkpoint config)"
    return "No NVIDIA GPU detected; CPU will likely be slow"


def detect_cam_slots(policy: str, rename_map: dict | None) -> tuple[str, str]:
    """Resolve the camera-slots mode for a checkpoint: ("map"|"native", note).

    The checkpoint's config.json input_features names the camera keys the policy was
    trained on; the yaml rollout.rename_map maps robot-side keys -> trained slots.
    Trained keys ⊆ map VALUES → "map" (the yaml rename applies, e.g. camera1/2/3
    FT checkpoints); trained keys ⊆ map KEYS → "native" (the policy wants the raw
    robot names — eval.sh then neutralizes the yaml map with an identity override,
    because draccus MERGES dict CLI overrides so an empty map cannot clear it). Anything unreadable/unrecognized (hub repo ids, exotic key
    sets like pi05's base_0_rgb) falls back to "map" — today's behavior — with a
    note the form surfaces so a wrong guess is visible before the robot moves.
    """
    mode, note, _confident = detect_cam_detail(policy, rename_map)
    return mode, note


def detect_cam_detail(policy: str, rename_map: dict | None) -> tuple[str, str, bool]:
    """As :func:`detect_cam_slots`, plus whether the answer was DERIVED or GUESSED.

    The third element is the point. Three of the branches below are fallbacks — no rename
    map, unreadable config, unrecognised keys — and they return the same "map" as a
    confident match. Rendering a guess and a derivation identically is how a wrong image
    routing reaches the robot silently, so the caller marks the guesses.
    """
    rmap = {str(k): str(v) for k, v in (rename_map or {}).items()}
    if not rmap:
        return "map", "no yaml rename_map", False
    try:
        feats = json.loads((Path(policy) / "config.json").read_text()).get("input_features", {})
    except Exception:
        return "map", "checkpoint config unreadable", False
    cams = {k for k in feats if ".images." in k}
    if not cams:
        return "map", "no image features in checkpoint", False
    short = "/".join(sorted(k.rsplit(".", 1)[-1] for k in cams))
    if cams <= set(rmap.values()):
        return "map", short, True
    if cams <= set(rmap.keys()):
        return "native", short, True
    return "map", f"unknown keys ({short})", False


def _preprocessor_rename(data: dict) -> dict | None:
    for step in data.get("steps", []) or []:
        m = (step.get("config") or {}).get("rename_map")
        if isinstance(m, dict) and m:
            return m
    return None


def training_rename_map(policy: str) -> dict[str, str] | None:
    """The rename map the checkpoint was TRAINED with, or None if it does not record one.

    Two sources, checked in order: ``train_config.json``'s ``rename_map`` and the saved
    ``policy_preprocessor.json`` (its RenameObservations step). They agree in practice; the
    preprocessor is the fallback for checkpoints saved without the train config.

    This is the AUTHORITATIVE mapping. The yaml's ``rollout.rename_map`` is only what this
    workspace happens to be configured to send, and the two can disagree.
    """
    base = Path(policy)
    for name, pick in (("train_config.json", lambda d: d.get("rename_map")),
                       ("policy_preprocessor.json", _preprocessor_rename)):
        try:
            data = json.loads((base / name).read_text())
        except Exception:
            continue
        m = pick(data)
        if isinstance(m, dict) and m:
            return {str(k): str(v) for k, v in m.items()}
    return None


def cam_map_conflicts(policy: str, rollout_map: dict | None) -> list[tuple[str, str, str]]:
    """Per-slot disagreements as ``[(slot, trained_from, rollout_from)]``.

    THE failure this exists for: a set-subset check cannot see a PERMUTATION. If training
    fed top→camera2 and rollout sends wrist→camera2, both sides still use exactly
    {camera1, camera2, camera3}, so every "are these compatible" test based on membership
    passes while the policy receives two of its three views swapped. Nothing errors; the
    robot just behaves worse, which is the hardest kind of bug to notice.
    """
    trained_map = training_rename_map(policy)
    if not trained_map or not rollout_map:
        return []
    tail = lambda s: s.rsplit(".", 1)[-1]  # noqa: E731
    trained = {tail(v): tail(k) for k, v in trained_map.items()}
    live = {tail(v): tail(k) for k, v in
            {str(a): str(b) for a, b in rollout_map.items()}.items()}
    return [(slot, trained.get(slot, "—"), live.get(slot, "—"))
            for slot in sorted(set(trained) | set(live))
            if trained.get(slot) != live.get(slot)]


def cam_pairs(mode: str, rename_map: dict | None,
              trained_map: dict | None = None) -> list[tuple[str, str]]:
    """The concrete robot-camera → policy-slot pairs that *mode* will actually apply.

    Shown in full because the mapping is NOT guessable from the slot names, and it is not
    a property of the robot either: each training run picks its own pairing, so the same
    three cameras can be front→camera1/top→camera2/wrist→camera3 in one checkpoint and
    front→camera1/wrist→camera2/top→camera3 in the next. Anyone assuming the obvious
    alphabetical pairing has two views swapped and the policy sees the wrong images, with
    nothing raising. Hence "trained", which takes the pairing from the checkpoint itself;
    "native" applies no rename at all.
    """
    source = trained_map if mode == "trained" and trained_map else rename_map
    rmap = {str(k): str(v) for k, v in (source or {}).items()}
    tail = lambda s: s.rsplit(".", 1)[-1]  # noqa: E731
    if mode == "native":
        return [(tail(k), tail(k)) for k in rmap]
    return [(tail(k), tail(v)) for k, v in rmap.items()]


def _state(ctx: "Context") -> dict[str, Any]:
    """Per-session Run policy form memory.

    This intentionally does not write ``lekiwi.yaml``. It only keeps the last edited
    values while the TUI process is alive, so reopening Run policy feels continuous.
    """
    ui_state = getattr(ctx, "ui_state", None)
    if ui_state is None:
        ui_state = {}
        setattr(ctx, "ui_state", ui_state)
    return ui_state.setdefault(STATE_KEY, {})


def _state_int(state: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(state.get(key, default))
    except (TypeError, ValueError):
        return default


def _state_bool(state: dict[str, Any], key: str, default: bool) -> bool:
    val = state.get(key, default)
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


class EvalScreen(ScreenState):
    """Eval/rollout configuration form; Start suspends into lerobot-rollout (real TTY)."""

    title = "run policy"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._root = _policy_root_path(ctx.cfg)
        default_policy, self._fallback_note = resolve_eval_policy(ctx.cfg["POLICY_PATH"], self._root)
        self._default_abs = default_policy
        state = _state(ctx)
        remembered_policy = str(state.get("policy") or "")
        self._policy = remembered_policy or default_policy
        if remembered_policy:
            self._fallback_note = None
        self._task_text = str(state.get("task", cfg_get("rollout.task", doc=ctx.doc) or ""))
        backend = str(state.get("backend", ctx.cfg["INFERENCE"]))
        self._backend = backend if backend in ("sync", "rtc") else ctx.cfg["INFERENCE"]
        self._exec = NumberField(
            "Action horizon",
            _state_int(state, "exec_horizon", int(ctx.cfg["EXECUTION_HORIZON"])),
            minimum=1,
            step=1,
        )
        self._steps = NumberField(
            "Action steps",
            _state_int(state, "action_steps", 0),
            minimum=0,
            step=1,
            zero_label="checkpoint default",
        )
        self._flow = NumberField(
            "Flow steps",
            _state_int(state, "flow_steps", 0),
            minimum=0,
            step=1,
            zero_label="checkpoint default",
        )
        self._rename_map = cfg_get("rollout.rename_map", doc=ctx.doc) or {}
        cam_mode = str(state.get("cam_mode", "auto"))
        self._cam_mode = cam_mode if cam_mode in CAM_MODES else "auto"
        self._ckpt_cache: tuple[str, dict] | None = None
        self._refresh_ckpt_defaults()
        self._dur = NumberField(
            "Duration",
            _state_int(state, "duration", 0),
            minimum=0,
            step=5,
            unit="s",
            zero_label="saved default",
        )
        default_show = str(ctx.cfg["DISPLAY_DATA"]).lower() in ("1", "true", "yes", "on")
        self._show = _state_bool(state, "display", default_show)
        # Free-text passthrough: extra tokens forwarded verbatim to lerobot-rollout
        # (eval.sh collects unknown flags and appends them last, draccus last-wins).
        # Lets a power user set policy fields the form has no dedicated row for, e.g.
        # --policy.num_inference_timesteps=10 (EVO1's flow-step field) without new UI.
        self._extra_text = str(state.get("extra_flags", ""))
        self._err = ""
        self._fpos = 0
        self._fresh = True

    def _remember(self) -> None:
        _state(self.ctx).update({
            "policy": self._policy,
            "task": self._task_text,
            "backend": self._backend,
            "exec_horizon": self._exec.value,
            "action_steps": self._steps.value,
            "flow_steps": self._flow.value,
            "cam_mode": self._cam_mode,
            "duration": self._dur.value,
            "display": self._show,
            "extra_flags": self._extra_text,
        })

    def _ckpt_info(self) -> dict:
        """The current policy's parsed config.json ({} when unreadable), cached per
        path — the form re-reads it only when the policy changes, not per frame."""
        if self._ckpt_cache is not None and self._ckpt_cache[0] == self._policy:
            return self._ckpt_cache[1]
        try:
            info = json.loads((Path(self._policy) / "config.json").read_text())
        except Exception:
            info = {}
        if not isinstance(info, dict):
            info = {}
        self._ckpt_cache = (self._policy, info)
        return info

    def _refresh_ckpt_defaults(self) -> None:
        """Show the checkpoint's OWN values inside the 0-sentinel labels, so
        "checkpoint default" is never a surprise (a checkpoint may ship an atypical
        n_action_steps, e.g. 1 where the usual value is 50)."""
        n = self._ckpt_info().get("n_action_steps")
        self._steps.zero_label = (
            f"checkpoint default ({n})" if isinstance(n, int) else "checkpoint default"
        )
        k = self._ckpt_info().get("num_steps")
        self._flow.zero_label = (
            f"checkpoint default ({k})" if isinstance(k, int) else "checkpoint default"
        )

    def _fields(self) -> list[str]:
        """Navigation order, which MUST equal visual order (see the test of the same name).

        `cameras` sits next to `backend` because that is where it renders: both answer "how
        does the policy talk to the robot". It used to sit after `flow`, which was invisible
        while backend and cameras shared one row and became a skipped row the moment they
        stopped sharing.
        """
        f = ["policy", "task", "backend", "cameras"]
        if self._backend == "rtc":
            f.append("exec")
        else:
            f.append("steps")  # sync-only: rtc paces via exec-horizon, not n_action_steps
        # "flow" shows for BOTH backends and ALL checkpoints: a policy that never
        # reads num_steps ignores the override (inert) — no model-type gating.
        return f + ["flow", "duration", "display", "extra", "start"]

    def _cam_detail(self) -> tuple[str, str, bool]:
        """(effective mode, detection note, detection was confident).

        AUTO PREFERS "trained": when the checkpoint records the mapping it was trained
        with, that is authoritative and gets sent verbatim, which makes a yaml/checkpoint
        permutation impossible rather than merely visible. Only when the checkpoint records
        nothing does auto fall back to the map/native guess.
        """
        mode, note, confident = detect_cam_detail(self._policy, self._rename_map)
        if self._cam_mode != "auto":
            return self._cam_mode, note, confident
        if training_rename_map(self._policy):
            return "trained", "from the checkpoint", True
        return mode, note, confident

    def _cam_warning(self) -> str:
        """The one thing that must never be silent: images routed to the wrong slots.

        Two ways that happens — the detection fell back to a guess, or you forced a mode
        that contradicts what the checkpoint was trained on. Both are called out; a clean
        auto-detection says nothing.
        """
        # Checked FIRST and worded hardest: a per-slot disagreement means the policy is fed
        # the wrong views while every set-based check still passes. Nothing else on this
        # screen can go wrong this quietly.
        mode, _note, _conf = self._cam_detail()
        conflicts = cam_map_conflicts(self._policy, self._rename_map)
        # ONLY when the yaml map is what actually goes out. Under "trained" the checkpoint's
        # map is sent, so the disagreement is already resolved; under "native" nothing is
        # renamed at all, so "sending wrist" would be a false statement about the argv — the
        # forced-mode branch below describes that case correctly.
        if conflicts and mode == "map":
            detail = " · ".join(f"{slot}: trained {was}, sending {now}"
                                for slot, was, now in conflicts)
            return f"MAPPING MISMATCH — {detail}"
        auto_mode, auto_note, confident = detect_cam_detail(self._policy, self._rename_map)
        if self._cam_mode != "auto" and confident and self._cam_mode != auto_mode:
            return (f"forced {self._cam_mode}, but this checkpoint was trained for "
                    f"{auto_mode} ({auto_note})")
        if not confident:
            return f"not verified against the checkpoint: {auto_note}"
        return ""

    def _cam_resolved(self) -> tuple[str, str]:
        """The concrete --cam-slots value + display note for the current mode/policy.

        Delegates to :meth:`_cam_detail` so what the screen SHOWS and what argv SENDS can
        never disagree — including the "trained" resolution, which is the whole point of
        having it.
        """
        mode, note, _confident = self._cam_detail()
        return mode, note if self._cam_mode == "auto" else "forced"

    def _cur(self) -> str:
        fs = self._fields()
        return fs[min(self._fpos, len(fs) - 1)]

    def _num(self, key: str) -> "NumberField | None":
        return {"exec": self._exec, "steps": self._steps, "flow": self._flow,
                "duration": self._dur}.get(key)

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name == "s":
            return Invoke(self._start)
        if name in (UP, "k", BACKTAB):
            self._move(-1); return Nothing
        if name in (DOWN, "j", TAB):
            self._move(1); return Nothing
        cur = self._cur()
        if name in (LEFT, "h", RIGHT, "l"):
            delta = -1 if name in (LEFT, "h") else 1
            if cur == "backend":
                self._backend = "sync" if self._backend == "rtc" else "rtc"
                self._fpos = min(self._fpos, len(self._fields()) - 1)
                self._remember()
            elif cur in ("exec", "steps", "flow", "duration"):
                self._num(cur).step_by(delta); self._num(cur).sync_editor(); self._fresh = True
                self._remember()
            elif cur == "cameras":
                i = CAM_MODES.index(self._cam_mode)
                self._cam_mode = CAM_MODES[(i + delta) % len(CAM_MODES)]
                self._remember()
            elif cur == "display":
                self._show = not self._show
                self._remember()
            self._err = ""; return Nothing
        if name == ENTER:
            if cur == "policy":
                return Invoke(self._pick_policy)
            if cur == "task":
                return Invoke(self._edit_task)
            if cur == "extra":
                return Invoke(self._edit_extra)
            if cur == "backend":
                self._backend = "sync" if self._backend == "rtc" else "rtc"
                self._remember()
                self._fpos = min(self._fpos, len(self._fields()) - 1); return Nothing
            if cur == "cameras":
                i = CAM_MODES.index(self._cam_mode)
                self._cam_mode = CAM_MODES[(i + 1) % len(CAM_MODES)]
                self._remember(); return Nothing
            if cur == "display":
                self._show = not self._show; self._remember(); return Nothing
            if cur in ("exec", "steps", "flow", "duration"):
                self._commit_num(cur); return Nothing
            if cur == "start":
                return Invoke(self._start)
            return Nothing
        if cur in ("exec", "steps", "flow", "duration") and (is_char(key) or name == "Backspace"):
            ed = self._num(cur).editor
            if self._fresh and is_char(key):
                ed.clear()
            self._fresh = False
            if ed.handle_key(key):
                if ed.value.strip().isdigit():
                    self._num(cur).set_text(ed.value.strip())
                    self._remember()
                self._err = ""
                return Nothing
        return Nothing

    def _move(self, delta: int) -> None:
        fs = self._fields()
        self._fpos = (self._fpos + delta) % len(fs)
        self._err = ""
        nf = self._num(self._cur())
        if nf is not None:
            nf.sync_editor()
        self._fresh = True

    def _commit_num(self, cur: str) -> None:
        nf = self._num(cur)
        if not nf.set_text(nf.editor.value):
            self._err = nf.error
        else:
            self._err = ""; nf.sync_editor(); self._fresh = True; self._remember()

    # ── async flows ─────────────────────────────────────────────────────────────
    async def _edit_task(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Task instruction", value=self._task_text, multiline=True,
            hint="⏎ apply (blank = saved default) · ←→ move · esc keep current"))
        if ans is not None:
            self._task_text = ans.strip(); self._err = ""; self._remember()

    async def _edit_extra(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Extra lerobot flags", value=self._extra_text,
            hint="⏎ apply · forwarded to lerobot-rollout · e.g. --policy.num_inference_timesteps=10"))
        if ans is not None:
            self._extra_text = ans.strip(); self._err = ""; self._remember()

    async def _pick_policy(self) -> None:
        chosen = await self.app.run_modal(PolicyPicker(
            self._root, default_abs=self._default_abs, title="Pick a checkpoint"))
        if chosen is None:
            return
        if chosen == CUSTOM:
            ans = await self.app.run_modal(PromptModalState(
                "Policy path or model repo id", value=collapse_home(self._policy), hint="⏎ apply · esc cancel"))
            if ans is None or not ans:
                return
            policy = os.path.expanduser(ans)
            if not policy.startswith("/") and (self._root / policy).is_dir():
                policy = str(self._root / policy)
            self._policy = policy
        else:
            self._policy = chosen
        self._fallback_note = None
        self._err = ""
        self._refresh_ckpt_defaults()  # new checkpoint -> new 0-sentinel labels
        self._remember()

    async def _start(self) -> None:
        app = self.app
        try:
            extra_tokens = shlex.split(self._extra_text)
        except ValueError as exc:
            self._err = f"extra flags: {exc}"
            return
        if not self._policy:
            self._err = f"No policy found: POLICY_PATH is auto, and no checkpoint exists under '{self._root}'."
            return
        err = _checkpoint_error(self._policy)
        if err is not None:
            self._err = err
            return
        if not await confirm_preflight(
            app,
            "Run policy preflight",
            eval_issues(self.ctx, policy=self._policy),
        ):
            return
        self._remember()
        cam_slots, _note = self._cam_resolved()  # eval.sh takes only the concrete value
        argv = [
            "bash", str(EVAL_SCRIPT),
            "--policy", self._policy, "--task", self._task_text, "--backend", self._backend,
            "--exec-horizon", str(self._exec.value), "--action-steps", str(self._steps.value),
            "--flow-steps", str(self._flow.value),
            "--cam-slots", cam_slots, "--duration", str(self._dur.value),
            "--display", "on" if self._show else "off", "--gpu", self.ctx.gpu_name,
            *extra_tokens, *self._extra,
        ]
        rc = await app.suspend(argv, pause=True)
        await self._ask_verdict(rc)

    async def _ask_verdict(self, rc: int) -> None:
        """The post-run scoreboard verdict: one modal (Success / Failure / Skip), one
        JSONL append. Only after a REAL run that ended normally (0, or 130 = Ctrl+C,
        the usual end of an until-Ctrl+C rollout) — a crashed run isn't a verdict."""
        if runner.DRY_RUN or rc not in (0, 130):
            return
        label = ckpt_label(self._policy, self._root)
        task = (self._task_text or str(cfg_get("rollout.task", doc=self.ctx.doc) or "")).strip()
        choice = await self.app.run_modal(ConfirmModalState(
            f'Scoreboard — how did {label} do on "{_clip_end(task, 48)}"?',
            ["Success", "Failure", "Skip — no verdict"]))
        if choice not in ("Success", "Failure"):
            return
        ok = choice == "Success"
        if append_score(self._root, {"ts": time.time(), "label": label, "task": task,
                                     "success": ok}):
            s, n = score_tally(load_scores(self._root), label=label, task=task)
            self.app.notify(f"scoreboard: {label} on this task → {s}/{n}", "info")
        else:
            self.app.notify("✗ could not write the scoreboard file", "warn")

    # ── view ────────────────────────────────────────────────────────────────────
    _LABEL_W = 16

    def _header_right(self) -> list[Span]:
        """Eval needs BOTH live facts: the host (robot side) and the GPU (policy side)."""
        spans = slim_status_spans(self.ctx)
        gpu = [Span("GPU ", theme.MUTED_STYLE)]
        if self.ctx.gpu_name:
            gpu += [Span(f"{theme.status_dot()} ", theme.OK_STYLE),
                    Span("cuda", theme.TEXT_STYLE)]
        else:
            gpu.append(Span("cpu — slow", theme.WARN_STYLE))
        # host-dot spans come first, then the GPU, then the mode chip (already in spans).
        return spans[:-2] + gpu + [Span("   ", theme.BASE_STYLE)] + spans[-2:]

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Clear(), area)
        draw_form_page(frame, area, self.ctx, "run policy", self._body_lines(area.width),
                       header_right=self._header_right(),
                       msg=f"✗ {self._err}" if self._err else "",
                       hint=self._focused_hint())

    def _lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *keys: str) -> Span:
        on = self._cur() in keys
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _num_spans(self, key: str, pad: int = 0) -> list[Span]:
        nf = self._num(key)
        if key == self._cur():
            return [Span(f"{nf.editor.value}█", theme.HIGHLIGHT_TEXT_STYLE)]
        return [Span(f"{nf.display():<{pad}}" if pad else nf.display(), theme.TEXT_STYLE)]

    #: What each pacing number means, shown always. When the value is 0 the field's own
    #: zero_label wins (``number_line`` prefers it), so "checkpoint default (50)" still
    #: explains itself — including while the field is focused, which it previously did not.
    _NOTES = {
        "steps": "actions consumed per policy call",
        "exec": "actions executed before the next call",
        "flow": "flow-matching integration steps",
        "duration": "how long the rollout runs",
    }

    def _source_note(self, key: str) -> str:
        return self._NOTES.get(key, "")

    def _host_alive(self) -> bool | None:
        return host_alive(self.ctx)

    def _plan(self, width: int = 100) -> str:
        return _clip_end(
            f"run {ckpt_label(self._policy, self._root)} · {self._backend}"
            f" · {self._dur.display()} · hands the terminal to the policy",
            max(24, int(width) - 14))

    def _scoreboard_lines(self, width: int) -> list[Line]:
        """Per-task tallies for the SELECTED checkpoint (up to 3 tasks, most-judged
        first). Empty when the checkpoint has no verdicts yet — zero cost to ignore."""
        scores = load_scores(self._root)
        label = ckpt_label(self._policy, self._root)
        tasks: dict[str, tuple[int, int]] = {}
        for e in scores:
            if e.get("label") == label:
                t = str(e.get("task", ""))
                s, n = tasks.get(t, (0, 0))
                tasks[t] = (s + (1 if e.get("success") else 0), n + 1)
        if not tasks:
            return []
        out = [Line([]), Line([Span("  SCOREBOARD", theme.FAINT_STYLE),
                               Span(f"  {label}", theme.FAINT_STYLE)])]
        for t, (s, n) in sorted(tasks.items(), key=lambda kv: -kv[1][1])[:3]:
            frac_ok = n > 0 and s / n >= 0.5
            bar = "▓" * s + "░" * (n - s) if not theme.ASCII_MODE else "#" * s + "-" * (n - s)
            out.append(Line([
                Span(f"  {_clip_end(t, max(20, width - 40)):<{max(20, width - 40)}}",
                     theme.MUTED_STYLE),
                Span(f"  {s}/{n} ", theme.TEXT_STYLE),
                Span(bar, theme.OK_STYLE if frac_ok else theme.WARN_STYLE),
            ]))
        return out

    def _body_lines(self, width: int = 100) -> list[Line]:
        cur = self._cur()
        w = int(width)
        lines: list[Line] = [section_line("POLICY")]
        # Policy row — value middle-clipped, checkpoint age/size right-aligned.
        info = ""
        p = Path(self._policy) if self._policy else None
        if p is not None and p.is_dir():
            try:
                mtime = datetime.fromtimestamp((p / "config.json").stat().st_mtime)
                size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
                info = f"{mtime.strftime('%b %d')} · {size / 1e9:.1f} GB" if size else mtime.strftime("%b %d")
            except OSError:
                info = ""
        # Name on its own row with the checkpoint's age/size right-aligned, PATH underneath.
        # Middle-clipping one long string served neither: the name is what you recognise and
        # the path is what you verify, and the ellipsis ate the informative end of both.
        lines.append(padded_line(
            [self._gutter("policy"), self._lab("Policy", cur == "policy"),
             Span(ckpt_label(self._policy, self._root), theme.TEXT_STYLE)],
            [Span(info, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w))
        if self._policy:
            lines.append(Line([
                Span(" " * (2 + self._LABEL_W), theme.BASE_STYLE),
                Span(_clip_middle(collapse_home(self._policy), max(24, w - self._LABEL_W - 4)),
                     theme.FAINT_STYLE),
            ]))
        if self._fallback_note:
            lines.append(Line([Span(" " * (2 + self._LABEL_W), theme.BASE_STYLE),
                               Span(f"⚠ {self._fallback_note}", theme.WARN_STYLE)]))
        # Task wraps with a hanging indent rather than being clipped: the end of an
        # instruction is the part that distinguishes two similar tasks.
        task_segs = wrap_words(self._task_text or "(saved default)",
                               max(20, w - 4 - self._LABEL_W))
        lines.append(Line([self._gutter("task"), self._lab("Task", cur == "task"),
                           Span(task_segs[0], theme.TEXT_STYLE)]))
        lines.extend(Line([Span(" " * (2 + self._LABEL_W), theme.BASE_STYLE),
                           Span(s, theme.TEXT_STYLE)]) for s in task_segs[1:])
        lines.append(Line([]))
        lines.append(section_line("RUN"))
        # One setting per row. Backend and Cameras used to share a line with the resolved
        # mapping stranded underneath; each now carries its own always-visible note (a wrong
        # slot guess drives the robot on the wrong camera views, so it must never be clipped).
        lines.append(setting_line(
            "Backend",
            [seg("sync", self._backend == "sync"), Span(" ", theme.BASE_STYLE),
             seg("rtc", self._backend == "rtc")],
            "obs-gated, steadier" if self._backend == "sync" else "smoother for slow policies",
            focused=cur == "backend", label_width=self._LABEL_W, width=w))
        mode, note, confident = self._cam_detail()
        origin = "auto → " if self._cam_mode == "auto" else "forced "
        lines.append(setting_line(
            "Cameras",
            [s for m in CAM_MODES for s in
             (seg(m, self._cam_mode == m), Span(" ", theme.BASE_STYLE))][:-1],
            f"{origin}{mode} · {note}" if confident else f"{origin}{mode}",
            focused=cur == "cameras", label_width=self._LABEL_W, width=w))
        # The CONCRETE mapping, always on screen. It is not guessable from the slot names
        # (this robot maps wrist→camera2 and top→camera3, not the alphabetical pairing), and
        # a wrong routing shows up as the robot acting on the wrong view rather than as an error.
        trained_map = training_rename_map(self._policy)
        pairs = cam_pairs(mode, self._rename_map, trained_map)
        indent = " " * (2 + self._LABEL_W)
        if pairs:
            text = "   ".join(f"{a}→{b}" for a, b in pairs)
            for chunk in wrap_words(text, max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE),
                                   Span(chunk, theme.FAINT_STYLE)]))
        conflicts = cam_map_conflicts(self._policy, self._rename_map)
        if mode == "trained" and conflicts:
            slots = ", ".join(slot for slot, _was, _now in conflicts)
            for chunk in wrap_words(
                    f"using the checkpoint's own mapping — lekiwi.yaml disagrees on {slots}",
                    max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE),
                                   Span(chunk, theme.OK_STYLE)]))
        warning = self._cam_warning()
        if warning:
            style = theme.ERR_STYLE if warning.startswith("MAPPING MISMATCH") else theme.WARN_STYLE
            for chunk in wrap_words(f"⚠ {warning}", max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE), Span(chunk, style)]))
        # per-backend pacing field + flow steps share a row (rtc ⇄ sync swaps the label)
        pace_key = "exec" if self._backend == "rtc" else "steps"
        pace_lab = "Action horizon" if self._backend == "rtc" else "Action steps"
        for key, label in ((pace_key, pace_lab), ("flow", "Flow steps"),
                           ("duration", "Duration")):
            lines.append(number_line(self._num(key), label, cur == key,
                                     self._source_note(key), width=w,
                                     label_width=self._LABEL_W))
        lines.append(setting_line(
            "Display", toggle(self._show, focused=cur == "display"),
            "mirror the cameras in a window",
            focused=cur == "display", label_width=self._LABEL_W, width=w))
        lines.append(Line([
            self._gutter("extra"), self._lab("Extra flags", cur == "extra"),
            Span(self._value("extra", width=max(24, w - self._LABEL_W - 4)),
                 theme.TEXT_STYLE if self._extra_text else theme.FAINT_STYLE),
        ]))
        lines.append(Line([]))
        focused = cur == "start"
        if self._host_alive() is False:
            plan_span = Span("⚠ host not reachable — Start host first (menu 1)",
                             theme.WARN_STYLE)
        else:
            plan_span = Span(self._plan(w),
                             theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Run", [plan_span], focused=focused))
        lines.extend(self._scoreboard_lines(w))
        return lines

    def _focused_hint(self) -> str:
        field = self._cur()
        if field == "policy":
            return "⏎ pick a checkpoint · newest first"
        if field == "task":
            return "⏎ edit the task instruction (blank = saved default)"
        if field == "backend":
            return ("RTC: smoother control for slower policies · ←→/⏎ switch"
                    if self._backend == "rtc"
                    else "Sync: one policy forward per control tick · ←→/⏎ switch")
        if field == "exec":
            return "prediction window for RTC · ←→ ±1 · ⏎ type a number"
        if field == "steps":
            return "open-loop actions per forward (0 = checkpoint) · ←→ ±1 · ⏎ type"
        if field == "flow":
            return "FM integration steps (0 = checkpoint) · ←→ ±1 · ⏎ type"
        if field == "cameras":
            return "camera slots the policy expects · auto reads the checkpoint · ←→ cycle"
        if field == "duration":
            return "0 = the saved yaml default · ←→ ±5 · ⏎ type a number"
        if field == "display":
            return "show live Rerun view (off lowers CPU) · ←→/⏎ toggle"
        if field == "extra":
            return "⏎ edit extra lerobot flags passed through verbatim (e.g. --policy.num_inference_timesteps=10)"
        if self._host_alive() is False:
            return "preflight will stop the launch while the host is down"
        return f"device: {_device_note(self._policy, self.ctx.gpu_name)} · s also starts"

    def _policy_display_path(self) -> str:
        if not self._policy:
            return ""
        try:
            return str(Path(self._policy).relative_to(self._root))
        except ValueError:
            return collapse_home(self._policy)

    def _value(self, field: str, *, width: int = 60) -> str:
        if field == "policy":
            disp = self._policy_display_path() or collapse_home(self._policy) or "(none found)"
            suffix = "  default" if self._policy and self._policy == self._default_abs else ""
            disp = _clip_middle(disp, max(8, width - len(suffix))) + suffix
            return disp
        if field == "task":
            t = (self._task_text or "(saved default)").splitlines()[0]
            return _clip_end(t, width)
        if field == "backend":
            return theme.choice(self._backend)
        if field in ("exec", "steps", "flow", "duration"):
            nf = self._num(field)
            disp = (nf.editor.value + "█") if field == self._cur() else nf.display()
            return _clip_end(disp, width)
        if field == "cameras":
            mode, note = self._cam_resolved()
            if self._cam_mode == "auto":
                return _clip_end(f"{theme.choice('auto')} · {mode} · {note}", width)
            return _clip_end(f"{theme.choice(mode)} · {note}", width)
        if field == "display":
            return theme.choice("on") if self._show else theme.choice("off")
        if field == "extra":
            return _clip_end(self._extra_text or "(none)", width)
        return ""

    def _summary_value(self, field: str) -> str:
        if field == "policy":
            disp = self._policy_display_path() or collapse_home(self._policy) or "(none found)"
            if self._policy and self._policy == self._default_abs:
                disp += "  default"
            return disp
        if field == "task":
            return (self._task_text or "(saved default)").replace("\n", " ")
        if field == "backend":
            if self._backend == "rtc":
                return f"rtc · action horizon {self._exec.display()}"
            return f"sync · action steps {self._steps.display()}"
        if field == "flow":
            return self._flow.display()
        if field == "cameras":
            mode, note = self._cam_resolved()
            origin = "auto-detected" if self._cam_mode == "auto" else "forced"
            return f"{mode} · {note} · {origin}"
        if field == "duration":
            return self._dur.display()
        if field == "display":
            return "on · live Rerun view" if self._show else "off · lower CPU"
        if field == "extra":
            return self._extra_text or "(none)"
        return ""

def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY direct run of the eval action: resolve + validate the checkpoint, then front
    scripts/eval.sh directly (no app loop) with env defaults (INFERENCE / EXECUTION_HORIZON
    / DISPLAY_DATA). Emits the SAME flags the screen's _start computes from these defaults so
    headless and interactive front the script identically; --task is NOT passed (the config
    default applies) and --duration is 0 (the script omits --duration when 0). The script is
    the SOLE argv source; this path assembles no lerobot tokens.

    Ported from the Textual ``run_headless(app, extra)``; this port threads config through
    ``ctx`` (there is no ``app.cfg``), reading ``ctx.cfg[...]`` / ``ctx.gpu_name``. Carried
    for parity."""
    root = _policy_root_path(ctx.cfg)
    policy, _note = resolve_eval_policy(ctx.cfg["POLICY_PATH"], root)
    if not policy:
        print(f"✗ no policy found: POLICY_PATH is auto, and no checkpoint exists under '{root}'.")
        return 1
    err = _checkpoint_error(policy)
    if err is not None:
        print(f"✗ {err}")
        return 1
    show = str(ctx.cfg["DISPLAY_DATA"]).lower() in ("1", "true", "yes", "on")
    # Same auto camera-slots resolution as the form's default mode; action-steps has no
    # env knob, so headless always passes 0 (the script omits the token = checkpoint
    # default), exactly like an untouched form.
    cam_slots, _note = detect_cam_slots(policy, cfg_get("rollout.rename_map", doc=ctx.doc) or {})
    argv = [
        "bash", str(EVAL_SCRIPT),
        "--policy", policy, "--backend", ctx.cfg["INFERENCE"],
        "--exec-horizon", str(ctx.cfg["EXECUTION_HORIZON"]),
        "--action-steps", "0", "--flow-steps", "0", "--cam-slots", cam_slots,
        "--duration", "0", "--display", "on" if show else "off",
        "--gpu", ctx.gpu_name, *extra,
    ]
    return runner.headless_run(argv)


if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App


__all__ = ["EvalScreen", "detect_cam_slots", "resolve_eval_policy", "run_headless", "HEADLESS_HOOK"]
