"""policy_form.py — the half that Run policy and DAgger have in common.

Both screens configure the same thing: which checkpoint drives the robot, on which
instruction, through which camera slots, at which pacing. They are the same
``lerobot-rollout`` invocation and differ only in ``--strategy.type`` and in what the
session then does with the result. So the CONFIG half lives here and the two screens
are thin subclasses that add their own rows, launch and aftermath.

They deliberately stay two screens: Run policy observes, DAgger writes a dataset and
needs the leader arm, and the two have different preflights, different in-session
protocols and different post-run flows. Merging them into one screen with a mode row
would trade eleven duplicated rows for eight conditionals, and would hide DAgger from
anyone who does not already know it exists.

What the split buys, and what this module makes true: the shared settings are
remembered ONCE (``SHARED_STATE_KEY``), so a checkpoint, task string, camera mode and
pacing tuned while watching a rollout are already in place when you switch to
collecting corrections. Each screen keeps its own memory only for its own rows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Line, Span

from ..config import cfg_get, collapse_home
from ..datasets import record_root
from ..framework import theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key
from ..framework.modals import PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField, wrap_words
from ..hostprobe import host_alive
from ..scoreboard import ckpt_label
from ..widgets.pickers import CUSTOM, PolicyPicker
from ..widgets.task_choices import TaskChoices
from .chrome import clip_end as _clip_end
from .chrome import clip_middle as _clip_middle
from .chrome import (
    draw_form_page, number_line, padded_line, seg, setting_line,
    task_stepper_cell, toggle,
)

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: Session memory for the settings BOTH screens have. One key, so they carry across.
SHARED_STATE_KEY = "policy_form"

# Camera-slots modes, in ←→ cycle order. TWO choices on purpose.
#
#   auto   — send the mapping the CHECKPOINT recorded (the only authoritative source);
#            fall back to the yaml map, or to raw names, when it records none.
#   native — force NO renaming. The one case auto can get wrong in the other direction:
#            a policy trained on raw front/top/wrist whose config auto cannot read.
#
# "map" and "trained" were dropped from the picker and remain valid `--cam-slots`
# values for CLI/headless use. "trained" was identical to auto whenever it worked and a
# lie when it did not (it reported "trained" while the launcher silently emitted no
# token). "map" only ever DIFFERED from auto when the checkpoint recorded a map and the
# yaml contradicted it, i.e. a button whose sole distinct behaviour is the bug we detect.
CAM_MODES = ("auto", "native")


# ── checkpoint + camera-slot resolution (module-level: also used headless) ─────


def _policy_root_path(cfg) -> Path:
    from .. import ROOT

    p = Path(cfg["POLICY_ROOT"]).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def resolve_eval_policy(policy_path: str, root: Path) -> tuple[str, str | None]:
    """Resolve POLICY_PATH + the dead-path fallback."""
    from ..policies import discover_policies, resolve_policy

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
            note = (f"Configured policy path is missing ({collapse_home(policy)}); "
                    f"using newest checkpoint: {rel}")
            policy = str(newest)
    return policy, note


def _checkpoint_error(policy: str) -> str | None:
    from ..policies import is_valid_checkpoint

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
    # policy must not raise inside draw().
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
    """Resolve the camera-slots mode for a checkpoint: ("map"|"native", note)."""
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
    """The rename map the checkpoint was TRAINED with, or None if it records none.

    Two sources, checked in order: ``train_config.json``'s ``rename_map`` and the saved
    ``policy_preprocessor.json`` (its RenameObservations step). This is the
    AUTHORITATIVE mapping; the yaml's is only what this workspace happens to send.
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


def training_dataset_name(policy: str) -> str | None:
    """The bare NAME of the dataset the checkpoint was trained on, or None.

    Read from ``train_config.json``'s ``dataset.repo_id``. The task strings a policy
    actually understands are the ones in ITS training data, so this is a better default
    for the Base row than whatever dataset the yaml happens to point at.
    """
    try:
        data = json.loads((Path(policy) / "train_config.json").read_text())
    except Exception:
        return None
    repo = (data.get("dataset") or {}).get("repo_id")
    return str(repo).rsplit("/", 1)[-1] if repo else None


def resolve_base_dataset(policy: str, parent: str | Path) -> str | None:
    """The local dataset dir under *parent* matching the checkpoint's training dataset.

    None when the checkpoint records none or it is not on this machine. Paths stay
    RELATIVE (the app runs with cwd=ROOT), like ``discover_datasets``.
    """
    name = training_dataset_name(policy)
    if not name:
        return None
    candidate = Path(parent) / name
    return str(candidate) if candidate.is_dir() else None


def cam_map_conflicts(policy: str, rollout_map: dict | None) -> list[tuple[str, str, str]]:
    """Per-slot disagreements as ``[(slot, trained_from, rollout_from)]``.

    THE failure this exists for: a set-subset check cannot see a PERMUTATION. If training
    fed top→camera2 and rollout sends wrist→camera2, both sides still use exactly
    {camera1, camera2, camera3}, so every membership check passes while the policy
    receives two of its three views swapped. Nothing errors; the robot just behaves worse.
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
    a property of the robot either: each training run picks its own pairing.
    """
    source = trained_map if mode == "trained" and trained_map else rename_map
    rmap = {str(k): str(v) for k, v in (source or {}).items()}
    tail = lambda s: s.rsplit(".", 1)[-1]  # noqa: E731
    if mode == "native":
        return [(tail(k), tail(k)) for k in rmap]
    return [(tail(k), tail(v)) for k, v in rmap.items()]


# ── session memory ────────────────────────────────────────────────────────────


def _ui_state(ctx: "Context") -> dict[str, Any]:
    ui_state = getattr(ctx, "ui_state", None)
    if ui_state is None:
        ui_state = {}
        setattr(ctx, "ui_state", ui_state)
    return ui_state


def shared_state(ctx: "Context") -> dict[str, Any]:
    """The settings BOTH policy forms hold, remembered once for both.

    Per-session only: this intentionally does not write ``lekiwi.yaml``. It keeps the
    last edited values while the TUI process is alive, so moving between Run policy and
    DAgger — which is the normal way a session goes, watch it fail then correct it —
    does not mean configuring the same checkpoint and task twice.
    """
    return _ui_state(ctx).setdefault(SHARED_STATE_KEY, {})


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


class PolicyFormScreen(ScreenState):
    """Shared configuration form for the policy-driven screens.

    Subclasses supply: the yaml section their defaults come from, any extra rows, the
    launch, and the aftermath. Everything else — the policy picker, the base dataset and
    its task strings, backend, camera slots, pacing, duration, display and passthrough
    flags — lives here and is remembered once for both screens.
    """

    #: yaml block this screen's defaults come from ("rollout" / "dagger").
    CONFIG_SECTION = "rollout"
    #: session-memory key for the subclass's OWN rows (the shared ones are shared).
    STATE_KEY = "policy_form_own"
    #: the Duration row's step and 0-sentinel differ per screen.
    DURATION_STEP = 5
    DURATION_ZERO_LABEL = "saved default"

    _LABEL_W = 16

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._root = _policy_root_path(ctx.cfg)
        default_policy, self._fallback_note = resolve_eval_policy(ctx.cfg["POLICY_PATH"], self._root)
        self._default_abs = default_policy
        state = shared_state(ctx)
        remembered_policy = str(state.get("policy") or "")
        self._policy = remembered_policy or default_policy
        if remembered_policy:
            self._fallback_note = None
        # Base dataset: where the Task row's choices come from. Default = the dataset
        # THIS checkpoint was trained on (from its train_config.json), resolved under the
        # datasets dir; the yaml record dataset is the fallback. A base picked by hand is
        # PINNED, so switching checkpoints afterwards no longer moves it.
        self._ds_parent = str(Path(record_root(ctx.doc)).parent)
        self._base_pinned = bool(state.get("base_pinned", False))
        base = str(state.get("base") or "")
        if base:
            self._base_source = "pinned" if self._base_pinned else "remembered"
        else:
            derived = resolve_base_dataset(self._policy, self._ds_parent)
            base = derived or record_root(ctx.doc)
            self._base_source = "from checkpoint" if derived else "yaml default"
        self._task_choices = TaskChoices(base)
        self._task_text = str(state.get("task", self._cfg(f"{self.CONFIG_SECTION}.task") or ""))
        backend = str(state.get("backend")
                      or self._cfg(f"{self.CONFIG_SECTION}.inference.type")
                      or ctx.cfg["INFERENCE"])
        self._backend = backend if backend in ("sync", "rtc") else "sync"
        self._exec = NumberField(
            "Action horizon",
            _state_int(state, "exec_horizon", int(ctx.cfg["EXECUTION_HORIZON"])),
            minimum=1, step=1)
        self._steps = NumberField(
            "Action steps", _state_int(state, "action_steps", 0), minimum=0, step=1,
            zero_label="checkpoint default")
        self._flow = NumberField(
            "Flow steps", _state_int(state, "flow_steps", 0), minimum=0, step=1,
            zero_label="checkpoint default")
        self._rename_map = self._cfg(f"{self.CONFIG_SECTION}.rename_map") or {}
        cam_mode = str(state.get("cam_mode", "auto"))
        self._cam_mode = cam_mode if cam_mode in CAM_MODES else "auto"
        self._ckpt_cache: tuple[str, dict] | None = None
        self._refresh_ckpt_defaults()
        self._dur = NumberField(
            "Duration", _state_int(state, "duration", 0), minimum=0,
            step=self.DURATION_STEP, unit="s", zero_label=self.DURATION_ZERO_LABEL)
        default_show = str(ctx.cfg["DISPLAY_DATA"]).lower() in ("1", "true", "yes", "on")
        self._show = _state_bool(state, "display", default_show)
        # Free-text passthrough: extra tokens forwarded verbatim to lerobot-rollout
        # (the launcher collects unknown flags and appends them last, draccus last-wins).
        # Lets a power user set policy fields the form has no dedicated row for.
        self._extra_text = str(state.get("extra_flags", ""))
        self._err = ""
        self._fpos = 0
        self._fresh = True
        self._init_own(state)

    def _cfg(self, dotted: str) -> Any:
        return cfg_get(dotted, doc=self.ctx.doc)

    # ── hooks for subclasses ──────────────────────────────────────────────────
    def _init_own(self, shared: dict[str, Any]) -> None:
        """Build the subclass's own rows (called at the end of __init__)."""

    def _own_state(self) -> dict[str, Any]:
        """Session memory for the subclass's own rows."""
        return _ui_state(self.ctx).setdefault(self.STATE_KEY, {})

    def _remember_own(self) -> None:
        """Persist the subclass's own rows."""

    def _extra_fields(self) -> list[str]:
        """Rows the subclass inserts between the pacing rows and the tail."""
        return []

    def _handle_own_key(self, key: "Key", cur: str) -> Any | None:
        """Handle a key for a subclass row; return None to fall through to shared."""
        return None

    async def _start(self) -> None:
        raise NotImplementedError

    # ── session memory ────────────────────────────────────────────────────────
    def _remember(self) -> None:
        shared_state(self.ctx).update({
            "policy": self._policy,
            "base": self._task_choices.base,
            "base_pinned": self._base_pinned,
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
        self._remember_own()

    # ── checkpoint sentinels ──────────────────────────────────────────────────
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
            f"checkpoint default ({n})" if isinstance(n, int) else "checkpoint default")
        k = self._ckpt_info().get("num_steps")
        self._flow.zero_label = (
            f"checkpoint default ({k})" if isinstance(k, int) else "checkpoint default")

    # ── fields ────────────────────────────────────────────────────────────────
    def _common_fields(self) -> list[str]:
        """Navigation order, which MUST equal visual order.

        `cameras` sits next to `backend` because that is where it renders: both answer
        "how does the policy talk to the robot". The pacing row swaps with the backend
        (rtc paces via exec-horizon, sync via n_action_steps); `flow` shows for both,
        and for all checkpoints — a policy that never reads num_steps ignores it.
        """
        return ["policy", "base", "task", "backend", "cameras",
                "exec" if self._backend == "rtc" else "steps", "flow"]

    def _tail_fields(self) -> list[str]:
        return ["duration", "display", "extra"]

    def _blank_task_label(self) -> str:
        """What an empty Task row says. Overridden where blank is not a valid choice:
        Run policy falls back to the yaml default, DAgger refuses to start."""
        return "(saved default)"

    def _fields(self) -> list[str]:
        return self._common_fields() + self._extra_fields() + ["start"]

    def _cur(self) -> str:
        fs = self._fields()
        return fs[min(self._fpos, len(fs) - 1)]

    def _num(self, key: str) -> "NumberField | None":
        return {"exec": self._exec, "steps": self._steps, "flow": self._flow,
                "duration": self._dur}.get(key)

    # ── camera-slot resolution ────────────────────────────────────────────────
    def _cam_detail(self) -> tuple[str, str, bool]:
        """(effective mode, detection note, detection was confident).

        AUTO PREFERS "trained": when the checkpoint records the mapping it was trained
        with, that is authoritative and gets sent verbatim, which makes a yaml/checkpoint
        permutation impossible rather than merely visible.
        """
        mode, note, confident = detect_cam_detail(self._policy, self._rename_map)
        if self._cam_mode != "auto":
            return self._cam_mode, note, confident
        if training_rename_map(self._policy):
            return "trained", "from the checkpoint", True
        return mode, note, confident

    def _cam_warning(self) -> str:
        """The one thing that must never be silent: images routed to the wrong slots."""
        # Checked FIRST and worded hardest: a per-slot disagreement means the policy is
        # fed the wrong views while every set-based check still passes.
        mode, _note, _conf = self._cam_detail()
        conflicts = cam_map_conflicts(self._policy, self._rename_map)
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
        """The concrete --cam-slots value + display note for the current mode/policy."""
        mode, note, _confident = self._cam_detail()
        return mode, note if self._cam_mode == "auto" else "forced"

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
        own = self._handle_own_key(key, cur)
        if own is not None:
            return own
        if name in (LEFT, "h", RIGHT, "l"):
            delta = -1 if name in (LEFT, "h") else 1
            if cur == "task":
                self._cycle_task(delta)
            elif cur == "backend":
                self._toggle_backend()
            elif cur == "cameras":
                i = CAM_MODES.index(self._cam_mode)
                self._cam_mode = CAM_MODES[(i + delta) % len(CAM_MODES)]
                self._remember()
            elif cur == "display":
                self._show = not self._show
                self._remember()
            elif cur in ("exec", "steps", "flow", "duration"):
                nf = self._num(cur)
                nf.step_by(delta); nf.sync_editor(); self._fresh = True
                self._remember()
            self._err = ""; return Nothing
        if name == ENTER:
            if cur == "policy":
                return Invoke(self._pick_policy)
            if cur == "base":
                return Invoke(self._pick_base)
            if cur == "task":
                return Invoke(self._edit_task)
            if cur == "extra":
                return Invoke(self._edit_extra)
            if cur == "backend":
                self._toggle_backend(); return Nothing
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
        if cur in ("exec", "steps", "flow", "duration") and self._num(cur).type_key(
                key, fresh=self._fresh):
            self._fresh = False
            self._err = ""
            self._remember()
            return Nothing
        return Nothing

    def _toggle_backend(self) -> None:
        self._backend = "sync" if self._backend == "rtc" else "rtc"
        self._fpos = min(self._fpos, len(self._fields()) - 1)
        self._remember()

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

    def _cycle_task(self, delta: int) -> None:
        """←→ walks the base dataset's strings (inert when it has none)."""
        self._task_text = self._task_choices.cycle(self._task_text, delta)
        self._remember()

    # ── async flows ───────────────────────────────────────────────────────────
    async def _pick_policy(self) -> None:
        chosen = await self.app.run_modal(PolicyPicker(
            self._root, default_abs=self._default_abs, title="Pick a checkpoint"))
        if chosen is None:
            return
        if chosen == CUSTOM:
            ans = await self.app.run_modal(PromptModalState(
                "Policy path or model repo id", value=collapse_home(self._policy),
                hint="⏎ apply · esc cancel"))
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
        if not self._base_pinned:
            # Follow the new checkpoint's own training dataset (a pinned base stays put).
            derived = resolve_base_dataset(self._policy, self._ds_parent)
            if derived:
                self._task_choices.base = derived
                self._base_source = "from checkpoint"
        self._remember()

    async def _pick_base(self) -> None:
        """Pick which dataset's task strings the Task row offers. Picking PINS the
        choice, so a later checkpoint switch keeps it."""
        from ..dispatch import pick_dataset

        picked = await pick_dataset(self.app, self.ctx.doc,
                                    title="Task strings - pick the base dataset")
        if picked is None:
            return
        _repo, root = picked
        self._task_choices.base = root
        self._base_pinned = True
        self._base_source = "pinned"
        tasks = self._task_choices.tasks()
        # Keep a blank task (it means "use the saved yaml default"); only a stale
        # non-empty string snaps onto the new base's first choice.
        if tasks and self._task_text and self._task_text not in tasks:
            self._task_text = tasks[0]
        self._err = ""
        self._remember()

    async def _edit_task(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Task (language instruction)", value=self._task_text, multiline=True,
            hint="⏎ apply · ←→ move · ctrl+j newline · esc keep current"))
        if ans is not None:
            self._task_text = ans.strip(); self._err = ""; self._remember()

    async def _edit_extra(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Extra lerobot flags", value=self._extra_text,
            hint="⏎ apply · forwarded to lerobot-rollout · e.g. --policy.num_inference_timesteps=10"))
        if ans is not None:
            self._extra_text = ans.strip(); self._err = ""; self._remember()

    # ── view helpers ──────────────────────────────────────────────────────────
    def _lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *keys: str) -> Span:
        on = self._cur() in keys
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _host_alive(self) -> bool | None:
        return host_alive(self.ctx)

    def _indent(self) -> str:
        return " " * (2 + self._LABEL_W)

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Clear(), area)
        draw_form_page(frame, area, self.ctx, self.title, self._body_lines(area.width),
                       header_right=self._header_right(),
                       msg=f"✗ {self._err}" if self._err else "",
                       hint=self._focused_hint())

    def _header_right(self) -> list[Span] | None:
        return None

    # ── shared row groups ─────────────────────────────────────────────────────
    def _policy_rows(self, w: int) -> list[Line]:
        """Name (+ age/size right-aligned), path underneath, any fallback note."""
        from datetime import datetime

        cur = self._cur()
        info = ""
        p = Path(self._policy) if self._policy else None
        if p is not None and p.is_dir():
            try:
                mtime = datetime.fromtimestamp((p / "config.json").stat().st_mtime)
                size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
                info = (f"{mtime.strftime('%b %d')} · {size / 1e9:.1f} GB" if size
                        else mtime.strftime("%b %d"))
            except OSError:
                info = ""
        lines = [padded_line(
            [self._gutter("policy"), self._lab("Policy", cur == "policy"),
             Span(ckpt_label(self._policy, self._root), theme.TEXT_STYLE)],
            [Span(info, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w)]
        if self._policy:
            lines.append(Line([
                Span(self._indent(), theme.BASE_STYLE),
                Span(_clip_middle(collapse_home(self._policy), max(24, w - self._LABEL_W - 4)),
                     theme.FAINT_STYLE)]))
        if self._fallback_note:
            lines.append(Line([Span(self._indent(), theme.BASE_STYLE),
                               Span(f"⚠ {self._fallback_note}", theme.WARN_STYLE)]))
        return lines

    def _task_rows(self, w: int) -> list[Line]:
        """Base dataset + the task string it offers, with the divergence warning."""
        cur = self._cur()
        indent = self._indent()
        tasks = self._task_choices.tasks()
        base_note = (f"{len(tasks)} task string(s) · {self._base_source}" if tasks
                     else f"no task strings · {self._base_source}")
        lines = [padded_line(
            [self._gutter("base"), self._lab("Base", cur == "base"),
             Span(self._task_choices.name, theme.TEXT_STYLE)],
            [Span(base_note, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w)]
        position = self._task_choices.position(self._task_text)
        cell_spans, cell_cols = task_stepper_cell(position, len(tasks), focused=cur == "task")
        task_segs = wrap_words(self._task_text or self._blank_task_label(),
                               max(20, w - 4 - self._LABEL_W - cell_cols))
        lines.append(Line([self._gutter("task"), self._lab("Task", cur == "task"),
                           *cell_spans, Span(task_segs[0], theme.TEXT_STYLE)]))
        lines.extend(Line([Span(indent + " " * cell_cols, theme.BASE_STYLE),
                           Span(s, theme.TEXT_STYLE)]) for s in task_segs[1:])
        if tasks and position is None and self._task_text:
            # A string the base does not contain is a silently different prompt — the
            # exact divergence the picker exists to prevent, never left implicit.
            for chunk in wrap_words(
                    f"⚠ not one of {self._task_choices.name}'s strings — the policy was "
                    "conditioned on those; ←→ picks them",
                    max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE),
                                   Span(chunk, theme.WARN_STYLE)]))
        return lines

    def _backend_camera_rows(self, w: int) -> list[Line]:
        """Backend + camera slots, the concrete mapping, and any mapping warning."""
        cur = self._cur()
        indent = self._indent()
        lines = [setting_line(
            "Backend",
            [seg("sync", self._backend == "sync"), Span(" ", theme.BASE_STYLE),
             seg("rtc", self._backend == "rtc")],
            "obs-gated, steadier" if self._backend == "sync" else "smoother for slow policies",
            focused=cur == "backend", label_width=self._LABEL_W, width=w)]
        mode, note, confident = self._cam_detail()
        origin = "auto → " if self._cam_mode == "auto" else "forced "
        lines.append(setting_line(
            "Cameras",
            [s for m in CAM_MODES for s in
             (seg(m, self._cam_mode == m), Span(" ", theme.BASE_STYLE))][:-1],
            f"{origin}{mode} · {note}" if confident else f"{origin}{mode}",
            focused=cur == "cameras", label_width=self._LABEL_W, width=w))
        # The CONCRETE mapping, always on screen. It is not guessable from the slot
        # names, and a wrong routing shows up as the robot acting on the wrong view
        # rather than as an error.
        pairs = cam_pairs(mode, self._rename_map, training_rename_map(self._policy))
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
        return lines

    #: What each pacing number means, shown always. When the value is 0 the field's own
    #: zero_label wins, so "checkpoint default (50)" still explains itself.
    _NOTES = {
        "steps": "actions consumed per policy call",
        "exec": "actions executed before the next call",
        "flow": "flow-matching integration steps",
        "duration": "how long the rollout runs",
    }

    def _pacing_rows(self, w: int) -> list[Line]:
        cur = self._cur()
        pace = "exec" if self._backend == "rtc" else "steps"
        label = "Action horizon" if pace == "exec" else "Action steps"
        return [number_line(self._num(pace), label, cur == pace, self._NOTES[pace],
                            width=w, label_width=self._LABEL_W),
                number_line(self._flow, "Flow steps", cur == "flow", self._NOTES["flow"],
                            width=w, label_width=self._LABEL_W)]

    def _tail_rows(self, w: int, *, duration_note: str = "") -> list[Line]:
        cur = self._cur()
        return [
            number_line(self._dur, "Duration", cur == "duration",
                        duration_note or self._NOTES["duration"], width=w,
                        label_width=self._LABEL_W),
            setting_line("Display", toggle(self._show, focused=cur == "display"),
                         "mirror the cameras in a window",
                         focused=cur == "display", label_width=self._LABEL_W, width=w),
            Line([self._gutter("extra"), self._lab("Extra flags", cur == "extra"),
                  Span(_clip_end(self._extra_text or "(none)",
                                 max(24, w - self._LABEL_W - 4)),
                       theme.TEXT_STYLE if self._extra_text else theme.FAINT_STYLE)]),
        ]

    def _body_lines(self, width: int = 100) -> list[Line]:
        raise NotImplementedError

    # ── hints ─────────────────────────────────────────────────────────────────
    _COMMON_HINTS = {
        "policy": "⏎ pick a checkpoint · newest first",
        "base": "⏎ pick the dataset whose task strings this form offers",
        "exec": "prediction window for RTC · ←→ ±1 · ⏎ type a number",
        "steps": "open-loop actions per forward (0 = checkpoint) · ←→ ±1 · ⏎ type",
        "flow": "FM integration steps (0 = checkpoint) · ←→ ±1 · ⏎ type",
        "cameras": "camera slots the policy expects · auto reads the checkpoint · ←→ cycle",
        "display": "show live Rerun view (off lowers CPU) · ←→/⏎ toggle",
        "extra": "⏎ edit extra lerobot flags passed through verbatim",
    }

    def _common_hint(self, field: str) -> str:
        if field == "backend":
            return ("RTC: smoother control for slower policies · ←→/⏎ switch"
                    if self._backend == "rtc"
                    else "Sync: one policy forward per control tick · ←→/⏎ switch")
        if field == "task":
            return ("←→ cycle the base dataset's strings · ⏎ free text (blank = saved default)"
                    if self._task_choices.tasks()
                    else "⏎ edit the task instruction (base has no strings; blank = saved default)")
        return self._COMMON_HINTS.get(field, "")

    def _focused_hint(self) -> str:
        raise NotImplementedError


__all__ = [
    "CAM_MODES", "PolicyFormScreen", "SHARED_STATE_KEY", "cam_map_conflicts", "cam_pairs",
    "detect_cam_detail", "detect_cam_slots", "resolve_base_dataset", "resolve_eval_policy",
    "shared_state", "training_dataset_name", "training_rename_map",
]
