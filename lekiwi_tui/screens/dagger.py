"""dagger.py — DaggerScreen: configure + launch HIL correction collection
(lerobot-rollout --strategy.type=dagger, fronted by scripts/dagger.sh).

DAgger is launch-a-policy (like eval) plus produce-a-dataset (like record), so the
form borrows from both: a PolicyPicker + backend/cameras rows from eval, and a
dataset destination like record — except the destination is COMPUTED (a per-session
stamped dir under the datasets parent; rollout refuses to reuse a dir, so there is
nothing to name). Two rows are dagger's own:

  * Base — the dataset this session's corrections will later be merged into,
    picked with the shared DatasetPicker;
  * Task — cycles the BASE DATASET's own task strings (meta/tasks.parquet), so the
    stamped string can never typo-diverge from the training data. ⏎ still opens
    free text for a genuinely new instruction.

Start validates the checkpoint, runs the dagger preflight (leader + disk + the
dagger yaml block), shows the session cheat-sheet (the two protocol rules that are
expensive to learn live: squeeze the trigger when taking over mid-grasp; never use
tab to reposition — it records), then SUSPENDS into the rollout, which owns the
real TTY for its keys (space/tab/enter/ESC via lerobot's own reader; base wasd via
the composite leader's keyboard, exactly as in record). Fronts scripts/dagger.sh
(the sole argv source).
"""
from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Line, Span

from .. import ROOT
from ..config import cfg_get, collapse_home
from ..dagger_review import dagger_episode_report, session_summary, write_quality_flags
from ..datasets import dataset_episodes, record_root
from ..framework import theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField, wrap_words
from ..hostprobe import host_alive
from ..preflight import confirm_preflight, dagger_issues
from ..scoreboard import ckpt_label
from ..widgets.pickers import CUSTOM, PolicyPicker
from ..widgets.task_choices import TaskChoices
from .chrome import clip_end as _clip_end
from .chrome import clip_middle as _clip_middle
from .chrome import (
    draw_form_page, number_line, padded_line, plan_row, section_line, seg, setting_line,
    task_stepper_cell, toggle,
)
from .eval import (
    CAM_MODES, _checkpoint_error, _policy_root_path, cam_map_conflicts, cam_pairs,
    detect_cam_detail, resolve_eval_policy, training_rename_map,
)

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

DAGGER_SCRIPT = ROOT / "scripts" / "dagger.sh"
STATE_KEY = "dagger"

#: The session cheat-sheet, shown as the last gate before the terminal is handed
#: over — one key per line, then the three rules that cost data (or a dropped
#: grasped object) to discover live. Structured with \n (wrap_label honors breaks).
#: Deliberately NO `enter = push` line: this setup is local-only (push_to_hub off,
#: `local/` namespace), so advertising the hub push would be a lie.
_CHEATSHEET = (
    "space — pause / resume. Pausing glides the leader onto the follower (~2 s): "
    "hands off until it settles.\n"
    "tab — start / stop a correction (only while paused). Every stop saves one episode.\n"
    "esc — end the session; the arm returns to its start pose.\n"
    "\n"
    "Taking over mid-grasp? Squeeze the trigger as you grab, or the gripper eases open.\n"
    "Stop at a stable point: grasp = steady lifted hold; insertion = seated + released.\n"
    "Reset the scene by hand while paused — never teleop with tab, corrections record."
)
_GO = "Start session"
_STAY = "Cancel"

_MODE_LABELS = {False: "corrections-only", True: "record-all"}


def _state(ctx: "Context") -> dict[str, Any]:
    """Per-session form memory (same idiom as EvalScreen's _state)."""
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


def session_root(parent: str, *, now: "time.struct_time | None" = None) -> str:
    """The per-session dataset dir: ``<parent>/rollout_dagger_<YYYYMMDD_HHMMSS>``.

    Computed HERE (not left to dagger.sh's identical fallback) so the screen knows
    the exact root afterwards — the post-session review reads it. The ``rollout_``
    prefix is lerobot's own validation rule for deployment datasets.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return str(Path(parent) / f"rollout_dagger_{stamp}")


class DaggerScreen(ScreenState):
    """DAgger session form; Start gates on preflight + cheat-sheet then suspends."""

    title = "dagger"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._root = _policy_root_path(ctx.cfg)
        default_policy, self._fallback_note = resolve_eval_policy(ctx.cfg["POLICY_PATH"], self._root)
        self._default_abs = default_policy
        state = _state(ctx)
        remembered = str(state.get("policy") or "")
        self._policy = remembered or default_policy
        if remembered:
            self._fallback_note = None
        # Base dataset: the demos this session's corrections extend. Default = the
        # record dataset (what the policy was trained on in the usual loop).
        self._task_choices = TaskChoices(str(state.get("base") or record_root(ctx.doc)))
        self._task_text = str(state.get("task") or cfg_get("dagger.task", doc=ctx.doc) or "")
        backend = str(state.get("backend", cfg_get("dagger.inference.type", doc=ctx.doc) or "sync"))
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
        self._target = NumberField(
            "Corrections", _state_int(state, "target", 10), minimum=1, step=1)
        self._rename_map = cfg_get("dagger.rename_map", doc=ctx.doc) or {}
        cam_mode = str(state.get("cam_mode", "auto"))
        self._cam_mode = cam_mode if cam_mode in CAM_MODES else "auto"
        self._ckpt_cache: tuple[str, dict] | None = None
        self._refresh_ckpt_defaults()
        self._advanced = bool(state.get("advanced", False))
        self._record_all = bool(state.get("record_all", False))
        self._pedal = bool(state.get("pedal", False))
        self._dur = NumberField(
            "Duration", _state_int(state, "duration", 0), minimum=0, step=60, unit="s",
            zero_label="no time limit")
        self._show = bool(state.get("display", False))
        self._extra_text = str(state.get("extra_flags", ""))
        self._err = ""
        self._fpos = 0
        self._fresh = True

    def _remember(self) -> None:
        _state(self.ctx).update({
            "policy": self._policy, "base": self._base, "task": self._task_text,
            "backend": self._backend, "exec_horizon": self._exec.value,
            "action_steps": self._steps.value, "flow_steps": self._flow.value,
            "target": self._target.value, "cam_mode": self._cam_mode,
            "advanced": self._advanced, "record_all": self._record_all,
            "pedal": self._pedal, "duration": self._dur.value,
            "display": self._show, "extra_flags": self._extra_text,
        })

    # ── checkpoint-default sentinels (same contract as EvalScreen's) ──────────
    def _ckpt_info(self) -> dict:
        """The current policy's parsed config.json ({} when unreadable), cached per
        path — re-read only when the policy changes, not per frame."""
        if self._ckpt_cache is not None and self._ckpt_cache[0] == self._policy:
            return self._ckpt_cache[1]
        try:
            import json

            info = json.loads((Path(self._policy) / "config.json").read_text())
        except Exception:
            info = {}
        if not isinstance(info, dict):
            info = {}
        self._ckpt_cache = (self._policy, info)
        return info

    def _refresh_ckpt_defaults(self) -> None:
        """Show the checkpoint's OWN values inside the 0-sentinel labels, so
        "checkpoint default" is never a surprise."""
        n = self._ckpt_info().get("n_action_steps")
        self._steps.zero_label = (
            f"checkpoint default ({n})" if isinstance(n, int) else "checkpoint default"
        )
        k = self._ckpt_info().get("num_steps")
        self._flow.zero_label = (
            f"checkpoint default ({k})" if isinstance(k, int) else "checkpoint default"
        )

    # ── the base dataset's task strings (shared with the Run-policy form) ─────
    @property
    def _base(self) -> str:
        return self._task_choices.base

    @_base.setter
    def _base(self, root: str) -> None:
        self._task_choices.base = root

    def _tasks(self) -> list[str]:
        return self._task_choices.tasks()

    def _cycle_task(self, delta: int) -> None:
        self._task_text = self._task_choices.cycle(self._task_text, delta)
        self._remember()

    # ── field list (navigation order == visual order) ─────────────────────────
    def _fields(self) -> list[str]:
        f = ["policy", "base", "task", "backend", "cameras"]
        # Same pacing pair as EvalScreen: rtc paces via exec-horizon, sync via
        # n_action_steps; flow steps apply to both (inert for a policy without them).
        f.append("exec" if self._backend == "rtc" else "steps")
        f += ["flow", "target", "advanced"]
        if self._advanced:
            f += ["mode", "input", "duration", "display", "extra"]
        return f + ["start"]

    def _cur(self) -> str:
        fs = self._fields()
        return fs[min(self._fpos, len(fs) - 1)]

    def _num(self, key: str) -> "NumberField | None":
        return {"exec": self._exec, "steps": self._steps, "flow": self._flow,
                "target": self._target, "duration": self._dur}.get(key)

    # ── camera-slots resolution (same contract as EvalScreen, dagger's map) ───
    def _cam_detail(self) -> tuple[str, str, bool]:
        mode, note, confident = detect_cam_detail(self._policy, self._rename_map)
        if self._cam_mode != "auto":
            return self._cam_mode, note, confident
        if training_rename_map(self._policy):
            return "trained", "from the checkpoint", True
        return mode, note, confident

    def _cam_warning(self) -> str:
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
            if cur == "task":
                self._cycle_task(delta)
            elif cur == "backend":
                self._backend = "sync" if self._backend == "rtc" else "rtc"
                self._fpos = min(self._fpos, len(self._fields()) - 1)
                self._remember()
            elif cur == "cameras":
                i = CAM_MODES.index(self._cam_mode)
                self._cam_mode = CAM_MODES[(i + delta) % len(CAM_MODES)]
                self._remember()
            elif cur == "advanced":
                self._advanced = not self._advanced
                self._fpos = min(self._fpos, len(self._fields()) - 1)
                self._remember()
            elif cur == "mode":
                self._record_all = not self._record_all; self._remember()
            elif cur == "input":
                self._pedal = not self._pedal; self._remember()
            elif cur == "display":
                self._show = not self._show; self._remember()
            elif cur in ("exec", "steps", "flow", "target", "duration"):
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
                self._backend = "sync" if self._backend == "rtc" else "rtc"
                self._remember()
                self._fpos = min(self._fpos, len(self._fields()) - 1); return Nothing
            if cur == "cameras":
                i = CAM_MODES.index(self._cam_mode)
                self._cam_mode = CAM_MODES[(i + 1) % len(CAM_MODES)]
                self._remember(); return Nothing
            if cur == "advanced":
                self._advanced = not self._advanced
                self._fpos = min(self._fpos, len(self._fields()) - 1)
                self._remember(); return Nothing
            if cur == "mode":
                self._record_all = not self._record_all; self._remember(); return Nothing
            if cur == "input":
                self._pedal = not self._pedal; self._remember(); return Nothing
            if cur == "display":
                self._show = not self._show; self._remember(); return Nothing
            if cur in ("exec", "steps", "flow", "target", "duration"):
                self._commit_num(cur); return Nothing
            if cur == "start":
                return Invoke(self._start)
            return Nothing
        if cur in ("exec", "steps", "flow", "target", "duration") and self._num(cur).type_key(key, fresh=self._fresh):
            self._fresh = False
            self._err = ""
            self._remember()
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

    # ── async flows ────────────────────────────────────────────────────────────
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
            import os

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

    async def _pick_base(self) -> None:
        from ..dispatch import pick_dataset

        picked = await pick_dataset(self.app, self.ctx.doc, title="Base dataset (task strings + later merge)")
        if picked is None:
            return
        _repo, root = picked
        self._base = root
        # New base -> new task list; keep the current string only if the new base
        # also carries it (else snap to the base's first string, or free text).
        tasks = self._tasks()
        if tasks and self._task_text not in tasks:
            self._task_text = tasks[0]
        self._err = ""
        self._remember()

    async def _edit_task(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Task (language instruction)", value=self._task_text, multiline=True,
            hint="⏎ apply · ←→ move · ctrl+j newline · esc keep current"))
        if ans is not None and ans.strip():
            self._task_text = ans.strip(); self._err = ""; self._remember()

    async def _edit_extra(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Extra lerobot flags", value=self._extra_text,
            hint="⏎ apply · forwarded to lerobot-rollout · e.g. --strategy.pedal.device_path=…"))
        if ans is not None:
            self._extra_text = ans.strip(); self._err = ""; self._remember()

    async def _start(self) -> None:
        app = self.app
        try:
            extra_tokens = shlex.split(self._extra_text)
        except ValueError as exc:
            self._err = f"extra flags: {exc}"
            return
        if not self._policy:
            self._err = (f"No policy found: POLICY_PATH is auto, and no checkpoint "
                         f"exists under '{self._root}'.")
            return
        err = _checkpoint_error(self._policy)
        if err is not None:
            self._err = err
            return
        if not self._task_text:
            self._err = "Pick a task string first — dagger stamps it on every correction."
            return
        parent = str(Path(record_root(self.ctx.doc)).parent)
        if not await confirm_preflight(
            app, "DAgger preflight",
            dagger_issues(self.ctx, policy=self._policy, parent=parent),
        ):
            return
        # The last gate: the two session rules that cost data to learn live.
        if await app.run_modal(ConfirmModalState(_CHEATSHEET, [_GO, _STAY])) != _GO:
            return
        self._remember()
        root = session_root(parent)
        mode, _note, _conf = self._cam_detail()
        argv = [
            "bash", str(DAGGER_SCRIPT),
            "--policy", self._policy, "--task", self._task_text,
            "--backend", self._backend, "--exec-horizon", str(self._exec.value),
            "--action-steps", str(self._steps.value),
            "--flow-steps", str(self._flow.value),
            "--target", str(self._target.value),
            "--record-autonomous", "on" if self._record_all else "off",
            "--input", "pedal" if self._pedal else "keyboard",
            "--duration", str(self._dur.value),
            "--display", "on" if self._show else "off",
            "--gpu", self.ctx.gpu_name, "--cam-slots", mode,
            "--dataset-root", root,
            *extra_tokens, *self._extra,
        ]
        rc = await app.suspend(argv, pause=True)
        # 130 (Ctrl+C) is a normal end for an until-ESC session, not a crash.
        _state(self.ctx)["last_root"] = root
        if rc in (0, 130):
            await self._review_session(root)

    async def _review_session(self, root: str) -> None:
        """The post-session gate: per-episode stats, junk pre-flagged, one keypress
        into the dataset editor. Junk verdicts ride the editor's own quality.jsonl
        sidecar, so flagged episodes arrive there already marked — review is
        "open → D → typed delete", not a second triage UI."""
        app = self.app
        n = dataset_episodes(root)
        if n in ("?", "0"):
            app.notify(f"dagger session ended — nothing recorded ({root} stays empty)", "info")
            return
        report = dagger_episode_report(root)
        flagged = {r["index"]: r["junk"] for r in report if r["junk"]}
        if flagged and not write_quality_flags(root, flagged):
            app.notify("✗ could not write junk flags to meta/quality.jsonl", "warn")
        junk_note = (f" — {len(flagged)} look like junk (pre-marked for delete)"
                     if flagged else "")
        title = (f"Session saved {n} correction(s) → {Path(root).name}{junk_note}.  "
                 + session_summary(report))
        review = "Review in dataset editor"
        if await app.run_modal(ConfirmModalState(title, [review, "Done"])) == review:
            from .dataset_edit import DatasetEditScreen

            app.push(DatasetEditScreen(app, self.ctx, root=root))

    # ── view ────────────────────────────────────────────────────────────────────
    _LABEL_W = 14

    def _lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *keys: str) -> Span:
        on = self._cur() in keys
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _host_alive(self) -> bool | None:
        return host_alive(self.ctx)

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Clear(), area)
        draw_form_page(frame, area, self.ctx, "dagger", self._body_lines(area.width),
                       msg=f"✗ {self._err}" if self._err else "",
                       hint=self._focused_hint())

    def _body_lines(self, width: int = 100) -> list[Line]:
        cur = self._cur()
        w = int(width)
        indent = " " * (2 + self._LABEL_W)
        lines: list[Line] = [section_line("POLICY")]
        lines.append(Line([
            self._gutter("policy"), self._lab("Policy", cur == "policy"),
            Span(ckpt_label(self._policy, self._root), theme.TEXT_STYLE)]))
        if self._policy:
            lines.append(Line([
                Span(indent, theme.BASE_STYLE),
                Span(_clip_middle(collapse_home(self._policy), max(24, w - self._LABEL_W - 4)),
                     theme.FAINT_STYLE)]))
        if self._fallback_note:
            lines.append(Line([Span(indent, theme.BASE_STYLE),
                               Span(f"⚠ {self._fallback_note}", theme.WARN_STYLE)]))
        lines.append(Line([]))
        lines.append(section_line("TASK"))
        tasks = self._tasks()
        base_note = (f"{len(tasks)} task string(s) · {dataset_episodes(self._base)} ep"
                     if tasks else "no task strings found")
        lines.append(padded_line(
            [self._gutter("base"), self._lab("Base", cur == "base"),
             Span(Path(self._base).name, theme.TEXT_STYLE)],
            [Span(base_note, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w))
        # The task is ADJUSTABLE (←→ cycles the base dataset's strings), so it gets the
        # same guillemet stepper cell every adjustable row carries — ``‹ 2/8 ›`` — instead
        # of reading as static text. A custom/stale string shows ``‹ –/8 ›`` plus a
        # warning: stamping a string the base dataset does not contain is exactly the
        # silent divergence the picker exists to prevent.
        position = self._task_choices.position(self._task_text)
        cell_spans, cell_cols = task_stepper_cell(position, len(tasks), focused=cur == "task")
        task_warn = ""
        if tasks and position is None and self._task_text:
            task_warn = (f"not one of {self._task_choices.name}'s strings — "
                         "←→ picks them, ⏎ keeps editing")
        task_disp = self._task_text or ("(←→ pick a string · ⏎ type)" if tasks else "(⏎ to type)")
        task_segs = wrap_words(task_disp, max(20, w - 4 - self._LABEL_W - cell_cols))
        lines.append(Line([self._gutter("task"), self._lab("Task", cur == "task"),
                           *cell_spans, Span(task_segs[0], theme.TEXT_STYLE)]))
        text_indent = indent + " " * cell_cols
        lines.extend(Line([Span(text_indent, theme.BASE_STYLE), Span(s, theme.TEXT_STYLE)])
                     for s in task_segs[1:])
        if task_warn:
            lines.append(Line([Span(text_indent, theme.BASE_STYLE),
                               Span(f"⚠ {task_warn}", theme.WARN_STYLE)]))
        lines.append(Line([]))
        lines.append(section_line("SESSION"))
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
        pairs = cam_pairs(mode, self._rename_map, training_rename_map(self._policy))
        if pairs:
            text = "   ".join(f"{a}→{b}" for a, b in pairs)
            for chunk in wrap_words(text, max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE),
                                   Span(chunk, theme.FAINT_STYLE)]))
        warning = self._cam_warning()
        if warning:
            style = theme.ERR_STYLE if warning.startswith("MAPPING MISMATCH") else theme.WARN_STYLE
            for chunk in wrap_words(f"⚠ {warning}", max(20, w - 4 - self._LABEL_W)):
                lines.append(Line([Span(indent, theme.BASE_STYLE), Span(chunk, style)]))
        if self._backend == "rtc":
            lines.append(number_line(self._exec, "Action horizon", cur == "exec",
                                     "actions executed before the next call", width=w,
                                     label_width=self._LABEL_W))
        else:
            lines.append(number_line(self._steps, "Action steps", cur == "steps",
                                     "actions consumed per policy call", width=w,
                                     label_width=self._LABEL_W))
        lines.append(number_line(self._flow, "Flow steps", cur == "flow",
                                 "flow-matching integration steps", width=w,
                                 label_width=self._LABEL_W))
        lines.append(number_line(self._target, "Corrections", cur == "target",
                                 "session ends after this many saved corrections", width=w,
                                 label_width=self._LABEL_W))
        # Advanced — collapsed by default; the folded row names what it hides.
        folded = "mode · input · duration · display · extra flags"
        lines.append(Line([
            self._gutter("advanced"), self._lab("Advanced", cur == "advanced"),
            Span("▾ " if self._advanced else "▸ ", theme.TEXT_STYLE),
            Span("" if self._advanced else folded, theme.FAINT_STYLE)]))
        if self._advanced:
            lines.append(setting_line(
                "Mode",
                [seg("corrections-only", not self._record_all), Span(" ", theme.BASE_STYLE),
                 seg("record-all", self._record_all)],
                "record-all also saves autonomous frames (tagged)",
                focused=cur == "mode", label_width=self._LABEL_W, width=w))
            lines.append(setting_line(
                "Input",
                [seg("keyboard", not self._pedal), Span(" ", theme.BASE_STYLE),
                 seg("pedal", self._pedal)],
                "pedal = PCsensor footswitch (device via extra flags)",
                focused=cur == "input", label_width=self._LABEL_W, width=w))
            lines.append(number_line(self._dur, "Duration", cur == "duration",
                                     "hard time limit for the whole session", width=w,
                                     label_width=self._LABEL_W))
            lines.append(setting_line(
                "Display", toggle(self._show, focused=cur == "display"),
                "mirror the cameras in a window",
                focused=cur == "display", label_width=self._LABEL_W, width=w))
            lines.append(Line([
                self._gutter("extra"), self._lab("Extra flags", cur == "extra"),
                Span(_clip_end(self._extra_text or "(none)", max(24, w - self._LABEL_W - 4)),
                     theme.TEXT_STYLE if self._extra_text else theme.FAINT_STYLE)]))
        lines.append(Line([]))
        focused = cur == "start"
        if self._host_alive() is False:
            plan_span = Span("⚠ host not reachable — Start host first (menu 1)",
                             theme.WARN_STYLE)
        else:
            parent = Path(record_root(self.ctx.doc)).parent
            plan_span = Span(_clip_end(
                f"collect {self._target.value} corrections · {self._backend}"
                f" · → {parent}/rollout_dagger_<launch time>", max(24, w - 14)),
                theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Start", [plan_span], focused=focused))
        return lines

    def _focused_hint(self) -> str:
        field = self._cur()
        if field == "policy":
            return "⏎ pick a checkpoint · newest first"
        if field == "base":
            return "⏎ pick the base dataset — source of task strings, later merge target"
        if field == "task":
            return ("←→ cycle the base dataset's strings · ⏎ free text"
                    if self._tasks() else "⏎ type a task (base dataset has no strings)")
        if field == "backend":
            return "sync: one forward per tick · rtc: smoother for slow policies · ←→ switch"
        if field == "exec":
            return "prediction window for RTC · ←→ ±1 · ⏎ type a number"
        if field == "steps":
            return "open-loop actions per forward (0 = checkpoint) — long chunks also delay how fast space pauses · ←→ ±1 · ⏎ type"
        if field == "flow":
            return "FM integration steps (0 = checkpoint) · ←→ ±1 · ⏎ type"
        if field == "cameras":
            return "camera slots the policy expects · auto reads the checkpoint · ←→ cycle"
        if field == "target":
            return "saved corrections that end the session · ←→ ±1 · ⏎ type"
        if field == "advanced":
            return "←→/⏎ fold or unfold the advanced settings"
        if field == "mode":
            return "corrections-only = clean fine-tune data · record-all needs intervention-aware training"
        if field == "input":
            return "who sends pause/correction — keyboard (space/tab) or foot pedal"
        if field == "duration":
            return "0 = no time limit · ←→ ±60 · ⏎ type a number"
        if field == "display":
            return "show live Rerun view (off lowers CPU) · ←→/⏎ toggle"
        if field == "extra":
            return "⏎ edit extra lerobot flags passed through verbatim"
        if self._host_alive() is False:
            return "preflight will stop the launch while the host is down"
        return "space pauses · tab corrects · ESC ends — the cheat-sheet repeats before launch · s starts"


__all__ = ["DaggerScreen", "session_root"]
