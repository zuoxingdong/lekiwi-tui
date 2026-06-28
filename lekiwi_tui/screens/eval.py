"""eval.py — EvalScreen: configure + launch a policy rollout on the robot (lerobot-rollout).

A form with a DYNAMIC field list (the exec-horizon row appears only for the rtc backend), a
PolicyPicker, an editable Task, and Backend/Duration/Display. Start validates the checkpoint
then SUSPENDS into the rollout (pause=True) — it owns the real TTY for its keyboard controls.
Fronts scripts/eval.sh (the sole argv source). NO compile toggle (intentional). Ports
resolve_eval_policy / _device_note / _checkpoint_error verbatim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..config import cfg_get
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key, is_char
from ..framework.modals import PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, eval_issues
from ..policies import discover_policies, is_valid_checkpoint, resolve_policy
from ..widgets.pickers import CUSTOM, PolicyPicker
from .chrome import clip_end as _clip_end
from .chrome import clip_middle as _clip_middle
from .chrome import keycap_hint_line
from .chrome import runtime_chips

# The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

EVAL_SCRIPT = ROOT / "scripts" / "eval.sh"
STATE_KEY = "eval"
FORM_LABEL_WIDTH = 13
SUMMARY_LABEL_WIDTH = 10


def _tilde(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


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
            note = f"Configured policy path is missing ({_tilde(policy)}); using newest checkpoint: {rel}"
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
        self._err = ""
        self._fpos = 0
        self._fresh = True

    def _remember(self) -> None:
        _state(self.ctx).update({
            "policy": self._policy,
            "task": self._task_text,
            "backend": self._backend,
            "exec_horizon": self._exec.value,
            "duration": self._dur.value,
            "display": self._show,
        })

    def _fields(self) -> list[str]:
        f = ["policy", "task", "backend"]
        if self._backend == "rtc":
            f.append("exec")
        return f + ["duration", "display", "start"]

    def _cur(self) -> str:
        fs = self._fields()
        return fs[min(self._fpos, len(fs) - 1)]

    def _num(self, key: str) -> "NumberField | None":
        return {"exec": self._exec, "duration": self._dur}.get(key)

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
            elif cur in ("exec", "duration"):
                self._num(cur).step_by(delta); self._num(cur).sync_editor(); self._fresh = True
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
            if cur == "backend":
                self._backend = "sync" if self._backend == "rtc" else "rtc"
                self._remember()
                self._fpos = min(self._fpos, len(self._fields()) - 1); return Nothing
            if cur == "display":
                self._show = not self._show; self._remember(); return Nothing
            if cur in ("exec", "duration"):
                self._commit_num(cur); return Nothing
            if cur == "start":
                return Invoke(self._start)
            return Nothing
        if cur in ("exec", "duration") and (is_char(key) or name == "Backspace"):
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

    async def _pick_policy(self) -> None:
        chosen = await self.app.run_modal(PolicyPicker(
            self._root, default_abs=self._default_abs, title="Pick a checkpoint"))
        if chosen is None:
            return
        if chosen == CUSTOM:
            ans = await self.app.run_modal(PromptModalState(
                "Policy path or model repo id", value=_tilde(self._policy), hint="⏎ apply · esc cancel"))
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
        self._remember()

    async def _start(self) -> None:
        app = self.app
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
        argv = [
            "bash", str(EVAL_SCRIPT),
            "--policy", self._policy, "--task", self._task_text, "--backend", self._backend,
            "--exec-horizon", str(self._exec.value), "--duration", str(self._dur.value),
            "--display", "on" if self._show else "off", "--gpu", self.ctx.gpu_name, *self._extra,
        ]
        await app.suspend(argv, pause=True)

    # ── view ────────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Clear(), area)
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.length(1),
             Constraint.fill(1), Constraint.length(1), Constraint.length(1)]).split(area))
        frame.render_widget(Paragraph(Text([Line([
            Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE), Span("  run policy", theme.SUBTITLE_STYLE)])]
        )).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        body_rows = (Layout().direction(Direction.Vertical).constraints([
            Constraint.length(len(self._fields())),
            Constraint.length(1),
            Constraint.length(9 if self._fallback_note else 8),
            Constraint.fill(1),
        ]).split(rows[3]))
        frame.render_widget(self._body(body_rows[0].width), body_rows[0])
        frame.render_widget(self._summary(body_rows[2].width), body_rows[2])
        if self._err:
            frame.render_widget(Paragraph(Text([Line([Span(f"✗ {self._err}", theme.ERR_STYLE)])])
                                          ).style(theme.BASE_STYLE), rows[4])
        frame.render_widget(self._hint(), rows[5])

    def _policy_display_path(self) -> str:
        if not self._policy:
            return ""
        try:
            return str(Path(self._policy).relative_to(self._root))
        except ValueError:
            return _tilde(self._policy)

    def _value(self, field: str, *, width: int = 60) -> str:
        if field == "policy":
            disp = self._policy_display_path() or _tilde(self._policy) or "(none found)"
            suffix = "  default" if self._policy and self._policy == self._default_abs else ""
            disp = _clip_middle(disp, max(8, width - len(suffix))) + suffix
            return disp
        if field == "task":
            t = (self._task_text or "(saved default)").splitlines()[0]
            return _clip_end(t, width)
        if field == "backend":
            return theme.choice(self._backend)
        if field in ("exec", "duration"):
            nf = self._num(field)
            disp = (nf.editor.value + "█") if field == self._cur() else nf.display()
            return _clip_end(disp, width)
        if field == "display":
            return theme.choice("on") if self._show else theme.choice("off")
        return ""

    def _summary_value(self, field: str) -> str:
        if field == "policy":
            disp = self._policy_display_path() or _tilde(self._policy) or "(none found)"
            if self._policy and self._policy == self._default_abs:
                disp += "  default"
            return disp
        if field == "task":
            return (self._task_text or "(saved default)").replace("\n", " ")
        if field == "backend":
            if self._backend == "rtc":
                return f"rtc · action horizon {self._exec.display()}"
            return "sync · one policy forward per control tick"
        if field == "duration":
            return self._dur.display()
        if field == "display":
            return "on · live Rerun view" if self._show else "off · lower CPU"
        return ""

    def _hint_for(self, field: str) -> str:
        """The trailing muted per-field hint (port of the Textual _refresh hints). Backend
        and the numeric rows are DYNAMIC (rtc-vs-sync / NumberField.hint()), so this is a
        method, not a static dict like record.py's _HINTS."""
        if field == "backend":
            return ("RTC: smoother control for slower policies" if self._backend == "rtc"
                    else "Sync: one policy forward per control tick")
        if field == "exec":
            return self._exec.hint() + " · prediction window for RTC"
        if field == "duration":
            return self._dur.hint() + " · 0 = saved default"
        if field == "display":
            return "show live Rerun view (off lowers CPU)"
        if field == "policy":
            return "enter to pick a checkpoint or custom path"
        if field == "task":
            return "enter to edit the task instruction"
        return ""

    def _body(self, width: int) -> Paragraph:
        fs = self._fields()
        cur = self._cur()
        labels = {"policy": "Policy", "task": "Task", "backend": "Backend", "exec": "Action horizon",
                  "duration": "Duration", "display": "Display", "start": "Start"}
        lines = []
        for field in fs:
            focused = field == cur
            if field == "start":
                style = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.TEXT_STYLE
                hint = "validate preflight and launch rollout"
                if focused:
                    lines.append(Line([
                        Span(theme.selector(True), theme.HIGHLIGHT_LABEL_STYLE),
                        Span(f"{theme.play_mark()} Run policy", style),
                        Span("   ", theme.HIGHLIGHT_STYLE),
                        Span(hint, theme.HIGHLIGHT_TEXT_STYLE),
                    ], theme.HIGHLIGHT_STYLE))
                else:
                    lines.append(Line([
                        Span(theme.selector(False), theme.BASE_STYLE),
                        Span(f"{theme.play_mark()} Run policy", style),
                        Span("   ", theme.BASE_STYLE),
                        Span(hint, theme.MUTED_STYLE),
                    ]))
                continue
            lstyle = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.MUTED_STYLE
            vstyle = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.TEXT_STYLE
            selector_style = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.BASE_STYLE
            hint_style = theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE
            hint = self._hint_for(field)
            hint_text = f"   {hint}" if hint else ""
            prefix_width = 2 + FORM_LABEL_WIDTH + 2
            value_width = max(8, width - prefix_width - len(hint_text))
            if value_width < 18 and hint_text:
                hint_text = ""
                value_width = max(8, width - prefix_width)
            lines.append(Line([
                Span(theme.selector(focused), selector_style),
                Span(f"{labels[field]:<{FORM_LABEL_WIDTH}}", lstyle),
                Span(f"  {self._value(field, width=value_width)}", vstyle),
                Span(hint_text, hint_style),
            ], theme.HIGHLIGHT_STYLE if focused else None))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _summary(self, width: int) -> Paragraph:
        value_width = max(12, width - SUMMARY_LABEL_WIDTH - 2)
        rule_width = max(4, min(58, width - len("RUN SUMMARY ")))
        lines: list[Line] = [
            Line([
                Span("RUN SUMMARY ", theme.SECTION_STYLE),
                Span(theme.rule(rule_width), theme.BORDER_STYLE),
            ])
        ]
        if self._fallback_note:
            lines.append(Line([
                Span(f"{'Note':<{SUMMARY_LABEL_WIDTH}}", theme.WARN_STYLE),
                Span(_clip_end(self._fallback_note, value_width), theme.MUTED_STYLE),
            ]))
        rows = [
            ("Policy", self._summary_value("policy"), theme.STATUS_VALUE_STYLE),
            ("Task", self._summary_value("task"), theme.TEXT_STYLE),
            ("Backend", self._summary_value("backend"), theme.TEXT_STYLE),
            ("Duration", self._summary_value("duration"), theme.TEXT_STYLE),
            ("Display", self._summary_value("display"), theme.TEXT_STYLE),
            ("Device", _device_note(self._policy, self.ctx.gpu_name), theme.TEXT_STYLE),
        ]
        for label, value, style in rows:
            lines.append(Line([
                Span(f"{label:<{SUMMARY_LABEL_WIDTH}}", theme.MUTED_STYLE),
                Span(_clip_end(value, value_width), style),
            ]))
        lines.append(Line([
            Span(f"{'Host':<{SUMMARY_LABEL_WIDTH}}", theme.WARN_STYLE),
            Span(
                _clip_end("required; start Pi host in another terminal", value_width),
                theme.STATUS_VALUE_STYLE,
            ),
        ]))
        mode_style = theme.WARN_STYLE if runner.DRY_RUN else theme.OK_STYLE
        mode = "PREVIEW · wrapper prints argv" if runner.DRY_RUN else "REAL · controls robot"
        lines.append(Line([
            Span(f"{'Mode':<{SUMMARY_LABEL_WIDTH}}", theme.MUTED_STYLE),
            Span(_clip_end(mode, value_width), mode_style),
        ]))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _hint(self) -> Paragraph:
        return keycap_hint_line([
            ("↑↓/jk", "move"),
            ("←→/hl", "change"),
            ("⏎", "edit/run"),
            ("s", "run"),
            ("q", "back"),
        ])


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
    argv = [
        "bash", str(EVAL_SCRIPT),
        "--policy", policy, "--backend", ctx.cfg["INFERENCE"],
        "--exec-horizon", str(ctx.cfg["EXECUTION_HORIZON"]),
        "--duration", "0", "--display", "on" if show else "off",
        "--gpu", ctx.gpu_name, *extra,
    ]
    return runner.headless_run(argv)


if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App


__all__ = ["EvalScreen", "resolve_eval_policy", "run_headless", "HEADLESS_HOOK"]
