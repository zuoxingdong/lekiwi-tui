"""train.py — TrainScreen: configure + launch a local-GPU SmolVLA fine-tune (lerobot-train).

A form (run name, init checkpoint via PolicyPicker, steps/batch/save numbers, AMP toggle).
Start detects a resumable run (Resume/Cancel + new-total-steps prompt, an Invoke async flow)
or validates a fresh init, then STREAMS scripts/train.sh into a RunScreen via runner.stream_run
(with a step-count progress parse). The script owns the argv + the offline HF_* env.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..config import cfg_get
from ..datasets import args_have_resume, dataset_episodes
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key, is_char
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, train_issues
from ..policies import discover_policies, is_valid_checkpoint
from ..widgets.pickers import CUSTOM, PolicyPicker
from .chrome import keycap_hint_line, option_line, runtime_chips

TRAIN_SCRIPT = ROOT / "scripts" / "train.sh"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_STEP_RE = re.compile(r"\bstep[:\s]+(\d+)")
RULE = "─" * 54
HEADLESS_HOOK = "run_headless"


def train_yaml_int(key: str, fallback: int, doc: dict | None = None) -> int:
    v = cfg_get(f"train.{key}", doc=doc)
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return fallback


def train_dataset_root(doc: dict | None = None) -> str:
    r = cfg_get("train.dataset.root", doc=doc)
    return str(r) if r else "../datasets/lekiwi_dataset"


def _tilde(path: str) -> str:
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def _resume_config_path(outdir: str) -> str:
    return str(Path(outdir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json")


def _fresh_init_error(init: str) -> str | None:
    p = Path(init)
    if p.is_dir() and not is_valid_checkpoint(p):
        return f"'{init}' is not a valid checkpoint; expected config.json and model.safetensors."
    return None


def _has_resume_checkpoint(policy_root: Path, run: str) -> bool:
    return Path(_resume_config_path(str(policy_root / run))).is_file()


def _prev_steps(policy_root: Path, run: str) -> int:
    try:
        return int(json.loads(Path(_resume_config_path(str(policy_root / run))).read_text()).get("steps", 0))
    except Exception:
        return 0


_NUM_KEYS = {"steps", "batch", "save"}
_FIELDS = ["name", "init", "steps", "batch", "save", "amp", "start"]
_LABELS = {"name": "Run name", "init": "Init from", "steps": "Steps", "batch": "Batch size",
           "save": "Save every", "amp": "Mixed prec.", "start": "Start"}


class TrainScreen(ScreenState):
    """Train configuration form; Start resume-detects then streams lerobot-train."""

    title = "train"

    def __init__(self, app: "App", ctx: "Context", *, run: str | None = None,
                 extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._root = Path(ctx.cfg["POLICY_ROOT"])
        self._num = {
            "steps": NumberField("Steps", train_yaml_int("steps", 20000, ctx.doc), minimum=1000, step=1000),
            "batch": NumberField("Batch size", train_yaml_int("batch_size", 8, ctx.doc), minimum=1, step=1),
            "save": NumberField("Save freq", train_yaml_int("save_freq", 5000, ctx.doc), minimum=500, step=500),
        }
        self._run = run if run is not None else "local_" + datetime.now().strftime("%y%m%d")
        found = discover_policies(self._root)
        self._init = str(found[0]) if found else "lerobot/smolvla_base"
        self._amp = bool(cfg_get("train.policy.use_amp", doc=ctx.doc))
        self._fpos = 0
        self._msg = ""
        self._last_step = 0
        self._fresh = True

    def _cur(self) -> str:
        return _FIELDS[self._fpos]

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k", BACKTAB):
            self._move(-1); return Nothing
        if name in (DOWN, "j", TAB):
            self._move(1); return Nothing
        cur = self._cur()
        if name in (LEFT, "h", RIGHT, "l"):
            delta = -1 if name in (LEFT, "h") else 1
            if cur in _NUM_KEYS:
                self._num[cur].step_by(delta); self._num[cur].sync_editor(); self._fresh = True
            elif cur == "amp":
                self._amp = not self._amp
            self._msg = ""; return Nothing
        if name == ENTER:
            if cur == "name":
                return Invoke(self._edit_name)
            if cur == "init":
                return Invoke(self._pick_init)
            if cur == "amp":
                self._amp = not self._amp; return Nothing
            if cur in _NUM_KEYS:
                self._commit_num(cur); return Nothing
            if cur == "start":
                return Invoke(self._start)
            return Nothing
        if cur in _NUM_KEYS and (is_char(key) or name == "Backspace"):
            ed = self._num[cur].editor
            if self._fresh and is_char(key):
                ed.clear()
            self._fresh = False
            if ed.handle_key(key):
                if ed.value.strip().isdigit():
                    self._num[cur].set_text(ed.value.strip())
                self._msg = ""
                return Nothing
        return Nothing

    def _move(self, delta: int) -> None:
        self._fpos = (self._fpos + delta) % len(_FIELDS)
        self._msg = ""
        if self._cur() in _NUM_KEYS:
            self._num[self._cur()].sync_editor()
        self._fresh = True

    def _commit_num(self, cur: str) -> None:
        nf = self._num[cur]
        if not nf.set_text(nf.editor.value):
            self._msg = f"✗ {nf.error}"
        else:
            self._msg = ""; nf.sync_editor(); self._fresh = True

    # ── async flows ─────────────────────────────────────────────────────────────
    async def _edit_name(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Run name", value=self._run, hint="⏎ apply · esc cancel"))
        if ans is None or not ans:
            return
        if _NAME_RE.match(ans):
            self._run = ans; self._msg = ""
        else:
            self._msg = "✗ name must be letters/digits/._- only"

    async def _pick_init(self) -> None:
        chosen = await self.app.run_modal(PolicyPicker(
            self._root, default_abs=self._init, title="Choose initial policy"))
        if chosen is None:
            return
        if chosen == CUSTOM:
            ans = await self.app.run_modal(PromptModalState(
                "Initial checkpoint or model repo id", value="lerobot/smolvla_base",
                hint="⏎ apply · lerobot/smolvla_base for a fresh base model · esc cancel"))
            if ans is None or not ans:
                return
            self._init = os.path.expanduser(ans)
        else:
            self._init = chosen

    async def _start(self) -> None:
        app = self.app
        if _has_resume_checkpoint(self._root, self._run):
            prev = _prev_steps(self._root, self._run)
            choice = await app.run_modal(ConfirmModalState(
                f'Run "{self._run}" has checkpoints (trained to {prev} steps).',
                ["Resume", "Cancel"]))
            if choice != "Resume":
                return
            ans = await app.run_modal(PromptModalState(
                "New total training steps", str(prev), hint=f"⏎ keep {prev} · esc cancel"))
            if ans is None:
                return
            await self._launch("resume", ans if ans.isdigit() else "")
            return
        run_dir = self._root / self._run
        if run_dir.is_dir() and any(run_dir.iterdir()):
            self._msg = f"✗ '{self._run}' already exists without a checkpoint; choose another run name"
            return
        err = _fresh_init_error(self._init)
        if err is not None:
            self._msg = f"✗ {err}"
            return
        await self._launch("fresh", str(self._num["steps"].value))

    async def _launch(self, mode: str, steps: str) -> None:
        app = self.app
        if not await confirm_preflight(
            app,
            "Train preflight",
            train_issues(
                self.ctx,
                dataset_root=train_dataset_root(self.ctx.doc),
                policy_root=str(self._root),
            ),
        ):
            return
        argv = [
            "bash", str(TRAIN_SCRIPT),
            "--mode", mode, "--run", self._run, "--init", self._init,
            "--steps", str(steps), "--batch", str(self._num["batch"].value),
            "--save", str(self._num["save"].value), "--amp", "on" if self._amp else "off",
            "--gpu", self.ctx.gpu_name, "--policy-root", str(self._root), *self._extra,
        ]
        title = (f"Train · {self._run} · steps={steps or self._num['steps'].value} · "
                 f"batch={self._num['batch'].value} · {mode}")
        await runner.stream_run(app, argv, title=title, env=None, on_line=self._on_line)

    def _on_line(self, line: str) -> None:
        m = _STEP_RE.search(line)
        if m:
            self._last_step = int(m.group(1))

    # ── view ────────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.length(1),
             Constraint.length(2), Constraint.fill(1), Constraint.length(1),
             Constraint.length(1)]).split(area))
        frame.render_widget(Paragraph(Text([Line([
            Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE), Span("  train", theme.SUBTITLE_STYLE)])]
        )).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        dsroot = train_dataset_root(self.ctx.doc)
        gpu = (f"{theme.status_dot()} {self.ctx.gpu_name} (4 GB, lower Batch if GPU memory runs out)" if self.ctx.gpu_name
               else "none; CPU training is impractically slow")
        frame.render_widget(Paragraph(Text([
            Line([Span(f"dataset {dsroot} ({dataset_episodes(dsroot)} episodes) · GPU {gpu}", theme.MUTED_STYLE)]),
            Line([Span("training recipe: vision encoder frozen, action expert only · Weights & Biases and Hub upload off",
                       theme.MUTED_STYLE)]),
        ])).style(theme.BASE_STYLE), rows[3])
        frame.render_widget(self._body(rows[4].width), rows[4])
        if self._msg:
            frame.render_widget(Paragraph(Text([Line([Span(self._msg, theme.ERR_STYLE)])])
                                          ).style(theme.BASE_STYLE), rows[5])
        frame.render_widget(self._hint(), rows[6])

    def _value(self, field: str) -> str:
        if field == "name":
            return self._run
        if field == "init":
            try:
                return str(Path(self._init).relative_to(self._root))
            except ValueError:
                return _tilde(self._init)
        if field in _NUM_KEYS:
            nf = self._num[field]
            return (nf.editor.value + "█") if field == self._cur() else nf.display()
        if field == "amp":
            return theme.choice("on") if self._amp else theme.choice("off")
        return ""

    def _body(self, width: int = 120) -> Paragraph:
        lines = []
        for i, field in enumerate(_FIELDS):
            focused = i == self._fpos
            if field == "start":
                lines.append(option_line(
                    f"{theme.play_mark()} Start training",
                    "validate dataset and stream training",
                    focused=focused,
                    label_width=21,
                    width=width,
                    label_unfocused_style=theme.TEXT_STYLE,
                ))
                continue
            lines.append(option_line(
                _LABELS[field],
                self._value(field),
                self._hint_for(field),
                focused=focused,
                label_width=12,
                width=width,
                clip="middle" if field == "init" else "end",
            ))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _hint_for(self, field: str) -> str:
        """Per-field inline hint (the original train.py _set_row hints)."""
        if field == "name":
            return f"output folder: {_tilde(str(self._root))}/<name>"
        if field == "init":
            return "local checkpoint runs fully offline"
        if field == "steps":
            return self._num["steps"].hint()
        if field == "batch":
            return self._num["batch"].hint() + " · lower first if GPU memory runs out"
        if field == "save":
            return self._num["save"].hint() + " · steps between checkpoints"
        if field == "amp":
            return "mixed precision; faster and uses less VRAM"
        return ""

    def _hint(self) -> Paragraph:
        return keycap_hint_line([
            ("↑↓/jk", "move"),
            ("←→/hl", "adjust"),
            ("⏎", "edit/pick/start"),
            ("q", "back"),
        ])


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY train (mirrors the original train.py run_headless): an explicit --resume in
    extra is forwarded verbatim to lerobot-train with the offline env; otherwise a fresh
    local_<date> run from the yaml defaults + the newest checkpoint (or smolvla_base) is
    fronted through train.sh."""
    import os
    from datetime import datetime

    if args_have_resume(extra):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        return runner.headless_run(["lerobot-train", *extra], env=env)
    root = Path(ctx.cfg["POLICY_ROOT"])
    run = "local_" + datetime.now().strftime("%y%m%d")
    if (root / run).is_dir():
        print(f"✗ training run folder '{root / run}' already exists. Resume it or choose another name.")
        return 1
    found = discover_policies(root)
    init = str(found[0]) if found else "lerobot/smolvla_base"
    err = _fresh_init_error(init)
    if err is not None:
        print(f"✗ {err}")
        return 1
    amp = bool(cfg_get("train.policy.use_amp", doc=ctx.doc))
    argv = [
        "bash", str(TRAIN_SCRIPT), "--mode", "fresh", "--run", run, "--init", init,
        "--steps", str(train_yaml_int("steps", 20000, ctx.doc)),
        "--batch", str(train_yaml_int("batch_size", 8, ctx.doc)),
        "--save", str(train_yaml_int("save_freq", 5000, ctx.doc)),
        "--amp", "on" if amp else "off", "--gpu", ctx.gpu_name,
        "--policy-root", str(root), *extra,
    ]
    return runner.headless_run(argv)


if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App


__all__ = ["TrainScreen", "train_yaml_int", "train_dataset_root", "run_headless", "HEADLESS_HOOK"]
