"""train.py — TrainScreen: configure + launch a local-GPU SmolVLA fine-tune (lerobot-train).

A form (run name, init checkpoint via PolicyPicker, steps/batch/save numbers, AMP toggle).
Start detects a resumable run (Resume/Cancel + new-total-steps prompt, an Invoke async flow)
or validates a fresh init, then STREAMS scripts/train.sh into a RunScreen via runner.stream_run
(with a step-count progress parse). The script owns the argv + the offline HF_* env.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Line, Span

from .. import ROOT
from ..config import cfg_get, collapse_home
from ..datasets import args_have_resume, dataset_episodes
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, train_issues
from ..policies import discover_policies, is_valid_checkpoint
from ..widgets.pickers import CUSTOM, PolicyPicker
from .chrome import (
    clip_middle, draw_form_page, mode_chip_spans, padded_line, plan_row, section_line, seg,
)

TRAIN_SCRIPT = ROOT / "scripts" / "train.sh"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
# lerobot's tracker prints steps through format_big_number → `step:12K` means 12000.
# The old digits-only regex read that as step 12; the suffix is part of the number.
_STEP_RE = re.compile(r"\bstep[:\s]+([\d.]+)([KM]?)")
_LOSS_RE = re.compile(r"\bloss:([\d.eE+-]+)")
_GRDN_RE = re.compile(r"\bgrdn:([\d.eE+-]+)")
_LR_RE = re.compile(r"\blr:([\d.eE+-]+)")
_CKPT_RE = re.compile(r"Checkpoint policy after step (\d+)")
HEADLESS_HOOK = "run_headless"

_SUFFIX = {"": 1, "K": 1_000, "M": 1_000_000}


def parse_step(line: str) -> int | None:
    """Parse the tracker's `step:12K` / `step 12400` token, honoring K/M suffixes."""
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        return int(float(m.group(1)) * _SUFFIX[m.group(2)])
    except (ValueError, KeyError):
        return None


def _fmt_k(n: int) -> str:
    """12400 → `12.4k` (one decimal under 10k only when needed), 20000 → `20k`."""
    if n < 1000:
        return str(n)
    k = n / 1000
    return f"{k:.1f}k".replace(".0k", "k")


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
        self._fresh = True
        # ── stream telemetry (fed by _on_line; rendered by _telemetry_lines) ──
        self._last_step = 0
        self._target_steps = 0
        self._last_loss: float | None = None
        self._loss_hist: list[float] = []
        self._last_grdn = ""
        self._last_lr = ""
        self._last_ckpt = 0
        self._rate_t0: tuple[float, int] | None = None   # (monotonic, step) first sample
        self._rate_sps = 0.0                             # steps/second, live estimate
        self._result = ""                                # post-run summary shown on the form

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
        if cur in _NUM_KEYS and self._num[cur].type_key(key, fresh=self._fresh):
            self._fresh = False
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
        self._target_steps = int(steps) if str(steps).isdigit() else self._num["steps"].value
        self._last_step = self._last_ckpt = 0
        self._last_loss = None
        self._loss_hist = []
        self._last_grdn = self._last_lr = ""
        self._rate_t0 = None
        self._result = ""
        rc = await runner.stream_run(app, argv, title=title, env=None,
                                     on_line=self._on_line,
                                     telemetry=self._telemetry_lines)
        if self._rate_sps > 0:   # remember throughput for the next plan's ETA
            self.ctx.ui_state["train_rate_sps"] = self._rate_sps
        loss = f" · loss {self._last_loss:.3f}" if self._last_loss is not None else ""
        ckpt = f" · last ckpt {_fmt_k(self._last_ckpt)}" if self._last_ckpt else ""
        outcome = "✓ finished" if rc == 0 else f"✗ exited (rc={rc})"
        self._result = f"{outcome} · reached step {_fmt_k(self._last_step)}{loss}{ckpt}"

    def _on_line(self, line: str) -> None:
        step = parse_step(line)
        if step is not None:
            self._last_step = max(self._last_step, step)
            now = time.monotonic()
            if self._rate_t0 is None:
                self._rate_t0 = (now, step)
            elif now > self._rate_t0[0] + 5 and step > self._rate_t0[1]:
                self._rate_sps = (step - self._rate_t0[1]) / (now - self._rate_t0[0])
        m = _LOSS_RE.search(line)
        if m:
            try:
                self._last_loss = float(m.group(1))
                self._loss_hist.append(self._last_loss)
                del self._loss_hist[:-120]
            except ValueError:
                pass
        m = _GRDN_RE.search(line)
        if m:
            self._last_grdn = m.group(1)
        m = _LR_RE.search(line)
        if m:
            self._last_lr = m.group(1)
        m = _CKPT_RE.search(line)
        if m:
            self._last_ckpt = int(m.group(1))

    # ── live telemetry (rendered by the RunScreen between title and log) ────────
    def _eta_text(self, rate: float) -> str:
        if rate <= 0 or self._last_step >= self._target_steps:
            return ""
        s = (self._target_steps - self._last_step) / rate
        h, rem = divmod(int(s), 3600)
        return f"eta {h}:{rem // 60:02d}h" if h else f"eta {rem // 60}m"

    def _telemetry_lines(self, width: int) -> list[Line]:
        """Step meter + checkpoint facts, then loss + log-scaled sparkline. Empty
        until the first tracker line so the log opens full-height."""
        if self._last_step <= 0:
            return []
        w = int(width)
        target = max(self._target_steps, self._last_step, 1)
        fill, trough = theme.meter_segments(self._last_step / target, 22)
        ckpt = (f"last ckpt {_fmt_k(self._last_ckpt)}" if self._last_ckpt
                else "no checkpoint yet")
        eta = self._eta_text(self._rate_sps)
        right = f"{eta} · {ckpt}" if eta else ckpt
        lines = [padded_line(
            [Span("  steps     ", theme.MUTED_STYLE),
             Span(fill, theme.TITLE_STYLE), Span(trough, theme.BORDER_STYLE),
             Span(f"  {_fmt_k(self._last_step)}/{_fmt_k(target)}", theme.TEXT_STYLE)],
            [Span(right, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w)]
        if self._last_loss is not None:
            facts = " · ".join(x for x in (
                f"grdn {self._last_grdn}" if self._last_grdn else "",
                f"lr {self._last_lr}" if self._last_lr else "",
                f"{self._rate_sps:.1f} it/s" if self._rate_sps > 0 else "") if x)
            # log10-scale the sparkline: loss falls exponentially, so a linear scale
            # flatlines after the first stretch and hides late-training drift.
            logs = [math.log10(max(v, 1e-6)) for v in self._loss_hist]
            spark = theme.sparkline(logs, 16)
            lines.append(padded_line(
                [Span("  loss      ", theme.MUTED_STYLE),
                 Span(f"{self._last_loss:.3f}", theme.TEXT_STYLE),
                 Span(f"   {facts}", theme.FAINT_STYLE)],
                [Span("loss ", theme.FAINT_STYLE), Span(spark, theme.TITLE_STYLE),
                 Span("  ", theme.BASE_STYLE)], w))
        return lines

    # ── view ────────────────────────────────────────────────────────────────────
    _LABEL_W = 13

    def _header_right(self) -> list[Span]:
        """Train's live fact is the GPU, not the host — same slot, page-appropriate."""
        spans: list[Span] = [Span("GPU ", theme.MUTED_STYLE)]
        if self.ctx.gpu_name:
            spans += [Span(f"{theme.status_dot()} ", theme.OK_STYLE),
                      Span(self.ctx.gpu_name, theme.TEXT_STYLE)]
        else:
            spans.append(Span("none — CPU training is impractically slow", theme.WARN_STYLE))
        spans.append(Span("   ", theme.BASE_STYLE))
        spans.extend(mode_chip_spans())
        return spans

    def draw(self, frame: Any, area: Any) -> None:
        draw_form_page(frame, area, self.ctx, "train", self._body_lines(area.width),
                       header_right=self._header_right(), msg=self._msg,
                       hint=self._focused_hint())

    def _lab(self, key: str, focused: bool) -> Span:
        return Span(f"{_LABELS[key]:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *keys: str) -> Span:
        on = self._cur() in keys
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _num_text(self, key: str) -> tuple[str, Any]:
        nf = self._num[key]
        if key == self._cur():
            return f"{nf.editor.value}█", theme.HIGHLIGHT_TEXT_STYLE
        return nf.display(), theme.TEXT_STYLE

    def _init_display(self) -> str:
        try:
            return str(Path(self._init).relative_to(self._root))
        except ValueError:
            return collapse_home(self._init)

    def _run_status(self) -> str:
        """The Run-name row's right side: where the run lands + resumability."""
        if _has_resume_checkpoint(self._root, self._run):
            return f"resumable — trained to {_fmt_k(_prev_steps(self._root, self._run))} steps"
        return f"{collapse_home(str(self._root))}/{self._run} — new run"

    def _plan(self) -> str:
        """The Start plan: shape + dataset + an ETA once a throughput sample exists."""
        dsroot = train_dataset_root(self.ctx.doc)
        parts = [f"{_fmt_k(self._num['steps'].value)} steps",
                 f"batch {self._num['batch'].value}"]
        if self._amp:
            parts.append("amp")
        parts.append(f"dataset {Path(dsroot).name} ({dataset_episodes(dsroot)} ep)")
        rate = float(self.ctx.ui_state.get("train_rate_sps") or 0)
        if rate > 0:
            hours = self._num["steps"].value / rate / 3600
            parts.append(f"~{hours:.1f} h on {self.ctx.gpu_name or 'GPU'}")
        return " · ".join(parts)

    def _body_lines(self, width: int = 100) -> list[Line]:
        cur = self._cur()
        w = int(width)
        lines: list[Line] = [section_line("RUN")]
        lines.append(padded_line(
            [self._gutter("name"), self._lab("name", cur == "name"),
             Span(self._run, theme.TEXT_STYLE)],
            [Span(self._run_status(), theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w))
        lines.append(Line([
            self._gutter("init"), self._lab("init", cur == "init"),
            Span(clip_middle(self._init_display(), max(24, w - self._LABEL_W - 4)),
                 theme.TEXT_STYLE),
        ]))
        lines.append(Line([]))
        lines.append(section_line("OPTIMIZATION"))
        sv, ss = self._num_text("steps")
        bv, bs = self._num_text("batch")
        lines.append(Line([
            self._gutter("steps", "batch"),
            self._lab("steps", cur == "steps"), Span(f"{sv:<10}", ss),
            Span("   ", theme.BASE_STYLE),
            self._lab("batch", cur == "batch"), Span(bv, bs),
        ]))
        fv, fs = self._num_text("save")
        lines.append(Line([
            self._gutter("save", "amp"),
            self._lab("save", cur == "save"), Span(f"{fv:<10}", fs),
            Span("   ", theme.BASE_STYLE),
            self._lab("amp", cur == "amp"), seg("on" if self._amp else "off", self._amp),
        ]))
        lines.append(Line([]))
        focused = cur == "start"
        lines.append(plan_row("Start", self._plan(), focused=focused))
        if self._result:
            style = theme.OK_STYLE if self._result.startswith("✓") else theme.WARN_STYLE
            lines.append(Line([]))
            lines.append(Line([Span(f"  {self._result}", style),
                               Span("   Run policy (menu) evaluates it", theme.FAINT_STYLE)]))
        return lines

    def _focused_hint(self) -> str:
        field = self._cur()
        if field == "name":
            return f"training run folder name · ⏎ edit · output {collapse_home(str(self._root))}/<name>"
        if field == "init":
            return "⏎ pick a checkpoint · a local checkpoint runs fully offline"
        if field == "steps":
            return "total optimization steps · ←→ ±1000 · ⏎ type a number"
        if field == "batch":
            return "lower first if GPU memory runs out · ←→ ±1 · ⏎ type a number"
        if field == "save":
            return "steps between checkpoints · ←→ ±500 · ⏎ type a number"
        if field == "amp":
            return "mixed precision — faster, less VRAM · ←→/⏎ toggle"
        return "validates the run, then streams training (q backgrounds it)"


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
