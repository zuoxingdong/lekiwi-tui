"""record.py — RecordScreen: configure + launch lerobot-record (port of the Textual one).

A form (dataset name, task, episodes, episode/reset time, fps, image writers, display +
streaming + resume toggles). Start runs the existing-dataset safety gate (Resume / Delete
with a typed-'delete' confirm + unsafe-path guard / Cancel) as an Invoke async flow using
``app.run_modal``, persists the form values back to lekiwi.yaml (lossless ruamel), then
SUSPENDS into the live lerobot-record loop — the child owns the real TTY, which is the
guaranteed path for the base wasd keys (the kbd shim reads kitty-stdin/evdev there).
Fronts scripts/record.sh (the sole argv source). The former in-page HUD view (stream +
episode meters + G/B triage) was removed by choice: terminal view is the one path.
Post-session triage lives in the dataset editor (anomaly flags + Space marking).
"""
from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any

from pyratatui import Line, Paragraph, Span, Text

from .. import ROOT
from ..config import cfg_get, collapse_home, load_yaml
from ..datasets import (
    args_have_resume, dataset_defaults, dataset_episodes, dataset_present,
    dataset_stats_parts, persist_record_defaults, record_root, resolve_repo_root,
    safe_delete_target, valid_dataset_name, workspace_path,
)
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField, wrap_words
from ..preflight import confirm_preflight, record_issues
from .chrome import (
    clip_middle, draw_form_page, number_line, padded_line, plan_row, section_line, seg,
    setting_line, toggle,
)

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# The run_headless hook the dispatcher/__main__ calls for a no-TTY
# `python -m lekiwi_tui record` (parity with the Textual app's HEADLESS_HOOK).
HEADLESS_HOOK = "run_headless"

RECORD_SCRIPT = ROOT / "scripts" / "record.sh"


# Per-field hints, shown ONE at a time in the footer hint slot for the FOCUSED field
# only (the old design repeated them inline on every row — most of the visible
# characters were instructions for things the user was not doing).
_HINTS = {
    "name": "⏎ edit dataset name",  # the resolved path is appended dynamically
    "task": "⏎ edit language instruction",
    "episodes": "episodes to record · ←→ ±1 · ⏎ type a number",
    "ep_time": "recording time per episode · ←→ ±5 · ⏎ type a number",
    "reset_time": "reset pause between episodes · ←→ ±1 · ⏎ type a number",
    "fps": "recording frame rate · ←→ ±5 · ⏎ type a number",
    "img_threads": "image-writer threads per camera · ←→ ±1 · ⏎ type a number",
    "display": "show live Rerun view (off lowers CPU) · ←→/⏎ toggle",
    "streaming": "encode video while recording for faster saves · ←→/⏎ toggle",
    "resume": "append to an existing dataset · ←→/⏎ toggle",
    "start": "validates the dataset, then hands the terminal to lerobot-record",
}

# The existing-dataset gate's choice labels (fuller copy from the Textual original). The
# port's ConfirmModalState yields the chosen LABEL verbatim, so these are bound once and
# compared by equality at the call site (CANCEL or Esc/None both mean "stay put").
_RESUME = "Resume   append more episodes"
_DELETE = "Delete   remove it and start fresh"
_CANCEL = "Cancel   keep the dataset unchanged"


# Field spec: (key, label, kind) — kind in {text, task, num, toggle, start}.
# Navigation order == VISUAL order (grouped DATASET → SESSION → CAPTURE → Start).
_NUM_KEYS = {"episodes", "ep_time", "reset_time", "fps", "img_threads"}
_TOGGLE_KEYS = {"display", "streaming", "resume"}
_FIELDS = ["name", "task", "resume", "episodes", "ep_time", "fps", "reset_time",
           "streaming", "display", "img_threads", "start"]
_LABELS = {"name": "Name", "task": "Task", "episodes": "Episodes", "ep_time": "Episode time",
           "reset_time": "Reset time", "fps": "FPS", "img_threads": "Writers",
           "display": "Display", "streaming": "Streaming", "resume": "Resume",
           "start": "Start"}

# One setting per row, in _FIELDS navigation order. The note is always visible: it is what
# keeps a value's meaning on screen while the field is focused and its well shows a caret.
_SESSION_ROWS = [
    ("episodes", "Episodes", "takes to record before it stops"),
    ("ep_time", "Episode time", "how long each take runs"),
    ("fps", "FPS", "must match the robot's control-loop rate"),
    ("reset_time", "Reset time", "pause between takes to reposition"),
]
_CAPTURE_TOGGLES = [
    ("streaming", "Streaming", "send camera frames to the laptop"),
    ("display", "Display", "mirror them in a window"),
]


class RecordScreen(ScreenState):
    """Record-a-dataset form; Start gates on the existing-dataset safety modal then suspends."""

    title = "record"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        d = dataset_defaults(ctx.doc)
        self._ds_name = d["name"]; self._ds_ns = d["ns"]; self._ds_parent = d["parent"]
        self._ds_task = d["task"]
        self._num = {
            "episodes": NumberField("Episodes", d["episodes"], minimum=1, step=1),
            "ep_time": NumberField("Episode time", d["ep_time"], minimum=1, step=5, unit="s"),
            "reset_time": NumberField("Reset time", d["reset_time"], minimum=0, step=1, unit="s"),
            "fps": NumberField("FPS", d["fps"], minimum=1, step=5),
            "img_threads": NumberField("Image writers", d["img_threads"], minimum=1, step=1),
        }
        self._toggle = {"display": d["display"], "streaming": d["streaming"], "resume": d["resume"]}
        self._area = None
        self._fpos = 0
        self._msg = ""
        self._fresh = True

    def _cur(self) -> str:
        return _FIELDS[self._fpos]

    def _resolve(self) -> tuple[str, str]:
        return resolve_repo_root(self._ds_name, self._ds_ns, self._ds_parent)

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
            elif cur in _TOGGLE_KEYS:
                self._toggle[cur] = not self._toggle[cur]
            self._msg = ""; return Nothing
        if name == ENTER:
            if cur == "name":
                return Invoke(self._edit_name)
            if cur == "task":
                return Invoke(self._edit_task)
            if cur in _TOGGLE_KEYS:
                self._toggle[cur] = not self._toggle[cur]; return Nothing
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
        cur = self._cur()
        if cur in _NUM_KEYS:
            self._num[cur].sync_editor()
        self._fresh = True

    def _commit_num(self, cur: str) -> None:
        nf = self._num[cur]
        if not nf.set_text(nf.editor.value):
            self._msg = f"✗ {nf.error}"
        else:
            self._msg = ""; nf.sync_editor(); self._fresh = True

    # ── async flows (Invoke) ───────────────────────────────────────────────────
    async def _edit_name(self) -> None:
        ans = await self.app.run_modal(PromptModalState(
            "Dataset name", value=self._ds_name, hint="⏎ apply · esc cancel"))
        if ans is None or not ans:
            return
        if valid_dataset_name(ans):
            self._ds_name = ans; self._msg = ""
        else:
            self._msg = "✗ name must be a safe folder name using letters/digits/._-"

    async def _edit_task(self) -> None:
        # ctrl+j (NOT shift+⏎) is this port's PromptModalState newline key (see modals.py),
        # so the hint advertises ctrl+j — copying the Textual "shift+⏎" verbatim would be
        # wrong here. Empty apply is dropped (record's task is a PERSISTED label, no empty
        # guard in persist_record_defaults); Esc keeps the current value either way.
        ans = await self.app.run_modal(PromptModalState(
            "Task (language instruction)", value=self._ds_task, multiline=True,
            hint="⏎ apply · ←→ move · ctrl+j newline · esc keep current"))
        if ans is not None and ans.strip():
            self._ds_task = ans.strip(); self._msg = ""

    async def _start(self) -> None:
        app = self.app
        repo_id, root = self._resolve()
        exists = dataset_present(root)
        resume = self._toggle["resume"] or args_have_resume(self._extra)

        if not await confirm_preflight(
            app,
            "Record preflight",
            record_issues(self.ctx, root=root, parent=self._ds_parent),
        ):
            return

        if resume and not exists:
            app.notify("Resume is on, but no dataset exists yet; starting a new dataset.", "warn")
            resume = False
        elif not resume and exists:
            nep = dataset_episodes(root)
            choice = await app.run_modal(ConfirmModalState(
                f'Dataset already exists at "{root}" ({nep} episodes recorded).',
                [_RESUME, _DELETE, _CANCEL]))
            if choice == _RESUME:
                resume = True
            elif choice == _DELETE:
                target = safe_delete_target(root, self._ds_parent)
                if not target.ok:
                    app.notify(f"✗ refusing to delete unsafe dataset path: {target.reason}.", "error")
                    return
                typed = await app.run_modal(PromptModalState(
                    f"⚠ Delete {target.path} ({nep} episodes). This cannot be undone. Type 'delete' to confirm",
                    "", hint="type 'delete' then ⏎ confirm · esc cancel"))
                if typed != "delete":
                    app.notify("Dataset was not deleted; recording canceled.", "warn")
                    return
                shutil.rmtree(target.path, ignore_errors=True)
                app.notify(f"Removed dataset at {target.path}")
            else:
                return  # _CANCEL label OR None (Esc dismissed) → stay on the form

        try:
            persist_record_defaults(
                repo_id=repo_id, root=root, task=self._ds_task,
                episodes=self._num["episodes"].value, ep_time=self._num["ep_time"].value,
                reset_time=self._num["reset_time"].value, fps=self._num["fps"].value,
                img_threads=self._num["img_threads"].value, display=self._toggle["display"],
                streaming=self._toggle["streaming"])
            self.ctx.doc = load_yaml()
        except Exception as e:  # noqa: BLE001 - a save failure must never block recording
            app.notify(f"Could not save recording defaults to lekiwi.yaml: {e}", "warn")

        argv = [
            "bash", str(RECORD_SCRIPT),
            "--name", self._ds_name, "--task", self._ds_task,
            "--episodes", str(self._num["episodes"].value),
            "--episode-time", str(self._num["ep_time"].value),
            "--reset-time", str(self._num["reset_time"].value),
            "--fps", str(self._num["fps"].value),
            "--display", "on" if self._toggle["display"] else "off",
            "--streaming-encoding", "on" if self._toggle["streaming"] else "off",
            "--image-writer-threads", str(self._num["img_threads"].value),
            "--resume", "true" if resume else "false",
            *self._extra,
        ]
        await app.suspend(argv)

    def draw(self, frame: Any, area: Any) -> None:
        draw_form_page(frame, area, self.ctx, "record", self._body_lines(area.width),
                       msg=self._msg, hint=self._hint_text(area.width))

    def _encoder_summary(self) -> str:
        """`h264_nvenc · gpu` — the configured rgb_encoder vcodec, shown in the Start
        plan so a config copied to a GPU-less machine (nvenc aborts at startup) is
        visible BEFORE Start. Display-only: on resume, lerobot keeps the existing
        dataset's codec regardless of this setting."""
        vcodec = str(cfg_get("record.dataset.rgb_encoder.vcodec", doc=self.ctx.doc)
                     or "libsvtav1")  # lerobot's default when the yaml sets nothing
        if vcodec == "auto":
            return "auto (gpu if available)"
        gpu = vcodec.endswith(("_nvenc", "_qsv", "_vaapi", "_videotoolbox"))
        return f"{vcodec} · {'gpu' if gpu else 'cpu'}"

    def _resume_note(self) -> tuple[str, Any]:
        """What Start will do about existing data, stated inline on the Resume row —
        at the point of decision (the old yellow banner idiom is retired)."""
        root = self._resolve()[1]
        resume = self._toggle["resume"] or args_have_resume(self._extra)
        exists = dataset_present(root)
        if exists and resume:
            return f"will append after episode {dataset_episodes(root)}", theme.OK_STYLE
        if exists and not resume:
            return ("dataset exists → Start will ask: resume / delete / cancel",
                    theme.MUTED_STYLE)
        if resume and not exists:
            return "nothing to resume — starts a new dataset", theme.FAINT_STYLE
        return "starts a fresh dataset", theme.FAINT_STYLE

    @staticmethod
    def _wrap_task(text: str, width: int) -> list[str]:
        """Wrap the task to *width* for DISPLAY (the stored value is untouched). Word-wraps
        via the shared helper — this used to char-wrap, which split words in half and left
        a lone full stop on the last line."""
        return wrap_words(text, width)

    # ── form body: grouped one-page layout; the focused field's hint lives in the
    #    FOOTER slot, never inline (see _hint_line) ────────────────────────────────
    _LABEL_W = 14

    def _lab(self, key: str, focused: bool) -> Span:
        return Span(f"{_LABELS[key]:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *keys: str) -> Span:
        on = self._cur() in keys
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _num_text(self, key: str) -> str:
        nf = self._num[key]
        return f"{nf.editor.value}█" if self._cur() == key else nf.display()

    def _num_style(self, key: str) -> Any:
        return theme.HIGHLIGHT_TEXT_STYLE if self._cur() == key else theme.TEXT_STYLE

    def _pill(self, on: bool) -> Span:
        return seg("on" if on else "off", on)

    def _body_lines(self, width: int = 120) -> list[Line]:
        """Build the form's rows as a flat ``list[Line]`` — the testable view-model
        (a Paragraph seals its Text). Grouped DATASET → SESSION → CAPTURE → Start,
        matching _FIELDS navigation order; only the task block wraps to *width*."""
        cur = self._cur()
        w = int(width)
        lines: list[Line] = [section_line("DATASET")]
        # Name (right side: what is on disk at the target root, TTL-cached)
        parts = dataset_stats_parts(workspace_path(self._resolve()[1]))
        stats = (f"{parts['episodes']} ep · {parts['minutes']} min · {parts['size']}"
                 if parts else "new dataset — the first episode creates it")
        lines.append(padded_line(
            [self._gutter("name"), self._lab("name", cur == "name"),
             Span(self._ds_name, theme.TEXT_STYLE)],
            [Span(stats, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)], w))
        # Task — full text, hanging indent under the value column
        segs = self._wrap_task(self._ds_task or "(none)", max(1, w - 2 - self._LABEL_W))
        lines.append(Line([self._gutter("task"), self._lab("task", cur == "task"),
                           Span(segs[0], theme.TEXT_STYLE)]))
        lines.extend(Line([Span(" " * (2 + self._LABEL_W), theme.BASE_STYLE),
                           Span(seg, theme.TEXT_STYLE)]) for seg in segs[1:])
        note, nstyle = self._resume_note()
        lines.append(Line([self._gutter("resume"), self._lab("resume", cur == "resume"),
                           self._pill(self._toggle["resume"]),
                           Span(f"   {note}", nstyle)]))
        lines.append(Line([]))
        # One setting per row from here down. Packing two or three onto a line saved
        # vertical space this screen does not need, at the cost of an operable form.
        lines.append(section_line("SESSION"))
        for key, label, note in _SESSION_ROWS:
            lines.append(number_line(self._num[key], label, cur == key, note, width=w))
        lines.append(Line([]))
        lines.append(section_line("CAPTURE"))
        for key, label, note in _CAPTURE_TOGGLES:
            lines.append(setting_line(label, toggle(self._toggle[key], focused=cur == key),
                                      note, focused=cur == key, width=w))
        lines.append(number_line(self._num["img_threads"], "Writers", cur == "img_threads",
                                 "encoder threads per camera", width=w))
        lines.append(Line([]))
        # Start — the plan sentence; full-band tint when focused
        focused = cur == "start"
        eps, ept = self._num["episodes"].value, self._num["ep_time"].value
        plan = f"{eps} × {ept} s ≈ {max(1, round(eps * ept / 60))} min · enc {self._encoder_summary()}"
        lines.append(plan_row("Start", plan, focused=focused))
        return lines

    def _body(self, width: int = 120) -> Paragraph:
        return Paragraph(Text(self._body_lines(width))).style(theme.BASE_STYLE)

    def _hint_text(self, width: int) -> str:
        """The focused field's hint (plus the resolved dataset path for Name — a stale
        ../ in the yaml once split a dataset invisibly)."""
        cur = self._cur()
        hint = _HINTS.get(cur, "")
        if cur == "name":
            resolved = collapse_home(workspace_path(self._resolve()[1]).resolve())
            hint += " · " + clip_middle(resolved, max(24, int(width) - len(hint) - 44))
        return hint


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY `python -m lekiwi_tui record`: mirror do_record's non-interactive path
    (no form, so record.sh seeds every field from the yaml defaults and only honors a
    --resume passed in extra). If a dataset already exists at the resolved root and --resume
    was not passed, error (do not silently overwrite) and return 1; otherwise front the
    script through the shared headless runner (record.sh owns the argv assembly).

    Ported from the Textual `run_headless(app, extra)`; like sync.py/provision.py this port
    threads config through `ctx` (there is no `app.doc`), so it takes `ctx` and reads
    `record_root(ctx.doc, extra)`."""
    root = record_root(ctx.doc, extra)
    if dataset_present(root) and not args_have_resume(extra):
        print(
            f"✗ dataset already exists at '{root}' ({dataset_episodes(root)} episodes).\n"
            "  Re-run with --resume=true to append episodes, or delete the folder to start fresh."
        )
        return 1
    return runner.headless_run(["bash", str(RECORD_SCRIPT), *extra])


__all__ = [
    "RecordScreen",
    "dataset_defaults",
    "resolve_repo_root",
    "valid_dataset_name",
    "safe_delete_target",
    "persist_record_defaults",
    "run_headless",
    "HEADLESS_HOOK",
]
