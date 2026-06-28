"""record.py — RecordScreen: configure + launch lerobot-record (port of the Textual one).

A form (dataset name, task, episodes, episode/reset time, fps, image writers, display +
streaming + resume toggles). Start runs the existing-dataset safety gate (Resume / Delete
with a typed-'delete' confirm + unsafe-path guard / Cancel) as an Invoke async flow using
``app.run_modal``, persists the form values back to lekiwi.yaml (lossless ruamel), then
suspends into the live lerobot-record loop. Fronts scripts/record.sh (the sole argv source).
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import CFG_FILE, ROOT
from ..config import cfg_get, dump_yaml_rt, load_yaml, load_yaml_rt
from ..datasets import args_have_resume, dataset_episodes, dataset_present, record_root
from ..framework import runner, theme
from ..framework.events import BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key, is_char
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, record_issues
from .chrome import keycap_hint_line, option_line, runtime_chips

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# The run_headless hook the dispatcher/__main__ calls for a no-TTY
# `python -m lekiwi_tui record` (parity with the Textual app's HEADLESS_HOOK).
HEADLESS_HOOK = "run_headless"

RECORD_SCRIPT = ROOT / "scripts" / "record.sh"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RULE = "─" * 54


@dataclass(frozen=True)
class DeleteTarget:
    ok: bool
    path: Path
    reason: str = ""

# Per-field inline hints (one per non-start row), copied VERBATIM from the Textual
# original's _refresh (the literal "<name>" placeholders are intentional). Rendered as a
# trailing muted Span on each row by _body / the special-cased task row.
_HINTS = {
    "name": "edit dataset name · saves under datasets/<name>",
    "task": "edit language instruction",
    "episodes": "←→ ±1 · enter to type · episodes to record",
    "ep_time": "←→ ±5 · enter to type · recording time per episode",
    "reset_time": "←→ ±1 · enter to type · reset pause between episodes",
    "fps": "←→ ±5 · enter to type · recording frame rate",
    "img_threads": "←→ ±1 · enter to type · writer threads per camera",
    "display": "show live Rerun view (off lowers CPU)",
    "streaming": "encode video while recording for faster saves",
    "resume": "append to an existing dataset",
}

# The existing-dataset gate's choice labels (fuller copy from the Textual original). The
# port's ConfirmModalState yields the chosen LABEL verbatim, so these are bound once and
# compared by equality at the call site (CANCEL or Esc/None both mean "stay put").
_RESUME = "Resume   append more episodes"
_DELETE = "Delete   remove it and start fresh"
_CANCEL = "Cancel   keep the dataset unchanged"


def dataset_defaults(doc: dict | None = None) -> dict:
    """Per-run record defaults from the yaml `record` block (ported verbatim)."""
    repo_id = str(cfg_get("record.dataset.repo_id", doc=doc) or "local/lekiwi_dataset")
    root = str(cfg_get("record.dataset.root", doc=doc) or "../datasets/lekiwi_dataset")
    ns = repo_id.rsplit("/", 1)[0] if "/" in repo_id else "local"
    parent = str(Path(root).parent) if str(Path(root).parent) != "." else ""

    def _int(key: str, fallback: int) -> int:
        v = cfg_get(f"record.dataset.{key}", doc=doc)
        if isinstance(v, bool):
            return fallback
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return fallback

    def _bool(key: str, fallback: bool) -> bool:
        v = cfg_get(f"record.dataset.{key}", doc=doc)
        if v is None:
            return fallback
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "on", "yes")

    return {
        "name": Path(root).name or "lekiwi_dataset",
        "ns": ns, "parent": parent,
        "task": str(cfg_get("record.dataset.single_task", doc=doc) or ""),
        "episodes": _int("num_episodes", 5), "ep_time": _int("episode_time_s", 40),
        "reset_time": _int("reset_time_s", 5), "fps": _int("fps", 30),
        "img_threads": _int("num_image_writer_threads_per_camera", 3),
        "display": bool(cfg_get("record.display_data", doc=doc)),
        "streaming": _bool("streaming_encoding", True),
        "resume": bool(cfg_get("record.resume", doc=doc)),
    }


def resolve_repo_root(name: str, ns: str, parent: str) -> tuple[str, str]:
    """A dataset name -> (repo_id, root), preserving the yaml namespace + parent dir."""
    return f"{ns}/{name}", (f"{parent}/{name}" if parent else name)


def valid_dataset_name(name: str) -> bool:
    """Return True for a single safe dataset folder name."""
    return bool(_NAME_RE.fullmatch(name)) and name not in (".", "..")


def _workspace_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ROOT / p


def safe_delete_target(root: str | Path, parent: str | Path) -> DeleteTarget:
    """Validate a record delete target against its configured dataset parent."""
    raw = str(root).strip()
    if raw in ("", "/", os.path.expanduser("~")):
        return DeleteTarget(False, _workspace_path(root), "empty, root, or home path")
    target = _workspace_path(root).resolve(strict=False)
    parent_path = _workspace_path(parent or ".").resolve(strict=False)
    if target == parent_path:
        return DeleteTarget(False, target, "target is the dataset parent")
    if target.name in ("", ".", ".."):
        return DeleteTarget(False, target, "target has no safe folder name")
    try:
        target.relative_to(parent_path)
    except ValueError:
        return DeleteTarget(False, target, f"target is outside dataset parent {parent_path}")
    return DeleteTarget(True, target)


def persist_record_defaults(*, repo_id: str, root: str, task: str, episodes: int,
                            ep_time: int, reset_time: int, fps: int, img_threads: int,
                            display: bool, streaming: bool, cfg_path: Path = CFG_FILE) -> None:
    """Write the form values back into lekiwi.yaml (lossless ruamel round-trip). Ported
    verbatim from the Textual app (anchors/merges/comments survive; shared &task anchor)."""
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    doc = load_yaml_rt(cfg_path)
    rd = doc["record"]["dataset"]
    rd["num_episodes"] = int(episodes)
    rd["episode_time_s"] = int(ep_time)
    rd["reset_time_s"] = int(reset_time)
    rd["fps"] = int(fps)
    rd["num_image_writer_threads_per_camera"] = int(img_threads)
    rd["streaming_encoding"] = bool(streaming)
    doc["record"]["display_data"] = bool(display)
    doc["_dataset"]["repo_id"] = repo_id
    doc["_dataset"]["root"] = root
    ns = DoubleQuotedScalarString(task)
    ns.yaml_set_anchor("task", always_dump=True)
    doc["_task"] = ns
    doc["record"]["dataset"]["single_task"] = ns
    doc["rollout"]["task"] = ns
    dump_yaml_rt(doc, cfg_path)


# Field spec: (key, label, kind) — kind in {text, task, num, toggle, start}.
_NUM_KEYS = {"episodes", "ep_time", "reset_time", "fps", "img_threads"}
_TOGGLE_KEYS = {"display", "streaming", "resume"}
_FIELDS = ["name", "task", "episodes", "ep_time", "reset_time", "fps", "img_threads",
           "display", "streaming", "resume", "start"]
_LABELS = {"name": "Dataset", "task": "Task", "episodes": "Episodes", "ep_time": "Episode time",
           "reset_time": "Reset time", "fps": "FPS", "img_threads": "Image writers",
           "display": "Display", "streaming": "Streaming", "resume": "Resume", "start": "Start"}


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
                    "", hint="⏎ confirm · esc cancel"))
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

    # ── view ────────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        repo_id, root = self._resolve()
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.length(1),
             Constraint.length(2), Constraint.fill(1), Constraint.length(1),
             Constraint.length(1)]).split(area))
        frame.render_widget(Paragraph(Text([Line([
            Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE), Span("  record", theme.SUBTITLE_STYLE)])]
        )).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        frame.render_widget(Paragraph(Text([
            Line([Span(f"→ {root}  ·  {repo_id}", theme.MUTED_STYLE)]),
            Line([Span("⚠ start the Pi host before recording.", theme.STATUS_VALUE_STYLE)]),
        ])).style(theme.BASE_STYLE), rows[3])
        frame.render_widget(self._body(rows[4].width), rows[4])
        if self._msg:
            frame.render_widget(Paragraph(Text([Line([Span(self._msg, theme.ERR_STYLE)])])
                                          ).style(theme.BASE_STYLE), rows[5])
        frame.render_widget(self._hint(), rows[6])

    def _value(self, field: str) -> str:
        # NOTE: "task" is NOT handled here — it is special-cased in _body (multiline,
        # untruncated) so the value gets its own line(s). _value drives the single-line rows.
        if field == "name":
            return self._ds_name
        if field in _NUM_KEYS:
            nf = self._num[field]
            if field == self._cur():
                return nf.editor.value + "█"
            return nf.display()
        if field in _TOGGLE_KEYS:
            return theme.choice("on") if self._toggle[field] else theme.choice("off")
        return ""

    @staticmethod
    def _wrap_task(text: str, width: int) -> list[str]:
        """Wrap *text* to *width* cols for DISPLAY (the stored value is untouched). Splits on
        real "\\n" first (explicit breaks survive), then char-wraps each segment — mirrors
        ``PromptModalState._wrap`` so a long no-newline instruction spans lines instead of
        being truncated at the panel edge (the original Static got soft-wrap for free)."""
        if width < 1:
            width = 1
        out: list[str] = []
        for para in text.split("\n"):
            if para == "":
                out.append("")
                continue
            for i in range(0, len(para), width):
                out.append(para[i:i + width])
        return out or [""]

    def _task_lines(self, focused: bool, width: int = 120) -> list[Line]:
        """The Task row, rendered SPECIALLY (not via the single-line row builder): a label
        line, then the FULL task on its own line(s) — explicit "\\n"s AND soft-wrapped to the
        panel *width* (NOT truncated to 48 chars / not clipped at the edge) — then the hint on
        its own muted line, the original _set_task_row shape. Value/hint lines get a 2-space
        hanging indent so every (wrapped) line aligns under the label."""
        lstyle = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.MUTED_STYLE
        vstyle = theme.HIGHLIGHT_TEXT_STYLE if focused else theme.TEXT_STYLE
        selector_style = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.BASE_STYLE
        hint_style = theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE
        out = [Line([
            Span(theme.selector(focused), selector_style),
            Span("Task", lstyle),
        ], theme.HIGHLIGHT_STYLE if focused else None)]
        # The 2-space hanging indent eats 2 cols, so wrap the value to width-2.
        for seg in self._wrap_task(self._ds_task or "(none)", max(1, width - 2)):
            out.append(Line([Span(f"  {seg}", vstyle)], theme.HIGHLIGHT_STYLE if focused else None))
        out.append(Line([Span(f"  {_HINTS['task']}", hint_style)], theme.HIGHLIGHT_STYLE if focused else None))
        return out

    def _body_lines(self, width: int = 120) -> list[Line]:
        """Build the form's rows as a flat ``list[Line]`` (the pure view-model _body wraps).
        Split out from _body so the rendered content (labels, values, the verbatim per-field
        hints, the multiline+wrapped task block) is introspectable headlessly — a Paragraph
        seals its Text, so the Line list is the testable view-model. *width* is the panel
        width (only the task block wraps to it)."""
        lines: list[Line] = []
        for i, field in enumerate(_FIELDS):
            focused = i == self._fpos
            if field == "start":
                lines.append(option_line(
                    f"{theme.play_mark()} Start recording",
                    "validate dataset and launch capture",
                    focused=focused,
                    label_width=22,
                    width=width,
                    label_unfocused_style=theme.TEXT_STYLE,
                ))
                continue
            if field == "task":
                lines.extend(self._task_lines(focused, width))
                continue
            lines.append(option_line(
                _LABELS[field],
                self._value(field),
                _HINTS[field],
                focused=focused,
                label_width=13,
                width=width,
            ))
        return lines

    def _body(self, width: int = 120) -> Paragraph:
        return Paragraph(Text(self._body_lines(width))).style(theme.BASE_STYLE)

    def _hint(self) -> Paragraph:
        return keycap_hint_line([
            ("↑↓/jk", "move"),
            ("←→/hl", "adjust"),
            ("⏎", "edit/start"),
            ("q", "back"),
        ])


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
