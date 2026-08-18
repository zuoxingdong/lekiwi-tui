"""dataset_edit.py — DatasetEditScreen: browse, triage, and delete episodes IN PLACE.

The post-session cleanup home: a table of the record dataset's episodes (index,
duration, task) joined with the G/B triage verdicts the record HUD writes to
``meta/quality.jsonl``, plus a statistical anomaly column (a truncated take reads
as ``⚠ short`` without anyone remembering it). Flagged episodes arrive PRE-MARKED.

Deleting fronts ``scripts/edit.sh`` — the single argv source — which wraps
``lerobot-edit-dataset`` with guaranteed in-place semantics: the tool writes to a
sibling temp dir, then the script swaps original → timestamped ``.bak`` and temp →
original path (the untouched original IS the backup; no copy, no HF-cache surprise).

Keys: j/k move · Space mark · D delete marked (typed-'delete' confirm) ·
T retag marked (rewrite the task/language instruction) · V view episode in Rerun ·
d switch dataset (the shared replay/view picker) · r reload · q back.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..framework import runner, theme
from ..framework.events import DOWN, ENTER, ESC, SPACE, UP, Key
from ..framework.modals import PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from .chrome import clip_end, keycap_hint_line, padded_line, slim_status_spans
from ..datasets import dataset_defaults, resolve_repo_root, workspace_path

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

EDIT_SCRIPT = ROOT / "scripts" / "edit.sh"
VIEW_SCRIPT = ROOT / "scripts" / "view.sh"

# Length-anomaly thresholds vs the dataset's median episode length.
_SHORT_FRAC = 0.6
_LONG_FRAC = 1.8


@dataclass(frozen=True)
class EpRow:
    index: int
    frames: int
    seconds: float
    task: str


def load_episodes(root: Path) -> list[EpRow]:
    """Read (index, length, task) for every episode from the v3 episodes-metadata
    parquet files. Returns [] when the dataset (or pyarrow) is unavailable."""
    meta = Path(root) / "meta"
    try:
        fps = float(json.loads((meta / "info.json").read_text()).get("fps", 30) or 30)
    except (OSError, ValueError):
        return []
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []
    rows: list[EpRow] = []
    for f in sorted(meta.glob("episodes/**/*.parquet")):
        try:
            t = pq.read_table(f, columns=["episode_index", "length", "tasks"]).to_pydict()
        except Exception:  # noqa: BLE001 — one bad chunk must not hide the rest
            continue
        for idx, ln, tasks in zip(t["episode_index"], t["length"], t["tasks"]):
            task = str(tasks[0]) if isinstance(tasks, (list, tuple)) and tasks else str(tasks or "")
            rows.append(EpRow(int(idx), int(ln), round(int(ln) / fps, 1), task))
    return sorted(rows, key=lambda r: r.index)


def load_verdicts(root: Path) -> dict[int, str]:
    """The record HUD's triage sidecar (meta/quality.jsonl), latest line per episode
    wins. 'redo' audit lines resolve to no verdict (the take was replaced)."""
    out: dict[int, str] = {}
    try:
        for ln in (Path(root) / "meta" / "quality.jsonl").read_text().splitlines():
            try:
                e = json.loads(ln)
                out[int(e["episode"])] = str(e["verdict"])
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        return {}
    return {k: v for k, v in out.items() if v in ("good", "flagged")}


def anomalies(rows: list[EpRow]) -> dict[int, str]:
    """Flag episodes whose length is a clear outlier vs the median — the 'this take
    was cut short' signal that found the truncated episode by data, not memory."""
    if len(rows) < 3:
        return {}
    med = statistics.median(r.frames for r in rows)
    if med <= 0:
        return {}
    out: dict[int, str] = {}
    for r in rows:
        if r.frames < _SHORT_FRAC * med:
            out[r.index] = "short"
        elif r.frames > _LONG_FRAC * med:
            out[r.index] = "long"
    return out


def delete_argv(root: str, repo_id: str, indices: list[int]) -> list[str]:
    """The edit.sh invocation for a marked-episodes delete (single argv source)."""
    eps = "[" + ", ".join(str(i) for i in sorted(indices)) + "]"
    return ["bash", str(EDIT_SCRIPT), "--repo-id", repo_id, "--root", root,
            "--episodes", eps]


def retag_argv(root: str, repo_id: str, indices: list[int], task: str) -> list[str]:
    """The edit.sh invocation for a marked-episodes task rewrite. The task text is
    passed as ONE argv token — edit.sh JSON-encodes it per episode, so quotes,
    unicode, and newlines survive intact."""
    eps = "[" + ", ".join(str(i) for i in sorted(indices)) + "]"
    return ["bash", str(EDIT_SCRIPT), "--op", "retag", "--repo-id", repo_id,
            "--root", root, "--episodes", eps, "--task", task]


class DatasetEditScreen(ScreenState):
    """Episode browser + in-place delete for the record dataset."""

    title = "dataset"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None,
                 root: str | None = None, repo_id: str | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        if root:
            # Direct entry on a specific dataset (the dagger post-session review opens
            # the fresh session here, junk pre-marked via meta/quality.jsonl). `d` still
            # switches datasets afterwards, exactly as from the default entry.
            self._root = root
            self._repo_id = repo_id or f"local/{Path(root).name}"
        else:
            d = dataset_defaults(ctx.doc)
            self._repo_id, self._root = resolve_repo_root(d["name"], d["ns"], d["parent"])
        self._rows: list[EpRow] = []
        self._verdicts: dict[int, str] = {}
        self._anomalies: dict[int, str] = {}
        self._marks: set[int] = set()
        self._cursor = 0
        self._msg = ""
        self._area = None
        self.reload()

    # ── data ──────────────────────────────────────────────────────────────────
    def reload(self) -> None:
        root = workspace_path(self._root)
        self._rows = load_episodes(root)
        self._verdicts = load_verdicts(root)
        self._anomalies = anomalies(self._rows)
        # Triage flags arrive PRE-MARKED: post-session cleanup is open → D.
        self._marks = {r.index for r in self._rows
                       if self._verdicts.get(r.index) == "flagged"}
        self._cursor = min(self._cursor, max(0, len(self._rows) - 1))

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if not self._rows:
            if name == "r":
                self.reload()
            elif name == "d":       # empty root is exactly when you switch datasets
                return Invoke(self._choose_dataset)
            return Nothing
        if name in (UP, "k"):
            self._cursor = (self._cursor - 1) % len(self._rows); self._msg = ""
            return Nothing
        if name in (DOWN, "j"):
            self._cursor = (self._cursor + 1) % len(self._rows); self._msg = ""
            return Nothing
        if name == SPACE:
            idx = self._rows[self._cursor].index
            self._marks.symmetric_difference_update({idx})
            self._msg = ""
            return Nothing
        if name == "r":
            self.reload(); self._msg = ""
            return Nothing
        if name == "D":
            if not self._marks:
                self._msg = "nothing marked — Space marks the highlighted episode"
                return Nothing
            return Invoke(self._delete_marked)
        if name == "T":
            if not self._marks:
                self._msg = "nothing marked — Space marks episodes to retag"
                return Nothing
            return Invoke(self._retag_marked)
        if name in ("V", ENTER):
            return Invoke(self._view_current)
        if name == "d":
            return Invoke(self._choose_dataset)
        return Nothing

    # ── flows ─────────────────────────────────────────────────────────────────
    async def _delete_marked(self) -> None:
        marked = sorted(self._marks)
        eps = ", ".join(str(i) for i in marked)
        typed = await self.app.run_modal(PromptModalState(
            f"⚠ Delete {len(marked)} episode(s) [{eps}] from {self._repo_id}. "
            f"A timestamped .bak of the whole dataset is kept next to it. "
            f"Type 'delete' to confirm", "",
            # The requirement belongs in the HINT, not only in the label: the hint is short,
            # always on screen, and is where you look for "what do I press". A bare
            # "⏎ confirm" here reads as though Enter alone is enough, which silently
            # cancels.
            hint="type 'delete' then ⏎ confirm · esc cancel"))
        if typed != "delete":
            self.app.notify("Nothing deleted.", "warn")
            return
        argv = delete_argv(self._root, self._repo_id, marked)
        rc = await self.app.suspend(argv, pause=True)
        if rc == 0 and not runner.DRY_RUN:
            self.app.notify(f"✓ deleted {len(marked)} episode(s) — indices repacked", "info")
        elif rc != 0:
            self.app.notify(f"✗ edit failed (rc={rc}) — dataset untouched", "error")
        self.reload()

    async def _retag_marked(self) -> None:
        """Rewrite the task text on every marked episode. The prompt opens pre-filled
        with the FIRST marked episode's current task; Esc / empty cancels. Same
        backup+swap safety as delete (upstream modify_tasks alone edits in place
        with NO backup — edit.sh copies first, so a botched retag is one mv away)."""
        marked = sorted(self._marks)
        current = next((r.task for r in self._rows if r.index in self._marks), "")
        ans = await self.app.run_modal(PromptModalState(
            f"New task for {len(marked)} marked episode(s)", value=current,
            multiline=True,
            hint="⏎ apply · ctrl+j newline · esc cancel"))
        if ans is None or not ans.strip():
            self.app.notify("Retag canceled — tasks unchanged.", "warn")
            return
        argv = retag_argv(self._root, self._repo_id, marked, ans.strip())
        rc = await self.app.suspend(argv, pause=True)
        if rc == 0 and not runner.DRY_RUN:
            self.app.notify(f"✓ retagged {len(marked)} episode(s)", "info")
        elif rc != 0:
            self.app.notify(f"✗ retag failed (rc={rc}) — dataset untouched", "error")
        self.reload()

    async def _view_current(self) -> None:
        ep = self._rows[self._cursor].index
        argv = ["bash", str(VIEW_SCRIPT), "--repo-id", self._repo_id,
                "--root", self._root, "--episode-index", str(ep)]
        await self.app.suspend(argv)

    async def _choose_dataset(self) -> None:
        """Retarget the editor at another dataset via the shared DatasetPicker (the
        same picker replay/view use). Esc keeps the current one."""
        from ..dispatch import pick_dataset

        picked = await pick_dataset(self.app, self.ctx.doc, self._extra,
                                    title="Edit - choose dataset")
        if picked is None:
            return
        self._repo_id, self._root = picked
        self._cursor = 0
        self._msg = ""
        self.reload()

    # ── view ──────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        self._area = area
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1), Constraint.length(1)]).split(area))
        left = [Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
                Span(" · dataset", theme.SUBTITLE_STYLE)]
        frame.render_widget(Paragraph(Text([
            padded_line(left, slim_status_spans(self.ctx), rows[0].width)]
        )).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(Paragraph(Text(
            self._body_lines(rows[2].width, rows[2].height))).style(theme.BASE_STYLE), rows[2])
        if self._msg:
            frame.render_widget(Paragraph(Text([Line([Span(f"  {self._msg}", theme.ERR_STYLE)])])
                                          ).style(theme.BASE_STYLE), rows[3])
        frame.render_widget(keycap_hint_line([
            ("Space", "mark"), ("D", "delete"), ("T", "retag"), ("V", "view"),
            ("d", "dataset"), ("r", "reload"), ("j·k", "move"), ("q", "back"),
        ]), rows[4])

    def _body_lines(self, width: int = 100, height: int = 30) -> list[Line]:
        w = int(width)
        good = sum(1 for v in self._verdicts.values() if v == "good")
        flagged = sum(1 for v in self._verdicts.values() if v == "flagged")
        total_s = sum(r.seconds for r in self._rows)
        head = padded_line(
            [Span(f"  {self._repo_id}   ", theme.STATUS_VALUE_STYLE),
             Span(f"{len(self._rows)} ep · {total_s / 60:.1f} min", theme.MUTED_STYLE)],
            [Span("verdicts: ", theme.FAINT_STYLE), Span(f"{good} ✓", theme.OK_STYLE),
             Span(" · ", theme.FAINT_STYLE),
             Span(f"{flagged} ✗", theme.WARN_STYLE if flagged else theme.MUTED_STYLE),
             Span("  ", theme.BASE_STYLE)], w)
        lines = [head, Line([])]
        if not self._rows:
            lines.append(Line([Span("  no dataset on disk — record episodes first",
                                    theme.MUTED_STYLE)]))
            return lines
        lines.append(padded_line(
            [Span("       #     len    task", theme.FAINT_STYLE)],
            [Span("anomaly    verdict  ", theme.FAINT_STYLE)], w))
        # scroll window around the cursor
        avail = max(3, int(height) - 4)
        first = max(0, min(self._cursor - avail // 2, len(self._rows) - avail))
        window = self._rows[first:first + avail]
        if first > 0:
            lines.append(Line([Span(f"   … {first} earlier", theme.FAINT_STYLE)]))
        for r in window:
            lines.append(self._ep_line(r, w))
        rest = len(self._rows) - (first + len(window))
        if rest > 0:
            lines.append(Line([Span(f"   … {rest} more", theme.FAINT_STYLE)]))
        lines.append(Line([]))
        if self._marks:
            lines.append(Line([
                Span(f"  {len(self._marks)} marked", theme.WARN_STYLE),
                Span(" → D deletes in place via edit.sh · auto-backup first",
                     theme.MUTED_STYLE)]))
        return lines

    def _ep_line(self, r: EpRow, width: int) -> Line:
        focused = self._rows[self._cursor].index == r.index
        bg = theme.HIGHLIGHT_STYLE if focused else None
        sel_style = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.BASE_STYLE
        txt = theme.HIGHLIGHT_TEXT_STYLE if focused else theme.TEXT_STYLE
        mut = theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE
        verdict = self._verdicts.get(r.index)
        anom = self._anomalies.get(r.index)
        left = [
            Span(theme.selector(focused), sel_style),
            Span("● " if r.index in self._marks else "  ", theme.WARN_STYLE),
            Span(f"{r.index:>4}  ", mut),
            Span(f"{r.seconds:>6.1f} s  ", txt),
            Span(clip_end(r.task, max(10, width - 48)), mut),
        ]
        right: list[Span] = []
        if anom:
            right.append(Span(f"⚠ {anom}  ", theme.WARN_STYLE))
        if verdict == "good":
            right.append(Span("✓ good   ", theme.OK_STYLE))
        elif verdict == "flagged":
            right.append(Span("✗ flagged", theme.WARN_STYLE))
        else:
            right.append(Span("—        ", theme.FAINT_STYLE))
        right.append(Span("  ", theme.BASE_STYLE))
        return padded_line(left, right, width, pad_style=bg)


__all__ = ["DatasetEditScreen", "load_episodes", "load_verdicts", "anomalies",
           "delete_argv", "retag_argv"]
