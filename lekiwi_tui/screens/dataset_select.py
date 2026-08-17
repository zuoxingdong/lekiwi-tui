"""dataset_select.py — DatasetSelectScreen: the Edit-dataset entry page.

One list, two outcomes. ⏎ with nothing toggled opens the per-episode editor on the
cursor row (one toggle behaves the same); toggling SEVERAL datasets with Space and
pressing ⏎ starts the MERGE flow instead. Toggle ORDER is merge order — the first
toggle is the primary dataset: episodes concatenate after it and the output is
created next to it.

The merge is dagger-aware end to end (scripts/merge.sh): inputs carrying the
`intervention` feature (dagger sessions — tagged in the list) are stripped into
temp copies and their per-episode stats normalized before lerobot's aggregate
runs, so the output's features, stats and episode metadata are all consistent.
Sources are never modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Line, Span

from .. import ROOT
from ..datasets import dataset_features, discover_datasets, record_root
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, SPACE, UP, Key
from ..framework.modals import ConfirmModalState, PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from .chrome import clip_end as _clip_end
from .chrome import draw_form_page, padded_line, section_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

MERGE_SCRIPT = ROOT / "scripts" / "merge.sh"
_MERGE = "Merge"


def merge_out_default(names: list[str], any_dagger: bool, parent: Path) -> str:
    """The suggested output name: ``<primary>-dagger-vN`` when a dagger session is
    among the inputs (the fine-tune-set case), else ``<primary>-merged-vN`` — N one
    past the highest existing sibling with that stem."""
    stem = f"{names[0]}-{'dagger' if any_dagger else 'merged'}-v"
    taken = [d.name for d in parent.glob(f"{stem}*") if d.is_dir()]
    n = 1 + max((int(t[len(stem):]) for t in taken if t[len(stem):].isdigit()), default=0)
    return f"{stem}{n}"


class DatasetSelectScreen(ScreenState):
    """Pick one dataset to edit, or several (in order) to merge."""

    title = "datasets"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._parent = str(Path(record_root(ctx.doc)).parent)
        self._rows: list[tuple[str, str, str]] = []   # (name, root, episodes)
        self._feats: dict[str, set[str]] = {}          # root -> feature keys (cached)
        self._sel: list[int] = []                      # toggled row indices, IN ORDER
        self._cursor = 0
        self._msg = ""
        self.reload()

    def reload(self) -> None:
        self._rows = discover_datasets(self._parent)
        self._feats = {root: dataset_features(root) for _name, root, _eps in self._rows}
        self._sel = [i for i in self._sel if i < len(self._rows)]
        self._cursor = min(self._cursor, max(0, len(self._rows) - 1))

    def _is_dagger(self, root: str) -> bool:
        return "intervention" in self._feats.get(root, set())

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k"):
            self._cursor = (self._cursor - 1) % max(1, len(self._rows)); self._msg = ""
            return Nothing
        if name in (DOWN, "j"):
            self._cursor = (self._cursor + 1) % max(1, len(self._rows)); self._msg = ""
            return Nothing
        if name == SPACE and self._rows:
            if self._cursor in self._sel:
                self._sel.remove(self._cursor)
            else:
                self._sel.append(self._cursor)   # toggle order = merge order
            self._msg = ""
            return Nothing
        if name == "r":
            self.reload()
            self._msg = "rescanned"
            return Nothing
        if name == ENTER:
            return Invoke(self._proceed)
        return Nothing

    # ── flows ─────────────────────────────────────────────────────────────────
    async def _proceed(self) -> None:
        if not self._rows:
            self._msg = f"no datasets under {self._parent}"
            return
        picked = [self._rows[i] for i in self._sel] or [self._rows[self._cursor]]
        if len(picked) == 1:
            from .dataset_edit import DatasetEditScreen

            _name, root, _eps = picked[0]
            self.app.push(DatasetEditScreen(self.app, self.ctx, root=root))
            return
        await self._merge(picked)

    async def _merge(self, picked: list[tuple[str, str, str]]) -> None:
        app = self.app
        names = [name for name, _root, _eps in picked]
        any_dagger = any(self._is_dagger(root) for _name, root, _eps in picked)
        out = await app.run_modal(PromptModalState(
            f"Merged dataset name (created next to {names[0]})",
            value=merge_out_default(names, any_dagger, Path(self._parent)),
            hint="⏎ merge · esc cancel"))
        if not out or "/" in out:
            return
        dagger_note = (" Dagger inputs get `intervention` stripped and episode stats "
                       "normalized (temp copies)." if any_dagger else "")
        plan = (f"Merge {len(picked)} datasets in toggle order — {', '.join(names)} → "
                f"{out}. Sources untouched.{dagger_note}")
        if await app.run_modal(ConfirmModalState(plan, [_MERGE, "Cancel"])) != _MERGE:
            return
        argv = ["bash", str(MERGE_SCRIPT)]
        for _name, root, _eps in picked:
            argv += ["--dataset", root]
        argv += ["--out-name", out, *self._extra]
        rc = await app.suspend(argv, pause=True)
        self._sel = []
        self.reload()
        if rc == 0:
            app.notify(f"✓ merged into {Path(self._parent) / out}", "info")
        else:
            app.notify(f"✗ merge failed (rc={rc}) — sources are untouched", "error")

    # ── view ────────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Clear(), area)
        draw_form_page(frame, area, self.ctx, "datasets", self._body_lines(area.width),
                       msg=self._msg, hint=self._hint())

    def _body_lines(self, width: int = 100) -> list[Line]:
        w = int(width)
        lines: list[Line] = [section_line(f"DATASETS · {self._parent}")]
        if not self._rows:
            lines.append(Line([Span("  (none found — record something first)",
                                    theme.FAINT_STYLE)]))
            return lines
        name_w = max(20, w - 34)
        for i, (name, root, eps) in enumerate(self._rows):
            focused = i == self._cursor
            order = f"{self._sel.index(i) + 1}" if i in self._sel else " "
            box = f"[{'x' if i in self._sel else ' '}]{order} "
            tag = "dagger" if self._is_dagger(root) else ""
            lines.append(padded_line(
                [Span(theme.selector(focused),
                      theme.TITLE_STYLE if focused else theme.BASE_STYLE),
                 Span(box, theme.OK_STYLE if i in self._sel else theme.MUTED_STYLE),
                 Span(f"{_clip_end(name, name_w):<{name_w}}",
                      theme.HIGHLIGHT_TEXT_STYLE if focused else theme.TEXT_STYLE)],
                [Span(f"{tag:>6}  ", theme.WARN_STYLE if tag else theme.BASE_STYLE),
                 Span(f"{eps:>4} ep  ", theme.FAINT_STYLE)], w))
        return lines

    def _hint(self) -> str:
        n = len(self._sel)
        if n >= 2:
            return f"⏎ merge these {n} (toggle order = merge order; first = primary) · space untoggle · esc back"
        if n == 1:
            return "⏎ edit this dataset · toggle more with space to merge instead"
        return "⏎ edit the highlighted dataset · space toggle several to merge · r rescan"


__all__ = ["DatasetSelectScreen", "merge_out_default"]
