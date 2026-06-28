"""pickers.py — PolicyPicker + DatasetPicker + DirPicker, immediate-mode ports of the pickers.

All are :class:`ScreenState` modals driven by ``await app.run_modal(...)``: they return a
:class:`Pop` carrying the chosen value (or ``Pop(None)`` on cancel). The Textual versions
wrapped ``OptionList`` / ``DirectoryTree``; here the list + the directory walk are hand-built
over a plain ``_sel`` index (pyratatui has no DirectoryTree), rendered as a centered card.
Fully unit-testable with synthetic Keys (no Terminal).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from ..datasets import discover_datasets
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, UP, Key
from ..framework.screen import Nothing, Pop, ScreenState
from ..policies import discover_policies

#: Sentinel result for the PolicyPicker "Custom path or model repo…" row (caller then prompts).
CUSTOM = "__custom__"


def _card(area: Any, *, height_pct: int = 70, width: int = 80):
    """A centered sub-rect of *area* for a modal card (percentage tall, fixed-ish wide)."""
    rows = (Layout().direction(Direction.Vertical).constraints(
        [Constraint.percentage((100 - height_pct) // 2), Constraint.percentage(height_pct),
         Constraint.fill(1)]).split(area))
    mid = rows[1]
    w = min(width, mid.width)
    cols = (Layout().direction(Direction.Horizontal).constraints(
        [Constraint.fill(1), Constraint.length(w), Constraint.fill(1)]).split(mid))
    return cols[1]


def _render_list(frame: Any, area: Any, title: str, entries, sel: int, hint: str) -> None:
    """Render a titled, bordered list card with the selected row barred (shared by both)."""
    frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
    card = _card(area)
    block = theme.block(title, bordered=True)
    inner = block.inner(card)
    frame.render_widget(block, card)
    rows = (Layout().direction(Direction.Vertical).constraints(
        [Constraint.fill(1), Constraint.length(1)]).split(inner))
    body, hint_area = rows[0], rows[1]
    h = max(1, body.height)
    # Scroll so the selected row stays visible.
    top = max(0, min(sel - h + 1, len(entries) - h)) if len(entries) > h else 0
    lines = []
    for i in range(top, min(top + h, len(entries))):
        label = entries[i][0]
        if i == sel:
            lines.append(Line([Span(theme.selector(True), theme.HIGHLIGHT_LABEL_STYLE),
                               Span(label, theme.HIGHLIGHT_LABEL_STYLE)]))
        else:
            lines.append(Line([Span(theme.selector(False), theme.BASE_STYLE), Span(label, theme.TEXT_STYLE)]))
    frame.render_widget(Paragraph(Text(lines)).style(theme.BASE_STYLE), body)
    frame.render_widget(
        Paragraph(Text([Line([Span(hint, theme.HINT_STYLE)])])).style(theme.BASE_STYLE), hint_area)


class PolicyPicker(ScreenState):
    """Pick a checkpoint: discovered checkpoints (newest first, shown relative to root) +
    a 'Custom' row. Returns the chosen ABSOLUTE path, the :data:`CUSTOM` sentinel, or None."""

    def __init__(self, root: Path, *, default_abs: str = "", title: str = "Pick a checkpoint") -> None:
        self.root = Path(root)
        self.default_abs = default_abs
        self.title = title
        self.found = discover_policies(self.root)
        self.entries: list[tuple[str, str]] = []
        for p in self.found:
            try:
                rel = str(p.relative_to(self.root))
            except ValueError:
                rel = str(p)
            label = rel + ("  ← default" if str(p) == default_abs else "")
            self.entries.append((label, str(p)))
        self.entries.append(("Custom path or model repo…", CUSTOM))
        self._sel = 0

    def handle_key(self, key: "Key") -> Any:
        n = len(self.entries)
        name = key.name
        if name in (UP, "k"):
            self._sel = (self._sel - 1) % n; return Nothing
        if name in (DOWN, "j"):
            self._sel = (self._sel + 1) % n; return Nothing
        if name == ENTER:
            return Pop(self.entries[self._sel][1])
        if name in (ESC, "q"):
            return Pop(None)
        if len(name) == 1 and name.isdigit() and name != "0" and int(name) <= n:
            self._sel = int(name) - 1
            return Pop(self.entries[self._sel][1])
        return Nothing

    def draw(self, frame: Any, area: Any) -> None:
        _render_list(frame, area, self.title, self.entries, self._sel,
                     "↑↓/jk move · ⏎ select · q cancel")


class DatasetPicker(ScreenState):
    """Pick a recorded dataset to replay / view: the present datasets discovered directly
    under *parent* (newest first, the configured one tagged ``← default``) + a ``Custom
    path…`` row. Returns the chosen dataset ROOT (str, kept RELATIVE as discovered — the
    caller derives repo_id from it), the :data:`CUSTOM` sentinel, or None on cancel.

    Mirrors :class:`PolicyPicker` (same list/key model); the value carried is the root path
    rather than a checkpoint path. The default row is pre-selected so ⏎ keeps today's
    behaviour (the configured dataset) with zero extra keystrokes."""

    def __init__(self, parent: "str | Path", *, default_root: str = "",
                 title: str = "Pick a dataset") -> None:
        self.parent = Path(parent)
        self.default_root = str(default_root)
        self.title = title
        self.found = discover_datasets(self.parent)
        # Align the name column so the episode counts line up (purely cosmetic).
        width = max((len(name) for name, _, _ in self.found), default=0)
        self.entries: list[tuple[str, str]] = []
        self._sel = 0
        for i, (name, root, eps) in enumerate(self.found):
            # Path-compare so a trailing slash / "./" can't miss the default tag (both the
            # discovered root and default_root are relative, so this never mixes abs vs rel).
            tag = "   ← default" if self.default_root and Path(root) == Path(self.default_root) else ""
            self.entries.append((f"{name:<{width}}   {eps:>3} episodes{tag}", root))
            if tag:
                self._sel = i
        self.entries.append(("Custom path…", CUSTOM))

    def handle_key(self, key: "Key") -> Any:
        n = len(self.entries)
        name = key.name
        if name in (UP, "k"):
            self._sel = (self._sel - 1) % n; return Nothing
        if name in (DOWN, "j"):
            self._sel = (self._sel + 1) % n; return Nothing
        if name == ENTER:
            return Pop(self.entries[self._sel][1])
        if name in (ESC, "q"):
            return Pop(None)
        if len(name) == 1 and name.isdigit() and name != "0" and int(name) <= n:
            self._sel = int(name) - 1
            return Pop(self.entries[self._sel][1])
        return Nothing

    def draw(self, frame: Any, area: Any) -> None:
        _render_list(frame, area, self.title, self.entries, self._sel,
                     "↑↓/jk move · ⏎ select · q cancel")


class DirPicker(ScreenState):
    """Browse directories from *start*. ⏎ descends into the highlighted dir; 'u' picks the
    current directory; q/Esc cancels. Returns the chosen absolute path or None."""

    def __init__(self, start: "str | Path", title: str = "Pick a directory") -> None:
        sp = Path(start)
        if not sp.is_dir():
            sp = Path.cwd()
        self.cwd = sp.resolve()
        self.title = title
        self._sel = 0
        self._load()

    def _load(self) -> None:
        try:
            dirs = sorted([d for d in self.cwd.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
        except OSError:
            dirs = []
        self.entries: list[tuple[str, Path]] = [(".. (parent)", self.cwd.parent)]
        self.entries += [(d.name + "/", d) for d in dirs]
        self._sel = 0

    def handle_key(self, key: "Key") -> Any:
        n = len(self.entries)
        name = key.name
        if name in (UP, "k"):
            self._sel = (self._sel - 1) % n; return Nothing
        if name in (DOWN, "j"):
            self._sel = (self._sel + 1) % n; return Nothing
        if name == ENTER:
            self.cwd = Path(self.entries[self._sel][1]).resolve()
            self._load(); return Nothing
        if name == "u":
            return Pop(str(self.cwd))
        if name in (ESC, "q"):
            return Pop(None)
        return Nothing

    def draw(self, frame: Any, area: Any) -> None:
        _render_list(frame, area, f"{self.title}  ·  {self.cwd}", self.entries, self._sel,
                     "↑↓/jk move · ⏎ open dir · u use this dir · q cancel")


__all__ = ["CUSTOM", "PolicyPicker", "DatasetPicker", "DirPicker"]
