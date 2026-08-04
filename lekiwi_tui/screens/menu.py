"""menu.py — MenuScreen, the immediate-mode port of the Textual MenuScreen.

The canonical EXEMPLAR every other screen copies: it takes ``(app, ctx)``, subclasses
``ScreenState``, rebuilds its whole view each frame in :meth:`draw`, and turns keys into
:class:`Action` s in :meth:`handle_key` (pure → unit-testable with synthetic ``Key``).

LAYOUT (agreed design, 2026-08-03): a "◆ LEKIWI" header, an accent rule, then a full-width
**ROBOT status card**, then four **section cards two across** (HOST / COLLECT · DATA /
LEARN), then SETUP as a single strip, a light rule, and the hint line. Each action row is
``icon-cell + digit + label`` with an optional right-aligned live badge, over a muted
description line.

The card grid needs room. Below ``_MIN_CARD_W`` columns or ``_MIN_CARD_H`` rows the screen
falls back to the classic flat sectioned list, which is why :meth:`_flat_body` is still here
rather than deleted — a launcher has to render on a small terminal.

Keys: ↑↓ + j/k move within a card and spill into the next, ←→ + h/l switch column, a digit
jumps to and runs that action (daily-driver rows only — the SETUP strip has no digit, it is
not worth a slip), ⏎ runs the highlighted action, d toggles preview, q quits. ``self._sel``
stays an index into ``ACTIONS`` (tests and the dispatcher read ``.selected``); the grid
position is DERIVED from it each time, never stored alongside it and never allowed to drift.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from ..app_registry import ACTIONS, Action
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from ..framework.screen import Nothing, Quit, RunAction, ScreenState

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# Digit shortcuts cover the daily-driver rows only (HOST/COLLECT/LEARN, 1-9 today);
# SETUP rows are deliberate, low-frequency actions — no jump, no badge.
_JUMPABLE = [a for a in ACTIONS if a.section != "SETUP"]

# ── the card grid ─────────────────────────────────────────────────────────────
#: Which card sits where. Card CONTENTS and their order come from the registry, so this
#: holds the two-dimensional arrangement and nothing else.
_LAYOUT = [["HOST", "COLLECT"], ["DATA", "LEARN"]]
_STRIP_CARD = "SETUP"


def _grouped() -> dict[str, list[Action]]:
    out: dict[str, list[Action]] = {}
    for a in ACTIONS:
        out.setdefault(a.card or a.section, []).append(a)
    return out


_GROUPS = _grouped()
_GRID: list[list[list[Action]]] = [[_GROUPS.get(n, []) for n in row] for row in _LAYOUT]
_STRIP: list[Action] = _GROUPS.get(_STRIP_CARD, [])
_ROWS = len(_GRID)          # the SETUP strip lives at row index _ROWS

#: Status card: three content rows (robot / laptop / identity) plus its border.
_STATUS_H = 5

#: Smallest terminal the card grid is legible in; below either, fall back to the flat list.
#: Height = header + rule + gap + status card + 2 card rows with gaps + strip + rule + hint.
_MIN_CARD_W = 72
_MIN_CARD_H = 28

Pos = tuple[int, int, int]  # (grid row, column, index within that card)


def _pos_of(action: Action) -> Pos:
    for r, row in enumerate(_GRID):
        for c, cell in enumerate(row):
            if action in cell:
                return r, c, cell.index(action)
    if action in _STRIP:
        return _ROWS, 0, _STRIP.index(action)
    return 0, 0, 0


def _action_at(pos: Pos) -> Action:
    r, c, i = pos
    cell = _STRIP if r == _ROWS else _GRID[r][c]
    return cell[max(0, min(i, len(cell) - 1))]


def _move(pos: Pos, *, dr: int = 0, dc: int = 0) -> Pos:
    """Grid navigation, kept pure so it is testable without a terminal.

    Vertical moves walk within a card and spill into the card below/above, then wrap
    through the SETUP strip so every action stays reachable with ↑↓ alone. Horizontal moves
    switch column and clamp the index into the new card; inside the strip they walk ALONG
    it, because a one-row strip has no columns to switch between.
    """
    r, c, i = pos
    if dc:
        if r == _ROWS:
            return (r, 0, (i + dc) % len(_STRIP))
        c = (c + dc) % len(_GRID[r])
        return (r, c, min(i, len(_GRID[r][c]) - 1))
    if dr > 0:
        if r == _ROWS:
            return (0, 0, 0)
        if i + 1 < len(_GRID[r][c]):
            return (r, c, i + 1)
        if r + 1 < _ROWS:
            return (r + 1, c, 0)
        return (_ROWS, 0, 0)
    if dr < 0:
        if r == _ROWS:
            return (_ROWS - 1, 0, len(_GRID[_ROWS - 1][0]) - 1)
        if i > 0:
            return (r, c, i - 1)
        if r > 0:
            return (r - 1, c, len(_GRID[r - 1][c]) - 1)
        return (_ROWS, 0, len(_STRIP) - 1)
    return pos


class MenuScreen(ScreenState):
    """The main menu. Built from ACTIONS metadata; imports no screen (lazy dispatch)."""

    title = "LEKIWI"

    def __init__(self, app: "App", ctx: "Context") -> None:
        self.app = app
        self.ctx = ctx
        self._sel = 0  # index into ACTIONS (the real, selectable rows)

    # ── navigation (pure; returns an Action) ──────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (UP, "k"):
            self._go(dr=-1)
            return Nothing
        if name in (DOWN, "j"):
            self._go(dr=1)
            return Nothing
        if name in (LEFT, "h"):
            self._go(dc=-1)
            return Nothing
        if name in (RIGHT, "l"):
            self._go(dc=1)
            return Nothing
        if len(name) == 1 and name.isdigit() and name != "0":
            d = int(name)
            if 1 <= d <= len(_JUMPABLE):
                self._sel = ACTIONS.index(_JUMPABLE[d - 1])
                return RunAction(_JUMPABLE[d - 1].id)
            return Nothing
        if name == ENTER:
            return RunAction(ACTIONS[self._sel].id)
        if name == "d":  # toggle real-execution vs dry-run preview, live
            from ..framework import runner
            runner.DRY_RUN = not runner.DRY_RUN
            self.app.notify(
                "PREVIEW mode ON - actions show commands instead of running" if runner.DRY_RUN
                else "REAL mode ON - actions will run on the robot/GPU",
                "warn" if runner.DRY_RUN else "info",
            )
            return Nothing
        if name == "q" or name == ESC:
            return Quit()
        return Nothing

    def _go(self, *, dr: int = 0, dc: int = 0) -> None:
        target = _action_at(_move(_pos_of(ACTIONS[self._sel]), dr=dr, dc=dc))
        self._sel = ACTIONS.index(target)

    @property
    def selected(self) -> Action:
        """The currently-highlighted Action (used by tests / the dispatcher)."""
        return ACTIONS[self._sel]

    # ── live facts for the status card ────────────────────────────────────────
    def _live_spans(self) -> list[Span]:
        """Host reachability, session countdown, GPU. Everything here is already polled or
        cached elsewhere: nothing in this method may touch the disk, because draw() runs
        every frame."""
        from ..hostprobe import get_probe, session_remaining

        spans: list[Span] = []
        probe = get_probe(self.ctx)
        alive = None
        if probe is not None:
            probe.poll()
            alive = probe.alive
        if alive is True:
            spans += [Span(f"{theme.status_dot()} ", theme.OK_STYLE),
                      Span("host up", theme.OK_STYLE)]
            left = session_remaining(self.ctx)
            if left is not None:
                # Amber: the countdown is advisory, and a session lapsing mid-take is
                # exactly what is worth noticing BEFORE you start one.
                spans += [Span("  ·  ", theme.MUTED_STYLE),
                          Span(f"{left // 60}:{left % 60:02d} remaining", theme.WARN_STYLE)]
        elif alive is False:
            spans += [Span("○ " if not theme.ASCII_MODE else "o ", theme.MUTED_STYLE),
                      Span("host down", theme.MUTED_STYLE)]
        else:
            spans.append(Span("host unknown", theme.FAINT_STYLE))
        return spans

    def _machine_spans(self) -> list[Span]:
        """Laptop resources: CPU, RAM, GPU, VRAM.

        Labelled "laptop" on purpose. This card's other two rows are about the ROBOT, and an
        unlabelled percentage sitting under "host up" would read as the robot's. Sampling is
        throttled onto a background thread by :mod:`~lekiwi_tui.sysstat`; each field is
        omitted rather than faked when it cannot be read.
        """
        from ..sysstat import get_sysstat

        stat = get_sysstat(self.ctx)
        stat.poll()
        s = stat.sample
        spans: list[Span] = [Span("laptop  ", theme.MUTED_STYLE)]
        if s.cpu_pct is not None:
            spans += [Span("cpu ", theme.MUTED_STYLE),
                      Span(f"{s.cpu_pct:.0f}%", self._load_style(s.cpu_pct))]
        if s.ram_used_gb is not None and s.ram_total_gb:
            frac = 100.0 * s.ram_used_gb / s.ram_total_gb
            spans += [Span("   ram ", theme.MUTED_STYLE),
                      Span(f"{s.ram_used_gb:.1f}", self._load_style(frac)),
                      Span(f"/{s.ram_total_gb:.0f} GB", theme.MUTED_STYLE)]
        # The GPU's own name doubles as the label for its two numbers.
        gpu_label = self.ctx.gpu_name or "gpu"
        if s.gpu_pct is not None:
            spans += [Span(f"   {gpu_label} ", theme.MUTED_STYLE),
                      Span(f"{s.gpu_pct}%", self._load_style(s.gpu_pct))]
        elif self.ctx.gpu_name:
            spans += [Span("   ", theme.MUTED_STYLE), Span(gpu_label, theme.TEXT_STYLE)]
        if s.vram_used_gb is not None and s.vram_total_gb:
            frac = 100.0 * s.vram_used_gb / s.vram_total_gb
            spans += [Span("   vram ", theme.MUTED_STYLE),
                      Span(f"{s.vram_used_gb:.1f}", self._load_style(frac)),
                      Span(f"/{s.vram_total_gb:.0f} GB", theme.MUTED_STYLE)]
        if len(spans) == 1:                      # nothing readable — say so, do not lie
            spans.append(Span("resources unavailable", theme.FAINT_STYLE))
        return spans

    @staticmethod
    def _load_style(pct: float) -> Any:
        """Green / amber / red by load, so a machine that cannot take another training run
        reads at a glance instead of needing the number parsed."""
        if pct >= 90:
            return theme.ERR_STYLE
        if pct >= 70:
            return theme.WARN_STYLE
        return theme.OK_STYLE

    def _identity_spans(self) -> list[Span]:
        """Which hardware and which environment you are about to drive. Machine constants,
        so they come from the config snapshot rather than a probe.

        The lerobot cell is the exception: it is what every screen here actually launches,
        and a too-old one does not fail early — it fails inside draccus mid-launch, with
        the robot already involved. Cheap enough for a per-frame call (metadata lookup,
        cached for the process) and it flags a pre-release checkout too, since the version
        string alone cannot tell those apart."""
        from ..config import cfg_get
        from ..lerobot_env import summary

        out: list[Span] = []
        for label, key in (("robot ", "_launcher.ROBOT_TYPE"),
                           ("env ", "_launcher.LAPTOP_ENV")):
            value = cfg_get(key, doc=self.ctx.doc)
            if value:
                if out:
                    out.append(Span("     ", theme.BASE_STYLE))
                out += [Span(label, theme.MUTED_STYLE), Span(str(value), theme.TEXT_STYLE)]

        value, suffix, level = summary()
        if out:
            out.append(Span("     ", theme.BASE_STYLE))
        warn = level == "warn"
        out += [Span("lerobot ", theme.MUTED_STYLE),
                Span(("⚠ " if warn else "") + value, theme.WARN_STYLE if warn else theme.TEXT_STYLE)]
        if suffix:
            # muted, so a pre-release marker informs without competing with the vitals
            out.append(Span(suffix, theme.WARN_STYLE if warn else theme.MUTED_STYLE))
        return out

    def _badge(self, action: Action) -> tuple[str, Any] | None:
        """The optional right-aligned live number on an action row, or None.

        Only badges whose source is ALREADY cached belong here: draw() runs every frame, so
        a badge must never read the disk uncached. That is why Record carries the episode
        count (``dataset_stats_parts`` caches ~5s) while the flagged-episode count and the
        training checkpoint do not — both would need a fresh parquet read per frame.
        """
        if action.id == "record":
            from ..datasets import dataset_stats_parts, record_root, workspace_path

            parts = dataset_stats_parts(workspace_path(record_root(self.ctx.doc)))
            if parts and parts.get("episodes") not in (None, "", "?"):
                return f"{parts['episodes']} eps", theme.TEXT_STYLE
            return None
        if action.id == "host-kill":
            from ..hostprobe import get_probe

            probe = get_probe(self.ctx)
            if probe is not None and probe.alive is True:
                return f"{theme.status_dot()} up", theme.OK_STYLE
            return None
        return None

    # ── view (rebuilt fresh each frame) ───────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        from .chrome import draw_slim_header, mode_chip_spans

        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),   # header (title + mode chip)
                Constraint.length(1),   # accent rule
                Constraint.length(1),   # gap
                Constraint.fill(1),     # body
                Constraint.length(1),   # light rule
                Constraint.length(1),   # hint
            ])
            .split(area)
        )
        draw_slim_header(frame, rows[0], self.ctx, "mobile-manipulator control",
                         mode_chip_spans())
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE),
            rows[1],
        )
        if area.width >= _MIN_CARD_W and area.height >= _MIN_CARD_H:
            self._draw_cards(frame, rows[3])
        else:
            frame.render_widget(self._flat_body(), rows[3])
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[4].width, light=True))
            .style(theme.RULE_LIGHT_STYLE),
            rows[4],
        )
        frame.render_widget(self._hint_line(), rows[5])

    def _draw_cards(self, frame: Any, area: Any) -> None:
        # Two content lines per action, plus the card's own borders.
        heights = [max((len(c) for c in row), default=0) * 2 + 2 for row in _GRID]
        constraints = [Constraint.length(_STATUS_H), Constraint.length(1)]  # status card, gap
        for h in heights:
            constraints += [Constraint.length(h), Constraint.length(1)]
        constraints += [Constraint.length(1), Constraint.fill(1)]    # setup strip, slack
        band = Layout().direction(Direction.Vertical).constraints(constraints).split(area)

        self._draw_panel(frame, band[0], "ROBOT", [
            Line(self._live_spans()),       # the robot: reachable? session left?
            Line(self._machine_spans()),    # this laptop: can it take another run?
            Line(self._identity_spans()),   # which hardware, which environment
        ])
        for r, row in enumerate(_GRID):
            self._draw_card_row(frame, band[2 + r * 2], r, row)
        frame.render_widget(self._strip_line(), band[2 + len(_GRID) * 2])

    def _draw_card_row(self, frame: Any, area: Any, r: int, row: list[list[Action]]) -> None:
        cols = (
            Layout()
            .direction(Direction.Horizontal)
            .constraints([Constraint.fill(1), Constraint.length(2), Constraint.fill(1)])
            .split(area)
        )
        for c, cell in enumerate(row):
            rect = cols[c * 2]
            self._draw_panel(frame, rect, _LAYOUT[r][c],
                             self._card_lines(cell, max(0, rect.width - 4)))

    def _draw_panel(self, frame: Any, area: Any, title: str, lines: list[Line]) -> None:
        blk = theme.block(f" {title} ")
        inner = blk.inner(area)
        frame.render_widget(blk, area)
        frame.render_widget(Paragraph(Text(lines)).style(theme.BASE_STYLE), inner)

    #: Column the description text starts at inside a card, matching the label column.
    _DESC_INDENT = 6

    def _card_lines(self, cell: list[Action], width: int) -> list[Line]:
        """Two lines per action: the row itself, then its muted description."""
        out: list[Line] = []
        for action in cell:
            out.append(self._card_row(action, selected=action is ACTIONS[self._sel],
                                      width=width))
            out.append(Line([Span(" " * self._DESC_INDENT + self._desc(action, width),
                                  theme.FAINT_STYLE)]))
        return out

    def _desc(self, action: Action, width: int) -> str:
        """The card description: the short ``brief`` when the registry has one, else the
        full hint. Ends in "…" when it still does not fit, because a description that
        stops mid-word without saying so is the same bug the delete modal had."""
        text = action.brief or action.hint
        room = max(0, width - self._DESC_INDENT)
        if len(text) <= room:
            return text
        return text[: max(0, room - 1)].rstrip() + "…" if room else ""

    def _card_row(self, action: Action, *, selected: bool, width: int) -> Line:
        digit = _JUMPABLE.index(action) + 1 if action in _JUMPABLE else 0
        keycap = f"{digit}  " if 1 <= digit <= 9 else "   "
        cell = theme.icon_cell(action.icon)
        base = theme.HIGHLIGHT_STYLE if selected else theme.BASE_STYLE
        spans = [
            Span(theme.selector(selected),
                 theme.HIGHLIGHT_LABEL_STYLE if selected else theme.BASE_STYLE),
            Span(f"{cell} ", theme.HIGHLIGHT_ICON_STYLE if selected else theme.BASE_STYLE),
            Span(keycap, theme.HIGHLIGHT_TEXT_STYLE if selected else theme.KEYCAP_STYLE),
            Span(action.label, theme.HIGHLIGHT_LABEL_STYLE if selected else theme.TEXT_STYLE),
        ]
        badge_text, badge_style = self._badge(action) or ("", theme.BASE_STYLE)
        # An icon cell is ICON_CELL_W columns wide however many codepoints it holds, so
        # measure the cell rather than the string when padding the badge flush right.
        used = (len(theme.selector(selected)) + theme.ICON_CELL_W + 1
                + len(keycap) + len(action.label))
        spans.append(Span(" " * max(1, width - used - len(badge_text)), base))
        if badge_text:
            spans.append(Span(badge_text,
                              theme.HIGHLIGHT_TEXT_STYLE if selected else badge_style))
        return Line(spans, base) if selected else Line(spans)

    def _strip_line(self) -> Paragraph:
        spans: list[Span] = [Span("  SETUP   ", theme.FAINT_STYLE)]
        for action in _STRIP:
            selected = action is ACTIONS[self._sel]
            spans += [
                Span(f"{theme.icon_cell(action.icon)} ",
                     theme.HIGHLIGHT_ICON_STYLE if selected else theme.BASE_STYLE),
                Span(action.label,
                     theme.HIGHLIGHT_LABEL_STYLE if selected else theme.TEXT_STYLE),
                Span("   ", theme.BASE_STYLE),
            ]
        return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)

    # ── fallback: the classic flat list, for a terminal too small for cards ───
    def _flat_body(self) -> Paragraph:
        lines: list[Line] = []
        prev_section = ""
        for i, action in enumerate(ACTIONS):
            if action.section != prev_section:
                if prev_section:  # blank line between sections for breathing room
                    lines.append(Line([Span("", theme.BASE_STYLE)]))
                lines.append(Line([Span(f"  {action.section}", theme.FAINT_STYLE)]))
                prev_section = action.section
            lines.append(self._flat_row(action, selected=(i == self._sel)))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _flat_row(self, action: Action, *, selected: bool) -> Line:
        idx = _JUMPABLE.index(action) + 1 if action in _JUMPABLE else 0
        badge = f"{idx} " if 1 <= idx <= 9 else "  "
        if selected:
            return Line([
                Span(theme.selector(True), theme.HIGHLIGHT_LABEL_STYLE),
                Span(badge, theme.HIGHLIGHT_TEXT_STYLE),
                Span(f"{theme.icon_cell(action.icon)} ", theme.HIGHLIGHT_ICON_STYLE),
                Span(f"{action.label:<12}", theme.HIGHLIGHT_LABEL_STYLE),
                Span("   ", theme.HIGHLIGHT_STYLE),
                Span(action.hint, theme.HIGHLIGHT_TEXT_STYLE),
                Span("  ", theme.HIGHLIGHT_STYLE),
            ], theme.HIGHLIGHT_STYLE)
        return Line([
            Span(theme.selector(False), theme.BASE_STYLE),
            Span(badge, theme.KEYCAP_STYLE),
            Span(f"{theme.icon_cell(action.icon)} ", theme.BASE_STYLE),
            Span(f"{action.label:<12}", theme.TEXT_STYLE),
            Span("   ", theme.BASE_STYLE),
            Span(action.hint, theme.MUTED_STYLE),
        ])

    def _hint_line(self) -> Paragraph:
        spans: list[Span] = []
        for k, label in [("↑↓/jk", "move"), ("←→", "column"), ("⏎", "select"),
                         (f"1-{len(_JUMPABLE)}", "jump"), ("d", "preview"),
                         ("?", "help"), ("q", "quit")]:
            spans.append(Span(f" {theme.key_label(k)} ", theme.KEYCAP_STYLE))
            spans.append(Span(f" {label}  ", theme.HINT_STYLE))
        return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)


__all__ = ["MenuScreen"]
