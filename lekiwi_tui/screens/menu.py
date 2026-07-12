"""menu.py — MenuScreen, the immediate-mode port of the Textual MenuScreen.

The canonical EXEMPLAR every other screen copies: it takes ``(app, ctx)``, subclasses
``ScreenState``, rebuilds its whole view each frame in :meth:`draw`, and turns keys into
:class:`Action` s in :meth:`handle_key` (pure → unit-testable with synthetic ``Key``).

Look + keys mirror the original: a "◆ LEKIWI" header, an accent rule, a compact status
line, a sectioned action list (HOST / COLLECT / LEARN / SETUP), a footer hint, and
highlighted selected rows. ↑↓ + j/k move, a digit jumps to and runs that action
(daily-driver rows only — the SETUP section has no digit, it is not worth a slip),
⏎ runs the highlighted action, q quits. Navigation skips the section labels: the
selection index runs over the real ACTIONS only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from ..app_registry import ACTIONS
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, UP, Key
from ..framework.screen import Nothing, Quit, RunAction, ScreenState

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

SECTION_RULE_WIDTH = 46


# Digit shortcuts cover the daily-driver rows only (HOST/COLLECT/LEARN, 1-8 today);
# SETUP rows are deliberate, low-frequency actions — no jump, no badge.
_JUMPABLE = [a for a in ACTIONS if a.section != "SETUP"]


class MenuScreen(ScreenState):
    """The main menu. Built from ACTIONS metadata; imports no screen (lazy dispatch)."""

    title = "LEKIWI"

    def __init__(self, app: "App", ctx: "Context") -> None:
        self.app = app
        self.ctx = ctx
        self._sel = 0  # index into ACTIONS (the real, selectable rows)

    # ── navigation (pure; returns an Action) ──────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        n = len(ACTIONS)
        name = key.name
        if name in (UP, "k"):
            self._sel = (self._sel - 1) % n
            return Nothing
        if name in (DOWN, "j"):
            self._sel = (self._sel + 1) % n
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

    @property
    def selected(self):
        """The currently-highlighted Action (used by tests / the dispatcher)."""
        return ACTIONS[self._sel]

    # ── view (rebuilt fresh each frame) ───────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),   # header
                Constraint.length(1),   # accent rule
                Constraint.length(1),   # status chips
                Constraint.length(1),   # gap
                Constraint.fill(1),     # body (the action list)
                Constraint.length(1),   # light rule
                Constraint.length(1),   # hint
            ])
            .split(area)
        )
        # Header: "◆ LEKIWI  mobile-manipulator control"
        frame.render_widget(
            Paragraph(Text([Line([
                Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
                Span("  ", theme.BASE_STYLE),
                Span("mobile-manipulator control", theme.SUBTITLE_STYLE),
            ])])).style(theme.BASE_STYLE),
            rows[0],
        )
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1]
        )
        frame.render_widget(self._status_line(), rows[2])
        frame.render_widget(self._body(), rows[4])
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[5].width, light=True)).style(theme.RULE_LIGHT_STYLE), rows[5]
        )
        frame.render_widget(self._hint_line(), rows[6])

    def _status_line(self) -> Paragraph:
        cfg = self.ctx.cfg
        # Execution mode indicator (toggle with 'd'): REAL drives the robot; PREVIEW shows commands.
        from ..framework import runner

        spans: list[Span] = []
        spans.extend(self._chip("host", str(cfg["LEKIWI_HOST"]), theme.CHIP_VALUE_STYLE))
        spans.extend(self._chip("env", str(cfg["LAPTOP_ENV"]), theme.CHIP_VALUE_STYLE))
        # live robot chip (type + host ●/○ + session countdown) — shared with chrome
        from .chrome import robot_chip_spans

        spans.extend(robot_chip_spans(self.ctx))
        spans.extend(self._gpu_chip())
        spans.extend(
            self._chip(
                "mode",
                "PREVIEW" if runner.DRY_RUN else "REAL",
                theme.CHIP_WARN_STYLE if runner.DRY_RUN else theme.CHIP_OK_STYLE,
            )
        )
        return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)

    def _body(self) -> Paragraph:
        lines: list[Line] = []
        prev_section = ""
        for i, action in enumerate(ACTIONS):
            if action.section != prev_section:
                if prev_section:  # blank line between sections for breathing room
                    lines.append(Line([Span("", theme.BASE_STYLE)]))
                lines.append(self._section_line(action.section))
                prev_section = action.section
            lines.append(self._row(action, selected=(i == self._sel)))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)

    def _chip(self, label: str, value: str, value_style: Any) -> list[Span]:
        return [
            Span(f" {label} ", theme.CHIP_STYLE),
            Span(f"{value} ", value_style),
            Span(" ", theme.BASE_STYLE),
        ]

    def _gpu_chip(self) -> list[Span]:
        spans = [Span(" GPU ", theme.CHIP_STYLE)]
        if self.ctx.gpu_name:
            spans.append(Span(f"{theme.status_dot()} ", theme.CHIP_OK_STYLE))
            spans.append(Span(f"{self.ctx.gpu_name} ", theme.CHIP_TEXT_STYLE))
        else:
            spans.append(Span("none ", theme.CHIP_MUTED_STYLE))
        spans.append(Span(" ", theme.BASE_STYLE))
        return spans

    def _section_line(self, label: str) -> Line:
        rule = theme.rule(max(4, SECTION_RULE_WIDTH - len(label)))
        return Line([
            Span(f"{label:<7}", theme.SECTION_STYLE),
            Span(rule, theme.BORDER_STYLE),
        ])

    def _row(self, action, *, selected: bool) -> Line:
        # The digit shortcut, rendered as a visible per-row badge. SETUP rows are not
        # jumpable and get blank padding so the columns stay aligned.
        idx = _JUMPABLE.index(action) + 1 if action in _JUMPABLE else 0
        badge = f"{idx} " if 1 <= idx <= 9 else "  "
        if selected:
            return Line([
                Span(theme.selector(True), theme.HIGHLIGHT_LABEL_STYLE),
                Span(badge, theme.HIGHLIGHT_TEXT_STYLE),
                Span(f"{theme.action_icon(action.icon)}  ", theme.HIGHLIGHT_ICON_STYLE),
                Span(f"{action.label:<12}", theme.HIGHLIGHT_LABEL_STYLE),
                Span("   ", theme.HIGHLIGHT_STYLE),
                Span(action.hint, theme.HIGHLIGHT_TEXT_STYLE),
                Span("  ", theme.HIGHLIGHT_STYLE),
            ], theme.HIGHLIGHT_STYLE)
        return Line([
            Span(theme.selector(False), theme.BASE_STYLE),
            Span(badge, theme.KEYCAP_STYLE),
            Span(f"{theme.action_icon(action.icon)}  ", theme.BASE_STYLE),
            Span(f"{action.label:<12}", theme.TEXT_STYLE),
            Span("   ", theme.BASE_STYLE),
            Span(action.hint, theme.MUTED_STYLE),
        ])

    def _hint_line(self) -> Paragraph:
        spans: list[Span] = []
        for k, label in [("↑↓/jk", "move"), ("⏎", "select"),
                         (f"1-{len(_JUMPABLE)}", "jump"), ("d", "preview"), ("q", "quit")]:
            spans.append(Span(f" {theme.key_label(k)} ", theme.KEYCAP_STYLE))
            spans.append(Span(f" {label}  ", theme.HINT_STYLE))
        return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)


__all__ = ["MenuScreen"]
