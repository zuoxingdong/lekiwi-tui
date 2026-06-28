"""teleop.py — TeleopScreen, the FORM-screen exemplar (port of the Textual TeleopScreen).

The reference pattern every form screen copies (record / eval / calibrate / host / train all
share this shape). It demonstrates:
  * the ``Screen(app, ctx)`` contract (no ``app`` use in ``__init__``);
  * a multi-field form over :class:`~lekiwi_tui.framework.focus.FocusRing` (Display
    toggle, Duration + FPS :class:`NumberField` s, a Start row);
  * predictable keys — ↑↓/jk move focus, ←→/hl adjust the focused field, type digits to set a
    number, ⏎ activates (toggle / commit / start), q/esc backs out;
  * launching through an :class:`Invoke` flow (advisory preflight, then ``app.suspend``)
    — teleop owns the real TTY because it reads the keyboard to drive the base.

argv is built by fronting the carried-over ``scripts/teleop.sh`` (the SOLE argv source); the
form only gathers Display / Duration / FPS. handle_key is pure → unit-testable with synthetic
Key (no Terminal).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..config import cfg_get
from ..framework import runner, theme
from ..framework.events import (
    BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, SPACE, TAB, UP, Key, is_char,
)
from ..framework.focus import FocusRing
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, robot_runtime_issues
from .chrome import keycap_hint_line, option_line, runtime_chips

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

TELEOP_SCRIPT = ROOT / "scripts" / "teleop.sh"
RULE = "─" * 54


def _as_int(v: Any, default: int) -> int:
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    return default


class _Toggle:
    """A tiny on/off field (the Display row). Mirrors NumberField's 'object the screen
    renders + adjusts' shape so the FocusRing treats it uniformly."""

    def __init__(self, label: str, value: bool) -> None:
        self.label = label
        self.value = value

    def toggle(self) -> None:
        self.value = not self.value

    def display(self) -> str:
        return theme.choice("on") if self.value else theme.choice("off")


class _Start:
    """The Start pseudo-field — focusable, but activated by the screen (Enter), not itself."""

    label = "Start"


class TeleopScreen(ScreenState):
    """Config form for lerobot-teleoperate, then suspend into the live loop."""

    title = "teleop"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        doc = ctx.doc
        self.display = _Toggle("Display", bool(cfg_get("teleop.display_data", doc=doc)))
        self.dur = NumberField("Duration", 0, minimum=0, step=5, unit="s", zero_label="until Ctrl+C")
        self.fps = NumberField("FPS", _as_int(cfg_get("teleop.fps", doc=doc), 30), minimum=1, step=5)
        self.start = _Start()
        self.ring = FocusRing([self.display, self.dur, self.fps, self.start])
        self._msg = ""
        # True when a focused number field's editor mirrors its committed value, so the
        # next printable key REPLACES it (type-to-set) rather than appending; cleared after
        # the first edit, re-armed on focus move / step / commit.
        self._fresh = True

    # ── input (pure → returns an Action) ──────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k", BACKTAB):
            self._move(self.ring.prev); return Nothing
        if name in (DOWN, "j", TAB):
            self._move(self.ring.next); return Nothing

        cur = self.ring.current()
        if name in (LEFT, "h", RIGHT, "l"):
            delta = -1 if name in (LEFT, "h") else 1
            if isinstance(cur, NumberField):
                cur.step_by(delta); cur.sync_editor(); self._fresh = True
            elif isinstance(cur, _Toggle):
                cur.toggle()
            self._msg = ""
            return Nothing
        if name == ENTER:
            if cur is self.start:
                return Invoke(self._start)
            if isinstance(cur, _Toggle):
                cur.toggle(); return Nothing
            if isinstance(cur, NumberField):
                if not cur.set_text(cur.editor.value):
                    self._msg = f"✗ {cur.error}"
                else:
                    self._msg = ""; cur.sync_editor(); self._fresh = True
                return Nothing
            return Nothing
        if name == SPACE and isinstance(cur, _Toggle):
            cur.toggle(); return Nothing
        # Type digits / edit into a focused number field. The first printable key after a
        # focus move / step / commit REPLACES the mirrored value (type-to-set); subsequent
        # keys edit in place. Live-commit whenever the buffer parses as a number.
        if isinstance(cur, NumberField) and (is_char(key) or name == "Backspace"):
            if self._fresh and is_char(key):
                cur.editor.clear()
            self._fresh = False
            if cur.editor.handle_key(key):
                t = cur.editor.value.strip()
                if t.isdigit():
                    cur.set_text(t)
                self._msg = ""
                return Nothing
        return Nothing

    def _move(self, mover) -> None:
        """Rotate focus via *mover* (ring.next/prev), re-arm type-to-set, and sync a
        newly-focused number field's editor to its committed value."""
        mover()
        self._msg = ""
        cur = self.ring.current()
        if isinstance(cur, NumberField):
            cur.sync_editor()
        self._fresh = True

    def _argv(self) -> list[str]:
        return [
            "bash", str(TELEOP_SCRIPT),
            "--display", "on" if self.display.value else "off",
            "--fps", str(self.fps.value),
            "--duration", str(self.dur.value),
            *self._extra,
        ]

    async def _start(self) -> None:
        if not await confirm_preflight(
            self.app,
            "Teleop preflight",
            robot_runtime_issues(self.ctx, check_leader=True),
        ):
            return
        await self.app.suspend(self._argv(), title="teleop")

    # ── view (rebuilt each frame) ─────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (
            Layout().direction(Direction.Vertical).constraints([
                Constraint.length(1),   # header
                Constraint.length(1),   # rule
                Constraint.length(1),   # runtime chips
                Constraint.length(2),   # info
                Constraint.length(1),   # gap
                Constraint.length(1),   # display
                Constraint.length(1),   # duration
                Constraint.length(1),   # fps
                Constraint.length(1),   # gap
                Constraint.length(1),   # start
                Constraint.length(1),   # msg
                Constraint.fill(1),     # spacer
                Constraint.length(1),   # hint
            ]).split(area)
        )
        frame.render_widget(
            Paragraph(Text([Line([
                Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
                Span("  teleop", theme.SUBTITLE_STYLE),
            ])])).style(theme.BASE_STYLE), rows[0])
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        frame.render_widget(
            Paragraph(Text([
                Line([Span("leader arm controls the follower; keyboard drives the base (wasd + zx)", theme.MUTED_STYLE)]),
                Line([Span("⚠ start the Pi host first; after launch, Ctrl+C stops teleop", theme.STATUS_VALUE_STYLE)]),
            ])).style(theme.BASE_STYLE), rows[3])

        self._row(frame, rows[5], self.display, self.display.display(),
                  "show live Rerun view (off lowers CPU)")
        self._num_row(frame, rows[6], self.dur, self.dur.hint() + " · 0 = run until Ctrl+C")
        self._num_row(frame, rows[7], self.fps, self.fps.hint() + " · control-loop rate")
        self._start_row(frame, rows[9])

        if self._msg:
            frame.render_widget(
                Paragraph(Text([Line([Span(self._msg, theme.ERR_STYLE)])])).style(theme.BASE_STYLE),
                rows[10])
        self._hint(frame, rows[12])

    # ── row helpers (shared shape: a 2-col focus gutter + content) ────────────
    def _split_gutter(self, area: Any):
        cols = (Layout().direction(Direction.Horizontal)
                .constraints([Constraint.length(2), Constraint.fill(1)]).split(area))
        return cols[0], cols[1]

    def _gutter(self, frame: Any, area: Any, focused: bool) -> None:
        bar = theme.selector(focused)
        frame.render_widget(
            Paragraph(Text([Line([Span(bar, theme.HIGHLIGHT_LABEL_STYLE)])])).style(theme.BASE_STYLE),
            area)

    def _row(self, frame: Any, area: Any, field: Any, value: str, hint: str) -> None:
        focused = self.ring.is_focused(field)
        frame.render_widget(
            Paragraph(Text([option_line(
                field.label, value, hint, focused=focused, label_width=10, width=area.width
            )])).style(theme.BASE_STYLE),
            area,
        )

    def _num_row(self, frame: Any, area: Any, field: "NumberField", hint: str) -> None:
        focused = self.ring.is_focused(field)
        gut, content = self._split_gutter(area)
        self._gutter(frame, gut, focused)
        if focused:
            # NumberField.draw shows the live editor (caret) when focused.
            cols = (Layout().direction(Direction.Horizontal)
                    .constraints([Constraint.length(28), Constraint.fill(1)]).split(content))
            field.draw(frame, cols[0], focused=True)
            frame.render_widget(
                Paragraph(Text([Line([Span(f"  {hint}", theme.MUTED_STYLE)])])).style(theme.BASE_STYLE),
                cols[1])
        else:
            self._row(frame, area, field, field.display(), hint)

    def _start_row(self, frame: Any, area: Any) -> None:
        focused = self.ring.is_focused(self.start)
        frame.render_widget(
            Paragraph(Text([option_line(
                f"{theme.play_mark()} Start teleop",
                "launch live control",
                focused=focused,
                label_width=20,
                width=area.width,
                label_unfocused_style=theme.TEXT_STYLE,
            )])).style(theme.BASE_STYLE),
            area,
        )

    def _hint(self, frame: Any, area: Any) -> None:
        frame.render_widget(
            keycap_hint_line([
                ("↑↓/jk", "move"),
                ("←→/hl", "adjust"),
                ("⏎", "edit/toggle/start"),
                ("q", "back"),
            ]),
            area,
        )


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY ``python -m lekiwi_tui teleop``: front scripts/teleop.sh with no knob
    flags (no form), so the script seeds Display / FPS from the yaml `teleop` block and
    slices the same block into --config_path. Passthrough ``extra`` is forwarded verbatim.
    Run through the shared headless runner (no app loop). The script is the SOLE argv source: no
    lerobot-teleoperate argv is assembled here.

    Ported from the Textual ``run_headless(app, extra)``; this port threads context through
    ``ctx`` for signature parity with the other screens (sync/provision), but teleop fronts
    the script with no flags so ``ctx`` is unused."""
    return runner.headless_run(["bash", str(TELEOP_SCRIPT), *extra])


__all__ = ["TeleopScreen", "run_headless", "HEADLESS_HOOK"]
