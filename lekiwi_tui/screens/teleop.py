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

from pyratatui import Line, Span

from .. import ROOT
from ..config import as_int, cfg_get
from ..framework import runner, theme
from ..framework.events import (
    BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, SPACE, TAB, UP, Key,
)
from ..framework.focus import FocusRing
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, robot_runtime_issues
from ..hostprobe import host_alive
from .chrome import (
    draw_form_page,
    number_line,
    plan_row,
    section_line,
    setting_line,
    toggle,
)

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

TELEOP_SCRIPT = ROOT / "scripts" / "teleop.sh"


class _Toggle:
    """A tiny on/off field (the Display row). Mirrors NumberField's 'object the screen
    renders + adjusts' shape so the FocusRing treats it uniformly."""

    def __init__(self, label: str, value: bool) -> None:
        self.label = label
        self.value = value

    def toggle(self) -> None:
        self.value = not self.value


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
        self.fps = NumberField("FPS", as_int(cfg_get("teleop.fps", doc=doc), 30), minimum=1, step=5)
        self.start = _Start()
        # Ring order == VISUAL order: Duration · FPS · Display share the SESSION row.
        self.ring = FocusRing([self.dur, self.fps, self.display, self.start])
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
        if isinstance(cur, NumberField) and cur.type_key(key, fresh=self._fresh):
            self._fresh = False
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

    # ── view (rebuilt each frame; body lines are the testable view-model) ─────

    def _host_alive(self) -> bool | None:
        return host_alive(self.ctx)

    def _focused_hint(self) -> str:
        cur = self.ring.current()
        if cur is self.dur:
            return "0 = drive until Ctrl+C · ←→ ±5 · ⏎ type a number"
        if cur is self.fps:
            return "control-loop rate · ←→ ±5 · ⏎ type a number"
        if cur is self.display:
            return "show live Rerun view (off lowers CPU) · ←→/⏎ toggle"
        if self._host_alive() is False:
            return "preflight will stop the launch while the host is down"
        return "runs preflight, then hands the terminal to teleop (Ctrl+C ends it)"

    def _body_lines(self, width: int = 100) -> list[Line]:
        lines: list[Line] = [section_line("SESSION")]
        lines.append(number_line(self.dur, "Duration", self.ring.is_focused(self.dur),
                                 "0 = drive until you press Ctrl+C", width=width))
        lines.append(number_line(self.fps, "FPS", self.ring.is_focused(self.fps),
                                 "the robot's control-loop rate", width=width))
        lines.append(setting_line(
            "Display", toggle(self.display.value, focused=self.ring.is_focused(self.display)),
            "mirror the cameras in a window (off lowers CPU)",
            focused=self.ring.is_focused(self.display), width=width))
        lines.append(Line([]))
        # Start — the plan sentence; the host-down warning REPLACES it at the
        # decision point (the approved page-2 idiom; no banner rows).
        focused = self.ring.is_focused(self.start)
        if self._host_alive() is False:
            plan_span = Span("⚠ host not reachable — Start host first (menu 1)",
                             theme.WARN_STYLE)
        else:
            plan_span = Span("leader arm + wasd·zx base · no recording · full-TTY session",
                             theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Start", [plan_span], focused=focused))
        return lines

    def draw(self, frame: Any, area: Any) -> None:
        draw_form_page(frame, area, self.ctx, "teleop", self._body_lines(area.width),
                       msg=self._msg, hint=self._focused_hint())


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
