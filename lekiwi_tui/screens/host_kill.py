"""host_kill.py — HostKillScreen: kill the LeKiwi ZMQ host on the Pi, streaming the kill's
output IN-PAGE.

The original "Kill host" was a one-shot suspend action: one ``ssh`` invocation that runs the
remote ``host.sh emit-kill`` payload to stop ``lekiwi_host`` for this ROBOT_ID. This port keeps
the SAME single-ssh main path but, like the host-launch screen, watches it IN-PAGE: the ssh runs
under a PTY whose master is drained on the asyncio loop (no worker thread — pyratatui's
``Terminal``/``Frame`` are PyO3 *unsendable*), and its output streams into a log pane on the
same page. All of that PTY + pump + Ctrl+C→SIGKILL machinery lives in the reusable
:class:`~lekiwi_tui.framework.stream.StreamController`; this screen just embeds one.

Two phases (driven by ``self.stream.phase``):
  * **idle**    — a small confirm view ("Kill the LeKiwi host on <host> (robot <id>)?") + a
                  single "▶ Kill host" button. ⏎ fires the kill; q / Esc backs out.
  * **running** — the read-only target + a live log pane; ``s``/Ctrl+C stops the kill (writes
                  Ctrl+C to the remote PTY, then SIGKILLs the group after a grace window);
                  q / Esc also stops (leaving while it runs would orphan it).
  * **ended**   — the final log + the exit code; ⏎ relaunches, q goes back. If the ssh kill
                  exited non-zero we show a muted hint to kill the local tunnels / power-cycle
                  (the original's ``pkill`` fallback, here display-only — actually shelling out
                  would break the single-thread streaming model).

The kill argv is built EXACTLY like the original: ``host.sh emit-kill --robot-id R`` stdout is
used verbatim as the single ssh argv token, behind ``ssh -o ConnectTimeout=5 <host> <remote>``.
Under ``runner.DRY_RUN`` we notify the intent and do NOT ssh from the interactive screen; direct
no-TTY dispatch goes through ``run_headless`` and prints the ssh argv without spawning it. Either
way, an ssh argv is not a ``scripts/*.sh`` wrapper, so ``runner`` could not make it dry-run-safe.
"""
from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..framework import runner, theme
from ..framework.events import ENTER, ESC, Key
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.stream import StreamController
from ..remote import RemoteValueError, validate_remote_name, validate_ssh_host
from .chrome import draw_slim_header, hint_slot_line, keycap_hint_line, mode_chip_spans, plan_row

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

#: The launcher that owns the remote KILL bash — the SOLE argv source (fronted, never
#: re-translated). ``host.sh emit-kill --robot-id R`` stdout is the verbatim ssh payload.
HOST_SCRIPT = ROOT / "scripts" / "host.sh"

#: Default PTY winsize when the screen has not been drawn yet (no ``_area``).
_FALLBACK_WINSIZE = (40, 110)


def build_host_kill_argv(ctx: "Context") -> list[str]:
    """Build the host-stop ssh argv from the current context."""
    host = validate_ssh_host(ctx.cfg["LEKIWI_HOST"])
    robot_id = validate_remote_name(ctx.cfg["ROBOT_ID"], "robot id")
    remote = subprocess.check_output(
        ["bash", str(HOST_SCRIPT), "emit-kill", "--robot-id", robot_id],
        text=True,
    )
    return ["ssh", "-o", "ConnectTimeout=5", host, remote]


class _Kill:
    """The Kill pseudo-button — the one focusable thing; activated by the screen (⏎)."""

    label = "Stop host"


class HostKillScreen(ScreenState):
    """Confirm + kill the Pi host, streaming the kill's output IN-PAGE (idle/running/ended)."""

    title = "stop host"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        # Do NOT use ``app`` here; it can be None during root construction.
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self.kill = _Kill()
        #: The embedded subprocess streamer — owns the PTY + pump + stop escalation.
        self.stream = StreamController()
        #: Last drawn area, captured at the top of draw so the PTY gets a sane winsize.
        self._area: Any = None

    # ── input (pure → returns an Action), dispatched on the stream phase ────────
    def handle_key(self, key: "Key") -> Any:
        if self.stream.running:
            return self._handle_running(key)
        if self.stream.ended:
            return self._handle_ended(key)
        return self._handle_confirm(key)

    def _handle_confirm(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name == ENTER:                          # ⏎ on the single button fires the kill
            return Invoke(self._start)
        return Nothing

    def _handle_running(self, key: "Key") -> Any:
        # s / Ctrl+C stop via the controller; q / Esc also stop (leaving would orphan it).
        if self.stream.handle_stop_key(key):
            return Nothing
        if key.name in (ESC, "q"):
            self.stream.stop()
        return Nothing

    def _handle_ended(self, key: "Key") -> Any:
        if key.name in (ESC, "q"):
            return Pop()
        if key.name == ENTER:                      # relaunch the kill
            self.stream.reset()
            return Invoke(self._start)
        return Nothing

    # ── launch (the Invoke flow, run on the App loop) ──────────────────────────
    def _build_kill_argv(self) -> list[str]:
        """Build the real kill argv, EXACTLY like the original ``_run_host_kill``: capture the
        remote KILL bash from ``host.sh emit-kill`` (verbatim), then front it with a single
        ``ssh -o ConnectTimeout=5 <host> <remote>``. Factored out so the self-test can override
        it with a harmless local streamer (the subprocess.check_output + ssh are the only host
        I/O in the launch path)."""
        return build_host_kill_argv(self.ctx)

    async def _start(self) -> None:
        """Build the kill argv and stream it via the controller. Under ``--dry-run`` preview
        the intent instead of sshing (the ssh argv is not a dry-run-capable wrapper, R8)."""
        host = self.ctx.cfg["LEKIWI_HOST"]
        robot_id = self.ctx.cfg["ROBOT_ID"]
        if runner.DRY_RUN:
            self.app.notify(
                f"[preview] would stop the Pi host on {host} ({robot_id})", "info")
            return
        try:
            argv = self._build_kill_argv()
        except (RemoteValueError, OSError, subprocess.SubprocessError) as exc:
            self.app.notify(f"Could not build host stop command: {exc}", "error")
            return
        winsize = (self._area.height, self._area.width) if self._area else _FALLBACK_WINSIZE
        # Retire the announced session window: the robot chip stops counting down; the
        # live ●/○ keeps tracking reality through the probe regardless of the kill's rc.
        self.ctx.ui_state.pop("host_session", None)
        await self.stream.start(argv, winsize=winsize,
                                running_status=f"stopping the host on {host}…")

    # ── view (rebuilt fresh each frame) ────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        self._area = area
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        if self.stream.phase == "idle":
            self._draw_confirm(frame, area)
        else:
            self._draw_stream(frame, area)

    def _draw_confirm(self, frame: Any, area: Any) -> None:
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1)]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "stop host")
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        host = str(self.ctx.cfg["LEKIWI_HOST"])
        robot_id = str(self.ctx.cfg["ROBOT_ID"])
        lines = [
            Line([Span("  stops the ", theme.MUTED_STYLE),
                  Span(host, theme.STATUS_VALUE_STYLE),
                  Span(" host processes (robot id ", theme.MUTED_STYLE),
                  Span(robot_id, theme.STATUS_VALUE_STYLE),
                  Span(")", theme.MUTED_STYLE)]),
            Line([Span("  motors torque off · cameras released · teleop/record/policy "
                       "sessions end", theme.FAINT_STYLE)]),
            Line([]),
            # The destructive action wears RED (the plan_row danger accent).
            plan_row("Stop host",
                     f"ssh {host} · graceful stop, kill -9 fallback · session chip clears",
                     focused=True, accent="danger", label_pad=4),
        ]
        frame.render_widget(Paragraph(Text(lines)).style(theme.BASE_STYLE), rows[2])
        frame.render_widget(hint_slot_line(
            "KK from any screen is the same stop (panic)", rows[3].width,
            keys=(("⏎", "stop"), ("q", "back"))), rows[3])

    def _stream_header_right(self) -> list[Span]:
        target = f"{self.ctx.cfg['LEKIWI_HOST']} · {self.ctx.cfg['ROBOT_ID']}"
        if self.stream.running:
            state = [Span("● STOPPING", theme.WARN_STYLE),
                     Span(f"  {target}", theme.MUTED_STYLE)]
        else:
            st = self.stream.status
            state = [Span(st, theme.OK_STYLE if st.startswith("✓") else theme.WARN_STYLE),
                     Span(f"  {target}", theme.MUTED_STYLE)]
        return state + [Span("   ", theme.BASE_STYLE), *mode_chip_spans()]

    def _draw_stream(self, frame: Any, area: Any) -> None:
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1), Constraint.length(1)]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "stop host", self._stream_header_right())
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        self.stream.draw_log(frame, rows[2],
                             title="host stop · live" if self.stream.running else "host stop")
        self._ended_note(frame, rows[3])
        if self.stream.running:
            hint = [("s", "stop"), ("Ctrl+C", "stop"), ("q", "stop + back")]
        else:
            hint = [("⏎", "relaunch"), ("q", "back")]
        frame.render_widget(keycap_hint_line(hint), rows[4])

    def _ended_note(self, frame: Any, area: Any) -> None:
        """On a non-zero ended exit, show the original's pkill-fallback advice (display-only —
        actually shelling out would break the single-thread streaming model)."""
        if self.stream.ended and self.stream.returncode not in (0,):
            note = (f"  host stop exited with code {self.stream.returncode}; "
                    "stop local SSH tunnels for this robot or power-cycle it")
            frame.render_widget(Paragraph(Text([Line([Span(note, theme.WARN_STYLE)])])
                                          ).style(theme.BASE_STYLE), area)


HostKill = HostKillScreen


def run_headless(ctx: "Context", extra: list[str]) -> int:
    """Direct no-TTY host stop: build the same ssh argv and hand it to the headless runner."""
    del extra
    try:
        argv = build_host_kill_argv(ctx)
    except RemoteValueError as exc:
        print(f"Invalid remote setting: {exc}", file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not build host stop command: {exc}", file=sys.stderr)
        return 1
    return runner.headless_run(argv)


__all__ = [
    "HostKillScreen",
    "HostKill",
    "build_host_kill_argv",
    "run_headless",
    "HEADLESS_HOOK",
]
