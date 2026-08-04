"""host.py — HostLaunchScreen: configure + launch the Pi ZMQ host, streaming its log IN-PAGE.

Unlike the interactive actions (teleop/record/eval), the host is WATCH-ONLY: you start it,
watch its log, and stop it. So this screen does NOT suspend into a raw ssh terminal — it
streams the remote host's output into a log pane on the SAME page as the options, and a
single key stops it. This mirrors the original Textual app's streaming host sub-screen, but
rebuilt for pyratatui's single-thread model.

The streaming engine is the reusable :class:`~lekiwi_tui.framework.stream.StreamController`
(the child runs under a PTY whose master is drained by ``loop.add_reader`` on the asyncio
event loop — no worker thread ever touches the terminal, since ``Terminal``/``Frame`` are
PyO3 *unsendable*). This screen just embeds one controller, kicks it off from an ``Invoke``
flow, and delegates the log render + stop + status to it.

Three phases (driven by ``self.stream.phase``; ``idle`` == the pre-launch form):
  * **form**    — the Session-length field + a Launch button (the teleop form idiom).
  * **running** — the Session value (read-only) + a live log pane; ``s``/Ctrl+C stops it
                  (writes Ctrl+C to the remote PTY so the host's own trap stops gracefully,
                  then SIGKILLs the group after a grace window if it wedges).
  * **ended**   — the final log + the exit code; ⏎ relaunches, q goes back.

The launch is the original's real path: ship the host config to the Pi (``scp``), then
``ssh -t`` in with the ``host.sh emit-launch`` remote bash (``-t`` forces a remote PTY so
Ctrl+C reaches the host). Under ``--dry-run`` it shows the intent and does not ssh.
"""
from __future__ import annotations

import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import ROOT
from ..config import as_int, cameras_summary, cfg_for, cfg_get, resolve_workspace_path
from ..framework import runner, theme
from ..framework.events import (
    BACKTAB, DOWN, ENTER, ESC, LEFT, RIGHT, TAB, UP, Key, is_char,
)
from ..framework.focus import FocusRing
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.stream import StreamController
from ..framework.widgets import NumberField, TextField
from ..remote import RemoteValueError, validate_positive_int, validate_remote_name, validate_ssh_host
from .chrome import (
    plan_row,
    draw_slim_header, hint_slot_line, keycap_hint_line, mode_chip_spans,
    padded_line, section_line,
)

# The run_headless hook name (parity with the Textual app's HEADLESS_HOOK).
HEADLESS_HOOK = "run_headless"

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# The SOLE argv source: the carried-over host.sh emitter (fronted, never re-translated).
HOST_SCRIPT = ROOT / "scripts" / "host.sh"

# scp target on the Pi for the shipped host config (original _host_ssh _pi_cfg).
PI_CFG_PATH = "/tmp/lekiwi_host.yaml"

# The PincOpen plugin lives as a sibling project of this checkout (same default as
# sync.sh / pi_provision.sh LOCAL_PLUGIN).
PLUGIN_LOCAL = ROOT.parent / "lerobot_robot_lekiwi_pincopen"


def remote_script(conda_env, robot_id, connection_time, *, cfg_flag="", loop_flag="", robot_type="lekiwi_pincopen"):  # noqa: ANN001
    """The remote LAUNCH bash — `bash host.sh emit-launch …` stdout, used verbatim as the
    single ssh argv token (ported from the Textual original's remote_script). robot_type
    picks the host module via host.sh's whitelist (lekiwi_pincopen = the PincOpen plugin
    wrapper, lekiwi = stock lerobot); callers pass the ROBOT_TYPE config value."""
    conda_env = validate_remote_name(conda_env, "conda env")
    robot_id = validate_remote_name(robot_id, "robot id")
    robot_type = validate_remote_name(robot_type, "robot type")
    connection_time = validate_positive_int(connection_time, "connection time")
    return subprocess.check_output(
        ["bash", str(HOST_SCRIPT), "emit-launch",
         "--conda-env", conda_env, "--robot-id", robot_id,
         "--robot-type", robot_type,
         "--connection-time", str(connection_time),
         "--cfg-flag", cfg_flag, "--loop-flag", loop_flag],
        text=True,
    )


def build_host_ssh_argv(host, robot_id, connection_time, *, conda_env, cfg_flag="", loop_flag="", robot_type="lekiwi_pincopen"):  # noqa: ANN001
    """The exact ssh invocation: `ssh -t <host> <emit-launch remote bash>` (verbatim port).
    -t forces a remote PTY so Ctrl+C reaches the remote and its trap stops the host."""
    host = validate_ssh_host(host)
    remote = remote_script(conda_env, robot_id, connection_time,
                           cfg_flag=cfg_flag, loop_flag=loop_flag, robot_type=robot_type)
    return ["ssh", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3", "-t", host, remote]


def ship_plugin(host, *, repo, local=None):  # noqa: ANN001
    """rsync the PincOpen plugin to its place next to <repo> on the Pi (the same mirror
    recipe as sync.sh) so plugin edits are live at the next host launch without a full
    Sync. The plugin is editable-installed on the Pi, so new code applies immediately;
    only the FIRST install needs Set up Pi (this only moves bytes). ``local`` is the
    laptop-side plugin dir (the LOCAL_PLUGIN config value; sibling default when empty).
    Returns True on success; callers hard-fail the launch on False (a half-shipped
    robot is worse than no launch)."""
    host = validate_ssh_host(host)
    resolved = resolve_workspace_path(local or "")
    local_dir = Path(resolved) if resolved else PLUGIN_LOCAL
    if not local_dir.is_dir():
        return False
    r = str(repo).rstrip("/")
    remote = (r.rsplit("/", 1)[0] if "/" in r else ".") + "/lerobot_robot_lekiwi_pincopen/"
    argv = [
        "rsync", "-az", "--delete",
        "--exclude", ".git", "--exclude", "__pycache__/", "--exclude", "*.pyc",
        "--exclude", "*.egg-info/",
        f"{local_dir}/", f"{host}:{shlex.quote(remote)}",
    ]
    try:
        rc = subprocess.run(
            argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
    except OSError:
        rc = 1
    return rc == 0


def ship_host_config(host, *, doc=None):  # noqa: ANN001
    """scp the host config block (cfg_for('host')) to the Pi; return `--config_path=…` or ''
    on failure (the host then uses built-in defaults). Ported verbatim; -q + stderr hidden."""
    host = validate_ssh_host(host)
    src = cfg_for("host", doc=doc)
    if src is None:
        return ""
    try:
        rc = subprocess.run(["scp", "-q", str(src), f"{host}:{PI_CFG_PATH}"],
                            check=False, stderr=subprocess.DEVNULL).returncode
    except OSError:
        rc = 1
    return f"--config_path={PI_CFG_PATH}" if rc == 0 else ""


def _loop_flag(loop_hz):  # noqa: ANN001
    return f"--host.max_loop_freq_hz={loop_hz}" if loop_hz else ""


def _format_duration(seconds: float | int, *, ceiling: bool = False) -> str:
    """Format seconds as mm:ss, or h:mm:ss for long sessions."""
    total = math.ceil(seconds) if ceiling else int(seconds)
    total = max(0, total)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _session_progress(
    total_seconds: int, started_at: float | None, *, now: float | None = None
) -> tuple[float, float]:
    """Return ``(remaining_seconds, elapsed_fraction)`` for a host session."""
    total = max(float(total_seconds), 1.0)
    if started_at is None:
        return total, 0.0
    current = time.monotonic() if now is None else now
    elapsed = max(0.0, current - started_at)
    remaining = max(0.0, total - elapsed)
    fraction = max(0.0, min(1.0, elapsed / total))
    return remaining, fraction



class _Start:
    """The Launch pseudo-field — focusable, activated by the screen (Enter), not itself."""

    label = "Start"


class HostLaunchScreen(ScreenState):
    """Configure + launch the Pi host, streaming its log IN-PAGE (form/running/ended)."""

    title = "host"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        # Do NOT use ``app`` here; it can be None during root construction.
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        def_min = max(as_int(ctx.cfg["CONNECTION_TIME"], 600) // 60, 1)
        self.minutes = NumberField("Session", def_min, minimum=1, step=5, unit=" min")
        # Robot id (per-launch --robot.id override) — a free-text field, prefilled from cfg.
        self.robot = TextField(str(ctx.cfg["ROBOT_ID"]))
        # Host loop-freq Hz (per-launch --host.max_loop_freq_hz override), from the yaml.
        def_hz = as_int(cfg_get("host.host.max_loop_freq_hz", doc=ctx.doc), 30)
        self.hz = NumberField("Loop freq", def_hz, minimum=1, step=5, unit=" Hz")
        self.start = _Start()
        # Ring order == VISUAL order: (Session | Loop freq) row, then Robot id, Start.
        self.ring = FocusRing([self.minutes, self.hz, self.robot, self.start])
        self._msg = ""
        self._fresh = True
        # ── streaming engine (idle | running | ended; idle == the pre-launch form) ──
        # App-lifetime, NOT per-screen: the pump runs on the asyncio loop, so a running
        # host survives leaving this screen (q backgrounds it — go record!) and a fresh
        # HostLaunchScreen RE-ATTACHES to the same live log by adopting the shared
        # controller from ui_state.
        self.stream = ctx.ui_state.setdefault("host_stream", StreamController())
        self._area = None                         # last drawn area (for the PTY winsize)
        self._session_total_s = self.minutes.value * 60
        self._session_started_at: float | None = None
        # Re-entering while a backgrounded session runs: restore the countdown clock
        # from the announcement host launch published (see _start).
        info = ctx.ui_state.get("host_session")
        if self.stream.running and isinstance(info, dict):
            started = info.get("started_at")
            total = info.get("total_s")
            if isinstance(started, (int, float)) and isinstance(total, (int, float)):
                self._session_started_at = float(started)
                self._session_total_s = int(total)

    # ── input ─────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        # phase is idle | running | ended; idle falls through to the form handler.
        if self.stream.running:
            return self._handle_running(key)
        if self.stream.ended:
            return self._handle_ended(key)
        return self._handle_form(key)

    def _handle_form(self, key: "Key") -> Any:
        name = key.name
        cur = self.ring.current()
        editing_text = isinstance(cur, TextField)
        # ── focus nav (works regardless of which field is focused) ──
        if name == ESC:
            return Pop()
        if name in (TAB, DOWN):
            self._move(self.ring.next); return Nothing
        if name in (BACKTAB, UP):
            self._move(self.ring.prev); return Nothing
        if name == ENTER:
            if cur is self.start:
                return Invoke(self._start)
            if isinstance(cur, NumberField):
                self._msg = "" if cur.set_text(cur.editor.value) else f"✗ {cur.error}"
                if not cur.error:
                    cur.sync_editor(); self._fresh = True
            return Nothing
        # ── a focused TEXT field captures everything else (incl. j/k/h/l/q) as text ──
        if editing_text:
            if cur.handle_key(key):    # printable / Backspace / Left / Right / Home / End
                self._msg = ""
            return Nothing
        # ── non-text fields: q backs out; j/k move; ←→/hl adjust; digits type ──
        if name == "q":
            return Pop()
        if name == "k":
            self._move(self.ring.prev); return Nothing
        if name == "j":
            self._move(self.ring.next); return Nothing
        if name in (LEFT, "h", RIGHT, "l"):
            if isinstance(cur, NumberField):
                cur.step_by(-1 if name in (LEFT, "h") else 1); cur.sync_editor(); self._fresh = True
            self._msg = ""
            return Nothing
        if isinstance(cur, NumberField) and (is_char(key) or name == "Backspace"):
            if self._fresh and is_char(key):
                cur.editor.clear()
            self._fresh = False
            if cur.editor.handle_key(key):
                if cur.editor.value.strip().isdigit():
                    cur.set_text(cur.editor.value.strip())
                self._msg = ""
        return Nothing

    def _handle_running(self, key: "Key") -> Any:
        # s / Ctrl+C stop the host. q / Esc BACKGROUND it: the controller is app-lifetime
        # (ui_state["host_stream"]) and its pump runs on the loop, so the session keeps
        # going while you go record/teleop; the menu's robot chip carries the countdown,
        # and re-entering this screen re-attaches to the live log.
        if self.stream.handle_stop_key(key):
            return Nothing
        if key.name in (ESC, "q"):
            return Pop()
        return Nothing

    def _handle_ended(self, key: "Key") -> Any:
        if key.name in (ESC, "q"):
            return Pop()
        if key.name == ENTER:                      # relaunch with the same settings
            self.stream.reset(); self._msg = ""
            self._session_started_at = None
            return Invoke(self._start)
        return Nothing

    def _move(self, mover) -> None:
        mover()
        self._msg = ""
        cur = self.ring.current()
        if isinstance(cur, NumberField):
            cur.sync_editor()
        self._fresh = True

    # ── launch (the StreamController drives the PTY pump on the asyncio loop) ───
    def _build_launch_argv(self) -> list[str]:
        """Ship the plugin (rsync) + the host config (scp), then build the real
        `ssh -t … <remote bash>` argv. Returns [] to ABORT the launch when either ship
        fails: cameras and servo tuning live in that config, and running the robot with
        lerobot's built-in 2-camera defaults is worse than not launching.
        Factored out so a test can override it with a harmless local streamer."""
        cfg = self.ctx.cfg
        if not ship_plugin(cfg["LEKIWI_HOST"], repo=cfg["PI_REPO"], local=str(cfg["LOCAL_PLUGIN"])):
            self.app.notify("✗ could not ship the PincOpen plugin to the Pi — launch aborted "
                            "(check LOCAL_PLUGIN / run Set up Pi once)", "error")
            return []
        cfg_flag = ship_host_config(cfg["LEKIWI_HOST"], doc=self.ctx.doc)
        if not cfg_flag:
            self.app.notify("✗ could not ship the host config (cameras/tuning) to the Pi — "
                            "launch aborted (Pi reachable? check LEKIWI_HOST + ssh keys in Settings)", "error")
            return []
        robot_id = self.robot.value.strip() or str(cfg["ROBOT_ID"])
        return build_host_ssh_argv(
            cfg["LEKIWI_HOST"], robot_id, self.minutes.value * 60,
            conda_env=cfg["CONDA_ENV"], cfg_flag=cfg_flag,
            loop_flag=_loop_flag(str(self.hz.value)),
            robot_type=str(cfg["ROBOT_TYPE"]))

    async def _start(self) -> None:
        """Ship the config + build the launch argv, then hand it to the StreamController
        (which spawns it under a PTY and pumps its output into the in-page log via
        ``loop.add_reader``, so the App keeps drawing + handling keys while it streams).
        Under --dry-run, preview the command instead of sshing."""
        cfg = self.ctx.cfg
        session_seconds = self.minutes.value * 60
        self.ctx.cfg.values["CONNECTION_TIME"] = str(session_seconds)
        if runner.DRY_RUN:
            self.app.notify(
                f"[preview] would start the Pi host on {cfg['LEKIWI_HOST']} "
                f"({self.minutes.value} min)", "info")
            return
        try:
            argv = self._build_launch_argv()
        except RemoteValueError as exc:
            self.app.notify(f"Invalid remote setting: {exc} — fix it in Settings", "error")
            return
        if not argv:  # a ship step failed; _build_launch_argv already notified
            return
        rows, cols = (self._area.height, self._area.width) if self._area else (40, 110)
        self._session_total_s = session_seconds
        self._session_started_at = time.monotonic()
        await self.stream.start(
            argv, winsize=(rows, cols),
            running_status=f"running · session {_format_duration(session_seconds)}")
        if not self.stream.running:
            self._session_started_at = None
        else:
            # Announce the session window app-wide: the robot chip (chrome/menu) shows
            # the countdown next to the live ●. hostprobe.session_remaining reads it.
            self.ctx.ui_state["host_session"] = {
                "ends_at": time.monotonic() + session_seconds,
                "started_at": self._session_started_at,
                "total_s": session_seconds,
            }

    # ── view ────────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        self._area = area
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        # idle == the pre-launch form; running/ended share the stream view.
        if self.stream.running or self.stream.ended:
            self._draw_stream(frame, area)
        else:
            self._draw_form(frame, area)

    _LABEL_W = 14

    def _lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *fields: Any) -> Span:
        on = any(self.ring.is_focused(f) for f in fields)
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _num_text(self, field: "NumberField") -> tuple[str, Any]:
        if self.ring.is_focused(field):
            return f"{field.editor.value}█", theme.HIGHLIGHT_TEXT_STYLE
        return field.display(), theme.TEXT_STYLE

    def _focused_hint(self) -> str:
        cur = self.ring.current()
        if cur is self.minutes:
            return "how long the host stays available · ←→ ±5 · ⏎ type a number"
        if cur is self.hz:
            return "host control-loop limit · ←→ ±5 · ⏎ type a number"
        if cur is self.robot:
            return "robot identity for this launch · type to edit"
        return "ships the plugin + cameras/tuning, then starts the host"

    def _body_lines(self, width: int = 100) -> list[Line]:
        """The form's rows as the testable view-model (grouped, one hint slot)."""
        lines: list[Line] = [section_line("SESSION")]
        mv, ms = self._num_text(self.minutes)
        hv, hs = self._num_text(self.hz)
        lines.append(Line([
            self._gutter(self.minutes, self.hz),
            self._lab("Session", self.ring.is_focused(self.minutes)),
            Span(f"{mv:<10}", ms),
            Span("   ", theme.BASE_STYLE),
            self._lab("Loop freq", self.ring.is_focused(self.hz)),
            Span(hv, hs),
        ]))
        rf = self.ring.is_focused(self.robot)
        if rf:
            c = self.robot.cursor
            rspans = [Span(self.robot.value[:c], theme.HIGHLIGHT_TEXT_STYLE),
                      Span("█", theme.HIGHLIGHT_TEXT_STYLE),
                      Span(self.robot.value[c:], theme.HIGHLIGHT_TEXT_STYLE)]
        else:
            rspans = [Span(self.robot.value, theme.TEXT_STYLE)]
        pad = max(0, 10 - len(self.robot.value) - (1 if rf else 0))
        lines.append(Line([
            self._gutter(self.robot),
            self._lab("Robot id", rf),
            *rspans,
            Span(" " * pad, theme.BASE_STYLE),
            Span("   ", theme.BASE_STYLE),
            self._lab("Robot type", False),
            Span(str(self.ctx.cfg["ROBOT_TYPE"]), theme.FAINT_STYLE),
            Span("  (yaml)", theme.FAINT_STYLE),
        ]))
        lines.append(Line([]))
        focused = self.ring.is_focused(self.start)
        plan = (f"{self.minutes.value} min session on {self.ctx.cfg['LEKIWI_HOST']}"
                f" · ships plugin + cameras/tuning first")
        lines.append(plan_row("Start", plan, focused=focused))
        return lines

    def _draw_form(self, frame: Any, area: Any) -> None:
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1), Constraint.length(1)]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "host")
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(Paragraph(Text(self._body_lines(rows[2].width))
                                      ).style(theme.BASE_STYLE), rows[2])
        if self._msg:
            frame.render_widget(Paragraph(Text([Line([Span(f"  {self._msg}", theme.ERR_STYLE)])])
                                          ).style(theme.BASE_STYLE), rows[3])
        frame.render_widget(hint_slot_line(self._focused_hint(), rows[4].width), rows[4])

    def _stream_header_right(self) -> list[Span]:
        """The stream states' header status: ● HOST live / stopping / the outcome."""
        if self.stream.running and not self.stream.status.startswith("stopping"):
            state = [Span("● HOST", theme.OK_STYLE), Span("  session live", theme.MUTED_STYLE)]
        elif self.stream.running:
            state = [Span(self.stream.status, theme.WARN_STYLE)]
        else:
            st = self.stream.status
            state = [Span(st, theme.OK_STYLE if st.startswith("✓") else theme.WARN_STYLE),
                     Span(f"  session {_format_duration(self._session_total_s)}",
                          theme.MUTED_STYLE)]
        return state + [Span("   ", theme.BASE_STYLE), *mode_chip_spans()]

    def _session_meter_line(self, width: int) -> Line:
        """running: a countdown meter · stopping: the escalation status · ended: summary."""
        host = self.ctx.cfg["LEKIWI_HOST"]
        if self.stream.running and not self.stream.status.startswith("stopping"):
            remaining, fraction = _session_progress(self._session_total_s, self._session_started_at)
            fill, trough = theme.meter_segments(fraction, 22)
            remote_ip = cfg_get("_robot.remote_ip", doc=self.ctx.doc) or "?"
            return padded_line(
                [Span("  session   ", theme.MUTED_STYLE),
                 Span(fill, theme.TITLE_STYLE), Span(trough, theme.BORDER_STYLE),
                 Span(f"  {_format_duration(remaining, ceiling=True)} left"
                      f" / {_format_duration(self._session_total_s)}", theme.TEXT_STYLE)],
                [Span(f"client {remote_ip}", theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)],
                width)
        if self.stream.running:
            return Line([Span(f"  {self.stream.status}", theme.WARN_STYLE)])
        return padded_line(
            [Span(f"  session ended on {host} · "
                  f"{_format_duration(self._session_total_s)} elapsed", theme.MUTED_STYLE)],
            [Span("relaunch keeps these settings", theme.FAINT_STYLE),
             Span("  ", theme.BASE_STYLE)], width)

    def _facts_line(self, width: int) -> Line:
        """Shipped-config facts (cams · units · watchdog), verifiable against the log."""
        if not self.stream.running:
            return Line([])
        doc = self.ctx.doc
        units = "degrees" if cfg_get("host.robot.use_degrees", doc=doc) else "raw units"
        watchdog = cfg_get("host.host.watchdog_timeout_ms", doc=doc) or "?"
        facts = f"{cameras_summary(doc)} · {units} · watchdog {watchdog} ms"
        return padded_line([Span("  ", theme.BASE_STYLE)],
                           [Span(facts, theme.FAINT_STYLE), Span("  ", theme.BASE_STYLE)],
                           width)

    def _draw_stream(self, frame: Any, area: Any) -> None:
        rows = (Layout().direction(Direction.Vertical).constraints([
            Constraint.length(1), Constraint.length(1), Constraint.length(1),
            Constraint.length(1), Constraint.fill(1), Constraint.length(1),
        ]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "host", self._stream_header_right())
        frame.render_widget(Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(Paragraph(Text([self._session_meter_line(rows[2].width)])
                                      ).style(theme.BASE_STYLE), rows[2])
        frame.render_widget(Paragraph(Text([self._facts_line(rows[3].width)])
                                      ).style(theme.BASE_STYLE), rows[3])
        self.stream.draw_log(frame, rows[4],
                             title="host log · live" if self.stream.running else "host log")
        if self.stream.running:
            hint = [("s", "stop host"), ("Ctrl+C", "stop"), ("q", "menu · host keeps running")]
        else:
            hint = [("⏎", "relaunch"), ("q", "back")]
        frame.render_widget(keycap_hint_line(hint), rows[5])


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY host-launch (mirrors the original _launch_host headless path): ship the host
    config then ssh the launch directly (no app loop, no streaming). CONNECTION_TIME stays as
    configured; loop-freq Hz comes from the yaml."""
    cfg = ctx.cfg
    hz = cfg_get("host.host.max_loop_freq_hz", doc=ctx.doc)
    loop_hz = str(hz) if hz is not None else "30"
    try:
        if not runner.DRY_RUN and not ship_plugin(
            cfg["LEKIWI_HOST"], repo=cfg["PI_REPO"], local=str(cfg["LOCAL_PLUGIN"])
        ):
            print("✗ could not ship the PincOpen plugin to the Pi — launch aborted "
                  "(check LOCAL_PLUGIN / run Set up Pi once)", file=sys.stderr)
            return 2
        cfg_flag = "" if runner.DRY_RUN else ship_host_config(cfg["LEKIWI_HOST"], doc=ctx.doc)
        if not runner.DRY_RUN and not cfg_flag:
            print("✗ could not ship the host config (cameras/tuning) to the Pi — launch aborted",
                  file=sys.stderr)
            return 2
        argv = build_host_ssh_argv(
            cfg["LEKIWI_HOST"], cfg["ROBOT_ID"], cfg["CONNECTION_TIME"],
            conda_env=cfg["CONDA_ENV"], cfg_flag=cfg_flag, loop_flag=_loop_flag(loop_hz),
            robot_type=str(cfg["ROBOT_TYPE"]))
    except RemoteValueError as exc:
        print(f"Invalid remote setting: {exc}", file=sys.stderr)
        return 2
    return runner.headless_run(argv)


HostLaunch = HostLaunchScreen

__all__ = ["HostLaunchScreen", "HostLaunch", "run_headless", "HEADLESS_HOOK"]
