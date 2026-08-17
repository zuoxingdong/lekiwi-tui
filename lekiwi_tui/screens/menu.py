"""menu.py — MenuScreen, the immediate-mode port of the Textual MenuScreen.

The canonical EXEMPLAR every other screen copies: it takes ``(app, ctx)``, subclasses
``ScreenState``, rebuilds its whole view each frame in :meth:`draw`, and turns keys into
:class:`Action` s in :meth:`handle_key` (pure → unit-testable with synthetic ``Key``).

LAYOUT (agreed design, 2026-08-03; status grid 2026-08-15): a "◆ LEKIWI" header, an
accent rule, then a **2×2 status grid** — HARDWARE (robot, leader plug state, calibration
age, cameras) | SESSION (host up/down + a countdown meter) over SOFTWARE (lerobot, env) |
COMPUTE (gradient resource meters, 2×2) — on the same column split as the grid below, then four
**section cards two across** (HOST / COLLECT · DATA / LEARN), then SETUP as a single
strip, a light rule, and the hint line. Each action row is ``icon-cell + digit + label`` with an optional
right-aligned live badge (the per-action description line was dropped 2026-08-15: the
labels are self-explanatory to a daily user, and the full hints remain in the flat list
and the ``?`` help).

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

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from ..app_registry import ACTIONS, Action
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from ..framework.modals import ConfirmModalState
from ..framework.screen import Invoke, Nothing, Quit, RunAction, ScreenState

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

#: Status grid: 2×2 cards — HARDWARE | SESSION (4 content rows) over SOFTWARE | COMPUTE
#: (2 content rows; the meters pack 2×2 inside COMPUTE). Spacious puts a blank between
#: rows for breathing room; compact drops ONLY the blanks — never the cards — so a short
#: terminal keeps the grid instead of falling back to the flat list.
_ROW1_COMPACT, _ROW1_SPACIOUS = 4 + 2, 7 + 2
_ROW2_COMPACT, _ROW2_SPACIOUS = 2 + 2, 3 + 2

#: Body height (the fill row of draw()) each layout needs: two status rows + gap +
#: action rows with gaps + strip.
_GRID_TAIL_H = 1 + (2 + 2) + 1 + (3 + 2) + 1 + 1
_COMPACT_BODY_H = _ROW1_COMPACT + _ROW2_COMPACT + _GRID_TAIL_H
_SPACIOUS_BODY_H = _ROW1_SPACIOUS + _ROW2_SPACIOUS + _GRID_TAIL_H

#: Smallest terminal the card grid is legible in; below either, fall back to the flat
#: list. Height = the compact body + header/rule/gap above and rule/hint below (5).
_MIN_CARD_W = 72
_MIN_CARD_H = _COMPACT_BODY_H + 5

#: Eighth-block tips give the meters sub-cell resolution (index = eighths minus one).
_EIGHTHS = "▏▎▍▌▋▊▉█"

#: Meters wider than this read as decoration, not data; the slack goes to the value column.
_MAX_BAR_W = 28


def _grad_style(cell_frac: float) -> Any:
    """Bar-POSITION gradient: cells low in the track render green, the 60–85% band amber,
    the top red — so where the tip sits tells low/high at a glance, before the number."""
    if cell_frac >= 0.85:
        return theme.ERR_STYLE
    if cell_frac >= 0.60:
        return theme.WARN_STYLE
    return theme.OK_STYLE


def meter_spans(label: str, frac: float | None, value: str, *,
                label_w: int, value_w: int, bar_w: int,
                value_style: Any = None) -> list[Span]:
    """One meter row: ``label  ▮▮▮▯▯▯▯  value``.

    *frac* is 0..1 (clamped; ``None`` draws an empty track for a metric whose value
    exists but could not be read, e.g. a GPU whose utilisation query failed). The fill
    is colored by position (:func:`_grad_style`), the track is faint, and *value* is
    right-aligned into a fixed column so the numbers line up across rows. ASCII mode
    swaps the blocks for ``#``/``.`` and drops the sub-cell tip.
    """
    fill_ch, track_ch = ("#", ".") if theme.ASCII_MODE else ("█", "░")
    spans = [Span(label.ljust(label_w), theme.MUTED_STYLE), Span("  ", theme.BASE_STYLE)]
    f = 0.0 if frac is None else min(max(frac, 0.0), 1.0)
    if theme.ASCII_MODE:
        full, tip = round(f * bar_w), 0
    else:
        full, tip = divmod(round(f * bar_w * 8), 8)
    # Contiguous same-color runs, split at the gradient boundaries (cell counts).
    cut1, cut2 = int(bar_w * 0.60), int(bar_w * 0.85)
    for n, style in ((min(full, cut1), theme.OK_STYLE),
                     (min(full, cut2) - min(full, cut1), theme.WARN_STYLE),
                     (full - min(full, cut2), theme.ERR_STYLE)):
        if n > 0:
            spans.append(Span(fill_ch * n, style))
    used = full
    if tip:
        spans.append(Span(_EIGHTHS[tip - 1], _grad_style((full + 0.5) / bar_w if bar_w else 0.0)))
        used += 1
    if bar_w - used > 0:
        spans.append(Span(track_ch * (bar_w - used), theme.FAINT_STYLE))
    spans += [Span("  ", theme.BASE_STYLE),
              Span(value.rjust(value_w), value_style or theme.TEXT_STYLE)]
    return spans


# ── tiny TTL cache for the status cards' file-system probes ─────────────────────
# draw() runs every frame; an os.stat per frame is cheap but not free, and the values
# only matter at human timescales. Same pattern as datasets._STATS_CACHE.
_PROBE_CACHE: dict[str, tuple[float, Any]] = {}


def _probe_cached(key: str, ttl: float, fn: Any) -> Any:
    now = time.monotonic()
    hit = _PROBE_CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _PROBE_CACHE[key] = (now, value)
    return value


def _leader_port_present(port: str) -> bool:
    """Is the leader arm's serial device plugged in right now? THE classic teleop
    no-start, surfaced before the launch instead of inside its traceback."""
    return bool(port) and _probe_cached(f"leader:{port}", 2.0,
                                        lambda: os.path.exists(port))


def _leader_calib_age_days(leader_id: str) -> int | None:
    """Days since the LOCAL leader calibration file was written, or None when it does
    not exist. Same file calibrate.py stats for its age line; the follower's file lives
    on the Pi, so there is nothing local to stat for it."""
    from .calibrate import _calib_base

    path = Path(_calib_base()) / "teleoperators" / "so101_leader" / f"{leader_id}.json"

    def stat() -> int | None:
        try:
            return max(0, int((time.time() - path.stat().st_mtime) / 86400))
        except OSError:
            return None

    return _probe_cached(f"calib:{path}", 5.0, stat)

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
        if len(name) == 1 and name.isdigit():
            d = int(name) if name != "0" else 10  # phone-style: 0 = the 10th row
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
            return self._quit_action()
        return Nothing

    #: The confirm choice that stops the host — a class attribute so tests pin the exact label.
    STOP_AND_QUIT = "Stop the host gracefully, then quit"

    def _quit_action(self) -> Any:
        """Quit, gating on a live backgrounded host session.

        ``q`` on the host screen deliberately BACKGROUNDS a running session (see
        host.py) — but quitting the whole app would close the PTY master and the
        remote host would die on SIGHUP, skipping its finally-block (torque left on,
        cameras left claimed). So a live session gets a confirm modal first. The
        actual graceful stop runs AFTER the loop exits (``__main__`` calls
        ``shutdown_sync``): waiting here inside the Invoke flow would freeze the
        draw loop for the whole grace window, and the loop-bound ``stop()`` timers
        would not survive the quit anyway.
        """
        stream = self.ctx.ui_state.get("host_stream")
        if getattr(stream, "running", False):
            return Invoke(self._confirm_quit_with_host)
        return Quit()

    async def _confirm_quit_with_host(self) -> None:
        choice = await self.app.run_modal(ConfirmModalState(
            "The Pi host session is still running. Quit anyway?",
            [self.STOP_AND_QUIT, "Cancel"]))
        if choice == self.STOP_AND_QUIT:
            self.app.quit()

    def _go(self, *, dr: int = 0, dc: int = 0) -> None:
        target = _action_at(_move(_pos_of(ACTIONS[self._sel]), dr=dr, dc=dc))
        self._sel = ACTIONS.index(target)

    @property
    def selected(self) -> Action:
        """The currently-highlighted Action (used by tests / the dispatcher)."""
        return ACTIONS[self._sel]

    # ── the four status cards (draw() runs every frame: cached/polled sources only) ──
    @staticmethod
    def _spaced(rows: list[Line], spacious: bool) -> list[Line]:
        """Interleave blank lines between *rows* when the terminal has the height (the
        shared rhythm of all four status cards); compact returns them packed."""
        if not spacious:
            return rows
        out: list[Line] = []
        for i, row in enumerate(rows):
            if i:
                out.append(Line([Span("", theme.BASE_STYLE)]))
            out.append(row)
        return out

    #: Label column inside the HARDWARE card, sized to its widest label.
    _HW_LABEL_W = 13

    def _hardware_lines(self, width: int = 60, *, spacious: bool = True) -> list[Line]:
        """The HARDWARE card: the physical kit this laptop is about to drive — robot
        model, leader arm plug state, leader calibration age, the configured cameras BY
        NAME. One item per line, label column + value, plain glyphs only (no emoji).
        *width* bounds the cameras row so it elides ("front · wrist +1") instead of
        letting the panel clip a name mid-word."""
        from ..config import cfg_get

        ok_g, bad_g = ("+", "x") if theme.ASCII_MODE else ("✓", "✗")
        w = self._HW_LABEL_W
        rows: list[Line] = []

        robot = cfg_get("_launcher.ROBOT_TYPE", doc=self.ctx.doc)
        rows.append(Line([Span("robot type".ljust(w), theme.MUTED_STYLE),
                          Span(str(robot or "?"), theme.TEXT_STYLE)]))

        port = str(self.ctx.cfg["LEADER_PORT"])
        if _leader_port_present(port):
            rows.append(Line([Span("leader arm".ljust(w), theme.MUTED_STYLE),
                              Span(f"{ok_g} ", theme.OK_STYLE),
                              Span(port, theme.TEXT_STYLE)]))
        else:
            rows.append(Line([Span("leader arm".ljust(w), theme.MUTED_STYLE),
                              Span(f"{bad_g} ", theme.ERR_STYLE),
                              Span(f"{port} · unplugged", theme.ERR_STYLE)]))

        days = _leader_calib_age_days(str(self.ctx.cfg["LEADER_ID"]))
        if days is None:
            rows.append(Line([Span("calibration".ljust(w), theme.MUTED_STYLE),
                              Span(f"{bad_g} ", theme.ERR_STYLE),
                              Span("leader not calibrated", theme.ERR_STYLE)]))
        else:
            age = "today" if days == 0 else f"{days}d ago"
            rows.append(Line([Span("calibration".ljust(w), theme.MUTED_STYLE),
                              Span(f"{ok_g} ", theme.OK_STYLE),
                              Span(f"leader · {age}", theme.TEXT_STYLE)]))

        cams = cfg_get("_cameras", doc=self.ctx.doc)
        names = list(cams) if isinstance(cams, dict) else []
        if not names:
            rows.append(Line([Span("cameras".ljust(w), theme.MUTED_STYLE),
                              Span("none configured", theme.WARN_STYLE)]))
        else:
            # The names themselves, on one line; drop tail names into a "+N" before
            # the panel border would clip one mid-word.
            room = max(0, width - w)
            shown = list(names)
            text = " · ".join(shown)
            while shown and len(text) > room:
                shown.pop()
                text = " · ".join(shown) + f" +{len(names) - len(shown)}"
            if not shown:
                text = f"{len(names)} configured"   # no room for even one name
            rows.append(Line([Span("cameras".ljust(w), theme.MUTED_STYLE),
                              Span(text, theme.TEXT_STYLE)]))
        return self._spaced(rows, spacious)

    def _session_lines(self, width: int, *, spacious: bool = True) -> list[Line]:
        """The SESSION card: is the Pi host up, and how much session clock is left.

        The countdown renders as a gradient meter that FILLS as time elapses, so the
        tip crossing into amber/red is the "wrap up the take" signal. The probe and the
        session announcement are both already cached (hostprobe / ui_state)."""
        from ..config import cfg_get
        from ..hostprobe import get_probe, session_remaining

        probe = get_probe(self.ctx)
        alive = None
        if probe is not None:
            probe.poll()
            alive = probe.alive
        ip = cfg_get("_robot.remote_ip", doc=self.ctx.doc) or self.ctx.cfg["LEKIWI_HOST"]

        rows: list[Line] = []
        if alive is True:
            rows.append(Line([Span(f"{theme.status_dot()} host up", theme.OK_STYLE),
                              Span(f" · {ip}", theme.MUTED_STYLE)]))
            info = self.ctx.ui_state.get("host_session")
            left = session_remaining(self.ctx)
            total = info.get("total_s") if isinstance(info, dict) else None
            if left is not None and isinstance(total, (int, float)) and total > 0:
                value = f"{left // 60}:{left % 60:02d} left"
                bar_w = max(4, min(_MAX_BAR_W, width - len(value) - 4))
                rows.append(Line(meter_spans("", 1.0 - left / total, value, label_w=0,
                                             value_w=len(value), bar_w=bar_w,
                                             value_style=theme.TEXT_STYLE)))
            else:
                rows.append(Line([Span("no session clock", theme.FAINT_STYLE)]))
        elif alive is False:
            rows.append(Line([Span("○ " if not theme.ASCII_MODE else "o ", theme.MUTED_STYLE),
                              Span("host down", theme.MUTED_STYLE)]))
            rows.append(Line([Span(str(ip), theme.FAINT_STYLE)]))
        else:
            rows.append(Line([Span("host unknown", theme.FAINT_STYLE)]))
        return self._spaced(rows, spacious)

    def _compute_lines(self, width: int, *, spacious: bool = True) -> list[Line]:
        """The COMPUTE card: this machine's resources as gradient meters, packed 2×2
        (CPU | RAM over GPU | VRAM) so the card stays two rows tall.

        Its own card on purpose: an unlabelled percentage sitting under "host up" would
        read as the robot's. Sampling is throttled onto a background thread by
        :mod:`~lekiwi_tui.sysstat`; each meter is omitted rather than faked when it
        cannot be read — except a named GPU whose utilisation query failed, which keeps
        its cell (empty track, no number) so the hardware stays visible. The GPU's own
        FULL name is its label (e.g. "RTX 2050", never truncated).
        """
        from ..sysstat import get_sysstat

        stat = get_sysstat(self.ctx)
        stat.poll()
        s = stat.sample
        cells: list[tuple[str, float | None, str]] = []
        if s.cpu_pct is not None:
            cells.append(("CPU", s.cpu_pct / 100.0, f"{s.cpu_pct:.0f}%"))
        if s.ram_used_gb is not None and s.ram_total_gb:
            cells.append(("RAM", s.ram_used_gb / s.ram_total_gb,
                          f"{s.ram_used_gb:.1f}/{s.ram_total_gb:.0f} GB"))
        gpu_label = self.ctx.gpu_name or "GPU"
        if s.gpu_pct is not None:
            cells.append((gpu_label, s.gpu_pct / 100.0, f"{s.gpu_pct}%"))
        elif self.ctx.gpu_name:
            cells.append((gpu_label, None, ""))
        if s.vram_used_gb is not None and s.vram_total_gb:
            cells.append(("VRAM", s.vram_used_gb / s.vram_total_gb,
                          f"{s.vram_used_gb:.1f}/{s.vram_total_gb:.0f} GB"))
        if not cells:                            # nothing readable — say so, do not lie
            return [Line([Span("resources unavailable", theme.FAINT_STYLE)])]

        # Per-COLUMN label/value widths (left column carries the long GPU name, right
        # column the long GB values — a shared max would starve the bars), one shared
        # bar width so the meters read as one instrument.
        gap = 3
        cols = (cells[0::2], cells[1::2])
        label_ws = [max((len(c[0]) for c in col), default=0) for col in cols]
        value_ws = [max((len(c[2]) for c in col), default=0) for col in cols]
        fixed = sum(label_ws) + sum(value_ws) + 4 * len([c for c in cols if c])
        n_bars = 2 if cols[1] else 1
        bar_w = max(4, min(_MAX_BAR_W,
                           (width - (gap if cols[1] else 0) - fixed) // n_bars))
        lines: list[Line] = []
        for i in range(0, len(cells), 2):
            spans: list[Span] = []
            for j, (label, frac, value) in enumerate(cells[i:i + 2]):
                if j:
                    spans.append(Span(" " * gap, theme.BASE_STYLE))
                style = (self._load_style(100.0 * frac) if frac is not None
                         else theme.FAINT_STYLE)
                spans += meter_spans(label, frac, value, label_w=label_ws[j],
                                     value_w=value_ws[j], bar_w=bar_w, value_style=style)
            lines.append(Line(spans))
        return self._spaced(lines, spacious)

    @staticmethod
    def _load_style(pct: float) -> Any:
        """Green / amber / red by load, so a machine that cannot take another training run
        reads at a glance instead of needing the number parsed."""
        if pct >= 90:
            return theme.ERR_STYLE
        if pct >= 70:
            return theme.WARN_STYLE
        return theme.OK_STYLE

    #: Label column inside the SOFTWARE card, sized to its widest label ("conda env ").
    _SW_LABEL_W = 11

    def _software_lines(self, *, spacious: bool = True) -> list[Line]:
        """The SOFTWARE card: what every launch here actually runs — the lerobot version
        and the conda env. Machine constants from the config snapshot, except the lerobot
        row: a too-old lerobot does not fail early, it fails inside draccus mid-launch
        with the robot already involved. Cheap enough for a per-frame call (metadata
        lookup, cached for the process) and it flags a pre-release checkout too, since
        the version string alone cannot tell those apart."""
        from ..config import cfg_get
        from ..lerobot_env import summary

        out: list[Line] = []
        value, suffix, level = summary()
        warn = level == "warn"
        spans = [Span("lerobot".ljust(self._SW_LABEL_W), theme.MUTED_STYLE),
                 Span(("⚠ " if warn else "") + value,
                      theme.WARN_STYLE if warn else theme.TEXT_STYLE)]
        if suffix:
            # muted, so a pre-release marker informs without competing with the vitals
            spans.append(Span(suffix, theme.WARN_STYLE if warn else theme.MUTED_STYLE))
        out.append(Line(spans))

        env = cfg_get("_launcher.LAPTOP_ENV", doc=self.ctx.doc)
        if env:
            out.append(Line([Span("conda env".ljust(self._SW_LABEL_W), theme.MUTED_STYLE),
                             Span(str(env), theme.TEXT_STYLE)]))
        return self._spaced(out, spacious)

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
        # The blank lines inside the status cards are a luxury: keep them only when the
        # terminal has the height, and shed them BEFORE shedding the card grid itself.
        spacious = area.height >= _SPACIOUS_BODY_H
        row1_h = _ROW1_SPACIOUS if spacious else _ROW1_COMPACT
        row2_h = _ROW2_SPACIOUS if spacious else _ROW2_COMPACT
        # One content line per action, plus the card's own borders.
        heights = [max((len(c) for c in row), default=0) + 2 for row in _GRID]
        # Two status rows adjacent (their borders separate them), then the action grid.
        constraints = [Constraint.length(row1_h), Constraint.length(row2_h),
                       Constraint.length(1)]
        for h in heights:
            constraints += [Constraint.length(h), Constraint.length(1)]
        constraints += [Constraint.length(1), Constraint.fill(1)]    # setup strip, slack
        band = Layout().direction(Direction.Vertical).constraints(constraints).split(area)

        # The 2×2 status grid — HARDWARE | SESSION over SOFTWARE | COMPUTE — on the same
        # column split as the action-card rows below, so all card edges line up.
        def split(row_area: Any) -> Any:
            return (Layout()
                    .direction(Direction.Horizontal)
                    .constraints([Constraint.fill(1), Constraint.length(2),
                                  Constraint.fill(1)])
                    .split(row_area))

        top, bottom = split(band[0]), split(band[1])
        self._draw_panel(frame, top[0], "HARDWARE",
                         self._hardware_lines(max(0, top[0].width - 4),
                                              spacious=spacious))
        self._draw_panel(frame, top[2], "SESSION",
                         self._session_lines(max(0, top[2].width - 4), spacious=spacious))
        self._draw_panel(frame, bottom[0], "SOFTWARE",
                         self._software_lines(spacious=spacious))
        self._draw_panel(frame, bottom[2], "COMPUTE",
                         self._compute_lines(max(0, bottom[2].width - 4), spacious=spacious))
        for r, row in enumerate(_GRID):
            self._draw_card_row(frame, band[3 + r * 2], r, row)
        frame.render_widget(self._strip_line(), band[3 + len(_GRID) * 2])

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

    def _card_lines(self, cell: list[Action], width: int) -> list[Line]:
        """One line per action: the row itself. The muted description line was dropped by
        request (2026-08-15) — a daily driver does not need "Teleoperate" explained, and
        the full hint still shows in the flat list and the ``?`` help."""
        return [self._card_row(action, selected=action is ACTIONS[self._sel], width=width)
                for action in cell]

    def _card_row(self, action: Action, *, selected: bool, width: int) -> Line:
        digit = _JUMPABLE.index(action) + 1 if action in _JUMPABLE else 0
        # 10 daily-driver rows fit the digit row phone-style: the 10th shows (and is
        # jumped to by) "0". Beyond that a row simply has no keycap.
        keycap = f"{digit % 10}  " if 1 <= digit <= 10 else "   "
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
