"""calibrate.py — CalibrateScreen, the immediate-mode port of the Textual CalibrateScreen.

An INFO + selector form (the teleop FORM archetype): pick the arm — the SO101 leader
(local, this laptop) or the LeKiwi follower (on the Pi over ssh) — see its device, where
its calibration is saved, and the interactivity note, then Start. Calibration is
INTERACTIVE (lerobot-calibrate prompts you to move the arm to positions and press ENTER),
so Start hands the real terminal to the child (an :class:`Invoke` flow → ``app.suspend`` →
``runner.suspend_run``): rock-solid prompts on the inherited TTY, then control returns
here with the child's exit code shown as a one-line result, so you can calibrate the
other arm too. The leader runs locally; the follower's
motors are on the Pi (the laptop client's ``calibrate()`` is a no-op), so it runs over
``ssh -t`` — that whole split lives in the carried-over ``scripts/calibrate.sh``, the SOLE
argv source this screen fronts (same as teleop fronts ``teleop.sh``).

Save paths (lerobot ``HF_LEROBOT_CALIBRATION``, default ``~/.cache/huggingface/lerobot/
calibration``): leader → ``teleoperators/so101_leader/<id>.json`` (laptop); follower →
``robots/lekiwi/<id>.json`` (on the Pi). The host must be stopped before a follower
calibration (it holds the serial bus); calibration persists across runs.

Keys: ↑↓/jk move (focusing a leader/follower row makes it the arm Start will calibrate),
⏎ on a row selects that arm and jumps to Start, ⏎ on Start suspends into the live
calibration, ``e`` opens lekiwi.yaml in ``$EDITOR``, q/esc backs out. ``handle_key`` is
pure → unit-testable with a synthetic ``Key`` (no ``Terminal``).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Line, Span

from .. import CFG_FILE, ROOT
from ..config import collapse_home, resolve_editor
from ..hostprobe import host_alive
from ..framework import theme
from ..framework.events import DOWN, ENTER, ESC, UP, Key
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from .chrome import draw_form_page, padded_line, plan_row, section_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# Absolute path to the calibrate launcher the TUI fronts (scripts/calibrate.sh). Mirrors
# teleop.py's TELEOP_SCRIPT: the SOLE source of the lerobot-calibrate / ssh argv (both
# targets) and of the remote follower bash — the screen only picks the arm + supplies the
# resolved cfg scalars as flags. ``runner.safe_argv`` appends ``--dry-run`` (R8) because
# this is a ``scripts/*.sh`` wrapper.
CALIBRATE_SCRIPT = ROOT / "scripts" / "calibrate.sh"


def _calib_base() -> str:
    """lerobot's calibration dir (utils/constants.py): HF_LEROBOT_CALIBRATION, else
    HF_LEROBOT_HOME/calibration, else HF_HOME/lerobot/calibration, else the ~/.cache
    default. Replicated here so we do not import the (heavy) lerobot package. Ported
    verbatim from the Textual screen."""
    if cal := os.environ.get("HF_LEROBOT_CALIBRATION"):
        return cal
    home = os.environ.get("HF_LEROBOT_HOME")
    if not home:
        hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        home = os.path.join(hf, "lerobot")
    return os.path.join(home, "calibration")


class CalibrateScreen(ScreenState):
    """Info page + leader/follower selector. Start suspends into the live interactive
    calibration, then returns here so you can calibrate the other arm too."""

    title = "calibrate"

    #: The focusable rows, in order (ported verbatim from the Textual screen's FIELDS).
    #: The selection index ``_fpos`` runs over these; the two arm rows double as a radio
    #: group that drives ``_arm`` (the arm Start calibrates).
    FIELDS = ["leader", "follower", "start"]

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        self._fpos = 0
        self._arm = "leader"   # follows the focused arm row; Start calibrates this one
        self._msg = ""         # post-calibration result line (set by _start after suspend)

    # ── input (pure → returns an Action) ──────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k"):
            self._move(-1)
            return Nothing
        if name in (DOWN, "j"):
            self._move(1)
            return Nothing
        if name == "e":
            # Edit lekiwi.yaml in $EDITOR (suspend into it, then return). Safe as a global
            # key here: no field on this screen consumes printable chars. ``safe_argv``
            # leaves this argv untouched (not a scripts/*.sh), so it opens the real file.
            return Invoke(lambda: self.app.suspend([resolve_editor(), str(CFG_FILE)]))
        if name == ENTER:
            cur = self.FIELDS[self._fpos]
            if cur in ("leader", "follower"):
                # Select the focused arm, then jump to Start (verbatim action_activate).
                self._arm = cur
                self._fpos = self.FIELDS.index("start")
                return Nothing
            # cur == "start": suspend into the live interactive calibration (the child
            # owns the TTY for the prompts), then control returns here. We need the
            # child's exit code to build _msg, which a bare Suspend Action discards, so
            # ENTER returns Invoke(self._start) and _start awaits app.suspend for the rc
            # (the provision/host pattern).
            return Invoke(self._start)
        return Nothing

    def _move(self, delta: int) -> None:
        """Rotate focus by *delta* (with index wrap). Landing on a leader/follower row
        makes it the arm Start will calibrate (verbatim action_move semantics)."""
        self._fpos = (self._fpos + delta) % len(self.FIELDS)
        cur = self.FIELDS[self._fpos]
        if cur in ("leader", "follower"):
            self._arm = cur
        self._msg = ""   # clear the prior result when navigating (parity: action_move)

    # ── the suspend flow (async; awaited by the App via Invoke) ────────────────
    async def _start(self) -> None:
        """Suspend into the live interactive calibration, then surface the rc.

        Suspending hands the child the real TTY so the bus prompts (and the follower's
        ``ssh -t`` remote PTY) work; ``app.suspend`` → ``runner.suspend_run`` makes the
        argv demo-safe (R8: ``--dry-run`` appended to the ``scripts/*.sh`` wrapper while
        ``DRY_RUN`` is on). Control returns here, and the child's exit code becomes a
        one-line result so you know it finished before calibrating the other arm (the
        provision/host pattern; a bare ``Suspend`` Action discards the rc)."""
        rc = await self.app.suspend(self._argv())
        self._msg = (
            f"✓ {self._arm} calibration finished"
            if rc == 0
            else f"calibration exited with code {rc}"
        )

    def _argv(self) -> list[str]:
        """Front scripts/calibrate.sh per the selected arm (verbatim _start argv).

        Leader runs locally and forwards ``self._extra`` (passthrough lerobot overrides);
        follower runs over ssh and takes NO extra (the remote bash is fixed) — that
        asymmetry matches the script + the Textual original. The script assembles the
        exact lerobot-calibrate / ssh tokens; ``runner`` appends ``--dry-run`` (R8)."""
        cfg = self.ctx.cfg
        if self._arm == "leader":
            return [
                "bash",
                str(CALIBRATE_SCRIPT),
                "--target", "leader",
                "--leader-port", cfg["LEADER_PORT"],
                "--leader-id", cfg["LEADER_ID"],
                *self._extra,
            ]
        return [
            "bash",
            str(CALIBRATE_SCRIPT),
            "--target", "follower",
            "--host", cfg["LEKIWI_HOST"],
            "--conda-env", cfg["CONDA_ENV"],
            "--robot-id", cfg["ROBOT_ID"],
            "--robot-type", cfg["ROBOT_TYPE"],
        ]

    # ── view (rebuilt fresh each frame; _body_lines is the testable view-model) ─
    _LABEL_W = 12

    def _leader_calib_age(self) -> str:
        """`calibrated <date> (N d ago)` from the LOCAL leader calibration file's mtime
        (a stale calibration is a robot-quality issue nothing else surfaces), or
        'not calibrated yet'. The follower's file lives on the Pi — no local stat."""
        path = (Path(_calib_base()) / "teleoperators" / "so101_leader"
                / f"{self.ctx.cfg['LEADER_ID']}.json")
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return "not calibrated yet"
        days = max(0, int((time.time() - mtime) / 86400))
        stamp = datetime.fromtimestamp(mtime).strftime("%b %d")
        return f"calibrated {stamp} ({days} d ago)"

    def _host_alive(self) -> bool | None:
        return host_alive(self.ctx)

    def _focused_hint(self) -> str:
        cfg = self.ctx.cfg
        cur = self.FIELDS[self._fpos]
        if cur == "leader":
            saves = os.path.join(_calib_base(), "teleoperators", "so101_leader",
                                 f"{cfg['LEADER_ID']}.json")
            return f"saves {collapse_home(saves)}"
        if cur == "follower":
            return (f"saves ~/.cache/…/calibration/robots/lekiwi/{cfg['ROBOT_ID']}.json"
                    " on the Pi")
        return "the terminal guides each joint: move to the limit, press Enter"

    def _body_lines(self, width: int = 100) -> list[Line]:
        cfg = self.ctx.cfg
        cur = self.FIELDS[self._fpos]
        w = int(width)
        radio_on = "◉ " if not theme.ASCII_MODE else "(*) "
        radio_off = "○ " if not theme.ASCII_MODE else "( ) "
        lines: list[Line] = [section_line("ARM  (pick one)")]
        lines.append(padded_line(
            [Span(theme.selector(cur == "leader"),
                  theme.TITLE_STYLE if cur == "leader" else theme.BASE_STYLE),
             Span(radio_on if self._arm == "leader" else radio_off,
                  theme.TITLE_STYLE if self._arm == "leader" else theme.MUTED_STYLE),
             Span(f"{'leader':<{self._LABEL_W}}",
                  theme.TITLE_STYLE if cur == "leader" else theme.MUTED_STYLE),
             Span(f"SO101 · local {cfg['LEADER_PORT']} · {cfg['LEADER_ID']}",
                  theme.TEXT_STYLE)],
            [Span(self._leader_calib_age(), theme.FAINT_STYLE),
             Span("  ", theme.BASE_STYLE)], w))
        lines.append(Line([
            Span(theme.selector(cur == "follower"),
                 theme.TITLE_STYLE if cur == "follower" else theme.BASE_STYLE),
            Span(radio_on if self._arm == "follower" else radio_off,
                 theme.TITLE_STYLE if self._arm == "follower" else theme.MUTED_STYLE),
            Span(f"{'follower':<{self._LABEL_W}}",
                 theme.TITLE_STYLE if cur == "follower" else theme.MUTED_STYLE),
            Span(f"LeKiwi arm · robot.id {cfg['ROBOT_ID']} · on {cfg['LEKIWI_HOST']} over ssh",
                 theme.TEXT_STYLE),
        ]))
        lines.append(Line([]))
        focused = cur == "start"
        # INVERSE host condition: the FOLLOWER needs the host STOPPED (it holds the
        # arm's serial bus); the leader is a local device and ignores host state.
        if self._arm == "follower" and self._host_alive() is True:
            plan_span = Span("⚠ host is running — Stop host first; it owns the follower's motors",
                             theme.WARN_STYLE)
        else:
            plan_span = Span(
                f"interactive {self._arm} calibration in the terminal · move joints to limits",
                theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Start", [plan_span], focused=focused))
        if self._msg:
            lines.append(Line([]))
            lines.append(Line([Span(f"  {self._msg}", self._msg_style())]))
        return lines

    def draw(self, frame: Any, area: Any) -> None:
        draw_form_page(frame, area, self.ctx, "calibrate", self._body_lines(area.width),
                       hint=self._focused_hint(),
                       keys=(("↑↓/jk", "move"), ("⏎", "select·start"),
                             ("e", "edit config"), ("q", "back")))

    def _msg_style(self):
        # ✓ "finished" = ok (green); "calibration exited (rc≠0)" = warn (amber). A
        # non-zero rc here is usually the user aborting an interactive calibration, not a
        # crash, so WARN (not ERR) — matches sync.py's _msg_style discrimination.
        return theme.OK_STYLE if self._msg.startswith("✓") else theme.WARN_STYLE


__all__ = ["CalibrateScreen"]
