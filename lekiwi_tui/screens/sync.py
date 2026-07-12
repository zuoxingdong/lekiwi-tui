"""sync.py — SyncScreen ("Sync to Pi"), the THINNEST action-screen exemplar.

The immediate-mode port of the Textual ``SyncScreen``. It rsyncs the local lerobot clone
up to the Pi (the editable install there picks it up) by fronting ``scripts/sync.sh`` — a
routine SOURCE push, NO env rebuild / reinstall / smoke-test (use "Set up Pi" for first-time
bring-up or a dependency change). It is the lightweight sibling of provision.

Shape (the thinnest STREAMING archetype): a small read-only info/confirm page + ONE "Sync
now" row → a live in-page log. Unlike the form exemplar (teleop), there is no FocusRing and
no editable field — a single action.

Why STREAM (not suspend) now — sync is watch-only, like host-launch. The original suspended
into ``sync.sh`` so rsync owned the real TTY; this port instead streams rsync's output into
a log pane on the SAME page (via the shared :class:`~..framework.stream.StreamController`),
so the App keeps drawing + handling keys while it runs and a single key stops it. The
StreamController drives the child's PTY master from ``loop.add_reader`` on the asyncio loop —
no worker thread ever touches the terminal (``Terminal``/``Frame`` are PyO3 *unsendable*).

Three phases (``self.stream.phase``):
  * **idle**    — the info/confirm page + a "Sync now" row (⏎ launches).
  * **running** — the read-only target + a live log pane; ``s``/Ctrl+C/q/Esc stops it.
  * **ended**   — the final log + the exit code; ⏎ relaunches, q goes back.

CAVEAT (accepted limitation vs the old suspend): streaming assumes KEY-BASED ssh. An rsync
that prompts for a password can't be answered in-page (the log pane is read-only) — the old
suspend path handed rsync the real TTY so a prompt reached the user. Use key auth (the
contract the conda/lerobot provision stages already assume).

Host / repo come from the TUI config (``LEKIWI_HOST`` / ``PI_REPO``); ``sync_env`` exports them
so the script targets the same Pi the rest of the TUI uses (env > yaml > default — the in-memory
cfg leads the yaml until Save). The script is the SOLE argv source: ``build_sync_argv`` passes
NO flags (the no-arg invocation is the contract the SyncScreen + headless path share), so
``_argv`` deliberately does NOT append ``self._extra`` the way the form exemplar's ``_argv`` does.

Dry-run (contract R8): ``StreamController.start`` runs argv VERBATIM (it does NOT call
``runner.safe_argv``, unlike the old ``suspend`` path), so streaming ``sync.sh`` under
``runner.DRY_RUN`` would fire a REAL rsync to the Pi. So under ``DRY_RUN`` we NOTIFY the
intent and return without starting the stream (mirrors host-launch).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import CFG_FILE, ROOT
from ..config import Config, resolve_workspace_path
from ..framework import runner, theme
from ..framework.events import DOWN, ENTER, LEFT, RIGHT, UP, ESC, Key
from ..framework.modals import PromptModalState
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..framework.stream import StreamController
from .chrome import LABEL_W_ACTION, LABEL_W_FIELD, chip_spans, keycap_hint_line, option_line, runtime_chips
from .provision import checkout_provenance, local_checkout, pyproject_version
from ..remote import RemoteValueError, validate_ssh_host

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

#: Absolute path to the sync launcher this screen fronts.
SYNC_SCRIPT = ROOT / "scripts" / "sync.sh"

RULE = "─" * 54


def build_sync_argv(*, script: Any = SYNC_SCRIPT, install: bool = False) -> list[str]:
    """`bash sync.sh [--install]`. Run via bash (not the exec bit) so it works regardless
    of the file's mode. The script reads host/repo from lekiwi.yaml's `_launcher` (env
    var > yaml > default), so the TUI passes no destination flags. --install forces the
    editable installs on the Pi even when the dep fingerprints are unchanged (the
    screen's ‹ force › toggle)."""
    return ["bash", str(script), *(["--install"] if install else [])]


def sync_env(cfg) -> dict[str, str]:  # noqa: ANN001
    """sync.sh reads its knobs from the environment (env var > yaml > default); drive
    them from the TUI config so the script targets the same Pi + env + checkouts the
    rest of the TUI uses, even if the user changed them in Settings this session
    (the in-memory cfg can lead the yaml until Save). LOCAL_* pass resolved (absolute)
    and only when configured — empty keeps the script's sibling defaults."""
    env = {
        **os.environ,
        "LEKIWI_HOST": validate_ssh_host(cfg["LEKIWI_HOST"]),
        "PI_REPO": cfg["PI_REPO"],
        "CONDA_ENV": cfg["CONDA_ENV"],
    }
    local_repo = resolve_workspace_path(str(cfg["LOCAL_REPO"]))
    if local_repo:
        env["LOCAL_REPO"] = local_repo
    local_plugin = resolve_workspace_path(str(cfg["LOCAL_PLUGIN"]))
    if local_plugin:
        env["LOCAL_PLUGIN"] = local_plugin
    return env


class SyncScreen(ScreenState):
    """Info/confirm page → stream the rsync IN-PAGE (idle / running / ended).

    ⏎ on "Sync now" starts ``sync.sh`` under a PTY and streams rsync's output into a log
    pane on this page; ``s``/Ctrl+C/q/Esc stops it while running; ⏎ relaunches once ended,
    q goes back. q / Esc backs out (without syncing) only from the idle page."""

    title = "sync"

    #: Idle-page rows, in paint order. repo/plugin edit in a PromptModal (persisted to
    #: lekiwi.yaml `_launcher`, same storage Settings writes); install toggles per-run;
    #: sync launches. Focus starts on "sync" so a bare ⏎ still syncs immediately.
    FIELDS = ["repo", "plugin", "install", "sync"]

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        # Do NOT use ``app`` here; it can be None during root construction.
        self.app = app
        self.ctx = ctx
        # Stored for constructor parity with the other screens; deliberately NOT forwarded
        # into the argv — build_sync_argv takes only the install toggle (see _argv).
        self._extra = list(extra or [])
        self._msg = ""
        self._fpos = self.FIELDS.index("sync")
        self._force = False        # ‹ force ›: run sync.sh --install this run (not persisted)
        self._refresh_provenance()
        # The shared streaming engine: spawns rsync under a PTY, pumps its output into an
        # in-page log on the asyncio loop, routes Stop keys. Phase: idle | running | ended.
        self.stream = StreamController()
        self._area = None                          # last drawn area (for the PTY winsize)

    # ── provenance (recomputed on entry + after a path edit; 2 quick git calls) ─
    def _refresh_provenance(self) -> None:
        cfg = self.ctx.cfg
        self._repo_dir = local_checkout(cfg, "LOCAL_REPO", "lerobot")
        self._plugin_dir = local_checkout(cfg, "LOCAL_PLUGIN", "lerobot_robot_lekiwi_pincopen")
        try:
            self._repo_prov = checkout_provenance(self._repo_dir)
            self._plugin_prov = (
                pyproject_version(self._plugin_dir) if self._plugin_dir.is_dir() else "✗ not found"
            )
        except Exception:  # best-effort display; never block the screen
            self._repo_prov = self._plugin_prov = "?"

    # ── input (pure → returns an Action) ──────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        if self.stream.running:
            return self._handle_running(key)
        if self.stream.ended:
            return self._handle_ended(key)
        return self._handle_idle(key)

    def _handle_idle(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k"):
            self._fpos = (self._fpos - 1) % len(self.FIELDS)
            self._msg = ""
            return Nothing
        if name in (DOWN, "j"):
            self._fpos = (self._fpos + 1) % len(self.FIELDS)
            self._msg = ""
            return Nothing
        cur = self.FIELDS[self._fpos]
        if name in (LEFT, "h", RIGHT, "l") and cur == "install":
            self._force = not self._force
            return Nothing
        if name == ENTER:
            if cur in ("repo", "plugin"):
                # Invoke so the PromptModal flow runs on the App loop.
                return Invoke(self._edit_path)
            if cur == "install":
                self._force = not self._force
                return Nothing
            # cur == "sync": Invoke (not a bare Suspend) so _start streams on the loop.
            return Invoke(self._start)
        return Nothing

    # ── path editing (PromptModal → persist to lekiwi.yaml, the Settings storage) ─
    async def _edit_path(self) -> None:
        cur = self.FIELDS[self._fpos]
        key = "LOCAL_REPO" if cur == "repo" else "LOCAL_PLUGIN"
        ans = await self.app.run_modal(PromptModalState(
            key, value=str(self.ctx.cfg[key]),
            hint="path (absolute, ~, or relative to lekiwi-tui) · empty = sibling default · ⏎ apply · esc cancel"))
        if ans is None:                      # esc = cancel (blank ⏎ applies: back to auto)
            return
        ans = ans.strip()
        if ans == str(self.ctx.cfg[key]):
            return
        self.ctx.cfg.values[key] = ans
        try:
            Config(values=dict(self.ctx.cfg.values), env_set=set(self.ctx.cfg.env_set)).save(CFG_FILE)
            self._msg = f"✓ {key} saved to lekiwi.yaml"
        except OSError:
            self._msg = f"✗ could not write lekiwi.yaml ({key} kept for this session)"
        self._refresh_provenance()

    def _handle_running(self, key: "Key") -> Any:
        # s / Ctrl+C stop via the controller; q / Esc also stop (leaving while it runs
        # would orphan the rsync), but do NOT Pop — only stop. The stream ends on its own.
        if self.stream.handle_stop_key(key):
            return Nothing
        if key.name in (ESC, "q"):
            self.stream.stop()
        return Nothing

    def _handle_ended(self, key: "Key") -> Any:
        if key.name in (ESC, "q"):
            return Pop()
        if key.name == ENTER:                      # relaunch with the same settings
            self.stream.reset()
            self._msg = ""
            return Invoke(self._start)
        return Nothing

    async def _start(self) -> None:
        """The launch flow, run on the App loop: guard the script exists, then stream
        ``sync.sh`` under a PTY into the in-page log. Under ``runner.DRY_RUN`` preview the
        intent and return WITHOUT streaming (StreamController runs argv verbatim — it does
        NOT inject ``--dry-run`` like the old suspend path, so streaming here would fire a
        real rsync; see the module docstring)."""
        cfg = self.ctx.cfg
        if not SYNC_SCRIPT.exists():
            self._msg = f"✗ {SYNC_SCRIPT} not found"
            return
        if runner.DRY_RUN:
            self.app.notify(
                f"[preview] would copy the local LeRobot source to {cfg['LEKIWI_HOST']}:"
                f"{cfg['PI_REPO']}", "info")
            return
        try:
            env = sync_env(cfg)
        except RemoteValueError as exc:
            self._msg = f"✗ invalid remote setting: {exc} — fix it in Settings"
            return
        self._msg = ""
        winsize = (self._area.height, self._area.width) if self._area else (40, 110)
        await self.stream.start(
            self._argv(), env=env, winsize=winsize,
            running_status=f"syncing to {cfg['LEKIWI_HOST']}…")

    def _argv(self) -> list[str]:
        """The argv to stream: `bash sync.sh` plus the per-run ‹ force › toggle. No
        destination flags / no self._extra — the script reads host/repo/checkouts from
        the env (sync_env) + yaml (do NOT append self._extra here). Factored out as the
        test seam: a subclass overrides it with a harmless local streamer."""
        return build_sync_argv(install=self._force)

    # ── view (rebuilt fresh each frame) ───────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        self._area = area                          # remember the area for the PTY winsize
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        if self.stream.phase == "idle":
            self._draw_idle(frame, area)
        else:
            self._draw_stream(frame, area)

    def _draw_idle(self, frame: Any, area: Any) -> None:
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),   # 0 header
                Constraint.length(1),   # 1 heavy rule
                Constraint.length(1),   # 2 runtime chips
                Constraint.length(1),   # 3 info line
                Constraint.length(1),   # 4 gap
                Constraint.length(4),   # 5 repo + provenance, plugin + provenance
                Constraint.length(1),   # 6 dest line
                Constraint.length(1),   # 7 install toggle
                Constraint.length(1),   # 8 gap
                Constraint.length(1),   # 9 sync-now row
                Constraint.length(1),   # 10 result msg
                Constraint.fill(1),     # 11 spacer
                Constraint.length(1),   # 12 light rule
                Constraint.length(1),   # 13 hint
            ])
            .split(area)
        )
        self._header(frame, rows[0])
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1]
        )
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        frame.render_widget(self._info(), rows[3])
        self._source_rows(frame, rows[5])
        frame.render_widget(self._dest_line(), rows[6])
        frame.render_widget(self._install_row(rows[7].width), rows[7])
        frame.render_widget(self._sync_row(), rows[9])
        if self._msg:
            frame.render_widget(
                Paragraph(Text([Line([Span(self._msg, self._msg_style())])])).style(theme.BASE_STYLE),
                rows[10],
            )
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[12].width, light=True)).style(theme.RULE_LIGHT_STYLE), rows[12]
        )
        self._hint(frame, rows[13], [
            ("↑↓/jk", "move"), ("⏎", "edit/toggle/sync"), ("q", "back"),
        ])

    def _draw_stream(self, frame: Any, area: Any) -> None:
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),   # header
                Constraint.length(1),   # heavy rule
                Constraint.length(1),   # target (read-only context)
                Constraint.length(1),   # status
                Constraint.fill(1),     # log pane
                Constraint.length(1),   # hint
            ])
            .split(area)
        )
        self._header(frame, rows[0])
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1]
        )
        # Target, read-only, on the same page as the log.
        host = self.ctx.cfg["LEKIWI_HOST"]
        repo = self.ctx.cfg["PI_REPO"]
        frame.render_widget(
            Paragraph(Text([Line(chip_spans([
                ("target", f"{host}:{repo}", theme.CHIP_VALUE_STYLE),
            ]))])).style(theme.BASE_STYLE),
            rows[2],
        )
        # Status line: green when finished ok, amber while stopping / ended-nonzero.
        status = self.stream.status
        st_style = theme.OK_STYLE if status.startswith("✓") else (
            theme.WARN_STYLE if self.stream.ended else theme.STATUS_VALUE_STYLE)
        frame.render_widget(
            Paragraph(Text([Line([Span(status, st_style)])])).style(theme.BASE_STYLE),
            rows[3],
        )
        self.stream.draw_log(frame, rows[4], title="rsync")
        if self.stream.running:
            hint = [("s", "stop"), ("Ctrl+C", "stop"), ("q", "stop + back")]
        else:
            hint = [("⏎", "relaunch"), ("q", "back")]
        self._hint(frame, rows[5], hint)

    def _header(self, frame: Any, area: Any) -> None:
        frame.render_widget(
            Paragraph(Text([Line([
                Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
                Span("  sync to Pi", theme.SUBTITLE_STYLE),
            ])])).style(theme.BASE_STYLE),
            area,
        )

    def _info(self) -> Paragraph:
        return Paragraph(Text([
            Line([Span(
                "mirror laptop → Pi; the editable installs re-run automatically when deps change",
                theme.MUTED_STYLE,
            )]),
        ])).style(theme.BASE_STYLE)

    @staticmethod
    def _collapse(path: "Path") -> str:
        home = str(Path.home())
        s = str(path)
        return "~" + s[len(home):] if s.startswith(home) else s

    def _path_display(self, key: str, resolved: "Path") -> str:
        raw = str(self.ctx.cfg[key]).strip()
        return raw if raw else f"{self._collapse(resolved)}  (auto)"

    def _source_rows(self, frame: Any, area: Any) -> None:
        """The two editable source rows, each with a muted provenance sub-line (what
        version/branch that path currently holds — a wrong checkout is caught by eye)."""
        cur = self.FIELDS[self._fpos]
        lines = [
            option_line(
                "lerobot",
                self._path_display("LOCAL_REPO", self._repo_dir),
                "",
                focused=cur == "repo",
                label_width=LABEL_W_FIELD,
                width=area.width,
            ),
            Line([Span(f"           └ {self._repo_prov}", theme.MUTED_STYLE)]),
            option_line(
                "plugin",
                self._path_display("LOCAL_PLUGIN", self._plugin_dir),
                "",
                focused=cur == "plugin",
                label_width=LABEL_W_FIELD,
                width=area.width,
            ),
            Line([Span(f"           └ {self._plugin_prov}", theme.MUTED_STYLE)]),
        ]
        frame.render_widget(Paragraph(Text(lines)).style(theme.BASE_STYLE), area)

    def _dest_line(self) -> Paragraph:
        host = self.ctx.cfg["LEKIWI_HOST"]
        repo = self.ctx.cfg["PI_REPO"]
        return Paragraph(Text([Line(chip_spans([
            ("dest", f"{host}:{repo}  (+ plugin)", theme.CHIP_VALUE_STYLE),
        ]))])).style(theme.BASE_STYLE)

    def _install_row(self, width: int) -> Paragraph:
        return Paragraph(Text([option_line(
            "install",
            theme.choice("force" if self._force else "auto"),
            "auto = reinstall only when deps changed · force = reinstall this run",
            focused=self.FIELDS[self._fpos] == "install",
            label_width=LABEL_W_FIELD,
            width=width,
        )])).style(theme.BASE_STYLE)

    def _sync_row(self) -> Paragraph:
        return Paragraph(Text([option_line(
            f"{theme.play_mark()} Sync now",
            "mirror + install check over SSH",
            focused=self.FIELDS[self._fpos] == "sync",
            label_width=LABEL_W_ACTION,
            label_unfocused_style=theme.TEXT_STYLE,
        )])).style(theme.BASE_STYLE)

    def _msg_style(self):
        # ✓ = ok (green); the "✗ … not found" / other lines = warn (amber).
        return theme.OK_STYLE if self._msg.startswith("✓") else theme.WARN_STYLE

    def _hint(self, frame: Any, area: Any, pairs) -> None:
        frame.render_widget(keycap_hint_line(pairs), area)


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY direct run of the sync action: run sync.sh directly (no app loop). rsync
    needs no TTY for key-based ssh; a password prompt would still need one, so the no-TTY
    path assumes key auth (like the conda/lerobot provision stages).

    Ported from the Textual ``run_headless(app, extra)``; this port threads config through
    ``ctx`` (there is no ``app.cfg``), so it takes ``ctx`` and reads ``sync_env(ctx.cfg)``."""
    try:
        env = sync_env(ctx.cfg)
    except RemoteValueError as exc:
        print(f"Invalid remote setting: {exc}", file=sys.stderr)
        return 2
    return runner.headless_run(build_sync_argv(), env=env)


__all__ = ["SyncScreen", "build_sync_argv", "sync_env", "run_headless", "HEADLESS_HOOK"]
