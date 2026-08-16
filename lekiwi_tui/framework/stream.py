"""stream.py — StreamController: run a subprocess on the asyncio loop, streaming its output
into an in-page log pane. The reusable core behind the watch-only screens (host-launch /
host-kill / sync): a screen embeds one, kicks it off from an ``Invoke`` flow, renders its log
with :meth:`draw_log`, and routes Stop keys via :meth:`handle_stop_key`.

Why this shape (pyratatui constraints): ``Terminal``/``Frame`` are PyO3 *unsendable*, so the
original's worker-thread PTY pump is illegal. Instead the child runs under a PTY whose master
is drained by ``loop.add_reader`` ON THE EVENT LOOP — no thread ever touches the terminal, and
the App keeps drawing + handling keys while output streams. A PTY (not a pipe) is used so
``ssh -t`` allocates a remote tty (clean Ctrl+C / trap-based graceful stop).

Lifecycle: ``idle`` → :meth:`start` → ``running`` → (child exits OR :meth:`stop`) → ``ended``.
Stop writes Ctrl+C to the PTY (the remote's own trap stops it), then SIGKILLs the process
group after a grace window if it wedges.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import re
import signal
import struct
import termios
import time
from collections import deque
from typing import Any

from pyratatui import Line, Paragraph, Span, Text

from . import theme

#: Strip ANSI escape sequences from the PTY output for a clean in-page log.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][AB0]")


class StreamController:
    """Drives one subprocess under a PTY, pumping its output into an in-page log.

    Embed one in a screen:

        self.stream = StreamController()
        # ... in an Invoke flow, on the live loop:
        await self.stream.start(argv, env=env, winsize=(area.height, area.width),
                                running_status="killing the host on lekiwi…")
        # ... in handle_key while running:
        if self.stream.handle_stop_key(key): return Nothing
        # ... in draw:
        self.stream.draw_log(frame, area, title="host log")
        # read self.stream.phase ('idle'|'running'|'ended'), .status, .returncode, .lines
    """

    def __init__(self, *, maxlen: int = 2000, grace: float = 5.0) -> None:
        self._lines: "deque[str]" = deque(maxlen=maxlen)
        self._grace = grace
        self._partial = ""
        self._proc: "asyncio.subprocess.Process | None" = None
        self._master: int | None = None
        self._loop: "asyncio.AbstractEventLoop | None" = None
        self._stopping = False
        self.phase = "idle"            # idle | running | ended
        self.status = ""
        self.returncode: int | None = None

    # ── introspection ──────────────────────────────────────────────────────────
    @property
    def lines(self) -> list[str]:
        """The captured log lines (a snapshot copy)."""
        return list(self._lines)

    @property
    def running(self) -> bool:
        return self.phase == "running"

    @property
    def ended(self) -> bool:
        return self.phase == "ended"

    def reset(self) -> None:
        """Clear state so the same controller can be re-:meth:`start`ed (relaunch)."""
        self._lines.clear()
        self._partial = ""
        self.status = ""
        self.returncode = None
        self._stopping = False
        self.phase = "idle"

    # ── start the pump (call from an Invoke async flow on the live loop) ─────────
    async def start(
        self,
        argv: "list[str]",
        *,
        env: "dict[str, str] | None" = None,
        winsize: tuple[int, int] = (40, 110),
        running_status: str = "running…",
    ) -> None:
        """Spawn *argv* under a PTY and begin streaming. Sets ``phase='running'`` (or
        ``'ended'`` immediately on a spawn error). Output flows into the log via the loop's
        reader callback; the screen just keeps drawing."""
        self.reset()
        master, slave = pty.openpty()
        rows, cols = winsize
        try:
            fcntl.ioctl(slave, termios.TIOCSWINSZ,
                        struct.pack("HHHH", max(int(rows), 4), max(int(cols), 20), 0, 0))
        except OSError:
            pass
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdin=slave, stdout=slave, stderr=slave,
                env=(dict(env) if env is not None else None), start_new_session=True)
        except OSError as exc:
            os.close(master)
            os.close(slave)
            self._lines.append(f"[error] could not launch: {exc}")
            self.phase = "ended"
            self.status = "launch failed"
            self.returncode = 127
            return
        os.close(slave)                # the child holds its own copy of the slave
        self._proc, self._master = proc, master
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        self.phase = "running"
        self.status = running_status
        self._loop.add_reader(master, self._on_readable)

    # ── the loop-side pump ───────────────────────────────────────────────────────
    def _on_readable(self) -> None:
        try:
            data = os.read(self._master, 8192)   # type: ignore[arg-type]
        except OSError:                           # EIO on Linux when the child exits
            data = b""
        if not data:
            self._on_eof()
            return
        clean = _ANSI_RE.sub("", data.decode("utf-8", "replace"))
        text = self._partial + clean.replace("\r", "")
        *whole, self._partial = text.split("\n")
        self._lines.extend(whole)

    def _on_eof(self) -> None:
        if self._master is not None and self._loop is not None:
            try:
                self._loop.remove_reader(self._master)
            except (OSError, ValueError):
                pass
            try:
                os.close(self._master)
            except OSError:
                pass
            self._master = None
        if self._partial:
            self._lines.append(self._partial)
            self._partial = ""
        if self._loop is not None and self._proc is not None:
            self._loop.create_task(self._reap())

    async def _reap(self) -> None:
        rc = await self._proc.wait() if self._proc else -1
        self.returncode = rc
        self.phase = "ended"
        self.status = (f"✓ finished (rc={rc})" if rc in (0, 130) else f"exited (rc={rc})")

    # ── stop ─────────────────────────────────────────────────────────────────────
    def stop(self) -> None:
        """Ctrl+C the child (its trap stops it gracefully), then SIGKILL the group after
        the grace window if it ignores it. Idempotent."""
        if self._stopping or self._master is None:
            return
        self._stopping = True
        self.status = "stopping (Ctrl+C → SIGKILL after grace)…"
        try:
            os.write(self._master, b"\x03")
        except OSError:
            pass
        if self._loop is not None:
            self._loop.call_later(self._grace, self._escalate)

    def _escalate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    def shutdown_sync(self, grace: float | None = None) -> str | None:
        """Gracefully stop a still-running child WITHOUT the asyncio loop.

        :meth:`stop` cannot run once ``asyncio.run`` has returned — its escalation
        timer (``call_later``) and the ``_reap`` task need the loop. Yet the app can
        exit while a backgrounded session is still live: quitting from the menu with
        a running host, or Ctrl+C straight out of the TUI. If nothing intervenes,
        interpreter exit closes the PTY master and the child chain dies on SIGHUP,
        which on the remote side skips the host's finally-block (torque stays on,
        cameras stay claimed) — the exact ungraceful stop ``host.sh emit-kill``
        exists to avoid.

        Same escalation as :meth:`stop`, done synchronously with os-level waits:
        Ctrl+C down the PTY, poll up to *grace* seconds (default: this controller's
        grace), SIGKILL the process group if it wedged. Returns ``None`` when there
        was nothing to stop, ``'stopped'`` on a graceful exit, ``'killed'`` when it
        took the SIGKILL.
        """
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return None
        pid = proc.pid
        if self._master is not None:
            try:
                os.write(self._master, b"\x03")
            except OSError:
                pass
        deadline = time.monotonic() + (self._grace if grace is None else grace)
        while time.monotonic() < deadline:
            if self._reaped(pid):
                self.phase = "ended"
                return "stopped"
            time.sleep(0.1)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            self.phase = "ended"
            return "stopped"      # exited between the last poll and the kill
        self._reaped(pid, block=True)
        self.phase = "ended"
        return "killed"

    @staticmethod
    def _reaped(pid: int, *, block: bool = False) -> bool:
        """os-level child reap check — asyncio's child watcher is gone with the loop."""
        try:
            done, _ = os.waitpid(pid, 0 if block else os.WNOHANG)
        except ChildProcessError:  # already reaped elsewhere
            return True
        return done == pid

    def handle_stop_key(self, key: Any) -> bool:
        """Route a Stop key (``s`` or Ctrl+C) while running. Returns True iff it handled
        the key (so the screen can `return Nothing`)."""
        if not self.running:
            return False
        if key.name == "s" or (key.name == "c" and getattr(key, "ctrl", False)):
            self.stop()
            return True
        return False

    # ── render ─────────────────────────────────────────────────────────────────
    def draw_log(self, frame: Any, area: Any, *, title: str = "log") -> None:
        """Render the log into *area* as a bordered, bottom-pinned pane (newest visible).
        The body is demoted to muted so the screen's own telemetry stays the primary
        layer; WARN/ERROR lines keep their status color."""
        block = theme.block(title, bordered=True)
        inner = block.inner(area)
        frame.render_widget(block, area)
        h = max(1, inner.height)
        body = list(self._lines) or ["(waiting for output…)"]
        scroll_y = max(0, len(body) - h)

        def _style_for(ln: str) -> Any:
            if "ERROR" in ln or "Traceback" in ln:
                return theme.ERR_STYLE
            if "WARN" in ln:
                return theme.WARN_STYLE
            return theme.MUTED_STYLE

        frame.render_widget(
            Paragraph(Text([Line([Span(ln, _style_for(ln))]) for ln in body]))
            .style(theme.BASE_STYLE).scroll(scroll_y, 0), inner)


__all__ = ["StreamController"]
