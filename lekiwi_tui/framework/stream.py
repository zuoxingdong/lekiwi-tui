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
        # Optional output watchers, set by the screen BEFORE start():
        #   health_pattern — a compiled regex searched over every decoded chunk (pre
        #   newline-split, so tqdm-style \r updates are seen too); the LAST match's
        #   group(0) lands in .health (e.g. "27.3 fps"). The screens' loop gauge.
        #   line_hook — called with each COMPLETED line (ANSI-stripped); screens use it
        #   to track structured progress (record's "Recording episode N").
        self.health_pattern: "re.Pattern[str] | None" = None
        self.health = ""
        self.line_hook: "Any | None" = None

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
        if self.health_pattern is not None:
            # Search the raw chunk BEFORE \r removal: tqdm/loop meters redraw in place
            # with \r and rarely emit a newline, so the log-line path never sees them.
            for m in self.health_pattern.finditer(clean):
                self.health = m.group(0)
        text = self._partial + clean.replace("\r", "")
        *whole, self._partial = text.split("\n")
        self._lines.extend(whole)
        if self.line_hook is not None:
            for ln in whole:
                try:
                    self.line_hook(ln)
                except Exception:  # a watcher must never kill the pump
                    pass

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

    # ── input forwarding (interactive streams: the record HUD) ───────────────────
    #: Key.name → the terminal byte sequence the child's tty reader expects.
    _KEY_BYTES = {
        "Up": b"\x1b[A", "Down": b"\x1b[B", "Right": b"\x1b[C", "Left": b"\x1b[D",
        "Enter": b"\r", "Esc": b"\x1b", "Tab": b"\t", "Backspace": b"\x7f",
        "Home": b"\x1b[H", "End": b"\x1b[F",
    }

    def write_bytes(self, data: bytes) -> bool:
        """Write raw bytes to the child's PTY (its stdin). True iff written."""
        if self._master is None or not self.running:
            return False
        try:
            os.write(self._master, data)
            return True
        except OSError:
            return False

    def forward_key(self, key: Any) -> bool:
        """Encode a normalized :class:`~..events.Key` press as terminal input bytes and
        write it to the child. Presses only (that is all the App has); hold-to-move
        consumers in the child must use their own below-the-terminal backend (evdev).
        Returns True iff the key was representable AND written."""
        name = getattr(key, "name", "")
        if getattr(key, "ctrl", False) and len(name) == 1 and name.isalpha():
            return self.write_bytes(bytes([ord(name.lower()) & 0x1F]))
        seq = self._KEY_BYTES.get(name)
        if seq is not None:
            return self.write_bytes(seq)
        if len(name) == 1:
            return self.write_bytes(name.encode("utf-8", "replace"))
        return False

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
        """Render the log into *area* as a bordered, bottom-pinned pane (newest visible)."""
        block = theme.block(title, bordered=True)
        inner = block.inner(area)
        frame.render_widget(block, area)
        h = max(1, inner.height)
        body = list(self._lines) or ["(waiting for output…)"]
        scroll_y = max(0, len(body) - h)
        frame.render_widget(
            Paragraph(Text([Line([Span(ln, theme.TEXT_STYLE)]) for ln in body]))
            .style(theme.BASE_STYLE).scroll(scroll_y, 0), inner)


__all__ = ["StreamController"]
