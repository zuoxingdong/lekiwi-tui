"""runner.py — running child CLIs, two mechanisms (contract R7 + R8 SAFETY).

This module provides the two ways the standalone TUI runs a lerobot CLI, built for
pyratatui's single-thread, own-the-loop model — there is NO worker thread anywhere
(``Terminal``/``Frame`` are PyO3
*unsendable*; calling them off the loop thread PANICS — see ``async_terminal.py``). All
concurrency is asyncio tasks on the one loop thread.

  suspend_run  — the lerobot CLI owns the real TTY (keyboard base-driving, Rerun, live
                 calibration prompts) while the TUI is torn down, exactly like the bash
                 script. The child owns the foreground process group, so Ctrl+C reaches
                 it directly. Used by every "suspend" action (and phase-1 host).
  stream_run   — streams a subprocess's stdout into an on-screen log (a ``RunScreen``)
                 that we draw every frame on the live loop. Stop ('s'/'q'/Ctrl+C) sends
                 SIGINT to the child's process GROUP, then escalates to SIGKILL after a
                 grace window. Used by "stream" actions (train / eval).

The action layer calls these as ``runner.suspend_run(...)`` / ``runner.stream_run(...)``
(module attribute) so argv tests can monkeypatch the module attribute and capture argv
without ever spawning a real lerobot CLI.

Why stream_run does NOT use ``App.run_modal``
---------------------------------------------
``run_modal``'s loop only calls ``handle_key`` on a *keypress* and only finishes when a
screen returns ``Pop``/``Quit`` — a subprocess that exits on its own is not a keypress,
so a normally-completing child would never deliver its exit code (the screen would hang
until a key happened to be pressed). ``stream_run`` therefore drives its OWN draw+poll
loop whose completion condition is "child exited OR Stop pressed". Driving
``app.terminal.draw(...)`` / ``poll_event(...)`` directly here is legal for the same
reason ``run_modal`` is: ``stream_run`` is reached from the action dispatcher while the
main ``events()`` generator is parked at its ``await``, so nothing else touches the
terminal concurrently. The ``RunScreen`` is driven loop-locally; it is never pushed onto
the App's nav stack.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from .events import is_char, key_from_pyratatui
from .screen import Pop, ScreenState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .app import App
    from .events import Key


# ══════════════════════════════════════════════════════════════════════════════
# Real execution vs dry-run (contract R8)
# ══════════════════════════════════════════════════════════════════════════════
# Default is REAL execution, matching the Textual app: an action shells out to its
# scripts/*.sh wrapper, which execs the lerobot CLI on the real TTY. Run the TUI from the
# lerobot env (the same env the Textual app runs in) so the scripts' ``python`` can import
# lerobot.
#
# Set ``DRY_RUN = True`` (via the ``--dry-run`` CLI flag, the menu's ``d`` toggle, or
# ``runner.DRY_RUN = True`` in code) to make every wrapper PRINT the argv it would run
# instead of executing it — a safe preview. It is read at call time, not import time, so
# a session can toggle it live. The self-test NEVER executes a real lerobot CLI.
DRY_RUN: bool = False

#: Seconds to wait after SIGINT before escalating to SIGKILL (bash GRACE=5).
GRACE: float = 5.0

#: Cap the in-memory scrollback so a chatty child can't grow the buffer unbounded.
SCROLLBACK_MAXLEN: int = 2000

# Frame interval (seconds) for the stream_run loop — ~30fps, matching the main loop, so
# the log repaints smoothly as lines stream in and Stop reacts within ~33ms. NOTE: this
# is enforced by an explicit ``asyncio.sleep`` per iteration, NOT by ``poll_event``'s
# ``timeout_ms`` — pyratatui's ``AsyncTerminal.poll_event`` is *non-blocking* (it does
# ``await asyncio.sleep(0)`` then polls with timeout 0 and discards the arg; see
# ``async_terminal.py``). Without our own sleep the loop would busy-spin a whole core.
_STREAM_FRAME_S: float = 1.0 / 30.0


def _supports_dry_run(argv: "Sequence[str]") -> bool:
    """True iff *argv* targets one of the carried-over ``scripts/*.sh`` wrappers, which
    are the only commands known to accept ``--dry-run`` (per R8). We do NOT blindly
    append ``--dry-run`` to every argv: a bare lerobot python entrypoint (e.g.
    ``python -m lerobot.record ...``) would reject an unknown flag and fail. The check
    is a conservative whitelist — only a token that looks like a ``.sh`` script under a
    ``scripts/`` path opts in."""
    for tok in argv:
        s = str(tok)
        if s.endswith(".sh") and ("scripts/" in s or os.path.basename(os.path.dirname(s)) == "scripts"):
            return True
    return False


def safe_argv(argv: "Sequence[str]") -> list[str]:
    """Return *argv* made safe for a demo run (contract R8).

    When :data:`DRY_RUN` is on AND the command supports it (a ``scripts/*.sh`` wrapper,
    see :func:`_supports_dry_run`), append ``--dry-run`` unless it is already present, so
    the wrapper prints what it WOULD do instead of touching hardware. When ``DRY_RUN`` is
    off, or the command is not a known dry-run-capable wrapper, the argv passes through
    unchanged (callers/UX still surface the argv + a DRY-RUN banner; see ``RunScreen``).

    This is a pure function (no I/O, no globals beyond the module flag) so it is the unit
    that the self-test exercises.
    """
    out = list(argv)
    if DRY_RUN and _supports_dry_run(out) and "--dry-run" not in out:
        out.append("--dry-run")
    return out


def _preview_command(argv: "Sequence[str]") -> None:
    """Print a conservative no-exec preview for commands that do not have a wrapper
    ``--dry-run`` mode. One token per line matches the launcher scripts' dry-run shape."""
    print("[preview] command would run:")
    for tok in argv:
        print(str(tok))


def headless_run(
    argv: "Sequence[str]",
    *,
    env: "Mapping[str, str] | None" = None,
) -> int:
    """Run *argv* from a no-TTY path, honoring :data:`DRY_RUN`.

    Dry-run-capable ``scripts/*.sh`` wrappers are still executed with ``--dry-run`` so they
    print their exact final lerobot argv. Raw commands such as ``ssh`` or ``lerobot-train``
    have no safe preview flag, so dry-run prints the command tokens and returns success
    without spawning anything.
    """
    requested = list(argv)
    real_argv = safe_argv(requested)
    run_env = dict(env) if env is not None else None

    if DRY_RUN:
        if _supports_dry_run(real_argv) and "--dry-run" in real_argv:
            return subprocess.run(real_argv, env=run_env).returncode
        _preview_command(requested)
        return 0

    return subprocess.run(real_argv, env=run_env).returncode


# ══════════════════════════════════════════════════════════════════════════════
# RunScreen — the streaming log view (a ScreenState, driven loop-locally)
# ══════════════════════════════════════════════════════════════════════════════


class RunScreen(ScreenState):
    """A full-screen scrolling log for one streamed subprocess.

    Port of the Textual ``RunScreen`` (which used a ``RichLog`` + a ``@work`` thread).
    Here it is a pure :class:`ScreenState`: it holds a bounded
    :class:`collections.deque` of output lines and renders them fresh each frame
    (a ``Paragraph`` whose ``.scroll`` is pinned to the bottom — newest visible). The
    subprocess pump and the SIGINT/SIGKILL escalation live in :func:`stream_run`'s loop,
    NOT in this class, so the view stays a pure view-model: feed it a synthetic
    :class:`~lekiwi_tui.framework.events.Key` and assert on the returned
    :class:`~lekiwi_tui.framework.screen.Pop`. ``handle_key`` returning ``Pop`` is
    the Stop request; :func:`stream_run` reads it (it does not go through the App).
    """

    #: Stop bindings, mirroring the original RunScreen ('s' / Ctrl+C / 'q').
    STOP_KEYS = frozenset({"s", "q"})

    def __init__(
        self,
        lines: "deque[str]",
        *,
        title: str,
        dry_run: bool = False,
        telemetry: "Callable[[int], list[Any]] | None" = None,
    ) -> None:
        self.title = title
        self._lines = lines
        self._dry_run = dry_run
        #: Optional live-telemetry hook: called each frame with the content width,
        #: returns pyratatui Lines rendered between the title bar and the log (the
        #: train screen's step meter + loss sparkline). Exceptions are swallowed —
        #: a bad parser must not take down the stream view.
        self._telemetry = telemetry
        #: Set True once Stop is requested; stream_run polls this to begin escalation.
        self.stop_requested = False
        #: Flipped by stream_run to update the title bar ("stopping…" / "[exit N]").
        self.status: str | None = None

    # ── view ────────────────────────────────────────────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        """Render a title bar + the scrollback. Built fresh every frame (immediate
        mode). The log ``Paragraph`` is scrolled so the LAST lines are visible, the
        equivalent of the original ``RichLog`` auto-scroll."""
        from pyratatui import (
            Constraint,
            Direction,
            Layout,
            Paragraph,
            Span,
            Line,
            Text,
        )

        from . import theme

        # Background fill so the modal-style full-screen view is opaque (R3: a modal
        # owns its whole area and fills its own background).
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)

        # Telemetry lines (if the hook yields any) sit between the title and the log.
        tele_lines: list[Any] = []
        if self._telemetry is not None:
            try:
                tele_lines = list(self._telemetry(int(area.width)))
            except Exception:  # noqa: BLE001 - telemetry must never kill the view
                tele_lines = []
        rows = Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(len(tele_lines)), Constraint.fill(1)]
        ).split(area)
        title_area, tele_area, log_area = rows[0], rows[1], rows[2]

        # Title bar: accent-on-bg, bold; a PREVIEW tag + any live status appended.
        title_spans = [Span(f" {self.title} ", theme.TITLE_STYLE)]
        if self._dry_run:
            title_spans.append(Span(" PREVIEW ", theme.WARN_STYLE.bold()))
        if self.status:
            title_spans.append(Span(f"  {self.status}", theme.MUTED_STYLE))
        frame.render_widget(
            Paragraph(Text([Line(title_spans)])).style(theme.BASE_STYLE),
            title_area,
        )
        if tele_lines:
            frame.render_widget(
                Paragraph(Text(tele_lines)).style(theme.BASE_STYLE), tele_area)

        # Body: the scrollback, bottom-pinned. We render the whole buffer and scroll so
        # the last `inner_h` lines show — Paragraph has no native tail/auto-scroll.
        # Log body wears muted (telemetry/title stay the primary layer); WARN/ERROR
        # lines keep their status color — same treatment as StreamController.draw_log.
        block = theme.block(bordered=True)
        inner = log_area.inner(1, 1)
        inner_h = max(1, inner.height)
        body_lines = list(self._lines)
        if not body_lines:
            body_lines = ["(waiting for output…)"]

        def _style_for(ln: str) -> Any:
            if "ERROR" in ln or "Traceback" in ln:
                return theme.ERR_STYLE
            if "WARN" in ln:
                return theme.WARN_STYLE
            return theme.MUTED_STYLE

        # Scroll offset = lines above the visible window (0 if everything fits).
        scroll_y = max(0, len(body_lines) - inner_h)
        text = Text([Line([Span(ln, _style_for(ln))]) for ln in body_lines])
        para = (
            Paragraph(text)
            .block(block)
            .style(theme.BASE_STYLE)
            .scroll(scroll_y, 0)
        )
        frame.render_widget(para, log_area)

    # ── input (pure; stream_run interprets) ───────────────────────────────────
    def handle_key(self, key: "Key") -> "Pop | None":
        """Map a Stop key to a :class:`Pop` (and flag :attr:`stop_requested`).

        Stop = 's', 'q', or Ctrl+C. Returning ``Pop`` here is how :func:`stream_run`
        learns the user wants to stop — it then runs the kill escalation. Repeat
        presses are idempotent (the flag is already set). Any other key is ignored
        (the log is non-interactive). Pure: no terminal, no subprocess access — so it
        is unit-testable with a synthetic ``Key``."""
        name = key.name
        is_ctrl_c = name == "c" and key.ctrl
        if is_ctrl_c or (is_char(key) and not key.ctrl and not key.alt and name in self.STOP_KEYS):
            self.stop_requested = True
            return Pop()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# stream_run — pump a subprocess into a RunScreen on the live loop (R7)
# ══════════════════════════════════════════════════════════════════════════════


async def stream_run(
    app: "App",
    argv: "Sequence[str]",
    *,
    title: str,
    env: "Mapping[str, str] | None" = None,
    on_line: "Callable[[str], None] | None" = None,
    telemetry: "Callable[[int], list[Any]] | None" = None,
) -> int:
    """Stream *argv* into an on-screen log and return its exit code (contract R7).

    Single-thread asyncio, no worker threads:

      * Spawn with :func:`asyncio.create_subprocess_exec` (``stdout=PIPE``,
        ``stderr=STDOUT``, ``start_new_session=True`` so the child leads its own
        process group — SIGINT/SIGKILL then hit the WHOLE group, not just the immediate
        child, since lerobot spawns Rerun/dataloader subprocs).
      * A reader coroutine does ``async for line in proc.stdout``, decodes each line,
        appends it to a bounded :class:`collections.deque` (``maxlen`` =
        :data:`SCROLLBACK_MAXLEN`) and calls *on_line* if given (e.g. a train-step →
        progress parser).
      * A :class:`RunScreen` renders that deque every frame in this function's own
        draw+poll loop (see the module docstring for why NOT ``run_modal``).

    Stop (the RunScreen returns :class:`Pop` on 's'/'q'/Ctrl+C) escalates INLINE:
    ``os.killpg(pgid, SIGINT)`` → ``await asyncio.wait_for(proc.wait(), GRACE)`` →
    ``os.killpg(pgid, SIGKILL)`` if it wedges (the bash GRACE escalation). The loop also
    ends when the child exits on its own. *argv* is passed through :func:`safe_argv`
    first, so a dry-run-capable wrapper gets ``--dry-run`` when :data:`DRY_RUN` is on.

    Returns the child's exit code (negative ``-N`` for signal-killed, per POSIX).
    """
    if app.terminal is None:
        raise RuntimeError("stream_run called outside the running App loop")
    term = app.terminal

    real_argv = safe_argv(argv)
    lines: "deque[str]" = deque(maxlen=SCROLLBACK_MAXLEN)
    screen = RunScreen(lines, title=title, dry_run=(real_argv != list(argv)),
                       telemetry=telemetry)
    screen.on_enter()

    # Spawn. A launch failure is surfaced in the log + a 127 exit, like the original.
    try:
        proc = await asyncio.create_subprocess_exec(
            *real_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=(dict(env) if env is not None else None),
            start_new_session=True,
        )
    except OSError as exc:
        lines.append(f"[error] could not launch: {exc}")
        # Still paint the error once so the user sees it, then return 127.
        _safe_draw(app, term, screen)
        screen.on_exit()
        return 127

    async def _pump() -> None:
        """Read stdout line-by-line into the deque (+ on_line). Runs as a task on the
        same loop; never touches the terminal."""
        assert proc.stdout is not None
        async for raw in proc.stdout:
            text = raw.decode("utf-8", "replace").rstrip("\n")
            lines.append(text)
            if on_line is not None:
                try:
                    on_line(text)
                except Exception:  # noqa: BLE001 - a bad parser must not kill the pump
                    pass

    reader = asyncio.create_task(_pump())
    wait_task = asyncio.create_task(proc.wait())
    stopping = False

    try:
        while True:
            _safe_draw(app, term, screen)

            # Poll one key. pyratatui's AsyncTerminal.poll_event is non-blocking (it
            # ignores timeout_ms — see async_terminal.py), so we pace the loop with our
            # own sleep below; passing 0 is honest about the non-blocking semantics.
            ev = await term.poll_event(timeout_ms=0)
            if ev is not None:
                key = key_from_pyratatui(ev)
                action = screen.handle_key(key)
                if isinstance(action, Pop):
                    screen.stop_requested = True

            # Begin the SIGINT→SIGKILL escalation once, on the first Stop request.
            if screen.stop_requested and not stopping:
                stopping = True
                screen.status = "stopping (SIGINT, then SIGKILL after grace)…"
                _safe_draw(app, term, screen)
                await _escalate(proc, wait_task)

            # Done when the child has exited (whether on its own or after escalation).
            if wait_task.done():
                break

            # Pace the frame (~30fps). The reader/wait tasks keep progressing during
            # this sleep; without it the loop would busy-spin (poll_event never blocks).
            await asyncio.sleep(_STREAM_FRAME_S)
    finally:
        # Tear down the reader; reap the child if somehow still alive (defensive). The
        # reader may still be pending here (wait_task can finish a tick before the
        # reader processes stdout EOF), so cancelling it raises CancelledError on the
        # await — which is a BaseException, NOT an Exception, so _suppress MUST list it
        # explicitly or it would propagate and skip `return rc`, tearing down the app on
        # a perfectly normal Stop/exit.
        reader.cancel()
        with _suppress(asyncio.CancelledError, Exception):
            await reader
        if proc.returncode is None:
            with _suppress(ProcessLookupError):
                proc.kill()
            with _suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(wait_task, timeout=2.0)
        screen.on_exit()

    rc = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    return rc


async def _escalate(
    proc: "asyncio.subprocess.Process", wait_task: "asyncio.Task[int]"
) -> None:
    """SIGINT the child's process group, wait up to :data:`GRACE`, then SIGKILL it if it
    ignored SIGINT (the bash cleanup() escalation). Group-signalling matches
    ``start_new_session=True``: lerobot's Rerun/dataloader children die too. Idempotent
    on a process that has already exited (``ProcessLookupError`` is swallowed)."""
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    with _suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGINT)
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=GRACE)
        return
    except asyncio.TimeoutError:
        pass
    # Did not exit within the grace window → force-kill the whole group.
    with _suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _safe_draw(app: "App", term: Any, screen: "RunScreen") -> None:
    """Paint *screen* full-screen via the App's shared frame routine (keeps toasts +
    overlays consistent with the rest of the app). Drawn as a modal layer (opaque,
    full-area). Same package → calling ``app._draw_frame`` is intended, not a reach."""
    term.draw(lambda frame: app._draw_frame(frame, active=screen, is_modal=True))


# ══════════════════════════════════════════════════════════════════════════════
# suspend_run — hand the real TTY to a child (R7)
# ══════════════════════════════════════════════════════════════════════════════


def suspend_run(
    app: "App",
    argv: "Sequence[str]",
    *,
    env: "Mapping[str, str] | None" = None,
    pause: bool = False,
) -> int:
    """Suspend the TUI and hand the real terminal to a child process, then return its
    exit code (contract R7).

    This is the bash behaviour for free: while suspended, the child owns the foreground
    process group, so Ctrl+C goes to the CHILD — calibration prompts, the teleop
    keyboard loop, and Rerun all work exactly as they did in the shell script.

    Mechanism (pyratatui has no ``AsyncTerminal.suspend``)
    ------------------------------------------------------
    Raw mode + the alt-screen are process-global terminal state, so we need a teardown
    CALL, not to unwind the ``async with`` (we can't, from sync code mid-loop). The live
    ``AsyncTerminal`` is a thin Python wrapper (see ``async_terminal.py``) holding the
    real PyO3 ``Terminal`` in ``._term``; that sync Terminal exposes ``restore()`` (leave
    raw/alt) and ``__enter__()`` (re-arm raw/alt). We therefore:

      1. ``restore()`` the underlying Terminal → back to the normal cooked TTY,
      2. run ``argv`` blocking with :class:`subprocess.Popen` (the child owns the TTY),
      3. ``__enter__()`` the SAME Terminal again + ``clear()`` → re-enter the alt-screen,

    all in a ``try/finally`` so a throw NEVER strands the user in raw mode. Re-using the
    AsyncTerminal's own ``._term`` (rather than constructing a fresh ``Terminal()``)
    matters: the AsyncTerminal still references it, so its ``__aexit__`` restores cleanly
    at app shutdown. The restore-to-re-enter handshake is inferred from pyratatui's
    wrapper shape and covered by runtime checks.

    *env*, when given, is the FULL environment for the child (callers build it from
    ``os.environ`` + overrides); ``None`` inherits the parent env. *pause=True* waits for
    Enter AFTER the child exits, while still restored, so the child's final output
    (errors, a long setup log) stays on screen instead of being wiped by the resume
    repaint. *argv* is passed through :func:`safe_argv` so PREVIEW mode can append wrapper
    ``--dry-run`` flags when requested.

    If the App is not currently running a live terminal, we fall back to the same blocking
    child runner (nothing to suspend) so the action still works headlessly.
    """
    real_argv = safe_argv(argv)
    run_env = dict(env) if env is not None else None

    async_term = app.terminal
    raw_term = getattr(async_term, "_term", None) if async_term is not None else None

    if raw_term is None:
        # No live terminal (e.g. called outside run()) — just run the child.
        return _blocking_child(real_argv, env=run_env)

    try:
        # 1) Leave raw mode + the alt-screen so the child gets a normal cooked TTY.
        raw_term.restore()
        # 2) Run the child, blocking, with the real terminal as its controlling tty.
        rc = _blocking_child(real_argv, env=run_env)
        # Pause when asked OR on a non-zero exit, so a failure's output stays on screen
        # instead of being wiped by the re-enter+repaint (the "flash, then nothing" fix:
        # a quick error no longer vanishes — the user sees the exit code + the message).
        if (pause and rc != 130) or rc not in (0, 130):
            try:
                input(f"\n[exit {rc}]  press Enter to return to the TUI… ")
            except (EOFError, KeyboardInterrupt):
                pass
        return rc
    finally:
        # 3) ALWAYS re-arm raw mode + the alt-screen on the SAME Terminal, then clear,
        #    so the main loop resumes onto a fresh frame. Guarded so a re-entry failure
        #    can't mask the child's exit / leave a half-broken state silently.
        with _suppress():
            raw_term.__enter__()
        with _suppress():
            raw_term.clear()


# ── tiny helpers ──────────────────────────────────────────────────────────────


def _blocking_child(argv: "Sequence[str]", *, env: "Mapping[str, str] | None" = None) -> int:
    """Run a foreground child and convert operator Ctrl+C into exit code 130.

    During ``suspend_run`` the TUI and child share the foreground process group. The
    terminal sends Ctrl+C to both, but ``asyncio.run`` has a parent-side SIGINT handler
    that would cancel the main app task even if the child shuts down cleanly. While the
    child owns the real TTY, temporarily replace that parent handler with one that only
    records the cancellation and nudges the child. The child resets SIGINT to default
    before exec, so lerobot still receives Ctrl+C normally.
    """
    proc: subprocess.Popen[Any] | None = None
    interrupted_at: float | None = None

    def note_sigint(_signum: int, _frame: Any) -> None:
        nonlocal interrupted_at
        if interrupted_at is None:
            interrupted_at = time.monotonic()
        if proc is not None:
            with _suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)

    old_handler: Any = None
    handler_installed = False
    try:
        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, note_sigint)
        handler_installed = True
    except (OSError, ValueError):
        # signal.signal only works in the main thread. The TUI path is main-threaded;
        # this fallback keeps the helper usable in unusual test/headless contexts.
        pass

    try:
        proc = _popen_child(argv, env=env)
        while True:
            try:
                rc = proc.wait(timeout=0.2)
                return 130 if interrupted_at is not None else rc
            except subprocess.TimeoutExpired:
                if interrupted_at is not None and time.monotonic() - interrupted_at >= GRACE:
                    _terminate_child(proc)
                    return 130
    except KeyboardInterrupt:
        if proc is not None:
            with _suppress(ProcessLookupError):
                proc.send_signal(signal.SIGINT)
            _reap_interrupted_child(proc)
        return 130
    finally:
        if handler_installed:
            with _suppress(Exception):
                signal.signal(signal.SIGINT, old_handler)


def _popen_child(
    argv: "Sequence[str]",
    *,
    env: "Mapping[str, str] | None" = None,
) -> "subprocess.Popen[Any]":
    kwargs: dict[str, Any] = {"env": env}
    if os.name == "posix":
        kwargs["preexec_fn"] = _restore_child_sigint
    return subprocess.Popen(list(argv), **kwargs)


def _restore_child_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _reap_interrupted_child(proc: "subprocess.Popen[Any]") -> None:
    """Wait for a Ctrl+C-interrupted child, escalating only if it does not exit."""
    try:
        proc.wait(timeout=GRACE)
        return
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        pass
    _terminate_child(proc)


def _terminate_child(proc: "subprocess.Popen[Any]") -> None:
    with _suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=2.0)
        return
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        pass

    with _suppress(ProcessLookupError):
        proc.kill()
    with _suppress(KeyboardInterrupt, Exception):
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.wait()


class _suppress:
    """A minimal ``contextlib.suppress`` (kept local to avoid an import just for this).
    Suppresses *exc_types* (default: everything) within the ``with`` block. Used on the
    terminal re-entry + reaper paths so a best-effort cleanup never raises."""

    def __init__(self, *exc_types: type[BaseException]) -> None:
        self._types: tuple[type[BaseException], ...] = exc_types or (Exception,)

    def __enter__(self) -> "_suppress":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return exc_type is not None and issubclass(exc_type, self._types)


__all__ = [
    "DRY_RUN",
    "GRACE",
    "SCROLLBACK_MAXLEN",
    "safe_argv",
    "headless_run",
    "RunScreen",
    "stream_run",
    "suspend_run",
]
