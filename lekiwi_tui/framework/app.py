"""app.py — App: the immediate-mode application shell (contract rule R3).

:class:`App` owns the single ``AsyncTerminal`` loop, a screen STACK, the toast list,
and the main tick loop. It is the ONLY component that drives the terminal; screens and
modals merely return :class:`~lekiwi_tui.framework.screen.Action` s that the App
interprets. The App is deliberately lekiwi-agnostic: it knows nothing about the action
registry, the config, or any concrete screen — the one hook into the application layer
is an injected async ``run_action`` dispatcher.

Why a hand-rolled loop (vs. Textual's reactive runtime)
-------------------------------------------------------
pyratatui is immediate mode: every frame the whole UI is rebuilt and drawn, then we
poll for at most one key. There is no widget tree, no event bubbling, no
``push_screen_wait`` Future machinery. We re-create the bits we need:

  * **screen stack** — :meth:`run` always draws and dispatches to the TOP screen;
    ``Push``/``Pop`` actions grow/shrink the stack (fire-and-forget navigation).
  * **run_modal** — the replacement for ``await push_screen_wait(...)``: a nested
    draw+poll sub-loop that redraws the UNDERLYING screen, a ``Clear()``, then the
    modal, and returns the modal's result. Re-entrant (Confirm -> picker chains).
  * **toasts** — a list with frame-time expiry, drawn on top every frame.

Concurrency rule (HARD): ``AsyncTerminal``/``Frame`` are PyO3 *unsendable* — touched
from exactly one thread (this loop). ``run_modal`` is safe to call from inside an
action coroutine because the main ``events()`` generator is suspended at its ``yield``
while that coroutine runs; nothing touches the terminal concurrently.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from pyratatui import AsyncTerminal

from .events import key_from_pyratatui
from .screen import (
    Invoke,
    Notify,
    Pop,
    Push,
    Quit,
    RunAction,
    Suspend,
    _Nothing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .screen import Action, ScreenState

# Type of the injected action dispatcher: (action_id, extra) -> awaitable[exit code].
RunActionFn = Callable[[str, "Sequence[str]"], Awaitable[int]]


class Toast:
    """One transient on-screen message with a frame-time expiry.

    *expires_at* is an ``time.monotonic()`` deadline; the App drops the toast once the
    current frame's clock passes it. *level* is ``"info"`` | ``"warn"`` | ``"error"``
    (the renderer maps it to a colour).
    """

    __slots__ = ("msg", "level", "expires_at")

    def __init__(self, msg: str, level: str, expires_at: float) -> None:
        self.msg = msg
        self.level = level
        self.expires_at = expires_at


class App:
    """The application shell.

    Parameters
    ----------
    root:
        The initial :class:`ScreenState` pushed onto the stack at startup.
    run_action:
        Optional async dispatcher ``(id, extra) -> awaitable[int]`` for
        :class:`RunAction`. The framework stays lekiwi-agnostic: the lekiwi layer
        injects its action registry here. If omitted, ``RunAction`` shows an
        "unhandled action" toast.
    fps:
        Target frame rate for the main loop's ``events(fps)`` generator.
    toast_seconds:
        Default on-screen lifetime for a toast (overridable per :meth:`notify`).
    draw_overlays:
        Optional callback ``(app, frame, area) -> None`` invoked AFTER the top screen
        draws and BEFORE the toasts, every frame (main loop and modal underlay alike).
        The lekiwi layer uses it for a persistent chrome layer if desired; the
        framework does not require it.
    """

    # Modal poll cadence (ms). Bounds how quickly a modal repaints when idle (e.g. a
    # throbber) and how snappily it reacts to a key; 30fps-ish like the main loop.
    MODAL_POLL_MS = 33

    def __init__(
        self,
        root: "ScreenState",
        *,
        run_action: RunActionFn | None = None,
        fps: float = 30.0,
        toast_seconds: float = 4.0,
        draw_overlays: Callable[["App", Any, Any], None] | None = None,
        global_key: "Callable[[App, Any], Any] | None" = None,
    ) -> None:
        self._stack: list["ScreenState"] = []
        self._toasts: list[Toast] = []
        self._run_action = run_action
        self._fps = fps
        self._toast_seconds = toast_seconds
        self._draw_overlays = draw_overlays
        # Optional app-wide key hook, consulted BEFORE the top screen (like the `?`
        # help intercept, and like it NOT active inside modal loops). Returns an
        # awaitable to run and swallow the key, or None to pass the key through.
        # The lekiwi shell uses this for the double-K panic stop.
        self._global_key = global_key
        self._running = False
        # The live terminal, set for the duration of run(); None outside it. Held so
        # suspend() can restore/re-enter and run_modal can draw+poll on the SAME one.
        self._term: AsyncTerminal | None = None
        # Seed the stack with the root screen (its on_enter fires when run() starts).
        self._stack.append(root)

    # ── stack ───────────────────────────────────────────────────────────────
    @property
    def stack(self) -> list["ScreenState"]:
        """The screen stack, bottom-to-top (live reference; the top is ``[-1]``)."""
        return self._stack

    def top(self) -> "ScreenState | None":
        """The current top-of-stack screen, or ``None`` if the stack is empty."""
        return self._stack[-1] if self._stack else None

    def push(self, screen: "ScreenState") -> None:
        """Push *screen* as the new top. Fires the old top's ``on_exit`` and the new
        screen's ``on_enter``."""
        if self._stack:
            self._stack[-1].on_exit()
        self._stack.append(screen)
        screen.on_enter()

    def pop(self, result: Any = None) -> Any:
        """Pop the top screen, fire its ``on_exit`` and the newly-exposed screen's
        ``on_enter``, and return *result* (so a caller could thread it onward).

        When the stack would become empty, the App stops (there is nothing left to
        draw) — :meth:`run` then returns.
        """
        if not self._stack:
            return result
        leaving = self._stack.pop()
        leaving.on_exit()
        if self._stack:
            self._stack[-1].on_enter()
        else:
            self._running = False
        return result

    # ── toasts ──────────────────────────────────────────────────────────────
    def notify(self, msg: str, level: str = "info", *, seconds: float | None = None) -> None:
        """Add a toast. *level* is ``"info"`` | ``"warn"`` | ``"error"``; it lives for
        *seconds* (default :attr:`toast_seconds`) of frame time, then auto-expires."""
        ttl = self._toast_seconds if seconds is None else seconds
        self._toasts.append(Toast(msg, level, time.monotonic() + ttl))

    @property
    def toasts(self) -> list[Toast]:
        """The live (unexpired-as-of-last-frame) toast list."""
        return self._toasts

    def _expire_toasts(self, now: float) -> None:
        if self._toasts:
            self._toasts = [t for t in self._toasts if t.expires_at > now]

    # ── drawing ───────────────────────────────────────────────────────────────
    def _render_toasts(self, frame: Any, area: Any) -> None:
        """Render the active toasts as a stack of bordered rows at the bottom-right of
        *area*, newest lowest. Toasts are framework-owned and self-contained (no theme
        needed): the colour is chosen from the toast level. A lekiwi layer that wants
        richer styling can draw its own via *draw_overlays* and keep toasts terse."""
        if not self._toasts:
            return
        # Lazy import so the module imports without the heavier widget surface; these
        # are cheap pyratatui constructors used only when a toast is on screen.
        from pyratatui import (
            Block,
            BorderType,
            Color,
            Paragraph,
            Rect,
            Style,
        )

        level_color = {
            "info": Color.cyan(),
            "warn": Color.yellow(),
            "error": Color.red(),
        }
        # Stack from the bottom up, newest lowest. Each toast is a single bordered row.
        row_h = 3  # bordered single line
        for i, toast in enumerate(reversed(self._toasts)):
            y = area.bottom - row_h * (i + 1)
            if y < area.top:
                break
            w = min(area.width, max(20, len(toast.msg) + 4))
            x = area.right - w
            rect = Rect(x, y, w, row_h)
            color = level_color.get(toast.level, Color.cyan())
            block = (
                Block()
                .bordered()
                .border_type(BorderType.Rounded)
                .border_style(Style().fg(color))
            )
            para = Paragraph.from_string(toast.msg).block(block).style(Style().fg(color))
            frame.render_widget(para, rect)

    def _draw_frame(self, frame: Any, *, active: "ScreenState", is_modal: bool) -> None:
        """The single per-frame paint routine, shared by the main loop and run_modal.

        Draws exactly ONE screen full-area, then toasts on top. The *active* screen is
        the top-of-stack screen (main loop) OR the modal (``run_modal``).

        A modal here is an OPAQUE, full-screen view — the faithful port of the original
        Textual ``Screen`` (e.g. ``EpisodeScreen``), which fully replaces the menu, so
        nothing shows behind it. We therefore do NOT draw the underlying screen under a
        modal, and do NOT issue a framework-level ``Clear()``: the modal owns the whole
        area and fills its own background. (A modal that wants a centered card on a
        blank ground simply fills *area* and draws the card; if a modal ever needs a
        transparent overlay it can ``Clear()`` only its own card rect inside its own
        ``draw`` — that is the only component that knows the card's rect.)

        Overlay chrome (*draw_overlays*) is part of the normal screen layer, so it is
        drawn for a top screen but skipped under a modal (the modal replaces it), again
        matching the original full-screen-modal look. Toasts always draw on top so a
        notification is visible over either layer.
        """
        area = frame.area
        active.draw(frame, area)
        if not is_modal and self._draw_overlays is not None:
            self._draw_overlays(self, frame, area)
        self._render_toasts(frame, area)

    # ── main loop (R3) ──────────────────────────────────────────────────────
    async def run(self) -> None:
        """Run the application: enter the alt-screen, then tick until the stack empties
        or a :class:`Quit` is returned. Each tick draws the top screen + toasts, polls
        one key via the ``events(fps)`` generator, adapts it to a :class:`Key`,
        dispatches to the top screen's ``handle_key``, and interprets the returned
        :class:`Action`.
        """
        self._running = True
        # Fire on_enter for the seeded root screen now that we are actually running.
        if self._stack:
            self._stack[-1].on_enter()
        async with AsyncTerminal() as term:
            self._term = term
            try:
                async for ev in term.events(fps=self._fps):
                    if not self._running:
                        break
                    screen = self.top()
                    if screen is None:
                        break
                    now = time.monotonic()
                    self._expire_toasts(now)
                    # Bind `screen` as a default arg so the closure is not flagged for
                    # capturing a loop variable (it is used synchronously here anyway).
                    term.draw(lambda frame, s=screen: self._draw_frame(frame, active=s, is_modal=False))
                    if ev is None:  # idle tick: redraw only (e.g. toast expiry)
                        continue
                    key = key_from_pyratatui(ev)
                    if key.name == "?" and not key.ctrl and not key.alt:
                        from .modals import HelpModalState

                        await self.run_modal(HelpModalState(screen))
                        continue
                    if self._global_key is not None:
                        handled = self._global_key(self, key)
                        if handled is not None:
                            await handled
                            continue
                    action = screen.handle_key(key)
                    await self._interpret(action)
            finally:
                self._term = None
                self._running = False

    async def _interpret(self, action: "Action | None") -> None:
        """Interpret an :class:`Action` returned by the TOP screen (main loop only).

        ``Push``/``Pop`` mutate the stack; ``RunAction`` delegates to the injected
        dispatcher; ``Suspend`` hands off the TTY; ``Notify`` toasts; ``Quit`` stops
        the loop; ``Nothing``/``None`` do nothing.
        """
        if action is None or isinstance(action, _Nothing):
            return
        if isinstance(action, Push):
            self.push(action.screen)
        elif isinstance(action, Pop):
            self.pop(action.result)
        elif isinstance(action, Notify):
            self.notify(action.msg, action.level)
        elif isinstance(action, Quit):
            self._running = False
        elif isinstance(action, RunAction):
            await self._dispatch_action(action)
        elif isinstance(action, Suspend):
            await self.suspend(
                action.argv, env=action.env, pause=action.pause, title=action.title
            )
        elif isinstance(action, Invoke):
            await self._safe_invoke(action)

    async def _safe_invoke(self, action: "Invoke") -> None:
        """Await an :class:`Invoke`'s async thunk (a screen's own modal/suspend flow). A
        raised exception becomes an error toast instead of tearing down the loop — the same
        containment as :meth:`_dispatch_action`."""
        try:
            await action.thunk()
        except Exception as exc:  # noqa: BLE001 - never let one flow kill the app
            self.notify(f"flow failed: {exc}", "error")

    async def _dispatch_action(self, action: RunAction) -> None:
        """Run a lekiwi action via the injected dispatcher, or toast if none was wired.
        Swallows the dispatcher's exit code here (the action layer surfaces failures via
        its own toasts/screens); a raised exception becomes an error toast rather than
        tearing down the loop."""
        if self._run_action is None:
            self.notify(f"unhandled action: {action.id!r}", "warn")
            return
        try:
            await self._run_action(action.id, action.extra)
        except Exception as exc:  # noqa: BLE001 - never let one action kill the app
            self.notify(f"action {action.id!r} failed: {exc}", "error")

    # ── modal sub-loop (R3) — replaces push_screen_wait ───────────────────────
    async def run_modal(self, modal: "ScreenState") -> Any:
        """Run *modal* full-screen and return its result.

        This is the immediate-mode stand-in for Textual's
        ``await self.app.push_screen_wait(modal)``: a nested draw+poll sub-loop that
        every frame draws the (opaque, full-screen) *modal* then toasts, and awaits one
        key via ``poll_event``. It returns when the modal closes itself by returning a
        :class:`Pop` from its ``handle_key`` — :class:`Pop.result` is the returned value
        (which may be ``None`` for a cancel).

        The modal is drawn ALONE (full area), matching the original Textual modals
        (``EpisodeScreen`` etc.) which are plain full-screen ``Screen`` s that replace
        the menu — nothing shows behind them. The modal does NOT go on the nav stack
        (it is purely a control-flow overlay), and the underlying screen's lifecycle is
        left untouched; only the modal's :meth:`on_enter`/:meth:`on_exit` fire.

        Re-entrant: a modal may itself return :class:`Push` to open a sub-modal (e.g.
        Confirm -> a picker), which is run with a recursive ``run_modal`` call before
        control returns to the outer modal. A :class:`Quit` from a modal stops the
        whole app.

        The done-vs-cancel distinction matters: completion is signalled by the modal
        returning ``Pop`` (an explicit flag), NOT by ``result is not None`` — a modal
        that cancels with ``Pop(None)`` must still terminate the loop.
        """
        if self._term is None:
            raise RuntimeError("run_modal called outside the running App loop")
        term = self._term

        modal.on_enter()
        done = False
        result: Any = None
        try:
            while self._running and not done:
                now = time.monotonic()
                self._expire_toasts(now)
                term.draw(lambda frame, m=modal: self._draw_frame(frame, active=m, is_modal=True))
                ev = await term.poll_event(timeout_ms=self.MODAL_POLL_MS)
                if ev is None:
                    continue
                key = key_from_pyratatui(ev)
                action = modal.handle_key(key)
                done, result = await self._interpret_modal(action, result)
        finally:
            modal.on_exit()
        return result

    async def _interpret_modal(
        self, action: "Action | None", current_result: Any
    ) -> tuple[bool, Any]:
        """Interpret an Action returned by a MODAL inside :meth:`run_modal`.

        Returns ``(done, result)``. ``Pop`` closes the modal with its result;
        ``Push`` opens a nested modal (recursive :meth:`run_modal`) and keeps the
        outer modal open; ``Notify`` toasts; ``Quit`` stops the app (and closes the
        modal); ``RunAction``/``Suspend`` are honoured the same as in the main loop
        but leave the modal open; ``Nothing``/``None`` keep looping.
        """
        if action is None or isinstance(action, _Nothing):
            return False, current_result
        if isinstance(action, Pop):
            return True, action.result
        if isinstance(action, Push):
            # Nested modal: run it to completion, then resume the outer modal. The
            # nested result is intentionally NOT auto-propagated to the outer one —
            # the outer modal reads it via its own state if it needs to.
            await self.run_modal(action.screen)
            return False, current_result
        if isinstance(action, Notify):
            self.notify(action.msg, action.level)
            return False, current_result
        if isinstance(action, Quit):
            self._running = False
            return True, current_result
        if isinstance(action, RunAction):
            await self._dispatch_action(action)
            return False, current_result
        if isinstance(action, Suspend):
            await self.suspend(
                action.argv, env=action.env, pause=action.pause, title=action.title
            )
            return False, current_result
        if isinstance(action, Invoke):
            await self._safe_invoke(action)
            return False, current_result
        return False, current_result

    # ── suspend (R3 / R7) ─────────────────────────────────────────────────────
    async def suspend(
        self,
        argv: "Sequence[str]",
        *,
        env: "Mapping[str, str] | None" = None,
        pause: bool = False,
        title: "str | None" = None,
    ) -> int:
        """Suspend the TUI and hand the real terminal to a child process, then return
        its exit code. Delegates to ``framework.runner.suspend_run`` (imported lazily
        to avoid an import cycle: ``runner`` may import App-side helpers). The runner
        restores the terminal, runs ``argv`` blocking, and re-enters the alt-screen.

        *title* is accepted (the :class:`Suspend` action carries it) but is intentionally
        NOT forwarded: per contract R7 the signature is
        ``suspend_run(app, argv, *, env=None, pause=False)`` — no ``title``. It is kept
        on the :class:`Suspend` action for callers/UX that want it (e.g. a pre-suspend
        banner) without making the framework depend on a runner kwarg it does not own.
        """
        from . import runner

        return runner.suspend_run(self, argv, env=env, pause=pause)

    # ── lifecycle helpers ─────────────────────────────────────────────────────
    def stop(self) -> None:
        """Request the main loop to stop after the current tick."""
        self._running = False

    @property
    def is_running(self) -> bool:
        """True while the main loop is active."""
        return self._running

    @property
    def terminal(self) -> AsyncTerminal | None:
        """The live ``AsyncTerminal`` while :meth:`run` is active, else ``None``.
        Exposed for ``framework.runner`` (suspend/re-enter); screens never need it."""
        return self._term


__all__ = ["App", "Toast", "RunActionFn"]
