"""screen.py — the ScreenState contract + the Action tagged union (contract rule R2).

A :class:`ScreenState` is one full-screen view in the app's screen STACK. Each tick
the :class:`~lekiwi_tui.framework.app.App` calls the top screen's
:meth:`ScreenState.draw` (rebuild every widget fresh — immediate mode, never store the
frame), and on a keypress calls :meth:`ScreenState.handle_key`, which returns an
:class:`Action` telling the App what to do next (or ``None`` / :data:`Nothing` to do
nothing). Screens NEVER touch the terminal, push other screens, or shell out directly;
they only *describe* intent via an :class:`Action`, and the App interprets it. That
indirection is what makes a screen unit-testable: feed it synthetic
:class:`~lekiwi_tui.framework.events.Key` s and assert on the returned Action.

This replaces Textual's ``Screen`` + its imperative ``self.app.push_screen`` /
``self.dismiss`` / ``self.notify`` calls with a small, inspectable return value.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from .events import Key


# ══════════════════════════════════════════════════════════════════════════════
# Action — the tagged union a screen returns from handle_key
# ══════════════════════════════════════════════════════════════════════════════
# Each variant is a tiny frozen dataclass; the App matches on type. This is the whole
# vocabulary a screen has for affecting the world. Mirrors the Textual app's verbs:
#   Push        ~ self.app.push_screen(other)         (fire-and-forget navigation)
#   Pop         ~ self.dismiss(result)                (return up the stack)
#   RunAction   ~ self.app.fire_action(id, extra)     (the lekiwi action registry)
#   Quit        ~ self.app.exit()
#   Notify      ~ self.app.notify(msg, severity=...)  (a toast)
#   Suspend     ~ runner.suspend_run(app, argv, ...)  (hand the TTY to a child)
#   Nothing     ~ (a no-op singleton; equivalent to returning None)


@dataclass(frozen=True, slots=True)
class Push:
    """Push *screen* onto the stack (it becomes the new top). Fire-and-forget
    navigation: there is NO awaiting caller and no result is delivered back — when the
    pushed screen later returns :class:`Pop`, the stack simply unwinds to whoever was
    below. (To push-AND-wait-for-a-result, a screen instead drives a modal via
    ``app.run_modal(...)`` from inside its own async flow; see app.py.)"""

    screen: "ScreenState"


@dataclass(frozen=True, slots=True)
class Pop:
    """Pop the current screen off the stack, optionally carrying *result* to the
    screen now exposed below. If the stack becomes empty the App exits."""

    result: Any = None


@dataclass(frozen=True, slots=True)
class RunAction:
    """Fire a lekiwi action by *id* (e.g. ``"teleop"``, ``"record"``), forwarding
    *extra* passthrough args. The App does not know what actions exist — it delegates
    to the async ``run_action`` dispatcher injected at construction (keeping the
    framework lekiwi-agnostic). If no dispatcher was wired, the App emits an
    "unhandled action" toast."""

    id: str
    extra: "Sequence[str]" = ()


@dataclass(frozen=True, slots=True)
class Quit:
    """Tear down the loop and exit the app."""


@dataclass(frozen=True, slots=True)
class Notify:
    """Show a transient toast. *level* is one of ``"info"`` | ``"warn"`` | ``"error"``
    (mirrors Textual's ``severity``)."""

    msg: str
    level: str = "info"


@dataclass(frozen=True, slots=True)
class Suspend:
    """Suspend the TUI and hand the real terminal to a child process.

    The App restores the terminal, runs ``argv`` (blocking) via
    ``framework.runner.suspend_run``, then re-enters the alt-screen. *env*, when given,
    is the FULL child environment; *pause* waits for Enter after the child exits (so a
    loud one-shot's final output survives the repaint); *title* is an optional banner.
    This is the immediate-mode equivalent of the Textual ``with app.suspend():`` block.
    """

    argv: "Sequence[str]"
    env: "Mapping[str, str] | None" = None
    pause: bool = False
    title: "str | None" = None


@dataclass(frozen=True, slots=True)
class Invoke:
    """Run an arbitrary async flow on the App's loop thread.

    *thunk* is a zero-arg coroutine FUNCTION (``async def`` or ``lambda: coro()``) the App
    awaits when it interprets this Action. This is how a (sync) ``handle_key`` kicks off an
    asynchronous flow that needs ``await app.run_modal(...)`` and/or ``await app.suspend(...)``
    — e.g. record's Resume/Delete confirm chain, settings' policy-picker chain, train's
    confirm-then-stream. The flow lives as an ``async def`` method ON THE SCREEN (which holds
    ``app`` + its form state), so each screen stays self-contained:

        def handle_key(self, key):
            if start: return Invoke(lambda: self._start())     # self._start is async
        async def _start(self):
            if await self.app.run_modal(ConfirmModalState(...)) == "Resume":
                await self.app.suspend(self._argv())

    Awaited safely because the main ``events()`` generator is parked at its ``yield`` while
    the thunk runs — nothing else touches the terminal concurrently (same guarantee as
    ``run_modal``). Exceptions are caught by the App and shown as an error toast."""

    thunk: "Callable[[], Awaitable[Any]]"


@dataclass(frozen=True, slots=True)
class _Nothing:
    """The no-op action. Use the :data:`Nothing` singleton; equivalent to returning
    ``None`` from ``handle_key`` (the App treats both identically)."""


#: The shared no-op action singleton (return this OR ``None`` to do nothing).
Nothing = _Nothing()

#: The Action tagged union (for annotations: ``-> Action | None``).
Action = Push | Pop | RunAction | Quit | Notify | Suspend | Invoke | _Nothing


# ══════════════════════════════════════════════════════════════════════════════
# ScreenState — one view in the stack
# ══════════════════════════════════════════════════════════════════════════════


class ScreenState(ABC):
    """Abstract base for every screen.

    Subclasses implement :meth:`draw` (render the whole view into *area* each frame)
    and :meth:`handle_key` (map a :class:`Key` to an :class:`Action`). Optional
    lifecycle hooks :meth:`on_enter` / :meth:`on_exit` fire when the screen becomes,
    or stops being, the top of the stack. ``title`` is a short label the App / a header
    may show.

    Hold ALL mutable view state as plain instance attributes — the screen instance
    persists across frames; only the pyratatui *widgets* are rebuilt each ``draw``.
    Never stash the ``frame`` (it is only valid inside the draw callback).
    """

    #: Short human-readable title for this screen (shown by the App where relevant).
    title: str = ""

    @abstractmethod
    def draw(self, frame: Any, area: Any) -> None:
        """Render the entire screen into *area* of *frame*.

        *frame* is a pyratatui ``Frame`` (call ``frame.render_widget(widget, rect)``
        etc.) and *area* is the ``Rect`` to draw within. Build fresh widgets every
        call — this is immediate mode. Typed as ``Any`` so this module needs no
        pyratatui import and stays trivially importable / testable.
        """
        raise NotImplementedError

    @abstractmethod
    def handle_key(self, key: "Key") -> "Action | None":
        """Handle one key press; return an :class:`Action` for the App to interpret,
        or ``None`` / :data:`Nothing` to consume it with no effect."""
        raise NotImplementedError

    def on_enter(self) -> None:
        """Called when this screen becomes the active top of the stack (on first push
        and again whenever a screen above it pops). Default: no-op."""

    def on_exit(self) -> None:
        """Called when this screen stops being the active top (it was popped, or
        another screen was pushed over it). Default: no-op."""


__all__ = [
    "ScreenState",
    "Action",
    "Push",
    "Pop",
    "RunAction",
    "Quit",
    "Notify",
    "Suspend",
    "Invoke",
    "Nothing",
]
