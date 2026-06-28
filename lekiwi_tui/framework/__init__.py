"""framework — the generic, lekiwi-agnostic immediate-mode application runtime.

This package is the SPINE every lekiwi screen builds on (contract rules R1-R4). It has
no knowledge of lekiwi, the config, the scripts, or any concrete screen — it provides
only the reusable machinery:

  * :mod:`.events` (R1) — the normalized :class:`Key` + the single pyratatui boundary
    adapter :func:`key_from_pyratatui`.
  * :mod:`.screen` (R2) — the :class:`ScreenState` ABC and the :class:`Action` tagged
    union (:class:`Push`/:class:`Pop`/:class:`RunAction`/:class:`Quit`/:class:`Notify`/
    :class:`Suspend`/:data:`Nothing`).
  * :mod:`.app` (R3) — the :class:`App` shell: the ``AsyncTerminal`` loop, the screen
    stack, toasts, and the re-entrant :meth:`App.run_modal` (the replacement for
    Textual's ``push_screen_wait``).
  * :mod:`.focus` (R4) — :class:`FocusRing` for keyboard focus over a form's fields.

Import the public surface from here, e.g.::

    from lekiwi_tui.framework import App, ScreenState, Key, Push, Pop, ENTER, ESC
"""
from __future__ import annotations

from .app import App, RunActionFn, Toast
from .events import (
    BACKSPACE,
    BACKTAB,
    DOWN,
    ENTER,
    ESC,
    FUNCTIONAL_NAMES,
    LEFT,
    RIGHT,
    SPACE,
    TAB,
    UP,
    Key,
    is_char,
    key_from_pyratatui,
)
from .focus import FocusRing
from .screen import (
    Action,
    Invoke,
    Notify,
    Nothing,
    Pop,
    Push,
    Quit,
    RunAction,
    ScreenState,
    Suspend,
)

__all__ = [
    # app (R3)
    "App",
    "Toast",
    "RunActionFn",
    # screen (R2)
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
    # focus (R4)
    "FocusRing",
    # events (R1)
    "Key",
    "is_char",
    "key_from_pyratatui",
    "ENTER",
    "ESC",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "TAB",
    "BACKTAB",
    "SPACE",
    "BACKSPACE",
    "FUNCTIONAL_NAMES",
]
