"""focus.py — FocusRing: keyboard focus over an ordered list of fields (rule R4).

A form screen holds several focusable widgets (text fields, number fields, a
"submit" pseudo-field). :class:`FocusRing` tracks which one is focused and rotates the
focus with Tab / BackTab / Up / Down, the ratatui input-form idiom (and the carried-
over Textual app's ``_fpos`` cursor pattern). It routes every other key to the focused
field only.

The subtle rule (R4, confirmed in review): an arrow key must reach the focused field
FIRST. The episode picker uses Up/Down to step a number field; if the ring swallowed
Up/Down to move focus unconditionally, stepping would break. So:

  * Tab / BackTab            -> ALWAYS rotate focus (next / prev), never reach the field.
  * Up / Down                -> offer to the focused field first; rotate focus ONLY if
                                the field did not consume the key.
  * everything else          -> go straight to the focused field.

A "focusable field" here is any object; routing is done through a caller-supplied
``handler(field, key) -> bool`` (return True = consumed), so :class:`FocusRing` itself
needs to know nothing about widget types and stays fully unit-testable with plain
stand-ins.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .events import BACKTAB, DOWN, TAB, UP

if TYPE_CHECKING:
    from .events import Key


class FocusRing:
    """An ordered set of focusable items plus a current index.

    Construct with the list of fields (any objects). :meth:`current` returns the
    focused field (or ``None`` when the ring is empty); :meth:`next` / :meth:`prev`
    rotate with wraparound. :meth:`route_key` implements the Tab/arrow policy above.
    """

    def __init__(self, items: list[Any] | None = None, index: int = 0) -> None:
        self._items: list[Any] = list(items) if items else []
        # Clamp the starting index into range (empty ring -> 0, harmless).
        self._index: int = max(0, min(index, len(self._items) - 1)) if self._items else 0

    # ── introspection ──
    @property
    def items(self) -> list[Any]:
        """The focusable items, in order (live list reference)."""
        return self._items

    @property
    def index(self) -> int:
        """Index of the currently focused item (0 when empty)."""
        return self._index

    def current(self) -> Any | None:
        """The focused item, or ``None`` if the ring is empty."""
        if not self._items:
            return None
        return self._items[self._index]

    def is_focused(self, item: Any) -> bool:
        """True iff *item* is the currently focused one (identity-based, so duplicate
        values in the ring are distinguished). Handy when a field draws itself
        differently while focused."""
        return self.current() is item

    # ── rotation ──
    def next(self) -> Any | None:
        """Advance focus to the next item (wraps around). Returns the now-focused
        item (or ``None`` if empty)."""
        if not self._items:
            return None
        self._index = (self._index + 1) % len(self._items)
        return self.current()

    def prev(self) -> Any | None:
        """Move focus to the previous item (wraps around). Returns the now-focused
        item (or ``None`` if empty)."""
        if not self._items:
            return None
        self._index = (self._index - 1) % len(self._items)
        return self.current()

    def focus(self, item: Any) -> bool:
        """Focus *item* if it is in the ring (identity match). Returns True on success,
        False if not found."""
        for i, candidate in enumerate(self._items):
            if candidate is item:
                self._index = i
                return True
        return False

    def focus_index(self, index: int) -> None:
        """Focus the item at *index* (clamped into range)."""
        if self._items:
            self._index = max(0, min(index, len(self._items) - 1))

    # ── key routing (R4) ──
    def route_key(self, key: "Key", handler: Callable[[Any, "Key"], bool]) -> bool:
        """Route *key* per the focus policy. *handler(field, key) -> bool* delivers a
        key to a field and reports whether the field consumed it.

        Returns True iff the key was handled here (rotated focus OR consumed by the
        focused field), so the caller can fall through to screen-level bindings only on
        a False. With an empty ring, only Tab/BackTab are "handled" (they no-op);
        every other key returns False so the screen can deal with it.

        Policy:
          * Tab            -> :meth:`next`, return True.
          * BackTab        -> :meth:`prev`, return True.
          * Up / Down      -> try the focused field; if it consumes, return True;
                              else rotate (Up = prev, Down = next) and return True.
          * other          -> return whatever the focused field reports.
        """
        name = key.name
        if name == TAB:
            self.next()
            return True
        if name == BACKTAB:
            self.prev()
            return True

        field = self.current()

        if name in (UP, DOWN):
            # Offer the arrow to the field FIRST (e.g. NumberField stepping); only if
            # it declines do we use the arrow to move focus. This is the rule that
            # keeps ↑/↓-to-step working inside a focused numeric field.
            if field is not None and handler(field, key):
                return True
            if name == UP:
                self.prev()
            else:
                self.next()
            return True

        # Any other key goes straight to the focused field.
        if field is not None:
            return handler(field, key)
        return False


__all__ = ["FocusRing"]
