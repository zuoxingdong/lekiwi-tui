"""events.py — the normalized keyboard event (contract rule R1).

THE ONE VOCABULARY
------------------
Every ``ScreenState.handle_key`` and every modal in this app takes a :class:`Key`,
never pyratatui's native ``KeyEvent``. A :class:`Key` is a frozen dataclass with a
``name`` (string) plus three modifier flags. The single boundary adapter
:func:`key_from_pyratatui` runs ONLY inside the live ``AsyncTerminal`` loop (in
``framework/app.py`` and ``framework/app.run_modal``); it is the *only* place in the
whole codebase that touches a pyratatui ``KeyEvent``. Everything downstream is fed a
:class:`Key`, so screens, focus rings, and text fields are fully unit-testable with a
synthetic ``Key(...)`` — no ``Terminal`` required.

This mirrors the carried-over ``kbd_listener``'s unified ``KeyEvent`` idea (a
functional name + a separate modifier field), but in pyratatui's native vocabulary:
crochet rides crossterm, so a letter arrives as ``.code == "a"`` and the spacebar as
``.code == " "`` (``KeyCode::Char(c) -> c``). We keep that native form — ``name`` for a
printable key is the character itself.

THE CANONICAL COMPARISON IDIOM (copy this in screens)
-----------------------------------------------------
Compare against the NAME, not a whole Key, so the test is modifier-agnostic by
default and you opt into modifiers only when you care::

    if key.name == ENTER:            # Enter, with or without modifiers
        ...
    elif key.name == "q":            # the letter q
        ...
    elif key.name == UP and not key.ctrl:
        ...
    elif is_char(key):               # any single printable char (incl. space)
        field.insert(key.name)

``key == ENTER`` would force an exact-modifier match (a bare Key has all flags
False), silently dropping e.g. Ctrl+Enter — so prefer ``key.name == ENTER``.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Key: the normalized event ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Key:
    """A normalized, press-only key event.

    ``name`` is either a single printable character (``"a"``, ``"5"``, ``" "`` for
    the spacebar) or one functional name from the constants below (``"Enter"``,
    ``"Esc"``, ``"Up"`` …). The modifier flags are independent booleans; a "bare"
    key has all three False. Frozen + slotted so it is hashable and cheap, and so a
    screen can never mutate an event it was handed.
    """

    name: str
    ctrl: bool = False
    alt: bool = False
    shift: bool = False


# ── Functional-name constants ─────────────────────────────────────────────────
# These are NAME STRINGS (compare with ``key.name == ENTER``), and they map 1:1 to
# pyratatui's ``KeyEvent.code`` values, so the adapter below is mostly identity. Do
# NOT rename to spellings like "Escape"/"Return": crossterm emits "Esc"/"Enter".
ENTER = "Enter"
ESC = "Esc"
UP = "Up"
DOWN = "Down"
LEFT = "Left"
RIGHT = "Right"
TAB = "Tab"
BACKTAB = "BackTab"          # Shift+Tab (crossterm reports it as its own code)
SPACE = " "                  # the spacebar IS a printable char in the native form
BACKSPACE = "Backspace"

# The fixed set of NON-character functional names. A Key whose name is in this set
# is a "special" key; anything else of length 1 is a printable character. (SPACE is
# deliberately NOT here — a space is a typeable character, see is_char.)
FUNCTIONAL_NAMES = frozenset(
    {ENTER, ESC, UP, DOWN, LEFT, RIGHT, TAB, BACKTAB, BACKSPACE}
)


def is_char(key: Key) -> bool:
    """True iff *key* represents a single printable character (a letter, digit,
    punctuation, OR the spacebar) — i.e. something a text field should insert. False
    for the functional keys (Enter/Esc/arrows/Tab/Backspace). Modifier filtering is
    left to the caller: ``is_char(key) and not key.ctrl and not key.alt`` is the
    usual guard before inserting into a buffer.

    Note ``len(name) == 1`` already excludes every multi-letter functional name
    ("Enter", "Up", …); the ``isprintable`` check then drops lone control codes.
    """
    return len(key.name) == 1 and key.name.isprintable()


# ── Boundary adapter: pyratatui KeyEvent -> Key (R1) ──────────────────────────
# crossterm key-code spellings that need remapping to OUR constants. In practice
# pyratatui already uses "Enter"/"Esc"/"BackTab" verbatim, so this table only guards
# against a couple of plausible alternate spellings; everything else passes through
# as-is (a printable char stays itself, an unknown functional name stays itself).
_CODE_ALIASES = {
    "Escape": ESC,
    "Return": ENTER,
    "Backtab": BACKTAB,   # defensive: casing variant
    "Back": BACKSPACE,
    "Space": SPACE,       # if a build ever names it; we prefer the literal " "
    "Spacebar": SPACE,
}


def key_from_pyratatui(ev: object) -> Key:
    """Adapt a pyratatui ``KeyEvent`` to a normalized :class:`Key`.

    Reads ``ev.code`` (str) and ``ev.ctrl`` / ``ev.alt`` / ``ev.shift`` (bool),
    normalizing the code to our constants where spellings differ. This is the SOLE
    place that reads a native KeyEvent's attributes — keep it that way (R1).

    Typed as ``object`` so this module imports with zero dependency on the pyratatui
    extension (it is unit-testable with any duck-typed stand-in exposing the four
    attributes). The live loop passes the real ``pyratatui.KeyEvent``.
    """
    code = getattr(ev, "code", "")
    name = _CODE_ALIASES.get(code, code)
    return Key(
        name=name,
        ctrl=bool(getattr(ev, "ctrl", False)),
        alt=bool(getattr(ev, "alt", False)),
        shift=bool(getattr(ev, "shift", False)),
    )


__all__ = [
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
