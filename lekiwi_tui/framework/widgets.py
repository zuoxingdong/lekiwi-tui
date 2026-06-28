"""widgets.py — framework-free input widgets + reusable render helpers (rule R5).

Two unrelated-but-co-located concerns, both lekiwi-agnostic:

1. **Framework-free input widgets** — :class:`TextField` (a single-line editor) and
   :class:`NumberField` (a bounded integer with stepping + type-to-set). Neither knows
   about pyratatui's event loop: they take a normalized
   :class:`~lekiwi_tui.framework.events.Key` and own their own state, so they are
   fully unit-testable with synthetic ``Key(...)`` — no ``Terminal`` required. We pointedly
   do NOT use pyratatui's ``TextState.handle_key`` for editing: it wants a native
   ``KeyEvent`` we cannot construct in Python, so the field would be
   untestable headlessly. Instead the edit ops are hand-rolled here against the cursor +
   buffer, and rendering is plain ``Paragraph``/``Span`` (the cursor is one reverse-video
   cell). This mirrors the ratatui input-form idiom and the carried-over Textual
   ``NumberField``'s "logic object the screen renders + steps" split.

2. **Render helpers** — :func:`status_bar`, :func:`key_hints`, :func:`labeled_rows`,
   :func:`title_bar`: thin, themed paint routines that map the original Textual app's
   ``rich.Text`` ``.append(text, style)`` patterns (menu.py's ``_hint_line`` / status line /
   header) onto pyratatui ``Line([Span(text, style)])`` rendered through ``Paragraph``.
   They take a ``frame`` + ``area`` and draw in place, so screens stay declarative.

Headless-testability note: you CANNOT render a ``Paragraph`` into a
``Buffer`` without a live ``Terminal`` (there is no ``Frame`` constructor). So the cursor
composition lives in :meth:`TextField.segments`, a PURE function returning
``[(text, style), ...]`` that ``draw`` turns into a ``Line``; the view-model is unit-tested
directly (segments + edit ops), and a raw ``Buffer.set_string`` round-trip is exercised in
the self-verify to show the underlying mechanism works.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratatui import Line, Paragraph, Span, Style, Text

from .events import (
    BACKSPACE,
    ENTER,
    LEFT,
    RIGHT,
    is_char,
)
from .theme import (
    HINT_STYLE,
    KEYCAP_STYLE,
    MUTED_STYLE,
    STATUS_VALUE_STYLE,
    TEXT_STYLE,
    TITLE_STYLE,
    key_label,
    cursor_style,
)

# The caret cell's style: warm-bone text, reverse-video. Built ONCE on a FRESH throwaway
# Style — NOT by calling ``TEXT_STYLE.reversed()``, which would risk mutating the shared
# theme constant in place (pyratatui's builders return a new Style in 0.2.9, but we do not
# want to depend on that copy-on-write semantic — every screen imports TEXT_STYLE, so a
# future in-place build would silently turn ALL normal text reverse-video). Derived from
# the same TEXT colour so it matches TEXT_STYLE visually.
_CURSOR_STYLE = cursor_style()

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .events import Key

# ── Editing key NAMES that events.py does not export a constant for ──────────────
# events.py exports ENTER/ESC/UP/DOWN/LEFT/RIGHT/TAB/BACKTAB/SPACE/BACKSPACE only.
# The three "nicety" motions below need their crossterm code spellings inline. The
# pyratatui KeyEvent code set used by our tests does not exercise these three, so keep
# the spellings isolated here. They follow crossterm's CamelCase convention (consistent
# with "BackTab"). If a build ever spells them differently, only this table changes. We
# deliberately do NOT add these to events.py; its constant set is the contract surface.
_HOME = "Home"
_END = "End"
_DELETE = "Delete"


# ══════════════════════════════════════════════════════════════════════════════
# TextField — a framework-free single-line text editor
# ══════════════════════════════════════════════════════════════════════════════


class TextField:
    """A single-line text editor that owns its ``value`` and ``cursor`` and edits them
    in response to a normalized :class:`Key`.

    Public state (read freely; mutate via the methods):
      * ``value: str`` — the current text.
      * ``cursor: int`` — caret position, an index in ``[0, len(value)]`` (``len`` means
        "after the last char"; the caret then renders as a phantom reversed space).

    :meth:`handle_key` returns ``True`` iff it consumed the key. Crucially it returns
    ``False`` for Up/Down (and Enter/Esc/Tab/unknown): a single-line field has no vertical
    motion, so it must let :class:`~lekiwi_tui.framework.focus.FocusRing` use Up/Down
    to rotate focus (R4). It consumes printable chars, Backspace, Left/Right, Home/End,
    and Delete.

    The widget is intentionally framework-free and stateless w.r.t. pyratatui: build it,
    feed it ``Key`` s, read ``value``. Rendering is via :meth:`draw` (or compose your own
    from :meth:`segments`).
    """

    __slots__ = ("value", "cursor")

    def __init__(self, value: str = "") -> None:
        self.value: str = value
        self.cursor: int = len(value)

    # ── mutation API ──
    def set_value(self, value: str) -> None:
        """Replace the text and clamp the cursor to its end (the natural place after a
        programmatic set, e.g. when a NumberField pushes its committed value back)."""
        self.value = value
        self.cursor = len(value)

    def clear(self) -> None:
        """Empty the field and reset the cursor to 0."""
        self.value = ""
        self.cursor = 0

    def insert(self, ch: str) -> None:
        """Insert *ch* at the cursor and advance it. (Single chars in normal use; a
        longer string is accepted and the cursor advances past all of it.)"""
        self.value = self.value[: self.cursor] + ch + self.value[self.cursor :]
        self.cursor += len(ch)

    def backspace(self) -> None:
        """Delete the char BEFORE the cursor (no-op at column 0), moving the cursor left."""
        if self.cursor > 0:
            self.value = self.value[: self.cursor - 1] + self.value[self.cursor :]
            self.cursor -= 1

    def delete(self) -> None:
        """Delete the char AT the cursor (no-op at end-of-line); cursor stays put."""
        if self.cursor < len(self.value):
            self.value = self.value[: self.cursor] + self.value[self.cursor + 1 :]

    def left(self) -> None:
        """Move the cursor one column left (clamped at 0)."""
        if self.cursor > 0:
            self.cursor -= 1

    def right(self) -> None:
        """Move the cursor one column right (clamped at end-of-line)."""
        if self.cursor < len(self.value):
            self.cursor += 1

    def home(self) -> None:
        """Move the cursor to the start of the line."""
        self.cursor = 0

    def end(self) -> None:
        """Move the cursor to the end of the line."""
        self.cursor = len(self.value)

    # ── key handling (R1 / R5) ──
    def handle_key(self, key: "Key") -> bool:
        """Apply one :class:`Key`; return ``True`` iff it was consumed.

        Consumed: a printable char (inserted, modifier-free — Ctrl/Alt chars are left for
        the screen), Backspace, Left, Right, Home, End, Delete. NOT consumed (returns
        ``False`` so the screen / focus ring can act): Up, Down, Enter, Esc, Tab, BackTab,
        and any key with Ctrl/Alt held that is not a bare printable.
        """
        name = key.name
        if name == BACKSPACE:
            self.backspace()
            return True
        if name == _DELETE:
            self.delete()
            return True
        if name == LEFT:
            self.left()
            return True
        if name == RIGHT:
            self.right()
            return True
        if name == _HOME:
            self.home()
            return True
        if name == _END:
            self.end()
            return True
        # A printable char (incl. space) with no Ctrl/Alt → insert it. Shift is fine
        # (capitals already arrive as the upper-case char in the normalized name).
        if is_char(key) and not key.ctrl and not key.alt:
            self.insert(name)
            return True
        return False

    # ── rendering (pure view-model + a draw wrapper) ──
    def segments(
        self, *, focused: bool, label: str | None = None
    ) -> list[tuple[str, Style]]:
        """Return the field as a list of ``(text, style)`` segments, the PURE view-model
        :meth:`draw` paints. Headlessly unit-testable (no ``Frame`` needed).

        Layout: an optional *label* (in :data:`~lekiwi_tui.framework.theme.MUTED_STYLE`)
        then the value split around the caret. When *focused*, the character at the cursor
        is emitted as its OWN segment in reverse-video (:meth:`Style.reversed`); at
        end-of-line, where there is no character under the caret, a phantom reversed space
        stands in so the caret is still visible. When NOT focused, no cursor cell is drawn
        (the value renders as one plain segment after the label).
        """
        segs: list[tuple[str, Style]] = []
        if label is not None:
            segs.append((label, MUTED_STYLE))

        if not focused:
            segs.append((self.value, TEXT_STYLE))
            return segs

        cur = max(0, min(self.cursor, len(self.value)))
        before = self.value[:cur]
        if before:
            segs.append((before, TEXT_STYLE))
        # The cursor cell: the char under the caret, or a phantom space at end-of-line.
        # Uses the shared _CURSOR_STYLE (a fresh reversed Style) — never TEXT_STYLE.reversed().
        cursor_char = self.value[cur] if cur < len(self.value) else " "
        segs.append((cursor_char, _CURSOR_STYLE))
        after = self.value[cur + 1 :] if cur < len(self.value) else ""
        if after:
            segs.append((after, TEXT_STYLE))
        return segs

    def draw(
        self, frame: Any, area: Any, *, focused: bool, label: str | None = None
    ) -> None:
        """Render the field into *area* of *frame* as a single-line ``Paragraph``.

        Composes :meth:`segments` into a ``Line`` of ``Span`` s (each segment's text +
        style) and renders it. Immediate mode: build fresh every frame, never store the
        frame. *focused* drives the reverse-video caret cell; *label* (if given) is shown
        before the value in the muted style.
        """
        spans = [Span(text, style) for text, style in self.segments(focused=focused, label=label)]
        para = Paragraph(Text([Line(spans)]))
        frame.render_widget(para, area)


# ══════════════════════════════════════════════════════════════════════════════
# NumberField — bounded integer with stepping + type-to-set (logic ported VERBATIM)
# ══════════════════════════════════════════════════════════════════════════════
# The numeric contract below (__init__/_clamp/step_by/set_text/display/hint) is
# framework-free and is the unit-test oracle. The pyratatui-side editing glue (an embedded
# TextField + Up/Down stepping + a draw) is layered around that core.
# (The Textual file's `NumberInput`, which subclasses `textual.widgets.Input`, is
# deliberately NOT ported: it cannot import in this venv, and the embedded TextField fills
# its "type a value AND step it" role here.)


class NumberField:
    """A bounded integer setting with a step size, optional unit, and a type-to-set parser.

    Args:
        label: human label (e.g. "Duration").
        value: initial value (clamped into range on construction).
        minimum: lowest allowed value (default 0).
        maximum: highest allowed value, or None for unbounded above.
        step: increment applied by one ←→ / +- press (default 1).
        unit: suffix shown after the value (e.g. "s", " Hz"); "" for none.
        zero_label: shown INSTEAD of "0<unit>" when value == 0 (e.g. "config default").
    """

    def __init__(
        self,
        label: str,
        value: int,
        *,
        minimum: int = 0,
        maximum: int | None = None,
        step: int = 1,
        unit: str = "",
        zero_label: str | None = None,
    ) -> None:
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
        self.unit = unit
        self.zero_label = zero_label
        self.error = ""  # last set_text validation message; "" when clean
        self.value = self._clamp(int(value))

    def _clamp(self, v: int) -> int:
        v = max(self.minimum, v)
        if self.maximum is not None:
            v = min(self.maximum, v)
        return v

    def step_by(self, delta: int) -> None:
        """Increment/decrement by `delta * step`, clamped. (←→ / +- path.)"""
        self.value = self._clamp(self.value + delta * self.step)
        self.error = ""

    def set_text(self, text: str) -> bool:
        """Type-to-set: parse `text` as a whole number and clamp it. Returns True on
        success; on a non-number sets `error` and leaves the value unchanged."""
        t = (text or "").strip()
        if not t.isdigit():
            self.error = f"{self.label} must be a whole number"
            return False
        self.value = self._clamp(int(t))
        self.error = ""
        return True

    def display(self) -> str:
        """The value as shown in the row: `zero_label` at 0 (if given), else value+unit."""
        if self.value == 0 and self.zero_label is not None:
            return self.zero_label
        return f"{self.value}{self.unit}"

    def hint(self) -> str:
        """A consistent affordance hint: how to step + that you can type."""
        stepword = f"±{self.step}" if self.step != 1 else "±1"
        return f"←→ {stepword} · enter to type"

    # ── pyratatui editing glue (NOT in the Textual original) ──────────────────────
    # The integer `self.value` stays AUTHORITATIVE. An embedded TextField holds the live
    # typed buffer while the user is editing; on a valid edit (or on Enter) we commit it
    # back through `set_text`, and we mirror a committed value back into the buffer so the
    # display and the editor never diverge. Up/Down step the value (and are CONSUMED — the
    # focus.py example: a focused numeric field eats the arrow to step). This is the only
    # place NumberField touches a Key; the numeric core above is untouched.

    def _ensure_editor(self) -> TextField:
        ed = getattr(self, "_editor", None)
        if ed is None:
            ed = TextField(str(self.value))
            self._editor = ed  # type: ignore[attr-defined]
        return ed

    @property
    def editor(self) -> TextField:
        """The embedded :class:`TextField` (created lazily, seeded with the value)."""
        return self._ensure_editor()

    def sync_editor(self) -> None:
        """Reset the editor buffer to the current committed value (call when (re)focusing
        the field so a stale half-typed buffer from a previous visit is cleared)."""
        self._ensure_editor().set_value(str(self.value))

    def handle_key(self, key: "Key") -> bool:
        """Apply one :class:`Key`; return ``True`` iff consumed.

        * Up → step up by ``step`` (consumed); Down → step down (consumed). Per R4 a
          focused numeric field eats Up/Down to step, so FocusRing does NOT rotate.
        * Enter → commit the typed buffer via :meth:`set_text`; on success mirror the
          clamped value back into the buffer. Returns ``True`` (consumed) on a valid commit,
          ``False`` on an invalid one so the screen can surface ``self.error`` / keep editing.
        * any TextField-handled key (printable / Backspace / Left / Right / Home / End /
          Delete) → edit the buffer, then opportunistically commit if the buffer parses as a
          number (so ``value`` tracks typing live); a transient empty/invalid buffer is
          tolerated (``value`` simply holds its last good number). Consumed.
        * everything else (Esc / Tab / BackTab / unknown) → ``False``.

        Compare ``key.name`` (modifier-agnostic, the events.py idiom).
        """
        from .events import DOWN, UP  # local: keep module import surface minimal

        name = key.name
        if name == UP:
            self.step_by(1)
            self.sync_editor()
            return True
        if name == DOWN:
            self.step_by(-1)
            self.sync_editor()
            return True

        ed = self._ensure_editor()
        if name == ENTER:
            ok = self.set_text(ed.value)
            if ok:
                ed.set_value(str(self.value))  # reflect the clamped value
            return ok

        if ed.handle_key(key):
            # Opportunistic live commit: only when the whole buffer is a clean number.
            t = ed.value.strip()
            if t.isdigit():
                self.set_text(t)
            return True
        return False

    def draw(self, frame: Any, area: Any, *, focused: bool) -> None:
        """Render the field into *area* as ``label  <value-or-editor>``.

        When *focused*, the live :class:`TextField` editor draws (so the caret shows and
        typing is visible); when not, the read-only :meth:`display` string is shown (which
        honours ``zero_label`` and the unit). The label is always the muted prefix.
        """
        ed = self._ensure_editor()
        if focused:
            ed.draw(frame, area, focused=True, label=f"{self.label}  ")
        else:
            spans = [
                Span(f"{self.label}  ", MUTED_STYLE),
                Span(self.display(), STATUS_VALUE_STYLE),
            ]
            frame.render_widget(Paragraph(Text([Line(spans)])), area)


# ══════════════════════════════════════════════════════════════════════════════
# Render helpers — themed paint routines (port of menu.py's rich.Text patterns)
# ══════════════════════════════════════════════════════════════════════════════


def status_bar(frame: Any, area: Any, text: str, style: Style) -> None:
    """Draw a single-line status string *text* in *style* across *area*.

    A plain one-string/one-style bar (honouring the requested signature). The original
    menu's multi-colour ``host · env · GPU`` line is a screen's concern (it interleaves
    several styles) and is NOT this helper's job — this is the generic substrate a screen
    uses for a uniform status line.
    """
    para = Paragraph(Text([Line([Span(text, style)])]))
    frame.render_widget(para, area)


def key_hints(frame: Any, area: Any, pairs: "Sequence[tuple[str, str]]") -> None:
    """Draw a footer hint line from ``(key, label)`` *pairs* — the immediate-mode port of
    Textual's ``Footer`` / menu.py's ``_hint_line``.

    Each key renders as a compact keycap and its label in
    :data:`~lekiwi_tui.framework.theme.HINT_STYLE` (muted), matching the current menu
    footer.
    """
    spans: list[Span] = []
    for key, label in pairs:
        spans.append(Span(f" {key_label(key)} ", KEYCAP_STYLE))
        spans.append(Span(f" {label.strip()}  ", HINT_STYLE))
    frame.render_widget(Paragraph(Text([Line(spans)])), area)


def labeled_rows(frame: Any, area: Any, rows: "Sequence[tuple[str, str]]") -> None:
    """Draw a read-only key/value panel from ``(label, value)`` *rows* — one row per line.

    Each row is ``<label>  <value>`` with the label in
    :data:`~lekiwi_tui.framework.theme.MUTED_STYLE` and the value in
    :data:`~lekiwi_tui.framework.theme.STATUS_VALUE_STYLE` (SAND), the same pairing
    the menu's status line uses for its host/env values. Rows stack top-to-bottom into
    *area* (a multi-line ``Paragraph``); the caller sizes *area* for the row count.
    """
    lines: list[Line] = []
    for label, value in rows:
        lines.append(
            Line([Span(f"{label}  ", MUTED_STYLE), Span(value, STATUS_VALUE_STYLE)])
        )
    frame.render_widget(Paragraph(Text(lines)), area)


def title_bar(frame: Any, area: Any, title: str) -> None:
    """Draw the app/screen title (e.g. ``◆ LEKIWI``) in
    :data:`~lekiwi_tui.framework.theme.TITLE_STYLE` (bold accent) across *area* —
    the port of menu.py's ``#hdr`` mark."""
    para = Paragraph(Text([Line([Span(title, TITLE_STYLE)])]))
    frame.render_widget(para, area)


__all__ = [
    "TextField",
    "NumberField",
    "status_bar",
    "key_hints",
    "labeled_rows",
    "title_bar",
]
