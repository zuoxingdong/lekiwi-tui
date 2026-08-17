"""modals.py — modal STATE objects driven by ``App.run_modal`` (contract R3 + R5).

This is the pyratatui port of the Textual app's ``widgets/modals.py`` (``ConfirmModal`` /
``PromptModal`` / ``_PromptTextArea``). A Textual ``ModalScreen`` is an imperative widget
that calls ``self.dismiss(value)``; in immediate mode there is no widget tree and no
``push_screen_wait`` Future. Instead a modal is a plain :class:`ScreenState` whose
``handle_key`` returns a :class:`Pop` to close itself, and the *underlying* screen drives
it with::

    choice = await app.run_modal(ConfirmModalState(...))   # returns Pop.result (or None)

The modal therefore lives entirely in the value world: ``handle_key`` mutates ``.result``
AND returns ``Pop(self.result)`` in the SAME step, so ``run_modal``'s return value always
equals ``.result``. That dual path (a returned Action for the live App, a public ``.result``
/ :meth:`is_done` for tests) is what makes these fully unit-testable with synthetic
:class:`Key` s — no ``Terminal`` ever (R1).

API redesign vs. the Textual original (deliberate, per the port spec)
--------------------------------------------------------------------
The Textual ``ConfirmModal`` returned an *index* (0,1,2…) in choice mode and a *bool* in a
``confirm_word`` typed-gate mode. This port re-specifies both:

  * :class:`ConfirmModalState` is choice-only and yields the chosen *label* string (the
    choice sets are unique human strings, so a label is unambiguous and reads better at the
    call site than ``choice == 1``).
  * There is NO ``confirm_word`` mode. The typed-"delete" confirmation gate is reproduced at
    the consumer layer as ``PromptModalState("Type delete to confirm …")`` + a
    ``result == "delete"`` compare — :class:`PromptModalState` already supports it, so this
    module grows nothing extra for it.

Two asymmetries from the original are preserved EXACTLY (both have dedicated self-tests):

  * Confirm: selecting a literal ``'Cancel'`` *choice* yields ``.result == 'Cancel'`` (its
    label), but pressing **Esc** yields ``.result is None``. A 'Cancel' choice and an Esc
    are different outcomes — mirrors the original where Cancel was a real index and Esc was
    ``None``. (Do not collapse both to ``None``.)
  * Prompt: Enter on an **empty** buffer yields ``.result == ''`` (blank is a real value the
    settings flows treat as "keep current"), Esc yields ``.result is None`` (abort). Never
    ``Pop(value or None)`` — that would destroy the blank-vs-abort distinction.
"""
from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from pyratatui import (
    Clear,
    Constraint,
    Direction,
    Layout,
    Line,
    Paragraph,
    Span,
    Text,
)

from .events import (
    BACKSPACE,
    BACKTAB,
    DOWN,
    ENTER,
    ESC,
    LEFT,
    RIGHT,
    TAB,
    UP,
    Key,
    is_char,
)
from .screen import Pop, ScreenState
from .theme import (
    HINT_KEY_STYLE,
    HINT_STYLE,
    HIGHLIGHT_LABEL_STYLE,
    SECTION_STYLE,
    TEXT_STYLE,
    TITLE_STYLE,
    bg_style,
    block,
    cursor_style,
    key_label,
    selector,
    surface_style,
)
from .widgets import _DELETE, _END, _HOME, TextField

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .screen import Action

# ── card sizing ─────────────────────────────────────────────────────────────────
# The original modals are 72 cols wide (tcss ``width: 72``) and grow vertically with
# content (``height: auto``). In immediate mode we compute a centered Rect from the
# frame area: a fixed target width (clamped to the area) and a content-derived height,
# centered with Layout percentages + length splits. Both modals share this geometry so
# they read as the same family of cards.
_CARD_WIDTH = 72

#: Ceiling on wrapped prompt-label rows. A label longer than this is a design smell, not
#: something to render: the card would outgrow a short terminal. Generous on purpose
#: (raised 8 → 12 for the structured dagger cheat-sheet: 3 key lines + gap + 3 tips).
_MAX_LABEL_ROWS = 12


def wrap_label(text: str, width: int, max_rows: int = _MAX_LABEL_ROWS) -> list[str]:
    """Word-wrap a modal prompt to ``width`` columns, at most ``max_rows`` rows.

    A prompt used to be rendered on ONE fixed row, so anything past the card width was
    silently clipped. That is how the delete confirmation managed to hide its own
    "Type 'delete' to confirm" tail and left the user staring at an unexplained field.
    Never shorten a label without saying so: an over-long one ends in "…" rather than
    just stopping. Returns at least one (possibly empty) row so the caller's height
    arithmetic never sees zero.

    Explicit ``\\n`` are hard line breaks (a blank segment stays a blank row), so a
    caller can STRUCTURE a longer prompt — one key per line, a gap before the tips —
    instead of it collapsing into one dense wrapped blob (the dagger cheat-sheet's
    old fate).
    """
    rows: list[str] = []
    for segment in text.split("\n"):
        rows.extend(textwrap.wrap(segment, width=max(1, width)) or [""])
    if len(rows) > max_rows:
        rows = rows[:max_rows]
        rows[-1] = rows[-1][: max(0, width - 1)].rstrip() + "…"
    return rows


# Selected-row marker (the original used a left ``▌`` accent bar via the highlight style;
# the menu port uses "▌ " — we mirror that affordance so a highlighted choice reads the
# same as everywhere else in the app).
_SEL_PREFIX = selector(True)
_UNSEL_PREFIX = selector(False)

# The multiline editor's caret cell: warm-bone text, reverse-video. Same idiom and look as
# the single-line ``TextField`` caret (widgets._CURSOR_STYLE), so a prompt's caret reads the
# same whether it is single- or multi-line. Built on a FRESH throwaway Style (NOT
# ``TEXT_STYLE.reversed()``, which would risk mutating the shared theme constant in place);
# derived from the same TEXT colour so it matches the surrounding text.
_CURSOR_STYLE = cursor_style()


def _centered_rect(area: Any, width: int, height: int) -> Any:
    """Compute a centered child ``Rect`` of *width* x *height* inside *area*.

    Uses :class:`Layout` percentage + length splits (the spec's "centered Rect via Layout
    percentages" idiom): a vertical split puts a fixed-*height* band in the middle, then a
    horizontal split puts a fixed-*width* band in the centre of that band. Width/height are
    clamped to the area so a tiny terminal still produces a valid rect.
    """
    w = min(width, max(1, area.width))
    h = min(height, max(1, area.height))
    # Vertical: flex top / fixed band / flex bottom  → the middle band is `h` tall, centered.
    vbands = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([Constraint.fill(1), Constraint.length(h), Constraint.fill(1)])
        .split(area)
    )
    band = vbands[1]
    # Horizontal: flex left / fixed card / flex right → the card is `w` wide, centered.
    hbands = (
        Layout()
        .direction(Direction.Horizontal)
        .constraints([Constraint.fill(1), Constraint.length(w), Constraint.fill(1)])
        .split(band)
    )
    return hbands[1]


def _fill_background(frame: Any, area: Any) -> None:
    """Paint the whole *area* with the app ground colour so the opaque modal fully replaces
    whatever the App drew underneath (run_modal draws the modal alone — see app.py — but a
    short modal must still cover the full screen, matching the original full-screen
    ``ModalScreen``)."""
    frame.render_widget(Paragraph.from_string("").style(bg_style()), area)


# ══════════════════════════════════════════════════════════════════════════════
# ConfirmModalState — a single-choice picker yielding the chosen LABEL
# ══════════════════════════════════════════════════════════════════════════════


class ConfirmModalState(ScreenState):
    """A centered single-choice picker (the port of the Textual ``ConfirmModal`` choice
    mode, e.g. the Resume / Delete / Cancel gate).

    Selection: Up/Down (or k/j) move the highlight; a digit ``1``..``9`` jumps to that
    choice (1-based, matching bash ``choose_one``'s number-then-Enter); Enter confirms the
    highlighted choice, setting :attr:`result` to its *label* string. Esc (or ``q``) cancels
    with :attr:`result` ``None`` — note this differs from selecting a literal ``'Cancel'``
    choice, which yields ``'Cancel'``.

    Drive it from a screen's async flow::

        choice = await app.run_modal(ConfirmModalState("Dataset exists.", ["Resume","Delete","Cancel"]))
        if choice == "Resume": ...
        elif choice == "Delete": ...
        else: ...                     # "Cancel" OR None (dismissed) → stay put

    Unit-test it directly::

        m = ConfirmModalState("q", ["A", "B", "Cancel"])
        m.handle_key(Key(DOWN)); act = m.handle_key(Key(ENTER))
        assert m.result == "B" and m.is_done() and isinstance(act, Pop)
    """

    def __init__(self, question: str, choices: "Sequence[str]") -> None:
        self.question = question
        self.choices: list[str] = list(choices)
        self.index: int = 0  # highlighted row (internal nav state; not the result)
        self.result: str | None = None
        self._done: bool = False

    # ── public test surface ──
    def is_done(self) -> bool:
        """True once the modal has closed (a choice confirmed OR cancelled). After this,
        :attr:`result` holds the chosen label, or ``None`` on cancel."""
        return self._done

    # ── navigation ──
    def _move(self, delta: int) -> None:
        if self.choices:
            self.index = (self.index + delta) % len(self.choices)

    def _close(self, result: str | None) -> "Action":
        """Record the outcome and return the :class:`Pop` that closes the modal. Setting
        ``.result`` + returning ``Pop(.result)`` in one step keeps ``run_modal``'s return
        value and the test-facing ``.result`` identical."""
        self.result = result
        self._done = True
        return Pop(result)

    # ── key handling (R1) ──
    def handle_key(self, key: Key) -> "Action | None":
        name = key.name
        if name == ESC or name == "q":
            # Esc / q → cancel. Distinct from a 'Cancel' CHOICE (which returns its label).
            return self._close(None)
        if name == UP or name == "k":
            self._move(-1)
            return None
        if name == DOWN or name == "j":
            self._move(+1)
            return None
        if name == ENTER:
            if self.choices:
                return self._close(self.choices[self.index])
            return self._close(None)
        # Digit 1..9 → jump to (and confirm is NOT implied; matches choose_one where a
        # number selects the row and Enter confirms). We move the highlight only; a second
        # Enter confirms. (Original UX: type the number, then ⏎.)
        if len(name) == 1 and name.isdigit() and name != "0":
            i = int(name) - 1
            if 0 <= i < len(self.choices):
                self.index = i
            return None
        return None

    # ── drawing ──
    def draw(self, frame: Any, area: Any) -> None:
        _fill_background(frame, area)
        # Height: 1 title + (blank) 1 + N choices + (blank) 1 + 1 hint, + 2 for borders,
        # + top/bottom padding 1 each → matches the original's padded card feel.
        # The question wraps for the same reason the prompt label does: a clipped question
        # can hide the very thing that distinguishes the choices.
        q_rows = wrap_label(self.question, max(1, _CARD_WIDTH - 2 - 2 * 2))
        q_h = len(q_rows)
        body_lines = q_h + 1 + len(self.choices) + 1 + 1
        card_h = body_lines + 2 + 2  # borders + vertical padding
        card = _centered_rect(area, _CARD_WIDTH, card_h)

        # Wipe under the card, then draw the bordered surface card.
        frame.render_widget(Clear(), card)
        blk = block(bordered=True).style(surface_style()).padding(2, 2, 1, 1)
        inner = blk.inner(card)
        frame.render_widget(blk, card)

        # Lay the inner area out: title / gap / choices / gap / hint.
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints(
                [
                    Constraint.length(q_h),                  # title (wrapped)
                    Constraint.length(1),                    # gap
                    Constraint.length(len(self.choices)),    # the choice list
                    Constraint.fill(1),                      # gap (absorbs slack)
                    Constraint.length(1),                    # hint
                ]
            )
            .split(inner)
        )

        # Title (bold accent, like the menu section headers).
        frame.render_widget(
            Paragraph(Text([Line([Span(r, SECTION_STYLE)]) for r in q_rows])),
            rows[0],
        )

        # The choice list, one row per line; the highlighted row gets the accent left-bar +
        # a faint background tint (the same highlight idiom the menu uses).
        lines: list[Line] = []
        for i, label in enumerate(self.choices):
            if i == self.index:
                spans = [
                    Span(_SEL_PREFIX, HIGHLIGHT_LABEL_STYLE),
                    Span(label, HIGHLIGHT_LABEL_STYLE),
                ]
            else:
                spans = [Span(_UNSEL_PREFIX, TEXT_STYLE), Span(label, TEXT_STYLE)]
            lines.append(Line(spans))
        frame.render_widget(Paragraph(Text(lines)), rows[2])

        # Footer hint (key in accent, label muted) — mirrors the original "↑↓/jk move · ⏎
        # select · q cancel" affordance.
        hint = Line(
            [
                Span(key_label("↑↓/jk"), HINT_KEY_STYLE),
                Span(" move  ", HINT_STYLE),
                Span(key_label("⏎"), HINT_KEY_STYLE),
                Span(" select  ", HINT_STYLE),
                Span("q/esc", HINT_KEY_STYLE),
                Span(" cancel", HINT_STYLE),
            ]
        )
        frame.render_widget(Paragraph(Text([hint])), rows[4])


# ══════════════════════════════════════════════════════════════════════════════
# HelpModalState — contextual, read-only key help
# ══════════════════════════════════════════════════════════════════════════════


_HELP_NOTES: dict[str, list[tuple[str, str]]] = {
    "menu": [
        ("1-9", "jump directly to an action"),
        ("d", "toggle REAL / PREVIEW mode"),
    ],
    "teleop": [
        ("Start", "starts teleop in the terminal"),
        ("then", "wasd + zx drive the base; Ctrl+C stops teleop"),
    ],
    "record": [
        ("Start", "starts recording in the terminal"),
        ("then", "wasd + zx drive the base; left/right/esc control episodes"),
    ],
    "dataset": [
        ("Space", "mark / unmark the highlighted episode"),
        ("D", "delete marked episodes in place (typed confirm, auto-backup)"),
        ("T", "retag marked episodes — rewrite the task text (auto-backup)"),
        ("V / Enter", "open the highlighted episode in Rerun"),
        ("d", "switch to another dataset (picker)"),
        ("r", "reload the table from disk"),
    ],
    "host": [
        ("s / Ctrl+C", "stop a running host stream"),
        ("Enter", "relaunch after the stream ends"),
    ],
    "host-kill": [
        ("s / Ctrl+C", "stop the stop-host command if it is running"),
        ("Enter", "relaunch after the stream ends"),
    ],
    "stop host": [
        ("s / Ctrl+C", "stop the stop-host command if it is running"),
        ("Enter", "run stop host again after the stream ends"),
    ],
    "run policy": [
        ("Start", "runs the selected policy in the terminal"),
        ("then", "Ctrl+C stops the policy run"),
    ],
    "sync": [
        ("s / Ctrl+C", "stop a running rsync stream"),
        ("Enter", "relaunch after the stream ends"),
    ],
    "settings": [
        ("e", "open lekiwi.yaml in $EDITOR"),
        ("Save", "writes launcher settings; launch env vars still override"),
    ],
    "robot-config": [
        ("e", "open lekiwi.yaml in $EDITOR"),
        ("r", "reload after editing"),
    ],
}


def _screen_key(screen: Any) -> str:
    return str(getattr(screen, "title", "") or type(screen).__name__).strip().lower()


def _help_entries(screen: Any) -> list[tuple[str, str]]:
    custom = getattr(screen, "help_entries", None)
    if callable(custom):
        return list(custom())
    if custom is not None:
        return list(custom)
    key = _screen_key(screen)
    return _HELP_NOTES.get(key, [])


class HelpModalState(ScreenState):
    """A read-only contextual help overlay opened by the App on bare ``?``.

    It deliberately does not define any action-specific controls itself; it only describes
    the current screen and common TUI keys. Runtime robot controls are still owned by the
    suspended child processes (teleop/record/eval), so this overlay cannot intercept wasd
    or episode-arrow controls once those actions are launched.
    """

    title = "help"

    def __init__(self, screen: Any) -> None:
        self.screen_title = str(getattr(screen, "title", "") or type(screen).__name__)
        self.entries = _help_entries(screen)

    def handle_key(self, key: Key) -> "Action | None":
        if key.name in (ESC, "q", "?", ENTER):
            return Pop(None)
        return None

    def draw(self, frame: Any, area: Any) -> None:
        _fill_background(frame, area)
        base_rows = [
            ("?", "close this help"),
            ("q / esc", "back or cancel on most screens"),
            ("up/down or j/k", "move through lists and form rows"),
            ("left/right or h/l", "adjust the focused value"),
            ("enter", "select, edit, confirm, or start"),
        ]
        entries = base_rows + self.entries
        card_h = min(max(10, len(entries) + 7), max(1, area.height))
        card = _centered_rect(area, min(_CARD_WIDTH, 78), card_h)
        frame.render_widget(Clear(), card)
        blk = block(bordered=True).style(surface_style()).padding(2, 2, 1, 1)
        inner = blk.inner(card)
        frame.render_widget(blk, card)

        rows = Layout().direction(Direction.Vertical).constraints([
            Constraint.length(1),
            Constraint.length(1),
            Constraint.fill(1),
            Constraint.length(1),
        ]).split(inner)

        frame.render_widget(Paragraph(Text([Line([
            Span("Help", TITLE_STYLE),
            Span("  ", TEXT_STYLE),
            Span(self.screen_title, HINT_STYLE),
        ])])), rows[0])

        body_lines: list[Line] = []
        for key_name, label in entries[: max(1, rows[2].height)]:
            body_lines.append(Line([
                Span(f"{key_label(key_name):<18}", HINT_KEY_STYLE),
                Span(label, TEXT_STYLE),
            ]))
        frame.render_widget(Paragraph(Text(body_lines)), rows[2])

        frame.render_widget(Paragraph(Text([Line([
            Span(key_label("?"), HINT_KEY_STYLE),
            Span(" close   ", HINT_STYLE),
            Span("q/esc", HINT_KEY_STYLE),
            Span(" close", HINT_STYLE),
        ])])), rows[3])


# ══════════════════════════════════════════════════════════════════════════════
# PromptModalState — a single- or multi-line text prompt yielding the typed value
# ══════════════════════════════════════════════════════════════════════════════


class PromptModalState(ScreenState):
    """A centered text prompt (the port of the Textual ``PromptModal`` / ``_PromptTextArea``).

    Single-line (default): wraps a framework :class:`TextField`. Enter applies
    (:attr:`result` = the buffer text, possibly ``''``), Esc aborts (:attr:`result` ``None``).

    Multiline (``multiline=True``, the Task editor): a manually-wrapped buffer. Enter
    applies; **Ctrl+J** inserts a real newline (the dependable cross-terminal newline key —
    Shift+Enter is honoured too when the terminal delivers it, but Ctrl+J is the guaranteed
    path, matching the original ``_PromptTextArea`` which bound both); Esc aborts. pyratatui's
    ``TextArea`` has no soft-wrap, so we wrap for *display* by hand (the stored value keeps
    its real newlines and never gets hard-wrap characters inserted). A reverse-video caret
    marks the insertion point (the same cell idiom as the single-line :class:`TextField`):
    ``←``/``→`` move it, ``Home``/``End`` jump to the line ends, ``Delete`` removes the char
    under it, and the editor scrolls to keep the caret on-screen.

    The three-way contract the settings flows depend on is preserved verbatim:

      * aborted → :attr:`result` is ``None`` (Esc)
      * blank   → :attr:`result` is ``''`` (Enter on an empty buffer; "keep current")
      * value   → :attr:`result` is the typed string

    Drive + test it like :class:`ConfirmModalState` (``await app.run_modal(...)`` returns
    ``.result``).
    """

    def __init__(
        self,
        prompt: str,
        value: str = "",
        *,
        multiline: bool = False,
        hint: str = "",
        placeholder: str = "",
    ) -> None:
        self.prompt = prompt
        self.multiline = multiline
        self.hint = hint
        self.placeholder = placeholder
        self.result: str | None = None
        self._done: bool = False
        if multiline:
            # Manual multiline buffer: a single string with embedded "\n"s plus a caret
            # index. (We do NOT reuse TextField — it is single-line by contract and treats a
            # "\n" as an ordinary char; a hand-rolled buffer keeps newline handling explicit.)
            self._text: str = value
            self._caret: int = len(value)
            self.field = None  # type: ignore[assignment]
        else:
            self.field: TextField | None = TextField(value)
            self._text = value  # kept in sync from the field; unused for single-line reads
            self._caret = len(value)

    # ── public test surface ──
    def is_done(self) -> bool:
        """True once the prompt has closed. :attr:`result` then holds the typed string
        (possibly ``''``) or ``None`` on abort."""
        return self._done

    @property
    def value(self) -> str:
        """The current buffer contents (single- or multi-line). Read-only convenience for
        tests / a screen that wants to peek before submit."""
        if self.multiline:
            return self._text
        assert self.field is not None
        return self.field.value

    def _close(self, result: str | None) -> "Action":
        self.result = result
        self._done = True
        return Pop(result)

    # ── multiline buffer edit ops (single-line delegates to TextField) ──
    def _ml_insert(self, s: str) -> None:
        self._text = self._text[: self._caret] + s + self._text[self._caret :]
        self._caret += len(s)

    def _ml_backspace(self) -> None:
        if self._caret > 0:
            self._text = self._text[: self._caret - 1] + self._text[self._caret :]
            self._caret -= 1

    def _ml_delete_word_left(self) -> None:
        """Delete the non-whitespace run before the caret, plus any trailing whitespace."""
        if self._caret == 0:
            return
        start = self._caret
        while start > 0 and self._text[start - 1].isspace():
            start -= 1
        while start > 0 and not self._text[start - 1].isspace():
            start -= 1
        self._text = self._text[:start] + self._text[self._caret :]
        self._caret = start

    def _ml_left(self) -> None:
        if self._caret > 0:
            self._caret -= 1

    def _ml_right(self) -> None:
        if self._caret < len(self._text):
            self._caret += 1

    def _ml_home(self) -> None:
        """Caret to the start of the current logical line (just after the preceding "\\n",
        or column 0 on the first line)."""
        self._caret = self._text.rfind("\n", 0, self._caret) + 1

    def _ml_end(self) -> None:
        """Caret to the end of the current logical line (just before the next "\\n", or the
        end of the buffer on the last line)."""
        nl = self._text.find("\n", self._caret)
        self._caret = len(self._text) if nl == -1 else nl

    def _ml_delete(self) -> None:
        """Delete the char AT the caret, a "\\n" included; no-op at end of buffer."""
        if self._caret < len(self._text):
            self._text = self._text[: self._caret] + self._text[self._caret + 1 :]

    # ── key handling (R1) ──
    def handle_key(self, key: Key) -> "Action | None":
        name = key.name
        if name == ESC:
            return self._close(None)  # abort → None (NOT "")

        if self.multiline:
            # Ctrl+J (or Shift+Enter when delivered) → a real newline; plain Enter applies.
            if name == ENTER and key.shift:
                self._ml_insert("\n")
                return None
            if name == "j" and key.ctrl:
                self._ml_insert("\n")
                return None
            if name == ENTER:
                return self._close(self._text)
            if name == BACKTAB or name == TAB:
                return None  # focus keys are inert here (single editor)
            if key.ctrl and name in (BACKSPACE, "h"):
                self._ml_delete_word_left()
                return None
            if name == BACKSPACE:
                self._ml_backspace()
                return None
            if name == LEFT:
                self._ml_left()
                return None
            if name == RIGHT:
                self._ml_right()
                return None
            if name == _HOME:
                self._ml_home()
                return None
            if name == _END:
                self._ml_end()
                return None
            if name == _DELETE:
                self._ml_delete()
                return None
            if is_char(key) and not key.ctrl and not key.alt:
                self._ml_insert(name)
                return None
            return None

        # Single-line: Enter applies (blank → ""), else route to the TextField.
        assert self.field is not None
        if name == ENTER:
            return self._close(self.field.value)
        if name == TAB or name == BACKTAB:
            return None  # inert (one field, no ring)
        self.field.handle_key(key)  # consumes printable/Backspace/Left/Right/Home/End/Del
        return None

    # ── drawing ──
    def _wrap(self, text: str, width: int) -> list[str]:
        """Wrap *text* to *width* columns for DISPLAY only (the stored value is untouched).

        Splits on the real newlines first (so explicit line breaks survive), then hard-wraps
        each resulting line at *width* — a simple char wrap (no word boundaries), which is
        all the modal needs to keep a long Task instruction visible while typing.
        """
        if width < 1:
            width = 1
        out: list[str] = []
        for para in text.split("\n"):
            if para == "":
                out.append("")
                continue
            for i in range(0, len(para), width):
                out.append(para[i : i + width])
        return out or [""]

    def _layout(self, text: str, width: int, caret: int) -> tuple[list[str], int, int]:
        """Wrap *text* for display AND locate *caret* as ``(row, col)`` in that wrapped grid.

        Returns ``(lines, row, col)``: ``lines`` is exactly :meth:`_wrap`'s output (so display
        wrapping stays single-sourced), and ``(row, col)`` is where the caret sits within it.
        ``col`` may equal ``len(lines[row])`` — the caret is then past the last char of the
        row, drawn as a phantom reversed space, the same end-of-line rule :class:`TextField`
        uses. One special case: a caret at the very end of a line whose length is an exact
        multiple of *width* would map to a non-existent cell one past the last row of that
        paragraph; we append an empty display row there so the caret has a real cell to land
        on (instead of overflowing the editor's right edge).
        """
        if width < 1:
            width = 1
        lines = self._wrap(text, width)
        base_row = 0  # index in `lines` of the current paragraph's first row
        base_idx = 0  # index in `text` of the current paragraph's first char
        crow = ccol = 0
        for para in text.split("\n"):
            plen = len(para)
            nrows = 1 if plen == 0 else (plen + width - 1) // width
            # The caret belongs to this paragraph when it is within its char span; ``<= plen``
            # so a caret at the line end (just before the "\n") lands HERE, not at the next
            # line's start. The "\n" index itself is this paragraph's end position.
            if base_idx <= caret <= base_idx + plen:
                local = caret - base_idx
                if plen == 0:
                    crow, ccol = base_row, 0
                elif local == plen and plen % width == 0:
                    lines.insert(base_row + nrows, "")  # phantom trailing row for the caret
                    crow, ccol = base_row + nrows, 0
                else:
                    crow, ccol = base_row + local // width, local % width
                break
            base_row += nrows
            base_idx += plen + 1  # +1 for the "\n" consumed between paragraphs
        return lines, crow, ccol

    def draw(self, frame: Any, area: Any) -> None:
        _fill_background(frame, area)

        # Editor height: 1 line single-line; up to ~10 wrapped lines multiline (the original
        # capped the TextArea at max-height 12; we use the same ceiling, min 3).
        inner_w = max(1, _CARD_WIDTH - 2 - 2 * 2)  # card width minus borders + h-padding
        if self.multiline:
            # Lay the buffer out ONCE: the wrapped display lines plus the caret's (row, col),
            # so the reverse-video cursor lands on the exact cell being edited.
            ml_lines, ml_crow, ml_ccol = self._layout(self._text, inner_w, self._caret)
            editor_h = max(3, min(10, len(ml_lines)))
        else:
            ml_lines = None
            editor_h = 1

        # The label wraps instead of clipping: a prompt that carries an instruction in its
        # tail (the delete confirmation) must never lose it to the card edge.
        label_rows = wrap_label(self.prompt, inner_w)
        label_h = len(label_rows)

        body_lines = label_h + 1 + editor_h + 1 + 1  # label / gap / editor / gap / hint
        card_h = body_lines + 2 + 2                  # borders + v-padding
        card = _centered_rect(area, _CARD_WIDTH, card_h)

        frame.render_widget(Clear(), card)
        blk = block(bordered=True).style(surface_style()).padding(2, 2, 1, 1)
        inner = blk.inner(card)
        frame.render_widget(blk, card)

        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints(
                [
                    Constraint.length(label_h),   # label (wrapped)
                    Constraint.length(1),         # gap
                    Constraint.length(editor_h),  # editor
                    Constraint.fill(1),           # gap
                    Constraint.length(1),         # hint
                ]
            )
            .split(inner)
        )

        # Label (bold accent — the original ``#prompt-label`` is accent + bold).
        frame.render_widget(
            Paragraph(Text([Line([Span(r, TITLE_STYLE)]) for r in label_rows])),
            rows[0],
        )

        # Editor.
        if self.multiline:
            assert ml_lines is not None
            if self._text == "" and self.placeholder:
                # Empty buffer: show the caret (a reversed cell) THEN the muted placeholder,
                # so the insertion point is visible even before the first keystroke.
                editor_lines = [
                    Line([Span(" ", _CURSOR_STYLE), Span(self.placeholder, HINT_STYLE)])
                ]
            else:
                # Scroll to keep the caret row on-screen: when the text is taller than the
                # editor window, show the slice ending at the caret line (the original
                # TextArea auto-scrolled to the caret; this is the same, computed fresh from
                # the caret each frame so there is no stored scroll state to drift).
                top = max(0, ml_crow - editor_h + 1)
                editor_lines = []
                for r, ln in enumerate(ml_lines[top : top + editor_h]):
                    if r != ml_crow - top:
                        editor_lines.append(Line([Span(ln, TEXT_STYLE)]))
                        continue
                    # The caret row: split it into [before][caret cell][after]. The caret
                    # cell is the char under the caret, or a phantom space at the row's end.
                    before, after = ln[:ml_ccol], ln[ml_ccol + 1 :]
                    cur_ch = ln[ml_ccol] if ml_ccol < len(ln) else " "
                    spans = []
                    if before:
                        spans.append(Span(before, TEXT_STYLE))
                    spans.append(Span(cur_ch, _CURSOR_STYLE))
                    if after:
                        spans.append(Span(after, TEXT_STYLE))
                    editor_lines.append(Line(spans))
            frame.render_widget(Paragraph(Text(editor_lines)), rows[2])
        else:
            assert self.field is not None
            if self.field.value == "" and self.placeholder:
                frame.render_widget(
                    Paragraph(Text([Line([Span(self.placeholder, HINT_STYLE)])])),
                    rows[2],
                )
            else:
                self.field.draw(frame, rows[2], focused=True)

        # Footer hint.
        if self.hint:
            hint_line = Line([Span(self.hint, HINT_STYLE)])
        elif self.multiline:
            hint_line = Line(
                [
                    Span(key_label("⏎"), HINT_KEY_STYLE),
                    Span(" apply  ", HINT_STYLE),
                    Span(key_label("←→"), HINT_KEY_STYLE),
                    Span(" move  ", HINT_STYLE),
                    Span("ctrl+j", HINT_KEY_STYLE),
                    Span(" newline  ", HINT_STYLE),
                    Span("esc", HINT_KEY_STYLE),
                    Span(" cancel", HINT_STYLE),
                ]
            )
        else:
            hint_line = Line(
                [
                    Span(key_label("⏎"), HINT_KEY_STYLE),
                    Span(" confirm  ", HINT_STYLE),
                    Span("esc", HINT_KEY_STYLE),
                    Span(" cancel", HINT_STYLE),
                ]
            )
        frame.render_widget(Paragraph(Text([hint_line])), rows[4])


__all__ = ["ConfirmModalState", "HelpModalState", "PromptModalState"]
