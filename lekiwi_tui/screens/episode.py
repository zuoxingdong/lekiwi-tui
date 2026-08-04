"""episode.py — EpisodeScreen, the richer episode-index picker (port of the Textual one).

A small shared value-picker for replay + view. It is run via ``app.run_modal(...)`` and
returns ``Pop(str(ep))`` on confirm or ``Pop(None)`` on cancel, exactly mirroring the
Textual ``EpisodeScreen`` (which the owning ``_run_replay`` / ``_run_view`` flow awaited via
``push_screen_wait``, then built the CLI argv and suspended — only the bare episode prompt is
replaced by this richer picker: dataset name, total episodes, valid range, range-checked
input).

It is NOT a menu action (there is no ``HEADLESS_HOOK`` / ``run_headless`` and it touches no
action registry): it is a modal a flow constructs and drives, so the constructor takes the
dataset coordinates directly (``title`` / ``repo_id`` / ``root`` / ``episodes``) rather than
the usual ``extra=None``. ``app`` / ``ctx`` are accepted for contract uniformity but never
used in ``__init__``.

The one porting subtlety (see the inline note on Enter): :class:`NumberField` *clamps* its
value into ``[minimum, maximum]`` on every keystroke, so the committed ``self.field.value``
is always in range. The original validated the *typed* string against the range and showed
"out of range" for an over-large index. To preserve that, Enter validates the raw editor
**buffer** (``self.field.editor.value``), NOT the clamped value — so typing ``9`` with only
5 episodes still surfaces ``✗ episode 9 out of range (0–4)`` instead of silently clamping.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pyratatui import (
    Clear,
    Constraint,
    Direction,
    Layout,
    Line,
    Paragraph,
    Span,
    Style,
    Text,
)

from ..framework import theme
from ..framework.events import BACKSPACE, DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key, is_char
from ..framework.screen import Nothing, Pop, ScreenState
from ..framework.widgets import NumberField
from .chrome import keycap_hint_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

# Card geometry — same family as the framework modals (modals.py ``_CARD_WIDTH`` = 72).
_CARD_WIDTH = 76


def collapse_home(path: str) -> str:
    """Collapse a leading ``$HOME`` to ``~`` (verbatim from the Textual EpisodeScreen)."""
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def _centered_rect(area: Any, width: int, height: int) -> Any:
    """A centered child ``Rect`` of *width* x *height* inside *area* (the modals.py idiom:
    a vertical flex/length/flex split for the band, then a horizontal one for the card).
    Width/height are clamped so a tiny terminal still yields a valid rect."""
    w = min(width, max(1, area.width))
    h = min(height, max(1, area.height))
    vbands = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([Constraint.fill(1), Constraint.length(h), Constraint.fill(1)])
        .split(area)
    )
    band = vbands[1]
    hbands = (
        Layout()
        .direction(Direction.Horizontal)
        .constraints([Constraint.fill(1), Constraint.length(w), Constraint.fill(1)])
        .split(band)
    )
    return hbands[1]


class EpisodeScreen(ScreenState):
    """Centered card: dataset (``repo_id  ·  ~root``), episode count + valid range, and an
    index :class:`NumberField`. Confirm (⏎) validates a whole number within range and
    pops it (as a str); q / Esc pops ``None``.

    Drive it from a flow (mirrors the Textual ``push_screen_wait``)::

        ep = await app.run_modal(EpisodeScreen(self.app, ctx, title="📼 Replay episode",
                                               repo_id=repo, root=root, episodes=n))
        if ep is None:
            return          # cancelled
        argv = ["bash", str(REPLAY_SCRIPT), "--episode", ep, *extra]
    """

    def __init__(
        self,
        app: "App",
        ctx: "Context",
        *,
        title: str = "Episode",
        repo_id: str = "",
        root: str = "",
        episodes: int | None = 0,
    ) -> None:
        # ``app`` / ``ctx`` accepted for uniformity; NOT used in __init__ (app may be None
        # at construction). They are not needed at all here — this modal reads no config.
        self.app = app
        self.ctx = ctx
        self._title = title
        self._repo_id = repo_id
        self._root = root
        # ``episodes`` is the recorded COUNT (int), or None / a negative sentinel (-1) for
        # "unknown / no dataset metadata". Pin the predicate so the default 0 is NOT mistaken
        # for unknown (0 is a real, known count with no valid range).
        self._episodes = episodes
        self._known = episodes is not None and episodes >= 0
        self._n = int(episodes) if self._known else 0
        # Clamp ↑↓/←→ stepping to the valid range when the count is known and non-empty.
        max_ep = (self._n - 1) if (self._known and self._n > 0) else None
        self.field = NumberField("Episode", 0, minimum=0, maximum=max_ep, step=1)
        self._msg = ""
        # True when the editor still mirrors the seeded value, so the first printable key
        # REPLACES it (type-to-set) rather than appending; cleared on the first edit, re-armed
        # on a step. Mirrors teleop's ``_fresh`` (else "5" → buffer "05" → Pop("05")).
        self._fresh = True

    # ── input (pure → returns an Action) ──────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop(None)

        # ↑↓ AND ←→ step the index (clamped); re-arm type-to-set. We do NOT delegate to
        # NumberField.handle_key — it treats ←→ as cursor motion and Enter as a clamping
        # commit, both wrong for this single-field validator screen.
        if name in (UP, RIGHT):
            self.field.step_by(1); self.field.sync_editor(); self._fresh = True; self._msg = ""
            return Nothing
        if name in (DOWN, LEFT):
            self.field.step_by(-1); self.field.sync_editor(); self._fresh = True; self._msg = ""
            return Nothing

        if name == ENTER:
            return self._confirm()

        # Type digits / edit into the field's editor buffer. The first printable key after a
        # step REPLACES the mirrored value (type-to-set); subsequent keys edit in place. We
        # edit the BUFFER only and do NOT commit through set_text on every keystroke (set_text
        # clamps, which would erase an out-of-range typed value before Enter can flag it).
        ed = self.field.editor
        if is_char(key) or name == BACKSPACE:
            if self._fresh and is_char(key):
                ed.clear()
            self._fresh = False
            if ed.handle_key(key):
                self._msg = ""
                return Nothing
        return Nothing

    def _confirm(self) -> Any:
        """Validate the TYPED buffer (not the clamped value) and pop the index, or set an
        inline error and stay. Blank → "0" (the bash do_replay/do_view default)."""
        raw = self.field.editor.value.strip()
        ep = raw or "0"
        if not ep.isdigit():
            self._msg = "✗ episode must be a whole number"
            return Nothing
        if self._known and self._n > 0 and int(ep) >= self._n:
            self._msg = f"✗ episode {ep} out of range (0–{self._n - 1})"
            return Nothing
        return Pop(ep)

    # ── view (rebuilt each frame) ─────────────────────────────────────────────
    def _count_line(self) -> Line:
        """The episode-count + valid-range line. Unknown → a single muted sentence; known →
        ``{n} episodes`` (muted) and, when n > 0, ``  ·  valid `` (muted) + ``0–{n-1}`` (accent)."""
        if not self._known:
            return Line([
                Span("episode count unavailable (metadata not found)", theme.MUTED_STYLE)
            ])
        spans = [Span(f"{self._n} episodes", theme.MUTED_STYLE)]
        if self._n > 0:
            spans.append(Span("  ·  valid ", theme.MUTED_STYLE))
            spans.append(Span(f"0–{self._n - 1}", theme.HINT_KEY_STYLE))
        return Line(spans)

    def draw(self, frame: Any, area: Any) -> None:
        # Opaque ground so the card fully replaces whatever was underneath (run_modal draws
        # this screen alone, but a short card must still cover the full screen).
        frame.render_widget(Paragraph.from_string("").style(Style().bg(theme.BG)), area)

        # Height: title / gap / dataset / count / gap / label+field / msg / gap / hint, + the
        # bordered card's own padding (1) + borders (2). Build the card centered.
        body_lines = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        card_h = body_lines + 2 + 2  # borders + vertical padding
        card = _centered_rect(area, _CARD_WIDTH, card_h)

        frame.render_widget(Clear(), card)
        blk = theme.block(bordered=True).style(Style().bg(theme.SURFACE)).padding(2, 2, 1, 1)
        inner = blk.inner(card)
        frame.render_widget(blk, card)

        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),   # title
                Constraint.length(1),   # gap
                Constraint.length(1),   # dataset identity
                Constraint.length(1),   # count + range
                Constraint.length(1),   # gap
                Constraint.length(1),   # Episode label / field
                Constraint.length(1),   # error msg
                Constraint.fill(1),     # gap (absorbs slack)
                Constraint.length(1),   # hint
            ])
            .split(inner)
        )

        # Title (bold accent).
        frame.render_widget(
            Paragraph(Text([Line([Span(self._title, theme.TITLE_STYLE)])])), rows[0])

        # Dataset identity: repo_id (SAND) · ~root (muted).
        frame.render_widget(
            Paragraph(Text([Line([
                Span(self._repo_id, theme.STATUS_VALUE_STYLE),
                Span("  ·  ", theme.MUTED_STYLE),
                Span(collapse_home(self._root), theme.MUTED_STYLE),
            ])])), rows[2])

        # Episode count + valid range.
        frame.render_widget(Paragraph(Text([self._count_line()])), rows[3])

        # The Episode number field (always "focused" — it is the only field, so its caret
        # is always live). NumberField.draw shows the live editor when focused.
        self.field.draw(frame, rows[5], focused=True)

        # Inline error (clay-red), when set.
        if self._msg:
            frame.render_widget(
                Paragraph(Text([Line([Span(self._msg, theme.ERR_STYLE)])])), rows[6])

        frame.render_widget(
            keycap_hint_line([
                ("↑↓", "step"),
                ("⏎", "start"),
                ("esc", "cancel"),
            ]),
            rows[8],
        )


__all__ = ["EpisodeScreen"]
