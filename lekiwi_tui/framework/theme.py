"""Theme primitives for the terminal LeKiwi app.

Pyratatui has no CSS layer, so every screen imports named ``Color`` and ``Style`` roles
from this module instead of hard-coding hex values. The palette is intentionally
terminal-native: deep background, high-contrast text, bright control accents, and
status colors that remain readable in Ghostty, xterm-compatible terminals, and
``NO_COLOR`` / ASCII fallback modes.
"""
from __future__ import annotations

import os
import unicodedata

from pyratatui import Block, BorderType, Color, Style


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


COLOR_ENABLED = not (
    _truthy_env("NO_COLOR")
    or _truthy_env("LEKIWI_NO_COLOR")
    or os.environ.get("TERM", "").lower() == "dumb"
)
ASCII_MODE = (
    _truthy_env("LEKIWI_ASCII")
    or _truthy_env("LEKIWI_TUI_ASCII")
    or os.environ.get("TERM", "").lower() == "dumb"
)
EMOJI_ENABLED = not (
    ASCII_MODE
    or _truthy_env("LEKIWI_NO_EMOJI")
    or _truthy_env("LEKIWI_TUI_NO_EMOJI")
)


def _style(
    *,
    fg: Color | None = None,
    bg: Color | None = None,
    bold: bool = False,
    dim: bool = False,
    reverse: bool = False,
) -> Style:
    style = Style()
    if COLOR_ENABLED:
        if fg is not None:
            style = style.fg(fg)
        if bg is not None:
            style = style.bg(bg)
    if bold:
        style = style.bold()
    if dim:
        style = style.dim()
    if reverse:
        style = style.reversed()
    return style

# ── core palette ──────────────────────────────────────────────────────────────
BG = Color.rgb(0x0B, 0x10, 0x1F)        # #0b101f  deep app background
SURFACE = Color.rgb(0x11, 0x18, 0x27)   # #111827  raised panels
PANEL = Color.rgb(0x1F, 0x29, 0x37)     # #1f2937  chips and active controls
TEXT = Color.rgb(0xE5, 0xED, 0xF7)      # #e5edf7  primary text
MUTED = Color.rgb(0x94, 0xA3, 0xB8)     # #94a3b8  labels, hints, secondary text
ACCENT = Color.rgb(0x38, 0xBD, 0xF8)    # #38bdf8  main control accent
SUCCESS = Color.rgb(0x34, 0xD3, 0x99)   # #34d399  running/ready/real
WARNING = Color.rgb(0xF5, 0x9E, 0x0B)   # #f59e0b  preview/caution
ERROR = Color.rgb(0xFB, 0x71, 0x85)     # #fb7185  destructive/error
HAIRLINE = Color.rgb(0x33, 0x41, 0x55)  # #334155  rules / dividers / borders

# Role aliases requested by the contract (R6) so screens can use either name.
OK = SUCCESS
GREEN = SUCCESS
WARN = WARNING
YELLOW = WARNING
ERR = ERROR
RED = ERROR

PURPLE = Color.rgb(0xA7, 0x8B, 0xFA)    # #a78bfa  section / secondary accent
FAINT = Color.rgb(0x5B, 0x6B, 0x82)     # #5b6b82  tertiary text: notes, log meta

# ── derived shades ────────────────────────────────────────────────────────────
# Focus band: a desaturated accent tint (was #0e354a) — reads as "selected", not "lit".
_TINT_18_ACCENT = (0x13, 0x23, 0x3D)    # #13233d
HIGHLIGHT_BG = Color.rgb(*_TINT_18_ACCENT)

_SURFACE_LIGHTEN_2 = (0x24, 0x32, 0x44)  # #243244
SURFACE_LIGHTEN_2 = Color.rgb(*_SURFACE_LIGHTEN_2)

# ── composed styles ───────────────────────────────────────────────────────────
BASE_STYLE = _style(fg=TEXT, bg=BG)

TITLE_STYLE = _style(fg=ACCENT, bold=True)
SUBTITLE_STYLE = _style(fg=MUTED)

STATUS_STYLE = _style(fg=MUTED)
# Values wear text, never a decorative color: SAND here collided with the WARNING amber
# one Lab-step away, so every status row used to read as "caution" for no reason.
STATUS_VALUE_STYLE = _style(fg=TEXT, bold=True)
FAINT_STYLE = _style(fg=FAINT)
#: An adjustable value inside a ‹ › stepper — warmer than plain text so a settable number
#: reads differently from a reported one at a glance.
CHOICE_STYLE = _style(fg=ACCENT, bold=True)

SECTION_STYLE = _style(fg=PURPLE, bold=True)

HIGHLIGHT_STYLE = _style(bg=HIGHLIGHT_BG, reverse=not COLOR_ENABLED)
HIGHLIGHT_LABEL_STYLE = _style(
    fg=ACCENT, bg=HIGHLIGHT_BG, bold=True, reverse=not COLOR_ENABLED
)
HIGHLIGHT_TEXT_STYLE = _style(fg=TEXT, bg=HIGHLIGHT_BG, reverse=not COLOR_ENABLED)
HIGHLIGHT_MUTED_STYLE = _style(fg=MUTED, bg=HIGHLIGHT_BG, reverse=not COLOR_ENABLED)
HIGHLIGHT_ICON_STYLE = _style(
    fg=TEXT, bg=HIGHLIGHT_BG, bold=True, reverse=not COLOR_ENABLED
)
# Destructive action row (Stop host, delete confirms): the gutter + label wear the
# error red instead of the accent — the ONLY red Start-style rows in the app.
HIGHLIGHT_DANGER_STYLE = _style(
    fg=ERROR, bg=HIGHLIGHT_BG, bold=True, reverse=not COLOR_ENABLED
)

BORDER_STYLE = _style(fg=HAIRLINE)
RULE_HEAVY_STYLE = _style(fg=ACCENT, bold=not COLOR_ENABLED)
RULE_LIGHT_STYLE = _style(fg=SURFACE_LIGHTEN_2)

HINT_STYLE = _style(fg=MUTED)
HINT_KEY_STYLE = _style(fg=ACCENT, bold=True)
KEYCAP_STYLE = _style(fg=ACCENT, bg=PANEL, bold=True)

CHIP_MUTED_STYLE = _style(fg=MUTED, bg=PANEL)   # the PREVIEW mode chip
# Live-hardware mode chip: amber "armed" fill (robotics convention — the hot mode wears
# the caution color; green stays reserved for live/ready dots).
CHIP_ARMED_STYLE = _style(fg=BG, bg=WARNING, bold=True)

# Pills (toggles) + segmented controls: on = accent fill, off = quiet panel.
PILL_ON_STYLE = _style(fg=BG, bg=ACCENT, bold=True)
PILL_OFF_STYLE = _style(fg=MUTED, bg=PANEL)

OK_STYLE = _style(fg=SUCCESS)
WARN_STYLE = _style(fg=WARNING)
ERR_STYLE = _style(fg=ERROR, bold=not COLOR_ENABLED)
MUTED_STYLE = _style(fg=MUTED)
TEXT_STYLE = _style(fg=TEXT)


_ICON_FALLBACKS = {
    "🚀": ">",
    "🛑": "x",
    "🎮": "T",
    "🔴": "R",
    "📼": "P",
    "🎞️": "P",
    "🎞": "P",
    "🔎": "V",
    "🔍": "V",
    "🧠": "N",
    "📊": "E",
    "🛠": "W",
    "🧰": "W",
    "🔄": "S",
    "🎯": "C",
    "🤖": "@",
    "🦾": "A",
    "🎬": "P",
    "🔧": "*",
    "⚙️": "*",
    "⚙": "*",
    "💾": "S",
}

#: Width of the icon column, in terminal cells. Fixed so every label in a list starts at
#: the same column no matter which glyph precedes it.
ICON_CELL_W = 2


def action_icon(icon: str) -> str:
    """Return an action icon, falling back to one-cell ASCII when emoji are disabled."""
    return icon if EMOJI_ENABLED else _ICON_FALLBACKS.get(icon, "*")


def glyph_cols(glyph: str) -> int:
    """How many terminal cells *glyph* occupies: 2 for an East-Asian-Wide codepoint
    (which is what a default-presentation emoji is), 1 otherwise."""
    return 2 if glyph and unicodedata.east_asian_width(glyph[0]) == "W" else 1


def icon_unstable(icon: str) -> bool:
    """True when *icon*'s rendered width cannot be trusted, i.e. it carries a variation
    selector (U+FE0F).

    This is the whole of the icon-alignment bug that misaligned Replay, Settings and Edit
    dataset in the menu: for ``⚙️`` (U+2699 U+FE0F) the layout library measures ONE cell
    while the terminal draws TWO, so the glyph spills into the next cell and shoves the
    label one column right. Plain ``⚙`` (U+2699) is unambiguously one cell and both agree,
    so it is fine — it just needs padding. Emoji were never the problem; the invisible
    selector was. Enforced by a test over the whole registry.
    """
    return "️" in icon


def icon_cell(icon: str) -> str:
    """An action icon padded to exactly :data:`ICON_CELL_W` cells.

    Wide glyphs fill the cell; narrow ones (a plain ``⚙``, or any ASCII fallback when
    emoji are disabled) are padded into it. Computed from the glyph actually returned by
    :func:`action_icon`, so ASCII mode lines up the same as emoji mode.
    """
    glyph = action_icon(icon)
    return glyph + " " * max(0, ICON_CELL_W - glyph_cols(glyph))


def title_mark() -> str:
    return "*" if ASCII_MODE else "◆"


def selector(selected: bool) -> str:
    if not selected:
        return "  "
    return "> " if ASCII_MODE else "▌ "


def play_mark() -> str:
    return ">" if ASCII_MODE else "▶"


def status_dot() -> str:
    return "*" if ASCII_MODE else "●"


def choice(value: str) -> str:
    return f"< {value} >" if ASCII_MODE else f"‹ {value} ›"


def rule(width: int, *, light: bool = False) -> str:
    char = "-" if ASCII_MODE else "─"
    return char * max(1, int(width))


def meter_segments(fraction: float, width: int) -> tuple[str, str]:
    """Slim line-meter segments (fill, trough): `━━━────`. The modern sibling of
    :func:`progress_segments` — render fill in ACCENT and trough in BORDER/hairline."""
    filled_ch, empty_ch = ("=", "-") if ASCII_MODE else ("━", "─")
    width = max(1, int(width))
    filled = max(0, min(width, round(fraction * width)))
    return filled_ch * filled, empty_ch * (width - filled)


_SPARK_CHARS = "▁▂▃▄▅▆▇"
_SPARK_ASCII = ".:-=+*#"


def sparkline(values: "list[float]", width: int, *, lo: float | None = None,
              hi: float | None = None) -> str:
    """Render the last *width* samples as a one-row sparkline. Empty input → ''.
    lo/hi pin the scale (else min/max of the window, with a flat-line guard)."""
    if not values or width < 1:
        return ""
    window = list(values)[-int(width):]
    vlo = min(window) if lo is None else lo
    vhi = max(window) if hi is None else hi
    span = (vhi - vlo) or 1.0
    chars = _SPARK_ASCII if ASCII_MODE else _SPARK_CHARS
    top = len(chars) - 1
    out = []
    for v in window:
        idx = int(((v - vlo) / span) * top + 0.5)
        out.append(chars[max(0, min(top, idx))])
    return "".join(out)


def key_label(label: str) -> str:
    if not ASCII_MODE:
        return label
    return (
        label.replace("↑↓", "up/down")
        .replace("←→", "left/right")
        .replace("⏎", "enter")
    )


def cursor_style() -> Style:
    return _style(fg=TEXT, reverse=True)


def surface_style() -> Style:
    return _style(bg=SURFACE)


def bg_style() -> Style:
    return _style(bg=BG)


def block(title: str | None = None, *, bordered: bool = True) -> Block:
    """Return a consistently-themed pyratatui ``Block`` so every screen gets the same
    chrome (contract R6).

    - ``bordered=True`` (default): a rounded hairline border. ``bordered=False`` for
      borderless panels such as menu lists.
    - ``title`` (optional): rendered in the shared section style.

    The block intentionally does NOT set a background fill, so a screen can lift a card
    with its own ``.style(Style().bg(SURFACE))`` without fighting this helper.
    """
    blk = Block()
    if bordered:
        blk = blk.bordered().border_type(BorderType.Rounded).border_style(BORDER_STYLE)
    if title is not None:
        blk = blk.title(title).title_style(SECTION_STYLE)
    return blk
