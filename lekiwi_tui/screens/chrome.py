"""Shared screen chrome for the LeKiwi TUI.

The menu established the current visual language: compact status chips, keycap-style
footer hints, and selected rows with a visible accent gutter plus a subtle row tint.
Screens use these helpers so subpages do not drift back into one-off text styling.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pyratatui import Line, Paragraph, Span, Style, Text

from ..framework import runner, theme


# Shared option_line label widths, so field/action columns align ACROSS screens
# instead of each screen picking its own number (they had drifted: 9/20/42).
LABEL_W_FIELD = 9    # short knob rows: python / recreate / install / lerobot / plugin …
LABEL_W_ACTION = 20  # action rows: "▶ Sync now" / "▶ Run setup" style


def ellipsis() -> str:
    return "..." if theme.ASCII_MODE else "…"


def clip_end(text: str, width: int) -> str:
    width = max(0, int(width))
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    mark = ellipsis()
    if width <= len(mark):
        return text[:width]
    return text[: width - len(mark)] + mark


def clip_middle(text: str, width: int) -> str:
    width = max(0, int(width))
    if len(text) <= width:
        return text
    mark = ellipsis()
    if width < 12:
        return clip_end(text, width)
    keep = width - len(mark)
    min_head = min(5, keep)
    final_segment = text.rsplit("/", 1)[-1]
    wanted_tail = len(final_segment) if final_segment != text else keep // 2
    tail = min(len(text), max(4, wanted_tail), keep - min_head)
    head = keep - tail
    return f"{text[:head]}{mark}{text[-tail:]}"


def chip_spans(pairs: Sequence[tuple[str, str, Style | None]]) -> list[Span]:
    spans: list[Span] = []
    for label, value, value_style in pairs:
        spans.append(Span(f" {label} ", theme.CHIP_STYLE))
        spans.append(Span(f"{value} ", value_style or theme.CHIP_TEXT_STYLE))
        spans.append(Span(" ", theme.BASE_STYLE))
    return spans


def chip_line(pairs: Sequence[tuple[str, str, Style | None]]) -> Paragraph:
    return Paragraph(Text([Line(chip_spans(pairs))])).style(theme.BASE_STYLE)


def _cfg(ctx: Any, key: str, default: str = "") -> str:
    try:
        return str(ctx.cfg[key])
    except Exception:
        return default


def robot_chip_spans(ctx: Any) -> list[Span]:
    """The `robot` chip: type + LIVE host state. `●` (green) when the Pi host answers
    on its ZMQ command port, `○` (muted) when it does not, nothing until the first
    probe lands. While a launched session is running, the remaining time trails the
    dot (`● 23:41`). The probe is throttled + threaded (hostprobe.py) — calling this
    every frame is free."""
    from ..hostprobe import get_probe, session_remaining

    robot = _cfg(ctx, "ROBOT_TYPE")
    if not robot:
        return []
    spans = [Span(" robot ", theme.CHIP_STYLE), Span(f"{robot} ", theme.CHIP_VALUE_STYLE)]
    probe = get_probe(ctx)
    if probe is not None:
        probe.poll()
        if probe.alive is True:
            spans.append(Span(f"{theme.status_dot()} ", theme.CHIP_OK_STYLE))
            left = session_remaining(ctx)
            if left is not None:
                spans.append(Span(f"{left // 60}:{left % 60:02d} ", theme.CHIP_TEXT_STYLE))
        elif probe.alive is False:
            spans.append(Span("○ " if not theme.ASCII_MODE else "o ", theme.CHIP_MUTED_STYLE))
    spans.append(Span(" ", theme.BASE_STYLE))
    return spans


def runtime_chips(ctx: Any) -> Paragraph:
    spans: list[Span] = []
    host = _cfg(ctx, "LEKIWI_HOST")
    env = _cfg(ctx, "LAPTOP_ENV")
    if host:
        spans.extend(chip_spans([("host", host, theme.CHIP_VALUE_STYLE)]))
    if env:
        spans.extend(chip_spans([("env", env, theme.CHIP_VALUE_STYLE)]))
    # which follower the launchers drive + whether it is LIVE right now
    spans.extend(robot_chip_spans(ctx))

    spans.append(Span(" GPU ", theme.CHIP_STYLE))
    if getattr(ctx, "gpu_name", ""):
        spans.append(Span(f"{theme.status_dot()} ", theme.CHIP_OK_STYLE))
        spans.append(Span(f"{ctx.gpu_name} ", theme.CHIP_TEXT_STYLE))
    else:
        spans.append(Span("none ", theme.CHIP_MUTED_STYLE))
    spans.append(Span(" ", theme.BASE_STYLE))

    mode = "PREVIEW" if runner.DRY_RUN else "REAL"
    mode_style = theme.CHIP_WARN_STYLE if runner.DRY_RUN else theme.CHIP_OK_STYLE
    spans.extend(chip_spans([("mode", mode, mode_style)]))
    return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)


def keycap_hint_line(pairs: Sequence[tuple[str, str]]) -> Paragraph:
    spans: list[Span] = []
    for key, label in pairs:
        spans.append(Span(f" {theme.key_label(key)} ", theme.KEYCAP_STYLE))
        spans.append(Span(f" {label}  ", theme.HINT_STYLE))
    return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)


def option_line(
    label: str,
    value: str = "",
    hint: str = "",
    *,
    focused: bool = False,
    label_width: int = 12,
    width: int | None = None,
    clip: Literal["end", "middle"] = "end",
    label_unfocused_style: Style | None = None,
    value_unfocused_style: Style | None = None,
) -> Line:
    label_unfocused_style = label_unfocused_style or theme.MUTED_STYLE
    value_unfocused_style = value_unfocused_style or theme.TEXT_STYLE
    label_style = theme.HIGHLIGHT_LABEL_STYLE if focused else label_unfocused_style
    value_style = theme.HIGHLIGHT_TEXT_STYLE if focused else value_unfocused_style
    selector_style = theme.HIGHLIGHT_LABEL_STYLE if focused else theme.BASE_STYLE
    hint_style = theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE
    spacer_style = theme.HIGHLIGHT_STYLE if focused else theme.BASE_STYLE

    hint_text = f"   {hint}" if hint else ""
    value_text = value
    if width is not None and value_text:
        prefix_width = 2 + label_width + 2 + len(hint_text)
        value_width = max(4, int(width) - prefix_width)
        if value_width < 16 and hint_text:
            hint_text = ""
            value_width = max(4, int(width) - (2 + label_width + 2))
        clipper = clip_middle if clip == "middle" else clip_end
        value_text = clipper(value_text, value_width)

    spans = [
        Span(theme.selector(focused), selector_style),
        Span(f"{label:<{label_width}}", label_style),
    ]
    if value_text:
        spans.append(Span("  ", spacer_style))
        spans.append(Span(value_text, value_style))
    if hint_text:
        spans.append(Span(hint_text, hint_style))
    return Line(spans, theme.HIGHLIGHT_STYLE if focused else None)

