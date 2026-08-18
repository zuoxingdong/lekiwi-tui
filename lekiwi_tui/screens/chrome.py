"""Shared screen chrome for the LeKiwi TUI — the one design system every screen uses.

The vocabulary (screens must not re-invent styling, contract R6): a slim one-line
header with LIVE status right-aligned (`draw_slim_header` / `slim_status_spans` /
`mode_chip_spans`), quiet section groups (`section_line`), pill/segment controls
(`seg`), right-aligned composition (`padded_line`), keycap footers with a single
focused-field hint slot (`hint_slot_line` / `keycap_hint_line`), value clipping
(`clip_end` / `clip_middle`), and the classic field row (`option_line`).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pyratatui import Line, Paragraph, Span, Style, Text

from ..framework import runner, theme


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


def mode_chip_spans() -> list[Span]:
    """The execution-mode chip. REAL wears the amber ARMED fill (live hardware = the
    hot state; green stays reserved for live/ready dots); PREVIEW is a quiet chip."""
    if runner.DRY_RUN:
        return [Span(" PREVIEW ", theme.CHIP_MUTED_STYLE), Span(" ", theme.BASE_STYLE)]
    return [Span(" REAL ", theme.CHIP_ARMED_STYLE), Span(" ", theme.BASE_STYLE)]


def slim_status_spans(ctx: Any) -> list[Span]:
    """The one-line live status for slim headers: host ● + session countdown, then the
    mode chip. Machine constants (env, robot type, GPU model) deliberately stay off —
    they never change mid-session; Settings owns them."""
    from ..hostprobe import get_probe, session_remaining

    spans: list[Span] = []
    probe = get_probe(ctx)
    if probe is not None:
        probe.poll()
        if probe.alive is True:
            spans.append(Span(f"{theme.status_dot()} ", theme.OK_STYLE))
            spans.append(Span("host ", theme.MUTED_STYLE))
            left = session_remaining(ctx)
            if left is not None:
                spans.append(Span(f"{left // 60}:{left % 60:02d}", theme.TEXT_STYLE))
            else:
                spans.append(Span("live", theme.TEXT_STYLE))
        elif probe.alive is False:
            spans.append(Span("○ " if not theme.ASCII_MODE else "o ", theme.MUTED_STYLE))
            spans.append(Span("host down", theme.MUTED_STYLE))
    spans.append(Span("   ", theme.BASE_STYLE))
    spans.extend(mode_chip_spans())
    return spans


def keycap_hint_line(pairs: Sequence[tuple[str, str]]) -> Paragraph:
    spans: list[Span] = []
    for key, label in pairs:
        spans.append(Span(f" {theme.key_label(key)} ", theme.KEYCAP_STYLE))
        spans.append(Span(f" {label}  ", theme.HINT_STYLE))
    return Paragraph(Text([Line(spans)])).style(theme.BASE_STYLE)


# ── the record-page design system, shared (R6: screens must not re-invent) ────
def padded_line(left: list[Span], right: list[Span], width: int,
                pad_style: Style | None = None) -> Line:
    """Compose a Line with *right* right-aligned at *width* (drop right on overflow)."""
    used = sum(len(s.content) for s in left) + sum(len(s.content) for s in right)
    pad = int(width) - used
    if pad < 1:
        return Line(left)
    return Line(left + [Span(" " * pad, pad_style or theme.BASE_STYLE)] + right)


def section_line(name: str) -> Line:
    """A quiet group header (DATASET / SESSION / …) — spacing does the separating."""
    return Line([Span(f"  {name}", theme.FAINT_STYLE)])


def seg(label: str, active: bool) -> Span:
    """One pill / segmented-control cell: accent fill when active, quiet panel when
    not. Toggles are two calls (seg('on', v), seg('off', not v) is WRONG — a toggle
    renders ONE pill: seg('on' if v else 'off', v))."""
    return Span(f" {label} ", theme.PILL_ON_STYLE if active else theme.PILL_OFF_STYLE)


#: Width of a stepper's value well, so every stepper in a column lines up.
STEPPER_W = 6


def stepper(text: str, *, focused: bool, width: int = STEPPER_W) -> list[Span]:
    """The adjustable-number affordance: ``‹   30 ›``.

    Purely a display cell — the number was ALWAYS adjustable (←→ steps it, typing digits
    live-commits), the row just never said so. The guillemets are the whole point: they
    tell you the value takes input before you focus it.
    """
    style = theme.HIGHLIGHT_TEXT_STYLE if focused else theme.CHOICE_STYLE
    bracket = theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE
    return [Span("‹ ", bracket), Span(f"{text:>{width}}", style), Span(" ›", bracket)]


def task_stepper_cell(position: int | None, total: int, *,
                      focused: bool) -> tuple[list[Span], int]:
    """The ``‹ 2/8 ›`` cell that marks a Task row as ADJUSTABLE, plus its column width.

    The task row used to render as plain text, so nothing said ←→ cycles the base
    dataset's strings while every neighbouring row showed guillemets. ``position=None``
    renders ``‹ –/8 ›``: the current text is not one of those strings, which is the
    silent divergence the picker exists to prevent, and the caller pairs it with a warning.
    ``total <= 0`` (a base with no strings) yields no cell at all — there is nothing
    to cycle, so an affordance would lie.
    """
    if total <= 0:
        return [], 0
    body = f"{position}/{total}" if position else f"–/{total}"
    spans = [*stepper(body, focused=focused, width=len(body)), Span("  ", theme.BASE_STYLE)]
    return spans, len(body) + 6  # "‹ " + body + " ›" + the two trailing spaces


def toggle(on: bool, *, focused: bool, labels: tuple[str, str] = ("on", "off")) -> list[Span]:
    """A two-pill toggle: both states visible, the live one filled.

    Deliberately NOT the single pill :func:`seg` renders. A lone ``off`` pill leaves you
    guessing what the other state is called; showing both makes ←→ obvious."""
    del focused  # the row's own highlight already marks focus; a third emphasis is noise
    return [seg(labels[0], on), Span(" ", theme.BASE_STYLE), seg(labels[1], not on)]


def setting_line(label: str, control: list[Span], note: str = "", *,
                 focused: bool = False, label_width: int = 14,
                 width: int | None = None) -> Line:
    """ONE setting on ONE row: ``▌ Label   <control>   why it matters``.

    The row archetype for every form screen. Packing two or three settings onto a line
    saved vertical space the screens did not need and made the form hard to operate: no
    affordance, and no anchor for which value ←→ was about to change.

    *note* is always-visible explanation, which is also what keeps a zero-sentinel legible
    while the field is focused — the old behaviour swapped the label out for a raw ``0``
    at exactly the moment you were deciding what to type.
    """
    spans = [
        Span(theme.selector(focused),
             theme.HIGHLIGHT_LABEL_STYLE if focused else theme.BASE_STYLE),
        Span(f"{label:<{label_width}}",
             theme.HIGHLIGHT_LABEL_STYLE if focused else theme.MUTED_STYLE),
        *control,
    ]
    if note:
        used = sum(len(s.content) for s in spans)
        room = (int(width) - used - 3) if width is not None else len(note)
        if room >= 8:
            spans.append(Span("   ", theme.HIGHLIGHT_STYLE if focused else theme.BASE_STYLE))
            spans.append(Span(clip_end(note, room),
                              theme.HIGHLIGHT_MUTED_STYLE if focused else theme.FAINT_STYLE))
    return Line(spans, theme.HIGHLIGHT_STYLE if focused else None)


def number_line(field: Any, label: str, focused: bool, note: str = "", *,
                width: int | None = None, label_width: int = 14) -> Line:
    """A :class:`NumberField` as a setting row: ``▌ FPS   ‹   30 ›   what it means``.

    The stepper always shows the NUMBER, never the zero-label, and the meaning lives in the
    note where it stays visible. That is the fix for the old behaviour: ``FieldRow.num``
    swapped in the raw editor value on focus, so "until Ctrl+C" vanished at exactly the
    moment you moved onto the field to change it.

    While focused the well shows the live editor buffer plus a caret, because typing
    live-commits digit by digit and you need to see what you have typed so far.
    """
    if focused:
        text = f"{field.editor.value}█"
    else:
        text = f"{field.value}{field.unit}" if getattr(field, "unit", "") else str(field.value)
    zero_label = getattr(field, "zero_label", None)
    if field.value == 0 and zero_label:
        note = zero_label if zero_label.lstrip().startswith("0") else f"0 = {zero_label}"
    return setting_line(label, stepper(text, focused=focused), note,
                        focused=focused, label_width=label_width, width=width)


def draw_slim_header(frame: Any, area: Any, ctx: Any, title: str,
                     right: list[Span] | None = None) -> None:
    """The one-line screen header: `◆ LEKIWI · <title>` left, LIVE status right
    (host ● + countdown + mode chip by default). Replaces the old chips row +
    per-screen info blocks; machine constants live in Settings."""
    left = [Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
            Span(f" · {title}", theme.SUBTITLE_STYLE)]
    frame.render_widget(Paragraph(Text([
        padded_line(left, right if right is not None else slim_status_spans(ctx),
                    area.width)])).style(theme.BASE_STYLE), area)


def hint_slot_line(hint: str, width: int,
                   keys: Sequence[tuple[str, str]] = (("↑↓/jk", "move"), ("←→", "adjust"),
                                                      ("⏎", "edit·start"), ("q", "back")),
                   ) -> Paragraph:
    """The single footer hint slot: the FOCUSED element's documentation on the left,
    global keycaps on the right. The ONLY place interaction hints render.

    Degrades in two steps rather than one. padded_line drops the right side wholesale on
    overflow, so on a ~120-column terminal a four-key row took the keycaps with it and the
    screen silently stopped advertising its keys (observed on Robot config, where `p` went
    missing). Labels are dropped first, keeping the keycaps themselves, which are the part
    that cannot be guessed.
    """
    return Paragraph(Text([hint_slot_row(hint, width, keys)])).style(theme.BASE_STYLE)


def hint_slot_row(hint: str, width: int,
                  keys: Sequence[tuple[str, str]] = (("q", "back"),)) -> Line:
    """The hint row as a Line, so its degradation is testable (a Paragraph is opaque)."""
    left = [Span("  ", theme.BASE_STYLE), Span(theme.key_label(hint), theme.FAINT_STYLE)]
    labelled: list[Span] = []
    caps_only: list[Span] = []
    for k, lab in keys:
        cap = Span(f" {theme.key_label(k)} ", theme.KEYCAP_STYLE)
        labelled += [cap, Span(f" {lab}  ", theme.HINT_STYLE)]
        caps_only += [cap, Span(" ", theme.BASE_STYLE)]
    used_left = sum(len(sp.content) for sp in left)
    for right in (labelled, caps_only):
        if used_left + sum(len(sp.content) for sp in right) < int(width):
            return padded_line(left, right, int(width))
    return padded_line(left, [], int(width))


def plan_row(label: str, plan_spans: "list[Span] | str", *, focused: bool,
             accent: Literal["default", "ok", "danger"] = "default",
             label_pad: int = 8) -> Line:
    """The action row every screen ends with: `▶ <label>   <plan sentence>`.

    *plan_spans* is the plan text (or pre-built spans, e.g. a warning-as-plan Span).
    *accent* picks the label color family: "ok" = green safe-commit (Save), "danger" =
    red destructive (Stop host, delete) — the only red/green Start-style rows allowed.
    """
    if focused:
        lstyle = (theme.HIGHLIGHT_DANGER_STYLE if accent == "danger"
                  else theme.HIGHLIGHT_LABEL_STYLE)
    else:
        lstyle = {"ok": theme.OK_STYLE, "danger": theme.ERR_STYLE}.get(accent, theme.TEXT_STYLE)
    if isinstance(plan_spans, str):
        plan_spans = [Span(plan_spans,
                           theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)]
    return Line([
        Span(theme.selector(focused),
             lstyle if focused else theme.BASE_STYLE),
        Span(f"{theme.play_mark()} {label}", lstyle),
        Span(" " * max(1, label_pad), theme.HIGHLIGHT_STYLE if focused else theme.BASE_STYLE),
        *plan_spans,
    ], theme.HIGHLIGHT_STYLE if focused else None)


def draw_form_page(frame: Any, area: Any, ctx: Any, title: str, body_lines: "list[Line]",
                   *, header_right: "list[Span] | None" = None, msg: str = "",
                   msg_style: Any = None, hint: str = "",
                   keys: Sequence[tuple[str, str]] = (("↑↓/jk", "move"), ("←→", "adjust"),
                                                      ("⏎", "edit·start"), ("q", "back")),
                   ) -> None:
    """The standard form-page skeleton every screen shares: slim header · heavy rule ·
    body (fill) · optional message row · the footer hint slot. Screens supply only
    their body_lines + focused-field hint."""
    from pyratatui import Constraint, Direction, Layout

    frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
    rows = (Layout().direction(Direction.Vertical).constraints(
        [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
         Constraint.length(1), Constraint.length(1)]).split(area))
    draw_slim_header(frame, rows[0], ctx, title, header_right)
    frame.render_widget(
        Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
    frame.render_widget(Paragraph(Text(body_lines)).style(theme.BASE_STYLE), rows[2])
    if msg:
        frame.render_widget(Paragraph(Text([Line([Span(f"  {msg}", msg_style or theme.ERR_STYLE)])])
                                      ).style(theme.BASE_STYLE), rows[3])
    frame.render_widget(hint_slot_line(hint, rows[4].width, keys=keys), rows[4])


class FieldRow:
    """The label/gutter/value cell builders every form repeats, bound to one label
    width. `rows.lab("Steps", focused)` / `rows.gutter(on)` / `rows.num(field, focused)`."""

    def __init__(self, label_width: int = 14) -> None:
        self.label_width = label_width

    def lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self.label_width}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def gutter(self, on: bool) -> Span:
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def num(self, field: Any, focused: bool, pad: int = 0) -> Span:
        """A NumberField cell: live editor + caret when focused, display() otherwise."""
        if focused:
            return Span(f"{field.editor.value}█", theme.HIGHLIGHT_TEXT_STYLE)
        text = field.display()
        return Span(f"{text:<{pad}}" if pad else text, theme.TEXT_STYLE)


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

