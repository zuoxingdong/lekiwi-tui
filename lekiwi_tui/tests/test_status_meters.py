"""The gradient meter renderer behind the LAPTOP status card.

meter_spans is pure geometry + styling, so these tests pin the parts that would fail
silently on a live dashboard: fill proportional to the reading, the position gradient
(green low, amber 60–85%, red top), a fixed value column so numbers align across rows,
and the ASCII fallback for terminals without block glyphs.
"""
from __future__ import annotations

from lekiwi_tui.framework import theme
from lekiwi_tui.screens.menu import _EIGHTHS, _grad_style, meter_spans


def _text(spans) -> str:
    return "".join(sp.content for sp in spans)


def _bar(spans) -> str:
    """The bar cells alone: everything between the label+gutter and the gutter+value."""
    return _text(spans)[5 + 2 : -(2 + 6)]


def _row(frac, *, bar_w=20, value="42%"):
    return meter_spans("cpu", frac, value, label_w=5, value_w=6, bar_w=bar_w)


# ── geometry ──────────────────────────────────────────────────────────────────
def test_the_row_is_exactly_label_gutter_bar_gutter_value_wide():
    spans = _row(0.5)

    assert len(_text(spans)) == 5 + 2 + 20 + 2 + 6


def test_zero_is_all_track_and_full_is_all_fill():
    assert _bar(_row(0.0)) == "░" * 20
    assert _bar(_row(1.0)) == "█" * 20


def test_fill_is_proportional_with_a_subcell_tip():
    bar = _bar(_row(0.5))

    assert bar.startswith("█" * 10)
    assert bar[10:] == "░" * 10  # exactly half: no tip cell

    bar = _bar(_row(0.53))       # 10.6 cells → 10 full + a ▋-ish tip

    assert bar.startswith("█" * 10)
    assert bar[10] in _EIGHTHS


def test_readings_beyond_the_range_are_clamped_not_crashed():
    assert _bar(_row(1.7)) == "█" * 20
    assert _bar(_row(-0.3)) == "░" * 20


def test_none_draws_an_empty_track_for_an_unreadable_metric():
    """A named GPU whose utilisation query failed keeps its row: empty track, no lie."""
    assert _bar(_row(None)) == "░" * 20


def test_the_value_is_right_aligned_into_its_column():
    spans = _row(0.5, value="9%")

    assert _text(spans).endswith("    9%")


# ── the position gradient ─────────────────────────────────────────────────────
def test_gradient_thresholds():
    assert _grad_style(0.10) is theme.OK_STYLE
    assert _grad_style(0.70) is theme.WARN_STYLE
    assert _grad_style(0.90) is theme.ERR_STYLE


def test_a_full_bar_carries_all_three_gradient_runs():
    """The visual promise: a maxed meter shows green THEN amber THEN red, in order —
    a single-color full bar would lose the low/high telling the user asked for."""
    styles = [sp.style for sp in _row(1.0) if "█" in sp.content]

    assert styles == [theme.OK_STYLE, theme.WARN_STYLE, theme.ERR_STYLE]


def test_a_low_bar_is_green_only():
    styles = [sp.style for sp in _row(0.3) if "█" in sp.content]

    assert styles == [theme.OK_STYLE]


# ── ASCII fallback ────────────────────────────────────────────────────────────
def test_ascii_mode_swaps_glyphs_and_drops_the_tip(monkeypatch):
    monkeypatch.setattr(theme, "ASCII_MODE", True)

    bar = _bar(_row(0.53))

    assert set(bar) <= {"#", "."}
    assert len(bar) == 20
