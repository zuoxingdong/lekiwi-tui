from __future__ import annotations

from lekiwi_tui.screens.host import _format_duration, _progress_bar, _session_progress


def test_format_duration_uses_mmss_and_hmmss():
    assert _format_duration(30 * 60) == "30:00"
    assert _format_duration(65) == "01:05"
    assert _format_duration(90 * 60) == "1:30:00"


def test_format_duration_can_round_remaining_up():
    assert _format_duration(1799.2, ceiling=True) == "30:00"
    assert _format_duration(1799.0, ceiling=True) == "29:59"


def test_session_progress_counts_down_and_clamps():
    remaining, fraction = _session_progress(600, 100.0, now=250.0)
    assert remaining == 450.0
    assert fraction == 0.25

    remaining, fraction = _session_progress(600, 100.0, now=900.0)
    assert remaining == 0.0
    assert fraction == 1.0


def test_progress_bar_segments():
    filled, empty = _progress_bar(0.25, 12)
    assert filled == "███"
    assert empty == "░" * 9

    filled, empty = _progress_bar(2.0, 4)
    assert filled == "████"
    assert empty == ""
