"""The HARDWARE and SESSION status cards (menu 2×2 status grid, 2026-08-15).

HARDWARE surfaces the classic teleop no-start (leader arm unplugged) and the leader
calibration age before a launch fails on them; SESSION turns the host countdown into a
gradient meter. These tests pin the live states each card can be in.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from lekiwi_tui import hostprobe
from lekiwi_tui.screens import menu as menu_mod
from lekiwi_tui.screens.menu import MenuScreen

from conftest import make_ctx


def _screen() -> MenuScreen:
    return MenuScreen(MagicMock(), make_ctx())


def _texts(lines) -> list[str]:
    return ["".join(sp.content for sp in line.spans) for line in lines]


# ── HARDWARE: the leader plug state ────────────────────────────────────────────
def test_leader_row_shows_a_check_when_the_port_exists(monkeypatch):
    monkeypatch.setattr(menu_mod, "_leader_port_present", lambda port: True)

    row = _texts(_screen()._hardware_lines(spacious=False))[1]

    assert "✓" in row
    assert "unplugged" not in row


def test_leader_row_flags_an_unplugged_arm(monkeypatch):
    """THE classic teleop no-start, surfaced on the menu instead of in a traceback."""
    monkeypatch.setattr(menu_mod, "_leader_port_present", lambda port: False)

    row = _texts(_screen()._hardware_lines(spacious=False))[1]

    assert "✗" in row
    assert "unplugged" in row


def test_calibration_row_reports_age_or_absence(monkeypatch):
    monkeypatch.setattr(menu_mod, "_leader_calib_age_days", lambda _id: 3)
    assert "3d ago" in _texts(_screen()._hardware_lines(spacious=False))[2]

    monkeypatch.setattr(menu_mod, "_leader_calib_age_days", lambda _id: 0)
    assert "today" in _texts(_screen()._hardware_lines(spacious=False))[2]

    monkeypatch.setattr(menu_mod, "_leader_calib_age_days", lambda _id: None)
    assert "not calibrated" in _texts(_screen()._hardware_lines(spacious=False))[2]


def test_the_port_probe_is_cached_between_frames(monkeypatch):
    """draw() runs every frame; the os.stat must not."""
    calls: list[str] = []

    def exists(path):
        calls.append(path)
        return True

    monkeypatch.setattr(menu_mod.os.path, "exists", exists)
    menu_mod._PROBE_CACHE.clear()

    assert menu_mod._leader_port_present("/dev/ttyTEST0")
    assert menu_mod._leader_port_present("/dev/ttyTEST0")

    assert len(calls) == 1, "second frame within the TTL must hit the cache"


# ── SESSION: host down / up / counting down ────────────────────────────────────
class _FakeProbe:
    def __init__(self, alive):
        self.alive = alive

    def poll(self):
        return None


def _session_texts(alive, ui_state=None, monkeypatch=None):
    monkeypatch.setattr(hostprobe, "get_probe", lambda ctx: _FakeProbe(alive))
    screen = MenuScreen(MagicMock(), make_ctx(ui_state=ui_state or {}))
    return _texts(screen._session_lines(44, spacious=False))


def test_session_card_reports_a_down_host(monkeypatch):
    lines = _session_texts(False, monkeypatch=monkeypatch)

    assert "host down" in lines[0]


def test_session_card_shows_the_countdown_meter_when_up(monkeypatch):
    ends = time.monotonic() + 270
    lines = _session_texts(True, {"host_session": {"ends_at": ends, "total_s": 600}},
                           monkeypatch=monkeypatch)

    assert "host up" in lines[0]
    assert "left" in lines[1], "the countdown meter row"
    assert "█" in lines[1] or "▏" in lines[1], "elapsed time must render as fill"


def test_session_card_says_so_without_a_session_clock(monkeypatch):
    lines = _session_texts(True, monkeypatch=monkeypatch)

    assert "host up" in lines[0]
    assert "no session clock" in lines[1]
