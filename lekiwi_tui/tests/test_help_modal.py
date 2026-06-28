from __future__ import annotations

from lekiwi_tui.framework.events import ENTER, ESC, Key
from lekiwi_tui.framework.modals import HelpModalState
from lekiwi_tui.framework.screen import Pop


class _Screen:
    def __init__(self, title: str):
        self.title = title


def test_help_modal_closes_on_help_or_common_cancel_keys():
    for name in ("?", "q", ESC, ENTER):
        modal = HelpModalState(_Screen("menu"))

        action = modal.handle_key(Key(name))

        assert isinstance(action, Pop)
        assert action.result is None


def test_help_modal_keeps_robot_runtime_controls_documented():
    record = HelpModalState(_Screen("record"))
    teleop = HelpModalState(_Screen("teleop"))

    record_text = " ".join(label for _, label in record.entries)
    teleop_text = " ".join(label for _, label in teleop.entries)

    assert "wasd + zx drive the base" in record_text
    assert "left/right/esc control episodes" in record_text
    assert "wasd + zx drive the base" in teleop_text
    assert "Ctrl+C stops teleop" in teleop_text


def test_help_modal_accepts_screen_specific_entries():
    class _Custom:
        title = "custom"
        help_entries = [("x", "custom action")]

    modal = HelpModalState(_Custom())

    assert modal.entries == [("x", "custom action")]

