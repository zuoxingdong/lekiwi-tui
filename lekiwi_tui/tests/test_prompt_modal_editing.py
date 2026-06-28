from __future__ import annotations

from lekiwi_tui.framework.events import BACKSPACE, ENTER, LEFT, Key
from lekiwi_tui.framework.modals import PromptModalState
from lekiwi_tui.framework.screen import Pop


def test_multiline_task_prompt_ctrl_backspace_deletes_previous_word():
    modal = PromptModalState(
        "Task (language instruction)",
        value="pick up the red cube",
        multiline=True,
    )

    modal.handle_key(Key(name=BACKSPACE, ctrl=True))

    assert modal.value == "pick up the red "
    assert modal._caret == len("pick up the red ")

    modal.handle_key(Key(name=BACKSPACE, ctrl=True))

    assert modal.value == "pick up the "
    assert modal._caret == len("pick up the ")


def test_multiline_task_prompt_ctrl_backspace_deletes_trailing_space_and_word():
    modal = PromptModalState(
        "Task (language instruction)",
        value="move to the drawer   ",
        multiline=True,
    )

    modal.handle_key(Key(name=BACKSPACE, ctrl=True))

    assert modal.value == "move to the "
    assert modal._caret == len("move to the ")


def test_multiline_task_prompt_ctrl_h_deletes_previous_word():
    modal = PromptModalState(
        "Task (language instruction)",
        value="move blue cube",
        multiline=True,
    )

    modal.handle_key(Key(name="h", ctrl=True))

    assert modal.value == "move blue "
    assert modal._caret == len("move blue ")


def test_multiline_task_prompt_ctrl_left_only_moves_one_char():
    modal = PromptModalState(
        "Task (language instruction)",
        value="pick up cube",
        multiline=True,
    )

    modal.handle_key(Key(name=LEFT, ctrl=True))

    assert modal.value == "pick up cube"
    assert modal._caret == len("pick up cub")


def test_multiline_task_prompt_ctrl_backspace_then_enter_applies_remaining_text():
    modal = PromptModalState(
        "Task (language instruction)",
        value="place the block on shelf",
        multiline=True,
    )

    modal.handle_key(Key(name=BACKSPACE, ctrl=True))
    action = modal.handle_key(Key(name=ENTER))

    assert isinstance(action, Pop)
    assert action.result == "place the block on "
