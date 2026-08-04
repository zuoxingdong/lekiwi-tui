from __future__ import annotations

from lekiwi_tui.framework.events import BACKSPACE, ENTER, LEFT, Key
from lekiwi_tui.framework.modals import PromptModalState, wrap_label
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


# ── prompt label wrapping ─────────────────────────────────────────────────────
# Regression: the label was rendered on ONE fixed row, so the delete confirmation's
# "Type 'delete' to confirm" tail fell off the card edge and the user was left facing
# an unexplained text field with no way to know what to type.

def test_wrap_label_keeps_the_instruction_that_used_to_be_clipped():
    prompt = (
        "⚠ Delete 1 episode(s) [75] from local/lekiwi-plate. A timestamped .bak "
        "of the whole dataset is kept next to it. Type 'delete' to confirm"
    )
    inner_w = 66  # _CARD_WIDTH 72 minus borders (2) and horizontal padding (4)

    rows = wrap_label(prompt, inner_w)

    assert len(rows) > 1, "a 139-char prompt must wrap, not render on one row"
    assert all(len(r) <= inner_w for r in rows), "no row may overflow the card"
    assert "Type 'delete' to confirm" in " ".join(rows)


def test_wrap_label_leaves_a_short_prompt_on_one_row():
    # Short prompts must keep the original single-row geometry (card height is derived
    # from the row count, so a spurious second row would resize every existing modal).
    assert wrap_label("Task (language instruction)", 66) == ["Task (language instruction)"]


def test_wrap_label_returns_one_row_for_empty_text():
    assert wrap_label("", 66) == [""]


def test_wrap_label_marks_truncation_rather_than_dropping_text_silently():
    rows = wrap_label("word " * 400, 66, max_rows=3)

    assert len(rows) == 3
    assert rows[-1].endswith("…"), "an over-long label must SAY it was shortened"
