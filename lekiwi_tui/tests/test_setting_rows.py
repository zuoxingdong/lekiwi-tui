"""One-setting-per-row controls, word wrapping, and word-level text editing.

Three changes share this file because they share a purpose: making a form operable. The
row tells you a value is adjustable, the note keeps its meaning visible while you adjust
it, and the editor moves the way every other editor you use moves.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from lekiwi_tui.framework.events import BACKSPACE, LEFT, RIGHT, Key
from lekiwi_tui.framework.widgets import NumberField, TextField, wrap_words
from lekiwi_tui.screens.chrome import number_line, setting_line, stepper, toggle

from conftest import make_ctx


def _text(line) -> str:
    return "".join(sp.content for sp in line.spans)


def _body(screen, width: int = 90) -> str:
    return "\n".join(_text(ln) for ln in screen._body_lines(width))


# ── the stepper affordance ────────────────────────────────────────────────────
def test_stepper_shows_guillemets_so_a_value_reads_as_adjustable():
    assert _text_of(stepper("30", focused=False)) == "‹     30 ›"


def _text_of(spans) -> str:
    return "".join(sp.content for sp in spans)


def test_stepper_right_aligns_so_a_column_of_them_lines_up():
    widths = {len(_text_of(stepper(v, focused=False))) for v in ("1", "30", "254")}

    assert len(widths) == 1


def test_toggle_shows_both_states_not_just_the_live_one():
    """A lone 'off' pill leaves you guessing what the other state is called."""
    assert _text_of(toggle(True, focused=False)) == " on   off "


# ── the row ───────────────────────────────────────────────────────────────────
def test_setting_line_puts_label_control_and_note_on_one_row():
    line = _text(setting_line("FPS", stepper("30", focused=False), "loop rate", width=90))

    assert "FPS" in line and "‹     30 ›" in line and "loop rate" in line


def test_setting_line_drops_the_note_rather_than_overflowing_a_narrow_row():
    line = _text(setting_line("FPS", stepper("30", focused=False),
                              "a very long explanation that cannot fit", width=30))

    assert len(line) <= 30


# ── the sentinel fix ──────────────────────────────────────────────────────────
def test_number_line_keeps_the_zero_label_visible_while_focused():
    """The bug this fixes: FieldRow.num swapped the label for a raw 0 on focus, so the
    explanation vanished exactly when you moved onto the field to change it."""
    field = NumberField("Duration", 0, minimum=0, step=5, unit="s",
                        zero_label="until Ctrl+C")
    field.sync_editor()

    focused = _text(number_line(field, "Duration", True, "session length", width=90))

    assert "until Ctrl+C" in focused
    assert "█" in focused, "the live editor caret should still be shown"


def test_number_line_shows_the_number_not_the_zero_label_inside_the_stepper():
    field = NumberField("Duration", 0, minimum=0, unit="s", zero_label="until Ctrl+C")

    line = _text(number_line(field, "Duration", False, "", width=90))

    assert "‹" in line and "0s" in line
    assert "‹ until Ctrl+C" not in line


def test_number_line_uses_the_plain_note_once_the_value_is_not_zero():
    field = NumberField("FPS", 30, minimum=1, zero_label="unset")

    line = _text(number_line(field, "FPS", False, "control-loop rate", width=90))

    assert "control-loop rate" in line and "unset" not in line


# ── screens actually use it ───────────────────────────────────────────────────
def test_teleop_gives_every_setting_its_own_row():
    from lekiwi_tui.screens.teleop import TeleopScreen

    body = _body(TeleopScreen(MagicMock(), make_ctx()))

    for label in ("Duration", "FPS", "Display"):
        assert sum(1 for ln in body.splitlines() if label in ln) == 1
    row = next(ln for ln in body.splitlines() if "Duration" in ln)
    assert "FPS" not in row


def test_record_gives_every_setting_its_own_row():
    from lekiwi_tui.screens.record import RecordScreen

    body = _body(RecordScreen(MagicMock(), make_ctx()))

    for label in ("Episodes", "Episode time", "FPS", "Reset time", "Writers"):
        assert label in body
    row = next(ln for ln in body.splitlines() if "Episodes" in ln)
    assert "Episode time" not in row, "two settings still share a row"


def test_eval_gives_every_setting_its_own_row():
    from lekiwi_tui.screens.eval import EvalScreen

    body = _body(EvalScreen(MagicMock(), make_ctx(gpu_name="RTX 2050")))

    backend_row = next(ln for ln in body.splitlines() if "Backend" in ln)
    assert "Cameras" not in backend_row
    duration_row = next(ln for ln in body.splitlines() if "Duration" in ln)
    assert "Display" not in duration_row


def test_eval_camera_mapping_stays_on_its_own_row_and_is_never_clipped():
    """A wrong slot guess drives the robot on the wrong camera views, so the resolved
    mapping must always be readable."""
    from lekiwi_tui.screens.eval import EvalScreen

    body = _body(EvalScreen(MagicMock(), make_ctx()))

    assert "→" in body or "native" in body


# ── word wrapping ─────────────────────────────────────────────────────────────
def test_wrap_words_keeps_words_whole():
    out = wrap_words("Pick up the plate from right and place into the second slot", 46)

    assert all(len(ln) <= 46 for ln in out)
    assert not any(ln.endswith("se") for ln in out), "split 'second' mid-word"
    assert " ".join(out) == "Pick up the plate from right and place into the second slot"


def test_wrap_words_does_not_strand_trailing_punctuation():
    """The old char-wrapper left the task's full stop alone on the last line."""
    out = wrap_words("Pick up the plate from right and place into the shelf.", 53)

    assert out[-1] != "."


def test_wrap_words_hard_splits_an_unbreakable_run():
    out = wrap_words("a/very/long/path/with/no/spaces/at/all/anywhere", 10)

    assert all(len(ln) <= 10 for ln in out)
    assert "".join(out) == "a/very/long/path/with/no/spaces/at/all/anywhere"


def test_wrap_words_preserves_explicit_newlines():
    assert wrap_words("one\ntwo", 40) == ["one", "two"]


def test_record_task_no_longer_char_wraps():
    from lekiwi_tui.screens.record import RecordScreen

    scr = RecordScreen(MagicMock(), make_ctx())
    scr._ds_task = "Pick up the plate from right and place into the shelf."

    segs = scr._wrap_task(scr._ds_task, 46)

    assert segs[-1] != "."
    assert all(not ln.startswith(" ") for ln in segs)


# ── word-level editing ────────────────────────────────────────────────────────
def _field(value: str, cursor: int | None = None) -> TextField:
    f = TextField(value)
    if cursor is not None:
        f.cursor = cursor
    return f


def test_ctrl_left_jumps_to_the_start_of_the_previous_word():
    f = _field("pick up the plate")

    assert f.handle_key(Key(name=LEFT, ctrl=True)) is True
    assert f.cursor == len("pick up the ")


def test_ctrl_right_jumps_past_the_next_word():
    f = _field("pick up the plate", cursor=0)

    f.handle_key(Key(name=RIGHT, ctrl=True))

    assert f.cursor == len("pick")


def test_ctrl_left_from_inside_a_word_lands_on_that_word_start():
    f = _field("pick up the plate", cursor=len("pick up the pl"))

    f.handle_key(Key(name=LEFT, ctrl=True))

    assert f.cursor == len("pick up the ")


def test_ctrl_backspace_deletes_the_word_to_the_left():
    f = _field("pick up the plate")

    assert f.handle_key(Key(name=BACKSPACE, ctrl=True)) is True
    assert f.value == "pick up the "


def test_ctrl_h_also_deletes_a_word_because_terminals_send_it_for_ctrl_backspace():
    f = _field("pick up the plate")

    f.handle_key(Key(name="h", ctrl=True))

    assert f.value == "pick up the "


def test_alt_arrows_work_too():
    f = _field("pick up the plate")

    f.handle_key(Key(name=LEFT, alt=True))

    assert f.cursor == len("pick up the ")


def test_ctrl_u_and_ctrl_k_kill_to_start_and_end():
    f = _field("pick up the plate", cursor=len("pick up "))
    f.handle_key(Key(name="u", ctrl=True))
    assert f.value == "the plate"

    g = _field("pick up the plate", cursor=len("pick up "))
    g.handle_key(Key(name="k", ctrl=True))
    assert g.value == "pick up "


def test_word_keys_are_consumed_so_the_focus_ring_does_not_also_act():
    f = _field("hello world")

    for key in (Key(name=LEFT, ctrl=True), Key(name=RIGHT, ctrl=True),
                Key(name=BACKSPACE, ctrl=True), Key(name="u", ctrl=True)):
        assert f.handle_key(key) is True, key


def test_an_unhandled_ctrl_key_is_still_declined():
    """Up/Down and unknown ctrl combos must reach the screen, not be swallowed."""
    f = _field("hello")

    assert f.handle_key(Key(name="z", ctrl=True)) is False


def test_plain_typing_still_inserts():
    f = _field("")

    f.handle_key(Key(name="a"))

    assert f.value == "a"


# ── navigation order == visual order ──────────────────────────────────────────
# The bug this catches: eval's `cameras` sat after `flow` in the field list while
# rendering directly under `backend`. Invisible while the two shared a row; the moment
# they got their own rows, ↓ from Backend jumped past Cameras to Action steps and the
# Cameras row never highlighted. Any screen whose rows and field list can drift apart
# needs this.

def _highlighted_row(screen, width: int = 90) -> int:
    """Index of the row carrying the selection bar, or -1 if no row is highlighted."""
    for i, ln in enumerate(screen._body_lines(width)):
        if "▌" in _text(ln):
            return i
    return -1


def _assert_nav_matches_visual(screen, positions: int) -> None:
    seen = []
    for pos in range(positions):
        screen._fpos = pos
        seen.append((screen._cur(), _highlighted_row(screen)))
    unlit = [f for f, i in seen if i < 0]
    assert not unlit, f"these fields highlight no row at all: {unlit}"
    rows = [i for _, i in seen]
    assert rows == sorted(rows), f"navigation order != visual order: {seen}"
    assert len(set(rows)) == len(rows), f"two fields highlight the same row: {seen}"


def test_eval_navigation_order_matches_visual_order():
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx(gpu_name="RTX 2050"))

    _assert_nav_matches_visual(scr, len(scr._fields()))


def test_eval_navigation_order_matches_visual_order_on_the_rtc_backend():
    """rtc swaps Action steps for Action horizon, so the field list changes shape."""
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx(gpu_name="RTX 2050"))
    scr._backend = "rtc"

    _assert_nav_matches_visual(scr, len(scr._fields()))


def test_record_navigation_order_matches_visual_order():
    from lekiwi_tui.screens.record import RecordScreen
    from lekiwi_tui.screens.record import _FIELDS

    _assert_nav_matches_visual(RecordScreen(MagicMock(), make_ctx()), len(_FIELDS))


def test_every_eval_field_is_reachable_by_arrowing_down():
    from lekiwi_tui.framework.events import DOWN, Key
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx())
    seen = {scr._cur()}
    for _ in range(len(scr._fields()) * 2):
        scr.handle_key(Key(name=DOWN))
        seen.add(scr._cur())

    assert seen == set(scr._fields())


# ── camera mapping must never be implicit ─────────────────────────────────────
# A wrong routing does not raise: the policy just receives the wrong view and the robot
# acts on it. So the concrete pairs are always on screen, and anything unverified says so.

def _eval(cam_mode: str = "auto"):
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx(gpu_name="RTX 2050"))
    scr._cam_mode = cam_mode
    return scr


def test_the_concrete_camera_mapping_is_always_displayed():
    body = _body(_eval(), 88)

    assert "→" in body, "the robot-camera → policy-slot pairs must be on screen"


def test_the_mapping_shows_pairs_not_just_the_slot_names():
    """The pairing is not guessable: this robot maps wrist→camera2 and top→camera3, so a
    bare 'camera1/camera2/camera3' would let you assume the alphabetical pairing."""
    from lekiwi_tui.screens.eval import cam_pairs

    rmap = {"observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
            "observation.images.top": "observation.images.camera3"}

    assert cam_pairs("map", rmap) == [("front", "camera1"), ("wrist", "camera2"),
                                      ("top", "camera3")]


def test_native_mode_maps_every_camera_to_itself():
    from lekiwi_tui.screens.eval import cam_pairs

    rmap = {"observation.images.front": "observation.images.camera1"}

    assert cam_pairs("native", rmap) == [("front", "front")]


def test_forcing_a_mode_the_checkpoint_contradicts_warns(tmp_path):
    # A checkpoint whose training map AGREES with the yaml, so the only complaint left is
    # the forced mode. (These two used to run against the live config, which turned out to
    # have a real mapping mismatch — a test must not depend on the workspace being correct.)
    scr = _eval("native")
    scr._policy = _ckpt(tmp_path, _TRAINED)
    scr._rename_map = _TRAINED

    assert "⚠" in _body(scr, 88)
    assert "trained for map" in _body(scr, 88)


def test_a_clean_auto_detection_does_not_cry_wolf(tmp_path):
    scr = _eval("auto")
    scr._policy = _ckpt(tmp_path, _TRAINED)
    scr._rename_map = _TRAINED

    assert "⚠" not in _body(scr, 88)


def test_an_unverified_detection_is_marked_rather_than_shown_as_fact():
    from lekiwi_tui.screens.eval import detect_cam_detail

    mode, note, confident = detect_cam_detail("lerobot/smolvla_base", {"a": "b"})

    assert (mode, confident) == ("map", False)
    assert note == "checkpoint config unreadable"


def test_detect_cam_slots_keeps_its_two_tuple_contract():
    """eval.sh argv and the existing tests consume the 2-tuple; only the detail is new."""
    from lekiwi_tui.screens.eval import detect_cam_slots

    assert detect_cam_slots("lerobot/smolvla_base", {"a": "b"}) == (
        "map", "checkpoint config unreadable")


# ── the permutation a set check cannot see ────────────────────────────────────
_TRAINED = {"observation.images.front": "observation.images.camera1",
            "observation.images.top": "observation.images.camera2",
            "observation.images.wrist": "observation.images.camera3"}
_SWAPPED = {"observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
            "observation.images.top": "observation.images.camera3"}


def _ckpt(tmp_path, rename_map):
    import json

    d = tmp_path / "pretrained_model"
    d.mkdir()
    (d / "train_config.json").write_text(json.dumps({"rename_map": rename_map}))
    (d / "config.json").write_text(json.dumps(
        {"input_features": {f"observation.images.camera{i}": {"type": "VISUAL"}
                            for i in (1, 2, 3)}}))
    return str(d)


def test_training_rename_map_is_read_from_the_checkpoint(tmp_path):
    from lekiwi_tui.screens.eval import training_rename_map

    assert training_rename_map(_ckpt(tmp_path, _TRAINED)) == _TRAINED


def test_training_rename_map_falls_back_to_the_saved_preprocessor(tmp_path):
    import json
    from lekiwi_tui.screens.eval import training_rename_map

    d = tmp_path / "pretrained_model"
    d.mkdir()
    (d / "policy_preprocessor.json").write_text(json.dumps(
        {"steps": [{"config": {}}, {"config": {"rename_map": _TRAINED}}]}))

    assert training_rename_map(str(d)) == _TRAINED


def test_a_swapped_pair_is_detected_even_though_the_slot_SETS_match(tmp_path):
    """The whole point. Both sides use exactly {camera1, camera2, camera3}, so every
    membership check passes while two of the three views are crossed."""
    from lekiwi_tui.screens.eval import cam_map_conflicts, detect_cam_slots

    ckpt = _ckpt(tmp_path, _TRAINED)

    # the old set-subset test is perfectly happy
    assert detect_cam_slots(ckpt, _SWAPPED) == ("map", "camera1/camera2/camera3")
    # the pairwise test is not
    assert cam_map_conflicts(ckpt, _SWAPPED) == [
        ("camera2", "top", "wrist"),
        ("camera3", "wrist", "top"),
    ]


def test_no_conflict_when_the_two_maps_agree(tmp_path):
    from lekiwi_tui.screens.eval import cam_map_conflicts

    assert cam_map_conflicts(_ckpt(tmp_path, _TRAINED), _TRAINED) == []


def test_no_conflict_claimed_when_the_checkpoint_records_no_training_map(tmp_path):
    """Absence of evidence is not a mismatch — that would cry wolf on every hub policy."""
    import json
    from lekiwi_tui.screens.eval import cam_map_conflicts

    d = tmp_path / "pretrained_model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"input_features": {}}))

    assert cam_map_conflicts(str(d), _SWAPPED) == []


def test_forcing_map_over_a_disagreeing_yaml_is_an_error(tmp_path):
    """`map` sends the yaml's pairing verbatim, so a disagreement IS the hazard."""
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx())
    scr._policy = _ckpt(tmp_path, _TRAINED)
    scr._rename_map = _SWAPPED
    scr._cam_mode = "map"

    warning = scr._cam_warning()

    assert warning.startswith("MAPPING MISMATCH")
    assert "camera2: trained top, sending wrist" in warning
    assert "⚠" in _body(scr, 88)


def test_auto_resolves_to_the_checkpoint_mapping_and_stops_warning(tmp_path):
    """The fix, end to end: auto picks `trained`, so the disagreement is corrected rather
    than merely reported, and the screen says which mapping it is sending."""
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx())
    scr._policy = _ckpt(tmp_path, _TRAINED)
    scr._rename_map = _SWAPPED
    scr._cam_mode = "auto"

    assert scr._cam_resolved()[0] == "trained"
    assert scr._cam_warning() == ""
    body = _body(scr, 88)
    assert "top→camera2" in body and "wrist→camera3" in body, "must show the TRAINED pairs"
    assert "lekiwi.yaml disagrees" in body, "the override should still be stated"


def test_trained_mode_sends_the_checkpoint_pairs_not_the_yaml_pairs(tmp_path):
    from lekiwi_tui.screens.eval import cam_pairs, training_rename_map

    ckpt = _ckpt(tmp_path, _TRAINED)

    pairs = cam_pairs("trained", _SWAPPED, training_rename_map(ckpt))

    assert pairs == [("front", "camera1"), ("top", "camera2"), ("wrist", "camera3")]


def test_native_is_not_accused_of_sending_the_yaml_pairs(tmp_path):
    """Under native NOTHING is renamed, so a "sending wrist" message would describe argv
    that is never emitted. The forced-mode wording is the accurate one there."""
    from lekiwi_tui.screens.eval import EvalScreen

    scr = EvalScreen(MagicMock(), make_ctx())
    scr._policy = _ckpt(tmp_path, _TRAINED)
    scr._rename_map = _SWAPPED
    scr._cam_mode = "native"

    warning = scr._cam_warning()

    assert "MAPPING MISMATCH" not in warning
    assert "forced native" in warning and "trained for map" in warning


def test_the_picker_offers_exactly_two_modes():
    from lekiwi_tui.screens.eval import CAM_MODES

    assert CAM_MODES == ("auto", "native")
