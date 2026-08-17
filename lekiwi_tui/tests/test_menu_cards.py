"""The card-grid main menu: coverage, digit honesty, icon widths, and 2-D navigation.

These guard the three ways this screen can silently lie: an action that no longer appears
in any card, a row whose visible digit is not the digit that jumps to it, and an icon whose
rendered width is not the width the layout reserved.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from lekiwi_tui.app_registry import ACTIONS
from lekiwi_tui.framework import theme
from lekiwi_tui.framework.events import DOWN, LEFT, RIGHT, UP, Key
from lekiwi_tui.screens import menu as menu_mod
from lekiwi_tui.screens.menu import MenuScreen, _GRID, _JUMPABLE, _STRIP

from conftest import make_ctx


def _screen() -> MenuScreen:
    return MenuScreen(MagicMock(), make_ctx(gpu_name="RTX 4090"))


def _press(screen: MenuScreen, name: str) -> None:
    screen.handle_key(Key(name=name))


def _select(screen: MenuScreen, action_id: str) -> None:
    """Place the cursor on an action BY ID. Navigation tests must not address rows by
    digit: adding one action shifts every digit after it, which is a property of the
    registry, not of the navigation being tested."""
    screen._sel = next(i for i, a in enumerate(ACTIONS) if a.id == action_id)


# ── coverage: the grid must not lose an action ────────────────────────────────
def test_every_action_appears_exactly_once_in_the_grid_or_strip():
    """A card name in _LAYOUT that no action claims would silently drop those actions off
    the menu — reachable by alias, invisible on screen. This is the test that catches it."""
    placed = [a for row in _GRID for cell in row for a in cell] + list(_STRIP)

    assert len(placed) == len(ACTIONS), "grid + strip must hold every action"
    assert set(placed) == set(ACTIONS)
    assert len(set(placed)) == len(placed), "an action is placed twice"


def test_no_card_in_the_layout_is_empty():
    for row in _GRID:
        for cell in row:
            assert cell, "an empty card would render as a titled box with nothing in it"


# ── digit honesty ─────────────────────────────────────────────────────────────
def test_displayed_digit_matches_the_digit_that_jumps_there():
    """Cards regroup COLLECT into COLLECT + DATA, so the card order no longer matches
    ACTIONS order. The digit a row SHOWS must still be the digit that runs it."""
    screen = _screen()
    for n, action in enumerate(_JUMPABLE[:10], start=1):
        key = "0" if n == 10 else str(n)  # phone-style: the 10th row rides on 0
        result = screen.handle_key(Key(name=key))
        assert getattr(result, "id", None) == action.id, f"digit {key}"
        assert screen.selected is action
    # rows past the ten digits carry no keycap (arrow/alias-reachable only; the
    # reachability test below covers them) — no digit may steal their identity


def test_setup_strip_has_no_digits():
    for action in _STRIP:
        assert action not in _JUMPABLE


# ── icon widths ───────────────────────────────────────────────────────────────
def test_no_registry_icon_carries_a_variation_selector():
    """U+FE0F is the whole bug: the layout library measures one cell, the terminal draws
    two, and every label after it shifts a column right."""
    offenders = [(a.id, a.icon) for a in ACTIONS if theme.icon_unstable(a.icon)]

    assert not offenders, f"variation-selector icons misalign their rows: {offenders}"


def test_every_icon_cell_is_exactly_two_columns():
    for action in ACTIONS:
        cell = theme.icon_cell(action.icon)
        cols = theme.glyph_cols(cell[0]) + (len(cell) - 1)

        assert cols == theme.ICON_CELL_W, f"{action.id}: {action.icon!r} -> {cols} cols"


def test_icon_cell_pads_a_narrow_glyph_and_leaves_a_wide_one_alone():
    assert theme.icon_cell("⚙") == "⚙ "     # plain gear: 1 column, padded to 2
    assert theme.icon_cell("🚀") == "🚀"     # wide emoji: already 2


def test_icon_cell_is_two_columns_in_ascii_mode(monkeypatch):
    """ASCII fallbacks are one column, so the cell must pad them too or the labels in a
    no-emoji terminal sit one column left of where the cards expect them."""
    monkeypatch.setattr(theme, "EMOJI_ENABLED", False)
    for action in ACTIONS:
        assert len(theme.icon_cell(action.icon)) == theme.ICON_CELL_W


# ── navigation ────────────────────────────────────────────────────────────────
def test_down_walks_within_a_card_then_spills_into_the_card_below():
    screen = _screen()
    assert screen.selected.id == "host-launch"      # HOST, first row

    _press(screen, DOWN)
    assert screen.selected.id == "host-kill"        # still HOST
    _press(screen, DOWN)
    assert screen.selected.id == "edit-dataset"     # spilled into DATA, same column


def test_right_switches_column_keeping_the_index():
    screen = _screen()
    _press(screen, RIGHT)
    assert screen.selected.id == "teleop"           # HOST[0] -> COLLECT[0]

    _press(screen, DOWN)
    _press(screen, LEFT)
    assert screen.selected.id == "host-kill"        # COLLECT[1] -> HOST[1]


def test_right_clamps_the_index_into_a_shorter_card():
    screen = _screen()
    _select(screen, "view")                         # DATA[2]: the third row of its card

    _press(screen, RIGHT)                           # LEARN has only two actions
    assert screen.selected.id == "eval"


def test_up_from_the_first_card_wraps_into_the_setup_strip():
    screen = _screen()
    _press(screen, UP)

    assert screen.selected is _STRIP[-1]


def test_down_from_the_last_card_enters_the_setup_strip_then_wraps_to_the_top():
    screen = _screen()
    _select(screen, "eval")                         # LEARN[1]: the last row of the last card
    _press(screen, DOWN)
    assert screen.selected is _STRIP[0]

    _press(screen, DOWN)
    assert screen.selected.id == "host-launch"


def test_left_right_walk_along_the_setup_strip():
    screen = _screen()
    _press(screen, UP)                              # into the strip, last entry
    assert screen.selected is _STRIP[-1]

    _press(screen, RIGHT)
    assert screen.selected is _STRIP[0]             # wraps within the strip
    _press(screen, LEFT)
    assert screen.selected is _STRIP[-1]


def test_down_alone_walks_one_column_and_returns():
    """Grid semantics: ↓ stays in its column. Recorded deliberately — with the old flat
    list ↓ reached everything, and it no longer does, which is why ←→ are in the hints."""
    screen = _screen()
    walk = [screen.selected.id]
    for _ in range(6):
        _press(screen, DOWN)
        walk.append(screen.selected.id)

    assert walk == ["host-launch", "host-kill", "edit-dataset", "replay", "view",
                    "setup-pi", "host-launch"]


def test_every_action_is_reachable_with_the_arrows_alone():
    """The reachability guarantee that matters: no action may be stranded behind a digit.
    Breadth-first over all four moves from the default selection."""
    frontier, seen = [_screen().selected.id], set()
    while frontier:
        start = frontier.pop()
        if start in seen:
            continue
        seen.add(start)
        for key in (UP, DOWN, LEFT, RIGHT):
            screen = _screen()
            screen._sel = next(i for i, a in enumerate(ACTIONS) if a.id == start)
            _press(screen, key)
            frontier.append(screen.selected.id)

    assert seen == {a.id for a in ACTIONS}


def test_hjkl_mirror_the_arrows():
    a, b = _screen(), _screen()
    for key_a, key_b in ((DOWN, "j"), (RIGHT, "l"), (UP, "k"), (LEFT, "h")):
        _press(a, key_a)
        _press(b, key_b)
        assert a.selected is b.selected


# ── card rows ─────────────────────────────────────────────────────────────────
def test_cards_carry_no_description_lines():
    """Dropped by request (2026-08-15): a daily driver knows what Teleoperate is. One
    line per action — a reappearing muted description line is a regression."""
    screen = _screen()
    host_cell = _GRID[0][0]

    assert len(screen._card_lines(host_cell, 40)) == len(host_cell)


# ── the small-terminal fallback ───────────────────────────────────────────────
def test_a_short_terminal_falls_back_to_the_flat_list(monkeypatch):
    """The card grid needs ~27 rows. A launcher still has to render below that."""
    screen = _screen()
    drawn: list[str] = []
    monkeypatch.setattr(MenuScreen, "_draw_cards",
                        lambda self, f, a: drawn.append("cards"))
    monkeypatch.setattr(MenuScreen, "_flat_body",
                        lambda self: drawn.append("flat") or MagicMock())

    import pyratatui as p

    frame = MagicMock()
    screen.draw(frame, p.Rect(0, 0, 100, 40))
    assert drawn == ["cards"]

    drawn.clear()
    screen.draw(frame, p.Rect(0, 0, 100, menu_mod._MIN_CARD_H - 1))
    assert drawn == ["flat"]

    drawn.clear()
    screen.draw(frame, p.Rect(0, 0, menu_mod._MIN_CARD_W - 1, 40))
    assert drawn == ["flat"]


def test_the_real_card_draw_path_renders_every_panel():
    """Exercises _draw_cards for real (the fallback test stubs it out): five panels — the
    status card plus four section cards — and no exception from the layout arithmetic."""
    import pyratatui as p

    screen = _screen()
    panels: list[tuple[str, int]] = []
    original = MenuScreen._draw_panel
    MenuScreen._draw_panel = lambda self, f, a, t, ls: panels.append((t, len(ls)))
    try:
        screen.draw(MagicMock(), p.Rect(0, 0, 100, 40))
    finally:
        MenuScreen._draw_panel = original

    assert [t for t, _ in panels] == ["HARDWARE", "SESSION", "SOFTWARE", "COMPUTE",
                                      "HOST", "COLLECT", "DATA", "LEARN"]
    # One line per action in a section card; the HARDWARE card is 4 rows with a blank
    # between each at this height (the COMPUTE card's row count depends on what the live
    # sampler could read, so it is pinned in test_sysstat instead).
    assert dict(panels)["HARDWARE"] == 7
    assert dict(panels)["DATA"] == 3
    assert dict(panels)["LEARN"] == 2


def test_a_terminal_just_above_minimum_keeps_the_grid_with_compact_cards():
    """The regression from the band redesign: growing the status area must never demote a
    terminal that used to get cards down to the flat list. Just above _MIN_CARD_H the
    cards drop their blank lines (HARDWARE = 4 rows, not 7) and every panel renders."""
    import pyratatui as p

    screen = _screen()
    panels: list[tuple[str, int]] = []
    original = MenuScreen._draw_panel
    MenuScreen._draw_panel = lambda self, f, a, t, ls: panels.append((t, len(ls)))
    try:
        screen.draw(MagicMock(), p.Rect(0, 0, 100, menu_mod._MIN_CARD_H + 1))
    finally:
        MenuScreen._draw_panel = original

    assert [t for t, _ in panels] == ["HARDWARE", "SESSION", "SOFTWARE", "COMPUTE",
                                      "HOST", "COLLECT", "DATA", "LEARN"]
    assert dict(panels)["HARDWARE"] == 4, "compact cards: rows packed, no blanks"


def test_hardware_and_software_cards_read_real_config():
    """robot type, leader port, and conda env must come from the config rather than
    being hard-coded; one item per line so the values form a scannable column."""
    screen = _screen()

    hw = ["".join(sp.content for sp in line.spans)
          for line in screen._hardware_lines(spacious=False)]
    sw = ["".join(sp.content for sp in line.spans)
          for line in screen._software_lines(spacious=False)]

    assert hw[0].startswith("robot type")
    assert hw[1].startswith("leader arm")
    assert hw[2].startswith("calibration")
    assert hw[3].startswith("cameras")
    assert sw[0].startswith("lerobot")
    assert sw[1].startswith("conda env")


def test_the_cameras_row_names_the_cameras_without_emoji():
    """User rules (2026-08-15): show the cameras BY NAME on one line, plain glyphs only
    inside the status cards."""
    screen = _screen()

    cams = "".join(sp.content for sp in
                   screen._hardware_lines(60, spacious=False)[3].spans)

    assert "front" in cams, "the camera names from lekiwi.yaml, not a count"
    assert "wrist" in cams
    assert all(ord(ch) < 0x2700 for ch in cams), f"emoji-range glyph in {cams!r}"


def test_the_cameras_row_elides_instead_of_clipping():
    """A narrow card must drop tail names into '+N', never let the border cut a name
    mid-word (the delete-modal lesson, again)."""
    screen = _screen()

    cams = "".join(sp.content for sp in
                   screen._hardware_lines(24, spacious=False)[3].spans)

    assert "+" in cams, f"expected an elision marker in {cams!r}"
    assert len(cams) <= 24, "the row must fit the width it was given"
