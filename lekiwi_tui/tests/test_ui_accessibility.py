from __future__ import annotations

import json
import os
import subprocess
import sys


_THEME_ENV_KEYS = (
    "NO_COLOR",
    "LEKIWI_NO_COLOR",
    "LEKIWI_ASCII",
    "LEKIWI_TUI_ASCII",
    "LEKIWI_NO_EMOJI",
    "LEKIWI_TUI_NO_EMOJI",
    "TERM",
)


def _theme_probe(extra_env: dict[str, str]) -> dict:
    env = os.environ.copy()
    for key in _THEME_ENV_KEYS:
        env.pop(key, None)
    env.update(extra_env)
    code = """
import json
from lekiwi_tui.framework import theme
print(json.dumps({
    "ascii": theme.ASCII_MODE,
    "emoji": theme.EMOJI_ENABLED,
    "color": theme.COLOR_ENABLED,
    "icon": theme.action_icon("\\U0001f680"),
    "title": theme.title_mark(),
    "selector": theme.selector(True),
    "play": theme.play_mark(),
    "dot": theme.status_dot(),
    "choice": theme.choice("on"),
    "rule": theme.rule(3),
    "meter": theme.meter_segments(0.5, 4),
    "spark": theme.sparkline([0.0, 1.0], 2, lo=0.0, hi=1.0),
    "key": theme.key_label("\\u2191\\u2193/jk \\u2190\\u2192 \\u23ce"),
}))
"""
    out = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    return json.loads(out)


def test_ascii_mode_replaces_wide_glyphs_and_emoji():
    probe = _theme_probe({"LEKIWI_TUI_ASCII": "1", "TERM": "xterm-256color"})

    assert probe["ascii"] is True
    assert probe["emoji"] is False
    assert probe["color"] is True
    assert probe["icon"] == ">"
    assert probe["title"] == "*"
    assert probe["selector"] == "> "
    assert probe["play"] == ">"
    assert probe["dot"] == "*"
    assert probe["choice"] == "< on >"
    assert probe["rule"] == "---"
    assert probe["meter"] == ["==", "--"]
    assert probe["spark"] == ".#"
    assert probe["key"] == "up/down/jk left/right enter"


def test_term_dumb_forces_ascii_no_color_no_emoji():
    probe = _theme_probe({"TERM": "dumb"})

    assert probe["ascii"] is True
    assert probe["emoji"] is False
    assert probe["color"] is False


def test_no_emoji_keeps_unicode_line_art_when_ascii_is_off():
    probe = _theme_probe({"LEKIWI_TUI_NO_EMOJI": "1", "TERM": "xterm-256color"})

    assert probe["ascii"] is False
    assert probe["emoji"] is False
    assert probe["icon"] == ">"
    assert probe["title"] == "\u25c6"
    assert probe["rule"] == "\u2500" * 3

