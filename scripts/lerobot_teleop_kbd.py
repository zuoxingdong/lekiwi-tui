#!/usr/bin/env python
"""lerobot-teleoperate, but with an evdev keyboard listener (Wayland-safe hold-to-move).

scripts/teleop.sh execs this instead of the bare `lerobot-teleoperate` console script.
lerobot auto-attaches a KeyboardTeleop for the lekiwi base (lerobot_teleoperate.py:266),
which uses pynput's Listener — a silent no-op on Wayland, so the base never moves. We swap
pynput's Listener CLASS for an evdev-backed one that reads /dev/input directly (real
press/release BELOW the compositor), then hand off to lerobot's own main(). lerobot is
untouched.

Why a release-capable listener here and not the press-only path (record/eval use): teleop
base control is HOLD-to-move — lerobot reads the set of currently-held keys and maps it to a
continuous base velocity, so it needs real key-RELEASE events. A terminal in cbreak has none
(and OS key-repeat only repeats the last key), which would make the base run away + lose
diagonals. We patch in the unified KeyListener; because KeyboardTeleop passes on_release, it
opens a RELEASE-capable source: the kitty keyboard protocol over stdin if the terminal speaks
it (Ghostty), else evdev reading /dev/input directly below the compositor. Both give true
press/release and work on Wayland. evdev REQUIRES the user in the `input` group (/dev/input is
root:input); if neither kitty nor evdev is available, KeyListener opens an IDLE source that
emits nothing (is_alive() stays True so the arm still teleops, base idle — never a press-only
source that would latch a held key into a runaway base).
"""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from lekiwi_tui.kbd_listener import KeyListener  # noqa: E402

# Patch the pynput Listener CLASS (KeyboardTeleop resolves keyboard.Listener at connect()
# time, so this is picked up). Guard the import like the rollout shim: a headless run (no
# DISPLAY) raises ImportError acquiring an X connection — skip the patch and let lerobot's
# own PYNPUT_AVAILABLE path degrade exactly as stock.
try:
    import pynput.keyboard  # noqa: E402

    pynput.keyboard.Listener = KeyListener
except ImportError:
    pass

import lerobot.scripts.lerobot_teleoperate as teleop  # noqa: E402

if __name__ == "__main__":
    teleop.main()
