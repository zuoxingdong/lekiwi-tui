#!/usr/bin/env python
"""lerobot-rollout (eval), but with a Wayland-safe stdin keyboard listener.

scripts/eval.sh execs this instead of the bare `lerobot-rollout` console script. It
is a thin shim: on Wayland, pynput's Listener is a silent no-op (see the local
kbd_listener.py for the full why), which kills the eval/rollout keys —
episodic right/left/esc, highlight save/push/esc, dagger space/tab/enter/esc. We swap
pynput's Listener CLASS for a stdin reader that emits REAL pynput Key/KeyCode objects,
then hand off to lerobot's own main(). The lerobot repo is never modified.

Unlike the record shim (which patches lerobot's init_keyboard_listener function),
rollout's three keyboard consumers — control_utils.init_keyboard_listener, the
highlight strategy, and the dagger keyboard strategy — all do `from pynput import
keyboard` and look up `keyboard.Listener` at CALL time. So patching the class on the
real pynput.keyboard module (BEFORE main() runs setup) is transparently picked up by
all three. The real Key/KeyCode classes are left intact so every existing
`key == keyboard.Key.esc` / `key.char == save_key` comparison keeps working. The dagger
evdev foot-pedal path shares nothing with pynput and is untouched.

All command-line flags pass straight through to lerobot (draccus reads sys.argv).
"""

import os
import sys

# Make the lekiwi_tui package importable regardless of CWD: this file lives at
# <lekiwi-tui>/scripts/, so its grandparent dir holds the lekiwi_tui/ package.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from lekiwi_tui.kbd_listener import KeyListener  # noqa: E402

# Import the real pynput.keyboard module FIRST (so its real backend + Key/KeyCode
# exist), then replace only the Listener attribute. Consumers resolve keyboard.Listener
# lazily at construction time, so this swap is seen by highlight, dagger, AND episodic
# (control_utils) — one patch point, low blast radius. KeyListener is the unified
# pynput-Listener work-alike (over open_key_source: kitty stdin, else cbreak); the eval
# consumers pass on_press only, so it opens a press-only source (no hold-to-move needed).
# Importing pynput.keyboard eagerly acquires an X connection and raises ImportError
# headless (ssh w/o X, cron); skip the patch there and let lerobot's own is_headless()
# path degrade exactly as stock, instead of crashing the whole entrypoint.
try:
    import pynput.keyboard  # noqa: E402

    pynput.keyboard.Listener = KeyListener
except ImportError:
    pass

import lerobot.scripts.lerobot_rollout as rollout  # noqa: E402

if __name__ == "__main__":
    rollout.main()
