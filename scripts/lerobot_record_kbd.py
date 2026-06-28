#!/usr/bin/env python
"""lerobot-record, but with a Wayland-safe keyboard for BOTH the episode keys and the base.

`scripts/record.sh` execs this instead of the bare `lerobot-record` console script. It is
a thin shim that swaps lerobot's pynput keyboard listeners (a silent no-op on Wayland — see
lekiwi_tui/kbd_listener.py for the full why) for the package's unified stdin/evdev
reader, then hands off to lerobot's own `main()`. The lerobot repo is never modified.

LeKiwi record has TWO keyboard consumers running at once (lerobot_record.py):
  1. EPISODE CONTROL — `init_keyboard_listener()` (arrows / ESC: right=next, left=re-record,
     esc=stop). Patched below as a FUNCTION (see reasons 1+2).
  2. BASE NAVIGATION — a `KeyboardTeleop` auto-attached for the mobile base (wasd + zx + r/f),
     which builds a `pynput.keyboard.Listener` (the CLASS) at connect() time. The original
     shim patched ONLY #1, so on Wayland the base listener stayed the dead pynput one and
     wasd never moved the base (arrows worked, wasd didn't). We now ALSO patch the class.

Both consumers default to stdin (fd 0). On a kitty terminal each would otherwise open its
OWN stdin reader and the two threads would race for the bytes, so both patches set
`share=True`: the base + episode listeners subscribe to ONE shared stdin reader that fans
every key out to both (pynput's "every listener sees every key", over a single fd). On a
non-kitty Wayland tty they land on different devices anyway (base->evdev, episode->cbreak),
so sharing is a no-op there.

All command-line flags pass straight through to lerobot (draccus `@parser.wrap()` reads
sys.argv), so the argv built by record.sh is exactly what lerobot sees.
"""

import os
import sys

# Make the lekiwi_tui package importable regardless of CWD: this file lives at
# <lekiwi-tui>/scripts/, so its grandparent dir holds the lekiwi_tui/ package.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from lekiwi_tui.kbd_listener import KeyListener, make_stdin_listener  # noqa: E402


class _SharedBaseKeyListener(KeyListener):
    """The base KeyboardTeleop's listener, riding the SHARED stdin reader (share=True) so it
    fans out from the same reader as the episode-control listener instead of racing a second
    one on fd 0. KeyboardTeleop passes on_release, so KeyListener opens a release-capable
    source (kitty stdin if the terminal speaks it, else evdev) — hold-to-move base control.

    A real subclass, NOT a lambda: KeyboardTeleop.is_connected does
    `isinstance(self.listener, keyboard.Listener)`, so the patched name must be a class.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("share", True)
        super().__init__(*args, **kwargs)


# (2) BASE NAVIGATION — patch the pynput Listener CLASS (KeyboardTeleop resolves
# keyboard.Listener at connect() time, so this is picked up). Guard the import like the
# teleop/rollout shims: a headless run (no DISPLAY) raises ImportError acquiring an X
# connection — skip the patch and let lerobot's own PYNPUT_AVAILABLE path degrade as stock.
try:
    import pynput.keyboard  # noqa: E402

    pynput.keyboard.Listener = _SharedBaseKeyListener
except ImportError:
    pass

import lerobot.scripts.lerobot_record as rec  # noqa: E402

# (1) EPISODE CONTROL — patch the name as bound in the record module's namespace (it did
# `from lerobot.common.control_utils import init_keyboard_listener` at import, so patching
# control_utils now would be too late). record() looks this up as a module global at call
# time, so this swap takes effect for the run. share=True so it joins the base's shared
# stdin reader (see the module docstring) rather than opening a second one.
#
# Unlike the base (#2) and the rollout shim, episode control replaces the whole
# init_keyboard_listener FUNCTION rather than relying on the class patch. Two reasons:
#   1. Immune to a future is_headless() Wayland fix. lerobot only builds the pynput
#      Listener AFTER `if is_headless(): return None, events`. A class-patch works
#      today (is_headless() is False on Wayland), but the day lerobot teaches
#      is_headless() to detect Wayland, record would take that early return and the
#      patched Listener class would never be constructed — silently stranding the
#      episode keys again. Replacing the whole function bypasses is_headless() entirely.
#   2. Drops control_utils' per-key debug prints. lerobot's on_press prints
#      "Right arrow key pressed. Exiting loop..." etc. on every mapped key, which would
#      smear the record loop's tqdm bars. make_stdin_listener has no such prints.
rec.init_keyboard_listener = lambda: make_stdin_listener(share=True)

if __name__ == "__main__":
    rec.main()
