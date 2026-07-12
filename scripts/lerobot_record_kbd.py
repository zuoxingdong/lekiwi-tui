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
#
# lerobot 0.6 note: upstream's new pynput_can_capture() gate returns False on Wayland and
# KeyboardTeleop.connect() then never constructs keyboard.Listener, bypassing this patch.
# Upstream's Wayland fallback is press-only by design (no key-release in cbreak) and does
# not serve hold-to-move base control, so force the gate open where the class patch is in
# place (same try block; see lerobot_teleop_kbd.py for the full rationale).
try:
    import pynput.keyboard  # noqa: E402

    pynput.keyboard.Listener = _SharedBaseKeyListener

    import lerobot.teleoperators.keyboard.teleop_keyboard as _teleop_keyboard  # noqa: E402

    if hasattr(_teleop_keyboard, "pynput_can_capture"):  # 0.6+; absent on 0.5.x
        _teleop_keyboard.pynput_can_capture = lambda: True
except ImportError:
    pass

import lerobot.scripts.lerobot_record as rec  # noqa: E402

# (1) EPISODE CONTROL — patch the name as bound in the record module's namespace
# (`from lerobot.utils.keyboard_input import init_keyboard_listener` on 0.6, so patching
# the source module now would be too late). record() looks this up as a module global at
# call time, so this swap takes effect for the run.
#
# Since lerobot 0.6 upstream DOES ship its own Wayland fallback for these episode keys
# (TerminalKeyListener, cbreak reads on the controlling TTY), and standalone
# `lerobot-record` works fine with it. This shim still replaces the function for ONE
# reason: coexistence with the base listener (#2). Both consumers want fd 0 — the base
# rides the kitty-protocol stdin reader on Ghostty, and upstream's episode listener would
# open a SECOND reader on the same fd; the two threads then steal bytes from each other
# (arrows land in the base reader and vice versa). share=True subscribes the episode keys
# to the SAME single stdin reader as the base instead. The events contract is identical
# to upstream's (exit_early / rerecord_episode / stop_recording).
rec.init_keyboard_listener = lambda: make_stdin_listener(share=True)

if __name__ == "__main__":
    rec.main()
