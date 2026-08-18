#!/usr/bin/env python
"""lerobot-rollout (dagger), but with a Wayland-safe keyboard for BOTH key consumers.

`scripts/dagger.sh` execs this instead of the bare `lerobot-rollout` console script.
It is the dagger twin of lerobot_record_kbd.py: swap lerobot's pynput keyboard paths
for the package's unified stdin/evdev reader, then hand off to lerobot's own main().
The lerobot repo is never modified.

A dagger session has TWO keyboard consumers running at once:
  1. SESSION KEYS — the DAgger strategy's `create_key_listener(dispatch)` (space =
     pause/resume, tab = correction, enter = upload, ESC = stop). Discrete taps;
     lerobot 0.6's own terminal fallback would serve them fine ALONE.
  2. BASE NAVIGATION — the composite leader's `KeyboardTeleop` (wasd + zx + r/f),
     which builds a `pynput.keyboard.Listener` (the CLASS) at connect() time. On
     Wayland pynput cannot capture, upstream's gate then skips the listener entirely,
     and the base never moves during corrections. Hold-to-move needs real key
     RELEASE, which no terminal cbreak reader reports — the class patch below routes
     it to the kitty-protocol stdin reader (Ghostty) or evdev instead.

Both consumers default to stdin (fd 0). On a kitty terminal each would otherwise open
its OWN stdin reader and the two threads would race for the bytes (a tab landing in
the base reader is a lost correction toggle), so both patches set `share=True`: they
subscribe to ONE shared stdin reader that fans every key out to both. On a non-kitty
Wayland tty they land on different devices anyway (base → evdev, session keys →
cbreak), so sharing is a no-op there.

All command-line flags pass straight through to lerobot (draccus reads sys.argv), so
the argv built by dagger.sh is exactly what lerobot sees.
"""

import os
import sys

# Make the lekiwi_tui package importable regardless of CWD: this file lives at
# <lekiwi-tui>/scripts/, so its grandparent dir holds the lekiwi_tui/ package.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from lekiwi_tui.kbd_listener import KeyListener, make_dispatch_listener  # noqa: E402


class _SharedBaseKeyListener(KeyListener):
    """The base KeyboardTeleop's listener on the SHARED stdin reader (share=True) so it
    fans out from the same reader as the session-key listener instead of racing a second
    one on fd 0. KeyboardTeleop passes on_release, so KeyListener opens a release-capable
    source (kitty stdin if the terminal speaks it, else evdev) — hold-to-move base control.

    A real subclass, NOT a lambda: KeyboardTeleop.is_connected does
    `isinstance(self.listener, keyboard.Listener)`, so the patched name must be a class.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("share", True)
        super().__init__(*args, **kwargs)


# (2) BASE NAVIGATION — patch the pynput Listener CLASS + force upstream's
# pynput_can_capture() gate open, exactly as lerobot_record_kbd.py does (see there for
# the full 0.6 rationale). Guarded: a headless run may fail to import pynput at all —
# skip the patch and let lerobot's own PYNPUT_AVAILABLE path degrade as stock.
try:
    import pynput.keyboard  # noqa: E402

    pynput.keyboard.Listener = _SharedBaseKeyListener

    import lerobot.teleoperators.keyboard.teleop_keyboard as _teleop_keyboard  # noqa: E402

    if hasattr(_teleop_keyboard, "pynput_can_capture"):  # 0.6+; absent on 0.5.x
        _teleop_keyboard.pynput_can_capture = lambda: True
except ImportError:
    pass

# (1) SESSION KEYS — patch the name as bound in the dagger strategy's namespace
# (`from lerobot.utils.keyboard_input import create_key_listener`, so patching the source
# module would be too late). The strategy calls it at setup() with (dispatch,
# controls_help=...); the replacement honors the dispatch contract (canonical key names)
# over the SAME shared stdin reader as the base listener. controls_help is display-only.
import lerobot.rollout.strategies.dagger as _dagger_strategy  # noqa: E402

_dagger_strategy.create_key_listener = (
    lambda dispatch, **_kw: make_dispatch_listener(dispatch, share=True)
)

import lerobot.scripts.lerobot_rollout as _rollout  # noqa: E402

if __name__ == "__main__":
    _rollout.main()
