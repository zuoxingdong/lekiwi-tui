#!/usr/bin/env python
"""lerobot-rollout (eval) passthrough — upstream 0.6 handles Wayland keys itself.

scripts/eval.sh execs this instead of the bare `lerobot-rollout` console script (kept
so the launcher argv stays stable). Since lerobot 0.6, no keyboard patching is needed
here: the episodic rollout strategy calls lerobot's own init_keyboard_listener()
(utils/keyboard_input.py), which falls back to a TerminalKeyListener reading the
controlling TTY when pynput cannot capture (Wayland / headless SSH). Rollout keys are
DISCRETE taps (right/left/esc plus the n/r/q letter equivalents), which the press-only
TTY backend serves fine — unlike hold-to-move base teleop, which is why the record and
teleop shims still patch (see lerobot_record_kbd.py / lerobot_teleop_kbd.py).

History: on lerobot 0.5.x this shim swapped pynput's Listener class for the package's
stdin reader because the rollout strategies constructed pynput listeners directly (a
silent no-op on Wayland). 0.6 routed them through init_keyboard_listener, so the swap
would intercept nothing and upstream's fallback does the job — reuse it, don't patch.

All command-line flags pass straight through to lerobot (draccus reads sys.argv).
"""

import lerobot.scripts.lerobot_rollout as rollout

if __name__ == "__main__":
    rollout.main()
