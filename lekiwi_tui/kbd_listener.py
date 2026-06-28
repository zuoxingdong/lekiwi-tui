"""stdin keyboard listener — a drop-in replacement for lerobot's pynput one.

WHY THIS EXISTS
---------------
lerobot's `init_keyboard_listener` (lerobot/common/control_utils.py) uses
`pynput` to grab the arrow keys / ESC that drive the record loop:
  right  -> exit_early       (stop this episode, encode video, advance)
  left   -> rerecord_episode (redo the last episode)
  esc    -> stop_recording   (end the session)

On a **Wayland** desktop session this silently does nothing. lerobot's
`is_headless()` only checks whether `pynput` *imports* (it does), so it builds a
`pynput.keyboard.Listener`. On Linux pynput's only listener backend is `_xorg`,
which captures global keys via the X RECORD extension through XWayland. A native
Wayland terminal routes keystrokes through the compositor, never the X server, so
the listener receives ZERO events. Recording then only responds to Ctrl+C (a
TTY-delivered SIGINT, independent of pynput), which kills the process instead of
cleanly stopping + encoding the episode.

THE FIX
-------
record runs in the foreground with a real TTY, so we read the keys straight from
stdin instead of pynput. `scripts/lerobot_record_kbd.py` monkeypatches
`lerobot.scripts.lerobot_record.init_keyboard_listener` to `make_stdin_listener`
before calling lerobot's `main()`, so the lerobot repo is never touched. Works on
Wayland and X11 alike. Only requirement: the terminal must be focused (you are
sitting at it), since this reads the TTY rather than grabbing keys globally.

NOTES
-----
- cbreak mode (not raw) leaves ISIG on, so Ctrl+C keeps working, and leaves OPOST
  on, so lerobot's stdout (SVT logs, tqdm bars) renders normally.
- ESC is a prefix of the arrow escape sequences (ESC [ C etc.), so a bare ESC is
  disambiguated with a short read timeout.
- termios is restored on stop() AND via atexit, so a crash never wedges the tty.
"""

from __future__ import annotations

import atexit
import os
import re
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

ESC = b"\x1b"
# After an ESC, how long to wait for a CSI/SS3 tail before deciding it's a bare ESC.
# Assumes the arrow bytes (ESC [ C) arrive together — true for a local TTY. Over a
# laggy SSH link a split ESC...[C could misread as a bare ESC (=stop); bump this if
# record is ever driven remotely (record currently runs locally, SSH_TTY empty).
_ESC_TIMEOUT = 0.05
# Idle poll cadence for the reader thread (also bounds shutdown latency).
_POLL = 0.1

# CSI/SS3 arrow final byte -> token name (also a valid pynput Key.<name> attr).
_ARROW_FINALS = {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}


# ── shared TTY/escape constants ───────────────────────────────────────────────
# CbreakReader (below) reuses these: ESC (the escape byte), _ESC_TIMEOUT (bare-ESC
# disambiguation window), _ARROW_FINALS (CSI/SS3 final byte -> name). The unified
# KeyEvent layer (KeyEvent + the three backends + open_key_source) follows; the two
# thin adapters (KeyListener, make_stdin_listener) sit at the very bottom, after
# open_key_source, since they call it.


def _open_keyboard_devices():
    """Open every /dev/input device that looks like a keyboard (has the letter keys).
    Raises on permission errors (user not in `input` group) so start() can degrade."""
    from evdev import InputDevice, ecodes, list_devices
    devs = []
    for path in list_devices():  # raises/empty without /dev/input read access
        try:
            d = InputDevice(path)
        except Exception:
            continue
        keys = d.capabilities().get(ecodes.EV_KEY, [])
        if ecodes.KEY_A in keys and ecodes.KEY_Z in keys and ecodes.KEY_SPACE in keys:
            devs.append(d)
        else:
            try:
                d.close()
            except Exception:
                pass
    return devs


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED KEYBOARD LAYER (KeyEvent + three backends + open_key_source)
# ══════════════════════════════════════════════════════════════════════════════
# This is the single way to READ keys. The two thin ADAPTERS at the bottom of this
# file (KeyListener for eval+teleop, make_stdin_listener for record) are the only
# things the shims touch; both sit on top of open_key_source. The legacy listeners
# (StdinKeyListener / EvdevKeyListener / _CbreakReader) have been RETIRED — they are
# reimplemented as the adapters over the three backends. The unification only changes
# HOW keys are read; no keymap changes.
#
# Every backend speaks one vocabulary — KeyEvent — and pushes events to an
# on_event(KeyEvent) callback from a daemon reader thread. A normalized key is
# either a single printable char ("w","s",…) or one functional name from a fixed
# set {esc,up,down,left,right,space,tab,enter}. Modifiers are a bitmask (0 = none).
#
# ROBOT SAFETY (HARD RULE 2): release-capable backends (Kitty, Evdev) track a held
# set and, on stop / EOF / any teardown, emit a synthetic RELEASE for every key
# still down, so a dying stream can never leave the base latched "forward".


# ── the proven kitty parser, owned by the package ────────────────────────────
# parse_one + the CSI-u regex + the push/pop/query sequences + detect_support are
# the kitty primitives the USER confirmed on real hardware (press AND release, via
# dev/kitty_probe.py). They live HERE so the package is SELF-CONTAINED (it must not
# depend on a dev/ script at runtime). dev/kitty_probe.py keeps a byte-identical
# standalone copy as a runnable diagnostic. Do NOT re-derive these — this is the
# empirically-validated parser; any change must be matched + re-confirmed on hardware.
_CSI = "\x1b["
_KITTY_QUERY = f"{_CSI}?u"         # CSI ? u    query current progressive-enhancement flags
_KITTY_PUSH = f"{_CSI}>11u"        # CSI > 11 u push 1|2|8 (disambiguate|event-types|all-keys)
_KITTY_POP = f"{_CSI}<u"           # CSI < u    pop our flags off the terminal's stack
_KITTY_FLAGS_WANTED = 1 | 2 | 8    # minimum for plain-text-key PRESS+REPEAT+RELEASE

# CSI <key>[:alt][;mods[:event]][;text] <terminator> — group 1 key-code, 3 modifiers
# (1+bitmask), 4 event-type (1/2/3, default press), 6 terminator (u, ~, or a letter).
_CSI_U = re.compile(
    r"\x1b\["
    r"(\d+)"                      # 1: key-code
    r"(?::(\d+))?"                # 2: alternate key-code (ignored)
    r"(?:;(\d+)(?::(\d+))?)?"     # 3: modifiers, 4: event-type
    r"(?:;([\d:]+))?"             # 5: associated text
    r"([u~ABCDEFHPQRSEodc])"      # 6: terminator
)
_KITTY_RESP = re.compile(r"\x1b\[\?(\d+)u")  # query response: CSI ? <flags> u


def parse_one(seq: str):
    """Decode a single CSI-u sequence -> (codepoint, modifiers, event_type) or None.
    modifiers default to 1 (kitty: 1 == no modifiers); event_type defaults to press (1)."""
    m = _CSI_U.fullmatch(seq)
    if m is None:
        return None
    cp = int(m.group(1))
    mods = int(m.group(3)) if m.group(3) else 1
    et = int(m.group(4)) if m.group(4) else 1
    return cp, mods, et


def _kitty_detect_support(fd: int, timeout: float = 0.3) -> int | None:
    """Return the terminal's current kitty flags (int) or None if unsupported. Writes a
    CSI ? u query + a CSI 6n cursor-position sentinel (a non-kitty terminal still replies
    promptly, so we never hang the generous timeout)."""
    os.write(sys.stdout.fileno(), _KITTY_QUERY.encode())
    os.write(sys.stdout.fileno(), f"{_CSI}6n".encode())
    buf = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
        if not r:
            break
        buf += os.read(fd, 64).decode(errors="replace")
        if "R" in buf:  # cursor-position reply arrived => the terminal has answered
            break
    m = _KITTY_RESP.search(buf)
    return int(m.group(1)) if m else None

# kitty event-type sub-field values (group 4 of the CSI-u match).
_KP_PRESS, _KP_REPEAT, _KP_RELEASE = 1, 2, 3
_ET_NAME = {_KP_PRESS: "press", _KP_REPEAT: "repeat", _KP_RELEASE: "release"}

# u-form codepoints that normalize to a FUNCTIONAL name (not a printable char).
# Under report-all-keys these arrive as text-key-like CSI-u escapes: esc=27,
# space=32 (printable but we name it "space" to match the fixed functional set),
# tab=9, enter=13 (CR). Everything else printable maps to its char.
_KITTY_CODEPOINT_NAME = {27: "esc", 32: "space", 9: "tab", 13: "enter", 10: "enter"}
# Arrow terminators (CSI/SS3 final letter) -> functional name. Shared by the bare
# arrow branch (unmodified arrow PRESS = bare CSI A / SS3 OA per the kitty spec —
# the `;mods:event` fields are omitted at their defaults, so _CSI_U does NOT match
# it) and the functional-form branch (arrow RELEASE = CSI 1:3 A, which _CSI_U does
# match). Handling BOTH forms is what keeps record/eval arrows working on Ghostty.
_KITTY_ARROW_FINALS = {"A": "up", "B": "down", "C": "right", "D": "left"}

# kitty modifier bits that are LOCK STATE, not an intentional chord: caps_lock (64),
# num_lock (128). While a lock is ON, the terminal rides that bit on EVERY key's
# modifier field — so a bare Right arrow becomes CSI 1;129 C (num_lock) / CSI 1;65 C
# (caps_lock). These must NOT count toward "modified arrow" suppression: a bare arrow
# under a lock is still a bare arrow. Not masking them silently killed record/eval
# arrows whenever NumLock/CapsLock was on (ESC, not being an arrow, slipped through —
# the "ESC works, arrows don't" report). Ctrl/Shift/Alt+arrow are still suppressed.
_LOCK_MODS = 64 | 128


# ── KeyEvent: the one vocabulary every backend speaks ─────────────────────────
@dataclass(frozen=True)
class KeyEvent:
    """A normalized key event. `key` is either a single printable char ("w", "s",
    …) or a functional name from {esc, up, down, left, right, space, tab, enter}.
    `event_type` is one of {press, repeat, release}. `modifiers` is a bitmask
    (0 = none); the kitty backend reports it (kitty mods-1), evdev/cbreak leave 0."""

    key: str
    event_type: str  # "press" | "repeat" | "release"
    modifiers: int = 0


def _normalize_kitty(seq: str):
    """Map a complete u-form / functional-form CSI sequence (one _CSI_U match) to a
    normalized key string + event_type, reusing parse_one VERBATIM. Returns
    (key, event_type, modifiers) or None for sequences we don't map.

    The terminator (group 6 of the match) discriminates an arrow (A/B/C/D — group 1
    is a dummy "1") from a u-form key (group 1 is a real codepoint). parse_one
    discards the terminator, so we re-match _CSI_U here to recover group 6 (the regex
    is the same proven object; the cost is one fullmatch on a tiny string).
    """
    m = _CSI_U.fullmatch(seq)
    if m is None:
        return None
    parsed = parse_one(seq)
    if parsed is None:
        return None
    cp, mods, et = parsed
    term = m.group(6)
    etype = _ET_NAME.get(et)
    if etype is None:
        return None
    if term in _KITTY_ARROW_FINALS:        # functional arrow (e.g. release CSI 1:3 A)
        return _KITTY_ARROW_FINALS[term], etype, mods - 1
    if term != "u":                         # other functional terminators: not mapped
        return None
    name = _KITTY_CODEPOINT_NAME.get(cp)
    if name is not None:
        return name, etype, mods - 1
    ch = chr(cp) if 0 <= cp < 0x110000 else ""
    if ch.isprintable():
        return ch, etype, mods - 1
    return None


class _ReleaseTracker:
    """Shared held-set bookkeeping for release-capable backends (HARD RULE 2). A key
    is "down" from its press until its matching release; repeats keep it down. On
    teardown, drain() yields a synthetic RELEASE KeyEvent for every still-held key so
    a dying stream cannot leave the base latched. Thread-affine to the reader (no lock
    needed: the reader thread mutates it, stop() drains AFTER joining that thread)."""

    def __init__(self):
        self._held: set[str] = set()

    def note(self, ev: KeyEvent) -> None:
        if ev.event_type == "release":
            self._held.discard(ev.key)
        else:  # press or repeat -> held
            self._held.add(ev.key)

    def drain(self) -> list[KeyEvent]:
        rel = [KeyEvent(key=k, event_type="release") for k in sorted(self._held)]
        self._held.clear()
        return rel


# ── KittyReader (stdin, kitty progressive-enhancement protocol) ───────────────
class KittyReader:
    """Read the kitty keyboard protocol off a tty and emit KeyEvent. On start: cbreak
    the tty + PUSH flags 1|2|8 (CSI > 11 u) — bits 2 AND 8 together are REQUIRED to get
    release on plain text keys. Parse CSI-u with the proven parse_one/_CSI_U; also
    handle the bare CSI/SS3 arrow PRESS form (the kitty spec omits the default
    `;mods:event` fields, so an unmodified arrow press is the bare CSI A / SS3 OA that
    _CSI_U does not match). On stop AND atexit: emit synthetic releases for held keys
    (HARD RULE 2), POP the flags (CSI < u), restore termios (HARD RULE 6)."""

    def __init__(self, on_event, fd: int):
        self._on_event = on_event
        self._fd = fd
        self._stop_evt = threading.Event()
        self._thread = None
        self._old_attr = None
        self._tracker = _ReleaseTracker()
        self._pending = ""
        self._torn_down = False

    def start(self):
        if self._thread is not None:
            return
        self._old_attr = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # ISIG/OPOST stay on (Ctrl+C still SIGINTs; logs render)
        self._push_flags()
        atexit.register(self._teardown)  # last-resort: pop flags + releases + restore
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # reader exits before we drain/restore
            self._thread = None
        self._teardown()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _push_flags(self):
        try:
            os.write(self._fd, _KITTY_PUSH.encode())
        except OSError:
            pass

    def _teardown(self):
        # Idempotent (stop() then atexit). Order: synthetic releases (so the consumer
        # sees the base released) BEFORE we pop flags / restore termios.
        if self._torn_down:
            return
        self._torn_down = True
        for ev in self._tracker.drain():
            self._dispatch(ev)
        try:
            os.write(self._fd, _KITTY_POP.encode())
        except OSError:
            pass
        if self._old_attr is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attr)
            except Exception:
                pass

    def _loop(self):
        fd = self._fd
        while not self._stop_evt.is_set():
            r, _, _ = select.select([fd], [], [], _POLL)
            if not r:
                continue
            try:
                chunk = os.read(fd, 256)
            except OSError:
                continue
            if not chunk:  # EOF: emit synthetic releases + stop (HARD RULE 2)
                break
            self._pending += chunk.decode(errors="replace")
            self._drain_buffer()
        # Falling out of the loop (stop OR EOF) MUST drain held keys, or an EOF (the
        # stream dying) latches the base until process exit. _teardown is idempotent
        # (_torn_down); stop() joins this thread BEFORE its own _teardown, so on the
        # stop() path this runs first and stop()'s call is the no-op (clean ordering).
        self._teardown()

    def _drain_buffer(self):
        # Drain every complete sequence from the buffer; leave a partial tail. At the
        # buffer head, try the proven _CSI_U match FIRST (u-form + functional arrows,
        # incl. release/modified). If that doesn't match the head, try a bare CSI/SS3
        # arrow PRESS (\x1b[A / \x1bOA) — which _CSI_U intentionally does not match.
        while self._pending:
            m = _CSI_U.search(self._pending)
            if m is not None and m.start() == 0:
                seq = m.group(0)
                self._pending = self._pending[m.end():]
                norm = _normalize_kitty(seq)
                if norm is not None:
                    key, etype, mods = norm
                    # Suppress MODIFIED arrows (e.g. Ctrl+Right \x1b[1;5C) so the kitty tier
                    # matches the cbreak fallback (which drains them) and the pre-unify
                    # behavior: record/eval bind their discrete actions to BARE arrows, and a
                    # modified arrow accidentally ending an episode is a footgun. Modified
                    # CHARS (e.g. Shift+w) are kept — teleop's held-set needs them.
                    # BUT mask lock state first (_LOCK_MODS): CapsLock/NumLock ride on every
                    # key while on, so without this a bare arrow under a lock looks "modified"
                    # and is dropped — the ESC-works-arrows-don't bug. The KeyEvent still
                    # carries the full mods (informational); only the suppression test masks.
                    real_mods = mods & ~_LOCK_MODS
                    if not (key in ("up", "down", "left", "right") and real_mods != 0):
                        self._dispatch(KeyEvent(key=key, event_type=etype, modifiers=mods))
                continue
            consumed = self._try_bare_arrow()
            if consumed:
                continue
            if m is not None:
                # A complete _CSI_U match exists but not at the head: drop the leading
                # unmatched byte and retry (keeps the parser advancing past noise).
                self._pending = self._pending[1:]
                continue
            break  # no complete sequence yet: keep the tail for the next read

    def _dispatch(self, ev: KeyEvent):
        # Kitty flag 8 (report-all-keys) causes Ghostty to send Ctrl+C as a CSI-u
        # sequence (cp=99 'c', mods-1=4=Ctrl) instead of the raw \x03 byte, so
        # termios ISIG never fires. Re-raise as SIGINT on press so Ctrl+C exits the
        # subprocess (teleop/eval/record) exactly as expected.
        if ev.key == "c" and ev.modifiers & 4 and ev.event_type == "press":
            import signal
            os.kill(os.getpid(), signal.SIGINT)
            return
        self._tracker.note(ev)
        try:
            self._on_event(ev)
        except Exception:
            pass

    def _try_bare_arrow(self) -> bool:
        """Consume a bare CSI/SS3 arrow at the buffer head (unmodified arrow PRESS).
        Returns True if it consumed something. A lone trailing ESC / `ESC [` / `ESC O`
        is left as a partial tail (don't strand a half-arrived sequence)."""
        p = self._pending
        if not p.startswith("\x1b"):
            return False
        if len(p) < 2:
            return False  # lone ESC: wait for more (could be the start of a sequence)
        if p[1] not in "[O":
            # ESC + something that isn't a CSI/SS3 intro: not an arrow we map. Drop the
            # ESC so the parser advances (a u-form key after it still drains next pass).
            self._pending = p[1:]
            return True
        if len(p) < 3:
            return False  # ESC [ / ESC O with no final yet: wait for the final byte
        final = p[2]
        name = _KITTY_ARROW_FINALS.get(final)
        if name is None:
            # p[2] is a digit / ';' / ':' -> a PARTIAL CSI-u or functional sequence
            # still arriving (a COMPLETE one would have matched _CSI_U at the head).
            # Leave it buffered; consuming 3 bytes here would corrupt a split sequence
            # -> a dropped RELEASE -> latched base (the exact HARD RULE 2 hazard).
            return False
        self._pending = p[3:]
        self._dispatch(KeyEvent(key=name, event_type="press"))
        return True


# ── EvdevReader (teleop hold-to-move, below the compositor) ───────────────────
class EvdevReader:
    """Read /dev/input/event* directly (below the Wayland compositor) and emit KeyEvent
    with real press/repeat/release. evdev EV_KEY value 1=press, 2=repeat, 0=release.
    Reuses _evdev_keycode_to_obj's keycode→name knowledge but maps to the NORMALIZED
    key string. Keeps _open_keyboard_devices + the no-device idle-thread degrade
    (is_alive stays True so KeyboardTeleop.is_connected holds). Synthetic releases on
    teardown (HARD RULE 2)."""

    def __init__(self, on_event):
        self._on_event = on_event
        self._stop_evt = threading.Event()
        self._thread = None
        self._devices = []
        self._tracker = _ReleaseTracker()
        self._torn_down = False

    def start(self):
        if self._thread is not None:
            return
        try:
            self._devices = _open_keyboard_devices()
        except Exception:  # not in `input` group / evdev unavailable
            self._devices = []
        if not self._devices:
            import logging
            logging.warning(
                "evdev keyboard reader opened no keyboard device — base keyboard control "
                "is disabled (the arm still teleops). Add yourself to the 'input' group: "
                "`sudo usermod -aG input $USER`, then log out and back in."
            )
        # ALWAYS spawn the reader (idle when no devices) so is_alive() stays True and
        # lerobot's KeyboardTeleop.is_connected holds -> teleop runs with the base idle
        # rather than crashing in get_action() (@check_if_not_connected).
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._teardown()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _teardown(self):
        if self._torn_down:
            return
        self._torn_down = True
        for ev in self._tracker.drain():   # synthetic releases for any held key
            self._dispatch(ev)
        for d in self._devices:
            try:
                d.close()
            except Exception:
                pass
        self._devices = []

    def _dispatch(self, ev: KeyEvent):
        self._tracker.note(ev)
        try:
            self._on_event(ev)
        except Exception:
            pass

    def _emit(self, keycode_name, event_type):
        key = _evdev_keycode_to_name(keycode_name)
        if key is None:
            return
        self._dispatch(KeyEvent(key=key, event_type=event_type))

    def _loop(self):
        import select as _select
        fds = {d.fd: d for d in self._devices}
        if not fds:
            # No device opened: idle so is_alive() stays True (teleop runs, base idle).
            while not self._stop_evt.is_set():
                self._stop_evt.wait(_POLL)
            return
        from evdev import categorize, ecodes
        while not self._stop_evt.is_set():
            r, _, _ = _select.select(list(fds), [], [], _POLL)
            for fd in r:
                dev = fds.get(fd)
                if dev is None:
                    continue
                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        ke = categorize(event)
                        kc = ke.keycode
                        if isinstance(kc, (list, tuple)):
                            kc = kc[0]
                        if ke.keystate == ke.key_down:
                            self._emit(kc, "press")
                        elif ke.keystate == ke.key_up:
                            self._emit(kc, "release")
                        elif ke.keystate == ke.key_hold:
                            self._emit(kc, "repeat")
                except OSError:
                    pass


def _evdev_keycode_to_name(name):
    """Map an evdev keycode name (e.g. 'KEY_W', 'KEY_ESC') to a NORMALIZED key string
    (a single printable char or a functional name), or None for keys we don't map.
    Mirrors _evdev_keycode_to_obj's coverage but returns the string vocabulary. Pure +
    unit-testable (no device needed). Functional set here is the KeyEvent set; modifier
    keys (shift/ctrl) are NOT in that set, so they map to None (teleop never uses them)."""
    functional = {
        "KEY_ESC": "esc", "KEY_SPACE": "space", "KEY_TAB": "tab", "KEY_ENTER": "enter",
        "KEY_UP": "up", "KEY_DOWN": "down", "KEY_LEFT": "left", "KEY_RIGHT": "right",
    }
    if name in functional:
        return functional[name]
    if name and name.startswith("KEY_") and len(name) == 5:
        c = name[4]
        if c.isalpha():
            return c.lower()   # KEY_W -> "w" (lekiwi base keys)
        if c.isdigit():
            return c           # KEY_5 -> "5"
    return None


# ── CbreakReader (DISCRETE only; a terminal in cbreak has no release) ─────────
class CbreakReader:
    """Read a tty in cbreak and emit KeyEvent with event_type ALWAYS "press" (a terminal
    in cbreak has no key-release). Keeps the legacy ESC-vs-CSI/SS3 disambiguation + the
    parametrized-CSI-drain fix (Ctrl+Right / Delete emit nothing). DISCRETE ONLY — never
    fed to a require_release consumer (HARD RULE 3): a press-only stream would latch a
    held key = runaway base. open_key_source enforces that."""

    def __init__(self, on_event, fd: int):
        self._on_event = on_event
        self._fd = fd
        self._stop_evt = threading.Event()
        self._thread = None
        self._old_attr = None

    def start(self):
        if self._thread is not None:
            return
        if not os.isatty(self._fd):
            return  # no controlling TTY (piped/headless): no-op so callers still run
        self._old_attr = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        atexit.register(self._restore)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._restore()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def _restore(self):
        if self._old_attr is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attr)
            except Exception:
                pass

    def _press(self, key: str):
        try:
            self._on_event(KeyEvent(key=key, event_type="press"))
        except Exception:
            pass

    def _loop(self):
        fd = self._fd
        while not self._stop_evt.is_set():
            r, _, _ = select.select([fd], [], [], _POLL)
            if not r:
                continue
            b = os.read(fd, 1)
            if not b:
                continue
            if b == ESC:
                self._read_escape()
                continue
            if b == b" ":
                self._press("space")
            elif b == b"\t":
                self._press("tab")
            elif b in (b"\r", b"\n"):
                self._press("enter")
            else:
                try:
                    ch = b.decode()
                except UnicodeDecodeError:
                    continue
                if ch.isprintable():
                    self._press(ch)

    def _read_escape(self):
        # Same machine as the legacy _CbreakReader: bare-ESC vs CSI/SS3 arrow, with the
        # parametrized-CSI-drain fix (Ctrl+Right shares final 'C' with the right arrow,
        # so the discriminator is a param byte BEFORE the final, not the final itself).
        fd = self._fd
        r2, _, _ = select.select([fd], [], [], _ESC_TIMEOUT)
        if not r2:
            self._press("esc")
            return
        intro = os.read(fd, 1)
        if intro not in (b"[", b"O"):
            return
        r3, _, _ = select.select([fd], [], [], _ESC_TIMEOUT)
        if not r3:
            return
        n = os.read(fd, 1)
        if 0x40 <= n[0] <= 0x7E:  # n is itself the final byte (bare arrow / other)
            name = _ARROW_FINALS.get(n)
            if name is not None:
                self._press(name)
            return
        while True:  # parametrized seq: drain to its final, emit nothing
            r4, _, _ = select.select([fd], [], [], _ESC_TIMEOUT)
            if not r4:
                return
            f = os.read(fd, 1)
            if 0x40 <= f[0] <= 0x7E:
                return


# ── IDLE no-op backend (HARD RULE 3) ──────────────────────────────────────────
class _IdleReader:
    """Emits NOTHING but stays is_alive() True. Used when a release-capable consumer
    (teleop) has NEITHER kitty NOR evdev: feeding it a press-only CbreakReader would
    latch a held key (runaway base), so we keep the base idle = SAFE instead."""

    def __init__(self, *_a, **_k):
        self._stop_evt = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop_evt.is_set():
            self._stop_evt.wait(_POLL)

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


# ── shared stdin-tty source (record's base teleop + episode listener share one) ──
# pynput's Listener is a GLOBAL hook: every listener sees every key. A tty's stdin is
# NOT — two reader threads doing os.read() on one fd split the bytes between them,
# corrupting multi-byte CSI-u sequences. record drives the robot with TWO keyboard
# consumers at once (the base KeyboardTeleop AND the episode-control listener), both
# defaulting to fd 0; on a kitty terminal both would otherwise open their own KittyReader
# and race for stdin. A _SharedKeySource wraps ONE backend and fans each KeyEvent out to
# every subscriber, restoring pynput's "everyone sees everything" semantics over a single
# fd (and as a bonus: one cbreak + one flag-push, not two). OPT-IN (share=True) so teleop
# (one listener) and eval (its own multi-listener case) are untouched; only stdin-tty
# backends (kitty/cbreak) ever share — evdev (its own /dev/input device) and idle never do.
_SHARED_LOCK = threading.Lock()
_SHARED: dict = {}  # fd -> _SharedKeySource (the live shared stdin reader for that fd)


class _SharedKeySource:
    """One stdin-tty backend on `fd`, fanned out to N subscribers (see the block comment).
    Refcounted: the backend starts on the FIRST subscribe and tears down (synthetic
    releases + kitty-flag pop + termios restore, all owned by the backend) only when the
    LAST subscriber stops — so neither consumer can strand the other's tty state."""

    def __init__(self, fd: int, backend_name: str):
        self._fd = fd
        self._backend_name = backend_name
        # Only kitty reports real release; a shared cbreak source is press-only and must
        # NEVER be reused by a require_release consumer (can_serve guards it — HARD RULE 3).
        self._release_capable = backend_name == "kitty"
        self._subs: list = []
        self._lock = threading.Lock()
        self._backend = None

    def can_serve(self, require_release: bool) -> bool:
        return (not require_release) or self._release_capable

    def _fanout(self, ev: "KeyEvent") -> None:
        with self._lock:
            subs = list(self._subs)   # snapshot: a subscriber may stop mid-dispatch
        for cb in subs:               # one subscriber raising must not drop the others
            try:
                cb(ev)
            except Exception:
                pass

    def subscribe(self, on_event):
        with self._lock:
            self._subs.append(on_event)
            first = self._backend is None
            if first:
                self._backend = _make_backend(self._backend_name, self._fanout, self._fd)
        if first:
            self._backend.start()     # one reader thread + one cbreak/flag-push for all
        return _Subscription(self, on_event)

    def _unsubscribe(self, on_event) -> None:
        with self._lock:
            try:
                self._subs.remove(on_event)
            except ValueError:
                pass
            last = not self._subs
            backend = self._backend if last else None
            if last:
                self._backend = None
        if last:
            # Drop from the registry BEFORE teardown so a later subscribe re-probes fresh.
            with _SHARED_LOCK:
                if _SHARED.get(self._fd) is self:
                    del _SHARED[self._fd]
            if backend is not None:
                backend.stop()        # drains held keys + pops flags + restores termios

    def is_alive(self) -> bool:
        b = self._backend
        return b is not None and b.is_alive()


class _Subscription:
    """A subscriber handle on a _SharedKeySource matching the start/stop/is_alive interface
    the two adapters expect of a backend. start() is a no-op (the shared source already
    started at subscribe time); stop() unsubscribes (tearing the shared backend down when it
    was the last subscriber); is_alive() reflects the shared backend's liveness."""

    def __init__(self, shared: "_SharedKeySource", on_event):
        self._shared = shared
        self._on_event = on_event
        self._stopped = False

    def start(self):
        pass

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._shared._unsubscribe(self._on_event)

    def is_alive(self):
        return (not self._stopped) and self._shared.is_alive()


# ── open_key_source: capability-detect ONCE at start, pick a backend ──────────
def open_key_source(on_event, *, fd=None, require_release: bool, prefer=None, share=False):
    """Pick + START one backend for `on_event`. Capability detection runs ONCE at
    start, BEFORE any read loop, on the same fd (HARD RULE 4) with a GENEROUS kitty
    timeout so a real kitty terminal is NEVER misclassified (a false negative would
    downgrade teleop to no-hold-to-move = the exact bug being fixed). The CSI ? u query
    is read-only.

    Selection:
      * prefer set (tests): instantiate that backend directly (never shared).
      * else fd is a tty AND a kitty query succeeds          -> KittyReader
      * else require_release AND evdev devices open          -> EvdevReader
      * else (not require_release)                           -> CbreakReader
      * else (require_release, no kitty/evdev)               -> _IdleReader (SAFE; HARD RULE 3)

    NEVER feeds a press-only CbreakReader to a require_release consumer.

    share=True (record only): if the chosen backend is a stdin-tty one (kitty/cbreak),
    return a SUBSCRIPTION to a single shared reader on `fd` instead of a fresh backend, so
    record's base teleop + episode listener fan out from ONE reader rather than racing two
    on stdin (see the _SharedKeySource block comment). evdev/idle are never shared. A
    require_release consumer is never joined to a press-only shared cbreak source.
    """
    if fd is None:
        fd = 0  # stdin

    if prefer is not None:
        backend = _make_backend(prefer, on_event, fd)
        backend.start()
        return backend

    # Non-tty (piped/headless): no kitty, no cbreak. require_release -> idle (safe);
    # discrete -> CbreakReader, whose start() is a no-op on a non-tty (callers still run).
    if not os.isatty(fd):
        backend = _IdleReader() if require_release else CbreakReader(on_event, fd)
        backend.start()
        return backend

    # SHARED fast path: a live stdin reader already exists for this fd. Reuse it (subscribe)
    # WITHOUT re-probing — a second _kitty_capable() would steal bytes from the running
    # reader (HARD RULE 4: detect once, before any read loop). can_serve() refuses to hand a
    # press-only (cbreak) shared source to a require_release consumer (HARD RULE 3); that
    # consumer then falls through to its own evdev/idle backend below.
    if share:
        with _SHARED_LOCK:
            existing = _SHARED.get(fd)
        if existing is not None and existing.can_serve(require_release):
            return existing.subscribe(on_event)

    # Resolve which backend this request wants. Capability detection runs ONCE here, BEFORE
    # any reader thread exists for this fd (for a shared fd, only the first subscriber gets
    # this far — see the fast path above).
    if _kitty_capable(fd):
        name = "kitty"
    elif require_release and _evdev_available():
        name = "evdev"
    elif not require_release:
        name = "cbreak"
    else:
        name = "idle"  # require_release, neither kitty nor evdev -> base idle

    # Only stdin-tty backends (kitty/cbreak) on the SAME fd can be shared; evdev (its own
    # /dev/input device) and idle never share. record opts in so its base teleop + episode
    # listener ride ONE kitty reader on fd 0 instead of racing two.
    if share and name in ("kitty", "cbreak"):
        with _SHARED_LOCK:
            existing = _SHARED.get(fd)
            if existing is None:
                existing = _SharedKeySource(fd, name)
                _SHARED[fd] = existing
        return existing.subscribe(on_event)

    backend = _make_backend(name, on_event, fd)
    backend.start()
    return backend


def _make_backend(name, on_event, fd):
    """prefer= dispatch for tests: a backend instance by name."""
    name = name.lower()
    if name == "kitty":
        return KittyReader(on_event, fd)
    if name == "evdev":
        return EvdevReader(on_event)
    if name == "cbreak":
        return CbreakReader(on_event, fd)
    if name == "idle":
        return _IdleReader()
    raise ValueError(f"unknown prefer={name!r}")


def _kitty_capable(fd) -> bool:
    """Read-only CSI ? u round-trip with a GENEROUS timeout (HARD RULE 4). Needs cbreak
    to read the reply char-at-a-time, so set it, query, restore. Returns True iff the
    terminal answered with a CSI ? <flags> u reply. Never raises (any failure -> False)."""
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        return False
    try:
        tty.setcbreak(fd)
        # GENEROUS ~0.25s so a real kitty terminal is never a false negative; the probe's
        # detect_support also sends a CSI 6n cursor sentinel so a NON-kitty terminal
        # returns promptly (R reply) instead of burning the whole timeout.
        flags = _kitty_detect_support(fd, timeout=0.25)
        return flags is not None
    except Exception:
        return False
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass


def _evdev_available() -> bool:
    """True iff at least one keyboard device opens (user in `input` group). Closes the
    probe devices immediately; EvdevReader reopens them in its own start()."""
    try:
        devs = _open_keyboard_devices()
    except Exception:
        return False
    ok = bool(devs)
    for d in devs:
        try:
            d.close()
        except Exception:
            pass
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TWO THIN ADAPTERS (the only things the shims touch)
# ══════════════════════════════════════════════════════════════════════════════
# KeyListener         — a pynput.keyboard.Listener work-alike for BOTH eval and teleop
#                       (the rollout + teleop shims patch pynput.keyboard.Listener to it).
# make_stdin_listener — record's (listener, events) contract (the record shim patches
#                       lerobot_record.init_keyboard_listener to it; name KEPT stable).
# Both sit on open_key_source and map KeyEvent -> the consumer's expected shape. They are
# the single layer between the unified backends and lerobot; no consumer keymap changes.


# The fixed functional-name set (KeyEvent's non-char vocabulary). A KeyEvent.key in this
# set maps to getattr(Key, name); anything else is a single printable char -> KeyCode.
_FUNCTIONAL_NAMES = frozenset(
    {"esc", "up", "down", "left", "right", "space", "tab", "enter"}
)


class KeyListener:
    """Drop-in for ``pynput.keyboard.Listener`` used by BOTH eval and teleop.

    Replaces the retired StdinKeyListener (eval) and EvdevKeyListener (teleop): one
    adapter over open_key_source. Surface lerobot relies on: ctor(on_press=...,
    on_release=...), .start(), .stop(), .is_alive(). Per KeyEvent it builds a REAL
    pynput object (char -> KeyCode.from_char(char); functional name -> getattr(Key, name))
    so every existing `key == keyboard.Key.esc` / `key.char == save_key` comparison keeps
    working, then dispatches: press -> on_press(obj); release -> on_release(obj) if given;
    repeat -> NO-OP (HARD RULE 2: a repeat must NOT re-fire on_press, or a held key would
    re-trigger discrete actions / never appear released).

    require_release = (on_release is not None): teleop passes on_release (hold-to-move
    needs real release + diagonals) -> a release-capable source (Kitty/Evdev, else IDLE —
    NEVER a press-only Cbreak, HARD RULE 3). eval passes on_press only -> Cbreak is fine.
    The real Key/KeyCode classes are imported, never replaced. fd defaults to stdin (the
    suspended child owns the real TTY). Before start(), is_alive() is False (pynput parity).
    """

    def __init__(self, on_press=None, on_release=None, *, fd=None, share=False, **_ignored):
        self._on_press = on_press
        self._on_release = on_release
        self._require_release = on_release is not None
        self._fd = 0 if fd is None else fd
        # share=True (record's base teleop) makes this ride a shared stdin reader on `fd`
        # so it doesn't race the episode-control listener; teleop/eval leave it False.
        self._share = share
        self._source = None
        # Import the REAL pynput key classes here; never replace them. Done lazily in the
        # ctor (not at module import) so a headless import of this module never needs pynput.
        from pynput.keyboard import Key, KeyCode
        self._Key = Key
        self._KeyCode = KeyCode

    def start(self):
        if self._source is not None:
            return
        # Capability-detect + pick a backend ONCE here (HARD RULE 4), then start it.
        self._source = open_key_source(
            self._dispatch, fd=self._fd, require_release=self._require_release,
            share=self._share,
        )

    def stop(self):
        if self._source is not None:
            self._source.stop()

    def is_alive(self):
        # pynput parity: False before start(); after start, the backend's liveness. teleop's
        # KeyboardTeleop.is_connected gates get_action() on this (an IdleReader stays alive so
        # teleop runs with the base idle rather than crashing — see open_key_source).
        return self._source is not None and self._source.is_alive()

    def _dispatch(self, ev: KeyEvent):
        # KeyEvent.key -> the REAL pynput object lerobot's callbacks compare against.
        obj = (
            getattr(self._Key, ev.key)
            if ev.key in _FUNCTIONAL_NAMES
            else self._KeyCode.from_char(ev.key)
        )
        if ev.event_type == "press":
            cb = self._on_press
        elif ev.event_type == "release":
            cb = self._on_release   # None for eval (on_press-only) -> skipped below
        else:  # "repeat": NO-OP (HARD RULE 2) — never re-fires on_press for a held key
            return
        if cb is not None:
            try:
                cb(obj)             # consumers wrap their own cb in try/except too
            except Exception:
                pass


def make_stdin_listener(fd: int | None = None, *, share: bool = False):
    """Build lerobot's (listener, events) contract for the RECORD loop (name KEPT stable —
    the record shim patches lerobot_record.init_keyboard_listener to this).

    Returns a `(listener, events)` tuple, where `events` is the dict the record loop polls
    (`exit_early` / `rerecord_episode` / `stop_recording`) and `listener` exposes `.stop()`
    (called by lerobot at cleanup). record is DISCRETE (a key tap, not hold-to-move), so
    require_release=False -> a press-only source is fine. The mapping is EXACTLY today's:
      right -> exit_early
      left  -> rerecord_episode + exit_early
      esc   -> stop_recording + exit_early
    (up/down and every other key: ignored.) Accepts an explicit `fd` for testing; defaults
    to stdin. A non-tty fd yields a no-op source (start() is a no-op there) + the same events
    dict, so headless record still runs.

    share=True (the record shim sets it) makes this ride the SAME stdin reader as the base
    KeyboardTeleop's listener (which the shim also patches in with share=True), so on a kitty
    terminal the two consumers fan out from one reader instead of racing two on fd 0. The
    arrow/ESC -> events mapping is unchanged either way.
    """
    if fd is None:
        fd = 0  # stdin
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    def on_event(ev: KeyEvent):
        # Discrete: act on PRESS only (cbreak is press-only anyway; if a kitty source is
        # picked, repeats/releases for these keys are simply ignored — same net mapping).
        if ev.event_type != "press":
            return
        if ev.key == "right":
            events["exit_early"] = True
        elif ev.key == "left":
            events["rerecord_episode"] = True
            events["exit_early"] = True
        elif ev.key == "esc":  # bare ESC -> stop the session (sets BOTH, like before)
            events["stop_recording"] = True
            events["exit_early"] = True
        # up/down and every other key: ignore.

    source = open_key_source(on_event, fd=fd, require_release=False, share=share)

    class _StdinListener:
        def stop(self):
            source.stop()

    return _StdinListener(), events
