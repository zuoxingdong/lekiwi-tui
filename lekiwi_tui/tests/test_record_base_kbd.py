"""Regression test for record's base navigation (wasd) + episode-control keys.

THE BUG (fixed): record drives LeKiwi with TWO keyboard consumers at once — the base
`KeyboardTeleop` (wasd + zx) via the pynput `Listener` CLASS, and episode control
(arrows/ESC) via `init_keyboard_listener`. The record shim used to patch only the latter,
so on Wayland the base listener stayed the dead pynput one: arrows worked, wasd didn't.

THE FIX: the shim now also patches the pynput `Listener` class, and BOTH consumers open
their stdin source with `share=True` so they fan out from ONE shared reader instead of
racing two `os.read()`s on fd 0 (which on a kitty terminal corrupts the byte stream).

These tests pin, headlessly (a fake backend stands in for the real tty reader):
  * fan-out: two subscribers on one shared source both see every KeyEvent; ONE reader.
  * HARD RULE 3: a release-requiring consumer never joins a press-only (cbreak) share.
  * refcount: the reader tears down only when the LAST subscriber stops.
  * integration: a REAL lerobot KeyboardTeleop's held-keys AND the episode events dict
    both update from one shared reader — and arrows/ESC still reach episode control
    (the path that worked before the fix must not regress).

Run from the lerobot env (pynput + lerobot present):
  conda run -n lekiwi pytest lekiwi_tui/tests/test_record_base_kbd.py -q
"""
from __future__ import annotations

import pytest

from lekiwi_tui import kbd_listener as kl
from lekiwi_tui.kbd_listener import KeyEvent, make_stdin_listener, open_key_source


class FakeBackend:
    """Stands in for a real tty reader. _make_backend is patched to return these, capturing
    the on_event callback (the shared source's _fanout) so a test can push KeyEvents through
    it. Tracks live instances so a test can assert exactly ONE reader was created."""

    instances: list["FakeBackend"] = []

    def __init__(self, name, on_event, fd):
        self.name = name
        self.on_event = on_event  # == _SharedKeySource._fanout for a shared source
        self.fd = fd
        self.started = False
        self.stopped = False
        FakeBackend.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def is_alive(self):
        return self.started and not self.stopped

    def push(self, key, event_type="press"):
        self.on_event(KeyEvent(key=key, event_type=event_type))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Force the tty + kitty path, swap in FakeBackend, and reset the shared registry so
    every test starts from a clean fd->source map."""
    FakeBackend.instances = []
    kl._SHARED.clear()
    monkeypatch.setattr(kl.os, "isatty", lambda fd: True)
    monkeypatch.setattr(kl, "_kitty_capable", lambda fd: True)
    monkeypatch.setattr(kl, "_make_backend", lambda name, on_event, fd: FakeBackend(name, on_event, fd))
    yield
    kl._SHARED.clear()


# ── core sharing / fan-out ────────────────────────────────────────────────────
def test_two_subscribers_share_one_reader_and_both_see_every_event():
    seen_a, seen_b = [], []
    sub_a = open_key_source(lambda ev: seen_a.append(ev.key), fd=0, require_release=True, share=True)
    sub_b = open_key_source(lambda ev: seen_b.append(ev.key), fd=0, require_release=False, share=True)

    # ONE reader created and started for both subscribers.
    assert len(FakeBackend.instances) == 1
    backend = FakeBackend.instances[0]
    assert backend.started and backend.is_alive()
    assert sub_a.is_alive() and sub_b.is_alive()

    # Every event fans out to BOTH subscribers (pynput "everyone sees everything").
    backend.push("w")
    backend.push("right")
    assert seen_a == ["w", "right"]
    assert seen_b == ["w", "right"]


def test_refcounted_teardown_waits_for_the_last_subscriber():
    sub_a = open_key_source(lambda ev: None, fd=0, require_release=True, share=True)
    sub_b = open_key_source(lambda ev: None, fd=0, require_release=False, share=True)
    backend = FakeBackend.instances[0]

    sub_a.stop()  # one consumer leaves — reader stays up for the other
    assert backend.is_alive()
    assert not sub_a.is_alive() and sub_b.is_alive()
    assert kl._SHARED.get(0) is not None

    sub_b.stop()  # last consumer leaves — now the reader tears down + registry clears
    assert backend.stopped
    assert not sub_b.is_alive()
    assert kl._SHARED.get(0) is None


def test_release_consumer_never_joins_a_press_only_cbreak_share(monkeypatch):
    # Non-kitty tty: episode (press-only) lands on cbreak; the base (require_release) must
    # NOT join that press-only share — it falls through to its own evdev reader (HARD RULE 3).
    monkeypatch.setattr(kl, "_kitty_capable", lambda fd: False)
    monkeypatch.setattr(kl, "_evdev_available", lambda: True)

    episode = open_key_source(lambda ev: None, fd=0, require_release=False, share=True)  # cbreak
    base = open_key_source(lambda ev: None, fd=0, require_release=True, share=True)       # evdev

    names = [b.name for b in FakeBackend.instances]
    assert names == ["cbreak", "evdev"]          # two SEPARATE readers, not one shared cbreak
    assert episode is not base
    # the shared registry holds only the press-only cbreak source
    assert kl._SHARED[0].can_serve(require_release=False) is True
    assert kl._SHARED[0].can_serve(require_release=True) is False


# ── full integration: real KeyboardTeleop + episode listener, one shared reader ──
def test_record_base_and_episode_share_one_reader(monkeypatch):
    pytest.importorskip("pynput")
    keyboard_mod = pytest.importorskip("lerobot.teleoperators.keyboard")
    KeyboardTeleop = keyboard_mod.KeyboardTeleop
    KeyboardTeleopConfig = keyboard_mod.KeyboardTeleopConfig

    # Mimic the record shim: patch the pynput Listener class to the share=True base listener.
    import pynput.keyboard

    from lekiwi_tui.kbd_listener import KeyListener

    class _SharedBaseKeyListener(KeyListener):
        def __init__(self, *a, **k):
            k.setdefault("share", True)
            super().__init__(*a, **k)

    monkeypatch.setattr(pynput.keyboard, "Listener", _SharedBaseKeyListener)

    # Mimic the shim's second patch (lerobot 0.6+): connect() gates on
    # pynput_can_capture(), which is False in a headless/Wayland test env and would
    # skip Listener construction entirely — force it open exactly like the shim does.
    import lerobot.teleoperators.keyboard.teleop_keyboard as teleop_keyboard

    if hasattr(teleop_keyboard, "pynput_can_capture"):
        monkeypatch.setattr(teleop_keyboard, "pynput_can_capture", lambda: True)

    # (2) base navigation — connect the KeyboardTeleop FIRST (record's order: line 454 < 458).
    base = KeyboardTeleop(KeyboardTeleopConfig())
    base.connect()
    assert base.is_connected  # isinstance(self.listener, keyboard.Listener) + is_alive both hold

    # (1) episode control — joins the SAME shared reader (share=True).
    episode_listener, events = make_stdin_listener(share=True)

    assert len(FakeBackend.instances) == 1, "base + episode must share ONE stdin reader"
    backend = FakeBackend.instances[0]

    # wasd drives the base: press -> held, release -> cleared (hold-to-move needs release).
    backend.push("w", "press")
    assert "w" in base.get_action(), "base navigation 'w' must register as held"
    backend.push("w", "release")
    assert "w" not in base.get_action(), "releasing 'w' must clear the held base key"

    # episode control still reaches the events dict from the same reader (no regression).
    backend.push("right", "press")
    assert events["exit_early"] is True
    backend.push("left", "press")
    assert events["rerecord_episode"] is True
    backend.push("esc", "press")
    assert events["stop_recording"] is True

    episode_listener.stop()
    base.disconnect()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
