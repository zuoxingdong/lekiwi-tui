"""Tests for the 2026 UX upgrade set: live host chip, panic key, stream key
forwarding + health matcher, the record HUD state machine, and the dataset panel.
All hardware-free; the stream tests drive a bash stand-in for lerobot-record."""
from __future__ import annotations

import asyncio
import json
import time
import types


from lekiwi_tui.framework.stream import StreamController
from lekiwi_tui.hostprobe import HostProbe, get_probe, session_remaining


def _key(name: str, **mods) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, ctrl=False, alt=False, shift=False, **mods)


# ── hostprobe ─────────────────────────────────────────────────────────────────


def test_hostprobe_dead_and_alive():
    import socket

    dead = HostProbe("127.0.0.1", 1)          # port 1: nothing listens
    dead.poll()
    time.sleep(1.0)
    assert dead.alive is False

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    live = HostProbe("127.0.0.1", srv.getsockname()[1])
    live.poll()
    time.sleep(0.5)
    assert live.alive is True
    srv.close()


def test_get_probe_retargets_on_host_change():
    ctx = types.SimpleNamespace(cfg={"LEKIWI_HOST": "a"}, ui_state={})
    p1 = get_probe(ctx)
    ctx.cfg["LEKIWI_HOST"] = "b"
    p2 = get_probe(ctx)
    assert p1 is not p2 and p2.host == "b"
    ctx.cfg["LEKIWI_HOST"] = ""
    assert get_probe(ctx) is None


def test_session_remaining():
    ctx = types.SimpleNamespace(ui_state={})
    assert session_remaining(ctx) is None
    ctx.ui_state["host_session"] = {"ends_at": time.monotonic() + 90}
    assert 85 <= session_remaining(ctx) <= 90
    ctx.ui_state["host_session"] = {"ends_at": time.monotonic() - 1}
    assert session_remaining(ctx) is None


# ── panic key ─────────────────────────────────────────────────────────────────


def test_panic_arms_then_fires(monkeypatch):
    import lekiwi_tui.panic as panic

    toasts: list[str] = []
    fired: list[bool] = []
    monkeypatch.setattr(panic, "_armed_at", 0.0)
    monkeypatch.setattr(panic, "_fire", lambda app, ctx: fired.append(True))
    app = types.SimpleNamespace(notify=lambda m, lvl="info": toasts.append(m))
    ctx = types.SimpleNamespace(cfg={"LEKIWI_HOST": "lekiwi", "ROBOT_ID": "lekiwi"}, ui_state={})
    hook = panic.make_global_key(ctx)

    assert hook(app, _key("k")) is None               # lowercase passes through
    assert hook(app, _key("K")) is not None           # armed
    assert "armed" in toasts[-1]
    assert hook(app, _key("K")) is not None           # fired
    assert fired


# ── stream: forwarding + health + line hook ───────────────────────────────────


def test_stream_forwarding_health_and_hook():
    async def main():
        import re

        hits: list[str] = []
        s = StreamController()
        s.health_pattern = re.compile(r"\d+(?:\.\d+)?\s*it/s")
        s.line_hook = hits.append
        await s.start(["bash", "-c",
                       'printf "x 12.5it/s\\r"; echo marker; read -r l; echo "got:$l"'])
        await asyncio.sleep(0.4)
        assert s.health == "12.5it/s"
        # \r is stripped by the pump, so the meter chunk and the echo merge into one
        # log line — the hook still sees the marker text (health came from the raw chunk).
        assert any("marker" in h for h in hits)
        assert s.forward_key(_key("Enter"))
        await asyncio.sleep(0.6)
        assert s.ended
        assert any(line.startswith("got:") for line in s.lines)
        # forwarding after the end is a no-op, not an error
        assert s.forward_key(_key("Enter")) is False

    asyncio.run(main())


def test_forward_key_encodings():
    s = StreamController()
    captured: list[bytes] = []
    s.phase = "running"
    s._master = 1  # never written: monkeypatch write
    s.write_bytes = lambda b: (captured.append(b), True)[1]  # type: ignore[assignment]
    s.forward_key(_key("Left"))
    s.forward_key(_key("Esc"))
    s.forward_key(_key("w"))
    s.forward_key(types.SimpleNamespace(name="c", ctrl=True, alt=False, shift=False))
    assert captured == [b"\x1b[D", b"\x1b", b"w", b"\x03"]


# ── record HUD state machine + dataset panel ─────────────────────────────────


def _record_screen():
    from lekiwi_tui.config import Config, load_yaml
    from lekiwi_tui import CFG_FILE
    from lekiwi_tui.screens.record import RecordScreen

    ctx = types.SimpleNamespace(cfg=Config.load(CFG_FILE), doc=load_yaml(),
                                gpu_name="", ui_state={})
    return RecordScreen(None, ctx)


def test_record_marker_hook_drives_hud_state():
    scr = _record_screen()
    scr._on_record_line("Recording episode 7")
    assert scr._ep_cur == 7 and scr._phase_note == "recording"
    scr._on_record_line("blah Reset the environment blah")
    assert scr._phase_note.startswith("reset")
    scr._on_record_line("Stop recording")
    assert scr._phase_note == "stopping"


def test_record_hud_forwards_keys_and_blocks_local_stop_letters():
    scr = _record_screen()
    forwarded: list[str] = []
    scr.stream.phase = "running"
    scr.stream.forward_key = lambda k: forwarded.append(k.name)  # type: ignore[assignment]

    # "s" and "q" are base-backward / lerobot-quit: they must FORWARD, not stop/pop.
    for name in ("s", "q", "Left", "Esc"):
        scr.handle_key(_key(name))
    assert forwarded == ["s", "q", "Left", "Esc"]

    stopped: list[bool] = []
    scr.stream.stop = lambda: stopped.append(True)  # type: ignore[assignment]
    scr.handle_key(types.SimpleNamespace(name="c", ctrl=True, alt=False, shift=False))
    assert stopped


def test_record_view_toggle_and_value():
    scr = _record_screen()
    from lekiwi_tui.screens.record import _FIELDS

    scr._fpos = _FIELDS.index("view")
    before = scr._view
    scr.handle_key(_key("Right"))
    assert scr._view != before
    assert scr._view in scr._value("view")


def test_dataset_stats_line(tmp_path):
    from lekiwi_tui.datasets import dataset_stats

    assert dataset_stats(tmp_path / "nope") == ""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        {"total_episodes": 4, "total_frames": 3600, "fps": 30}))
    (root / "blob.bin").write_bytes(b"x" * 2_000_000)
    line = dataset_stats(root)
    assert line.startswith("4 episodes · 2.0 min · 2 MB · updated ")
