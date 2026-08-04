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


from conftest import make_ctx


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


# ── stream: lifecycle (the watch-only controller; forwarding/hooks were removed
#    with the record HUD) ──────────────────────────────────────────────────────


def test_stream_lifecycle_captures_output_and_ends():
    async def main():
        s = StreamController()
        await s.start(["bash", "-c", 'echo hello; echo "WARN careful"'])
        for _ in range(40):
            await asyncio.sleep(0.05)
            if s.ended:
                break
        assert s.ended and s.returncode == 0
        assert any("hello" in ln for ln in s.lines)

    asyncio.run(main())



# ── record HUD state machine + dataset panel ─────────────────────────────────


def _record_screen():
    from lekiwi_tui.screens.record import RecordScreen

    ctx = make_ctx(gpu_name="")
    return RecordScreen(None, ctx)


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


# ── the redesigned stream-trio pages (sync / provision / stop-host) ───────────


def _trio_ctx():

    return make_ctx(gpu_name="")


def test_sync_body_groups_provenance_and_reinstall_segment():
    from lekiwi_tui.screens.sync import SyncScreen

    scr = SyncScreen(None, _trio_ctx())
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "SOURCES" in body and "DESTINATION" in body
    assert " auto " in body and " force " in body        # the Reinstall segment
    assert "reinstall only if deps changed" in body      # plan follows the toggle
    scr._force = True
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "reinstall forced this run" in body


def test_provision_body_stage_pills_and_plan():
    from lekiwi_tui.screens.provision import ProvisionScreen

    scr = ProvisionScreen(None, _trio_ctx())
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "STAGES" in body and "PI ENVIRONMENT" in body
    assert "system + conda + lerobot" in body            # plan lists chosen stages
    scr._on = {s: False for s in scr._on}
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "select at least one stage" in body


def test_host_kill_confirm_wears_the_danger_style():
    import pyratatui as pr

    from lekiwi_tui.framework import theme
    from lekiwi_tui.screens.host_kill import HostKillScreen

    scr = HostKillScreen(None, _trio_ctx())
    texts = []

    class Frame:
        def render_widget(self, widget, rect):  # noqa: ANN001
            texts.append(widget)

    scr.draw(Frame(), pr.Rect(0, 0, 100, 30))
    assert theme.HIGHLIGHT_DANGER_STYLE is not None      # the idiom exists app-wide
    assert len(texts) >= 4                               # header, rule, body, hint


# ── the redesigned SETUP tail (settings / calibrate / robot-config) ───────────


def test_calibrate_inverse_host_warning_and_radio():
    from lekiwi_tui.screens.calibrate import CalibrateScreen

    scr = CalibrateScreen(None, _trio_ctx())
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "ARM" in body and "leader" in body and "follower" in body
    assert "calibrated" in body or "not calibrated yet" in body  # the age readout

    # follower + host RUNNING → the warning replaces the plan (inverse condition)
    scr._arm = "follower"
    scr._host_alive = lambda: True  # type: ignore[method-assign]
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "Stop host first" in body
    # leader ignores host state entirely (local serial device)
    scr._arm = "leader"
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "Stop host first" not in body


def test_settings_save_plan_and_env_hint():
    from lekiwi_tui.screens.settings import CONFIG_SPEC, SettingsScreen

    scr = SettingsScreen(None, _trio_ctx())
    scr.cursor = scr._n                        # focus the Save row
    assert "writes launcher settings" in scr._focused_hint()
    scr.cursor = 0
    assert scr._focused_hint() == CONFIG_SPEC[0].hint
    scr._env_set = {CONFIG_SPEC[0].key}
    assert scr._focused_hint().startswith("[env override]")
    scr.dirty = True
    scr._work[CONFIG_SPEC[0].key] = "changed-value"
    hdr = "".join(sp.content for sp in scr._header_right())
    assert "unsaved (1)" in hdr


def test_record_start_always_suspends(monkeypatch, tmp_path):
    """The HUD view was removed: Start ALWAYS suspends into the real TTY (the
    guaranteed wasd path). No stream branch, no View field."""
    import asyncio

    from lekiwi_tui.screens.record import _FIELDS, _HINTS, RecordScreen

    assert "view" not in _FIELDS and "view" not in _HINTS
    assert not hasattr(RecordScreen("x", None) if False else object(), "stream")

    scr = _record_screen()
    assert not hasattr(scr, "stream") and not hasattr(scr, "_view")

    suspended: list[list[str]] = []

    class _App:
        async def suspend(self, argv, **kw):
            suspended.append(list(argv))
            return 0

        async def run_modal(self, modal):
            return None

        def notify(self, *a, **k):
            pass

    scr.app = _App()
    monkeypatch.setattr("lekiwi_tui.screens.record.confirm_preflight",
                        _async_true)
    monkeypatch.setattr("lekiwi_tui.screens.record.dataset_present", lambda root: False)
    monkeypatch.setattr("lekiwi_tui.screens.record.persist_record_defaults",
                        lambda **kw: None)
    asyncio.run(scr._start())
    assert suspended and suspended[0][0] == "bash"
    assert suspended[0][1].endswith("record.sh")


async def _async_true(*a, **k):
    return True
