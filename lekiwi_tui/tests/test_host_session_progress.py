from __future__ import annotations

from lekiwi_tui.screens.host import _format_duration, _session_progress
from conftest import make_ctx


def test_format_duration_uses_mmss_and_hmmss():
    assert _format_duration(30 * 60) == "30:00"
    assert _format_duration(65) == "01:05"
    assert _format_duration(90 * 60) == "1:30:00"


def test_format_duration_can_round_remaining_up():
    assert _format_duration(1799.2, ceiling=True) == "30:00"
    assert _format_duration(1799.0, ceiling=True) == "29:59"


def test_session_progress_counts_down_and_clamps():
    remaining, fraction = _session_progress(600, 100.0, now=250.0)
    assert remaining == 450.0
    assert fraction == 0.25

    remaining, fraction = _session_progress(600, 100.0, now=900.0)
    assert remaining == 0.0
    assert fraction == 1.0


# ── the redesigned host page view-model (form + stream states) ────────────────


def _host_screen():

    from lekiwi_tui.screens.host import HostLaunchScreen

    ctx = make_ctx(gpu_name="")
    return HostLaunchScreen(None, ctx)


def test_host_form_view_model_groups_and_plan():
    scr = _host_screen()
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(90))
    assert "SESSION" in body
    assert "Session" in body and "Loop freq" in body and "Robot id" in body
    # Robot type is READ-ONLY config, shown dimmed with its provenance
    assert "Robot type" in body and "(yaml)" in body
    # the Start row carries the plan (the old 3-line info block, condensed)
    assert "ships plugin + cameras/tuning first" in body
    # ring order == visual order: Session → Loop freq → Robot id → Start
    ring = [scr.ring.current()]
    for _ in range(3):
        scr.ring.next()
        ring.append(scr.ring.current())
    assert ring == [scr.minutes, scr.hz, scr.robot, scr.start]


def test_host_stream_states_meter_and_header():
    import time

    scr = _host_screen()
    scr._session_total_s = 600
    scr._session_started_at = time.monotonic() - 150
    scr.stream.phase = "running"
    scr.stream.status = "running · session 10:00"

    meter = "".join(sp.content for sp in scr._session_meter_line(90).spans)
    assert "left / 10:00" in meter and "client" in meter
    hdr = "".join(sp.content for sp in scr._stream_header_right())
    assert "● HOST" in hdr and "REAL" in hdr

    scr.stream.status = "stopping (Ctrl+C → SIGKILL after grace)…"
    assert "stopping" in "".join(sp.content for sp in scr._session_meter_line(90).spans)

    scr.stream.phase = "ended"
    scr.stream.status = "✓ finished (rc=0)"
    ended = "".join(sp.content for sp in scr._session_meter_line(90).spans)
    assert "session ended" in ended and "relaunch keeps these settings" in ended
    assert "✓ finished" in "".join(sp.content for sp in scr._stream_header_right())
