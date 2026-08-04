"""The ~5 fps camera preview: wire protocol, half-block geometry, the remote capture
argv, and the screen's lifecycle.

Everything here runs without a Pi and without an image decoder: `frames` is fed a byte
stream, `half_blocks` is fed pixels, and the screen gets an injected `spawn`. The one
un-fakeable piece (`decode_jpeg`) is deliberately tiny and reports None rather than raising.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from lekiwi_tui import ROOT
from lekiwi_tui.camstream import CameraStream, frames, half_blocks
from lekiwi_tui.framework.events import ESC, TAB, Key
from lekiwi_tui.framework.screen import Nothing, Pop, Push
from lekiwi_tui.screens.camera_preview import (
    CameraPreviewScreen,
    build_stream_argv,
    configured_cameras,
    fit_grid,
    rotation_degrees,
)
from lekiwi_tui.screens.robot_config import RobotConfigScreen

from conftest import make_ctx

FIND_SH = ROOT / "scripts" / "find_cameras.sh"


def _wire(*payloads: bytes, junk: bytes = b"") -> io.BytesIO:
    buf = bytearray(junk)
    for p in payloads:
        buf += b"FRAME %d\n" % len(p) + p
    return io.BytesIO(bytes(buf))


# ── wire protocol ─────────────────────────────────────────────────────────────


def test_frames_yields_each_payload_exactly():
    assert list(frames(_wire(b"abc", b"defgh"))) == [b"abc", b"defgh"]


def test_frames_resynchronises_past_remote_noise():
    """The remote shell can print a warning on the same fd; that must not desync us."""
    noisy = _wire(b"xy", junk=b"Warning: setlocale failed\n")
    assert list(frames(noisy)) == [b"xy"]


def test_frames_stops_on_a_truncated_tail():
    truncated = io.BytesIO(b"FRAME 10\nonly4")
    assert list(frames(truncated)) == []


def test_frames_refuses_an_absurd_length():
    """A corrupted header must not make us allocate the machine away."""
    assert list(frames(io.BytesIO(b"FRAME 999999999\n"), max_bytes=1024)) == []


# ── half-block geometry ───────────────────────────────────────────────────────


def test_two_pixel_rows_become_one_cell_row():
    red, blue = (255, 0, 0), (0, 0, 255)
    cells = half_blocks([[red, red], [blue, blue]])
    assert len(cells) == 1
    glyph, fg, bg = cells[0][0]
    assert glyph == "▀" and fg == red and bg == blue, "upper pixel is fg, lower is bg"


def test_an_odd_last_row_pairs_with_itself_rather_than_inventing_black():
    grey = (128, 128, 128)
    cells = half_blocks([[grey], [grey], [grey]])
    assert len(cells) == 2
    _, fg, bg = cells[1][0]
    assert fg == bg == grey


# ── the remote capture argv ───────────────────────────────────────────────────


def test_the_stream_argv_carries_geometry_and_rotation():
    argv = build_stream_argv(make_ctx(gpu_name=""),
                            {"name": "wrist", "device": "/dev/video4", "rotation": 180})
    assert argv[0] == "ssh" and "-t" not in argv, "a PTY would mangle the JPEG bytes"
    payload = argv[-1]
    assert 'cv2.VideoCapture("/dev/video4")' in payload
    assert "cv2.rotate(frame, rot[180])" in payload
    assert "(640, 480)" in payload and "period = 1.0 / 5" in payload
    assert "IMWRITE_JPEG_QUALITY), 60" in payload


@pytest.mark.parametrize(("flag", "value", "message"), [
    ("--device", "$(whoami)", "must be a /dev path"),
    ("--device", "/dev/../etc/shadow", "must not contain"),
    ("--fps", "60", "30 or less"),
    ("--rotation", "45", "must be 0, 90, 180 or 270"),
    ("--quality", "200", "must be 1..100"),
])
def test_the_emitter_refuses_unsafe_stream_arguments(flag, value, message):
    """The device is interpolated into remote python, so it is validated hard."""
    args = ["bash", str(FIND_SH), "emit-stream", "--conda-env", "lekiwi",
            "--device", "/dev/video0", flag, value]
    out = subprocess.run(args, capture_output=True, text=True)
    assert out.returncode == 2 and message in out.stderr


def test_rotation_degrees_reads_both_yaml_and_numeric_forms():
    assert rotation_degrees("ROTATE_180") == 180
    assert rotation_degrees("NO_ROTATION") == 0
    assert rotation_degrees(270) == 270
    assert rotation_degrees("ROTATE_45") == 0     # unknown enum -> no rotation
    assert rotation_degrees(None) == 0


# ── the cameras it previews ───────────────────────────────────────────────────


def _doc(**cams) -> dict:
    return {"_cameras": cams}


def test_configured_cameras_keeps_yaml_order_and_drops_deviceless_entries():
    doc = _doc(front={"index_or_path": "/dev/video0"},
               wrist={"index_or_path": "/dev/video4", "rotation": "ROTATE_180"},
               broken={"width": 640})
    cams = configured_cameras(doc)
    assert [c["name"] for c in cams] == ["front", "wrist"]
    assert cams[1]["rotation"] == 180


# ── the stream object ─────────────────────────────────────────────────────────


class _Proc:
    def __init__(self, stream) -> None:
        self.stdout = stream
        self.killed = False

    def terminate(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:  # noqa: ANN001
        return 0

    def kill(self) -> None:
        self.killed = True


def test_the_stream_keeps_only_the_latest_frame():
    proc = _Proc(_wire(b"one", b"two", b"three"))
    stream = CameraStream(spawn=lambda: proc)
    stream.start(now=0.0)
    stream._thread.join(timeout=2)
    assert stream.latest == b"three", "a preview that lags is worse than one that skips"
    assert stream.frame_count == 3


def test_leaving_kills_the_remote_capture():
    """A preview that keeps a camera open after you walk away is a robot-side bug."""
    proc = _Proc(_wire(b"x"))
    stream = CameraStream(spawn=lambda: proc)
    stream.start(now=0.0)
    stream._thread.join(timeout=2)
    stream.stop()
    assert proc.killed


def test_a_stream_that_never_yields_says_so():
    stream = CameraStream(spawn=lambda: _Proc(io.BytesIO(b"")))
    stream.start(now=0.0)
    stream._thread.join(timeout=2)
    assert "no frames" in stream.error and "host stopped" in stream.error


# ── the screen ────────────────────────────────────────────────────────────────


def _screen(monkeypatch, payloads=(b"jpeg-bytes",)) -> tuple[CameraPreviewScreen, list[_Proc]]:
    ctx = make_ctx(gpu_name="")
    ctx.doc = _doc(front={"index_or_path": "/dev/video0"},
                   wrist={"index_or_path": "/dev/video4", "rotation": "ROTATE_180"})
    spawned: list[_Proc] = []

    def spawn():
        proc = _Proc(_wire(*payloads))
        spawned.append(proc)
        return proc

    screen = CameraPreviewScreen(None, ctx, spawn=spawn)
    return screen, spawned


def test_entering_starts_one_capture_and_tab_restarts_it_for_the_next_camera(monkeypatch):
    screen, spawned = _screen(monkeypatch)
    screen.on_enter()
    assert len(spawned) == 1 and screen.index == 0

    assert screen.handle_key(Key(name=TAB)) is Nothing
    assert screen.index == 1 and len(spawned) == 2, "switching camera restarts the capture"
    assert spawned[0].killed, "the previous camera is released"


def test_leaving_the_screen_stops_the_capture(monkeypatch):
    screen, spawned = _screen(monkeypatch)
    screen.on_enter()
    assert isinstance(screen.handle_key(Key(name=ESC)), Pop)
    assert spawned[0].killed


def test_undecodable_frames_are_reported_not_raised(monkeypatch):
    """CI has no cv2/PIL, and neither does a bare install: it must degrade in words."""
    screen, _ = _screen(monkeypatch)
    screen.on_enter()
    screen._stream._thread.join(timeout=2)
    lines = screen._image_lines(cols=20, rows=10)
    text = "".join(sp.content for ln in lines for sp in ln.spans)
    assert "cannot be decoded" in text or "waiting for the first frame" in text


# ── the way in ────────────────────────────────────────────────────────────────


def test_p_pushes_the_preview_when_the_host_is_down(monkeypatch):
    import lekiwi_tui.hostprobe as hostprobe

    monkeypatch.setattr(hostprobe, "host_alive", lambda ctx: False)
    ctx = make_ctx(gpu_name="")
    screen = RobotConfigScreen(None, ctx)
    screen.ctx.doc = _doc(front={"index_or_path": "/dev/video0"})
    action = screen.handle_key(Key(name="p"))
    assert isinstance(action, Push)
    assert isinstance(action.screen, CameraPreviewScreen)


def test_p_refuses_while_the_host_holds_the_cameras(monkeypatch):
    import lekiwi_tui.hostprobe as hostprobe

    monkeypatch.setattr(hostprobe, "host_alive", lambda ctx: True)

    class _App:
        def __init__(self) -> None:
            self.toasts: list[tuple[str, str]] = []

        def notify(self, msg, level="info", **kw):  # noqa: ANN001, ANN003
            self.toasts.append((msg, level))

    app = _App()
    screen = RobotConfigScreen(app, make_ctx(gpu_name=""))
    assert screen.handle_key(Key(name="p")) is Nothing
    assert app.toasts and "stop it first" in app.toasts[0][0]


def test_p_says_so_when_no_camera_has_a_device(monkeypatch):
    import lekiwi_tui.hostprobe as hostprobe

    monkeypatch.setattr(hostprobe, "host_alive", lambda ctx: False)

    class _App:
        def __init__(self) -> None:
            self.toasts: list[tuple[str, str]] = []

        def notify(self, msg, level="info", **kw):  # noqa: ANN001, ANN003
            self.toasts.append((msg, level))

    app = _App()
    screen = RobotConfigScreen(app, make_ctx(gpu_name=""))
    screen.ctx.doc = {"_cameras": {}}
    assert screen.handle_key(Key(name="p")) is Nothing
    assert "no cameras" in app.toasts[0][0]


def test_the_hint_line_names_the_preview_key_in_words_too():
    """The keycap row is the first thing dropped on a narrow terminal (observed at ~120
    columns), so the sentence has to carry the key as well."""
    source = Path(ROOT / "lekiwi_tui" / "screens" / "robot_config.py").read_text()
    assert "p previews the robot cameras" in source
    assert '("p", "preview")' in source


# ── the keyboard must stay with the TUI ────────────────────────────────────────


def test_ssh_is_told_not_to_read_our_stdin():
    """Without -n, ssh reads the terminal's stdin — the same fd the TUI reads keys from —
    and forwards it to the remote shell, so the screen stops answering q/tab/Ctrl+C."""
    argv = build_stream_argv(make_ctx(gpu_name=""),
                            {"name": "front", "device": "/dev/video0", "rotation": 0})
    assert argv[:2] == ["ssh", "-n"]


def test_the_real_spawn_detaches_stdin(monkeypatch):
    """Belt and braces at the process layer: neither ssh nor its child may claim the
    keyboard, whatever the argv says."""
    captured = {}
    real_popen = subprocess.Popen

    class _Popen:
        """Intercepts only the ssh spawn; the emitter's own check_output still runs for
        real, so this exercises the actual argv-building path rather than a stub of it."""

        def __new__(cls, argv, **kwargs):
            if not (argv and argv[0] == "ssh"):
                return real_popen(argv, **kwargs)
            captured["argv"], captured["kwargs"] = argv, kwargs
            return super().__new__(cls)

        def __init__(self, argv, **kwargs):
            self.stdout = io.BytesIO(b"")

        def terminate(self): pass

        def wait(self, timeout=None): return 0  # noqa: ANN001

        def kill(self): pass

    monkeypatch.setattr(subprocess, "Popen", _Popen)
    ctx = make_ctx(gpu_name="")
    ctx.doc = _doc(front={"index_or_path": "/dev/video0"})
    screen = CameraPreviewScreen(None, ctx)          # no injected spawn: the real path
    screen.on_enter()
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.PIPE
    screen.on_exit()


# ── geometry ──────────────────────────────────────────────────────────────────


def test_the_grid_keeps_the_frame_from_being_stretched():
    """A cell is 1 pixel wide and 2 tall and the frame is 4:3, so the grid must be ~2.67x
    wider than tall; a naive fill would squash a face into a rectangle."""
    cols, rows = fit_grid(200, 80)
    assert abs(cols / rows - 320 / 240 * 2) < 0.2
    # never larger than the pane it was given
    cols, rows = fit_grid(40, 10)
    assert cols <= 40 and rows <= 10


# ── which renderer ────────────────────────────────────────────────────────────


def test_kitty_is_detected_from_the_environment_not_by_querying():
    """A capability QUERY writes an escape and reads the reply on the fd the TUI reads keys
    from, mid-run, in raw mode. Env detection cannot deadlock or eat a keystroke."""
    from lekiwi_tui.screens.camera_preview import PROTOCOL_ENV, kitty_capable

    assert kitty_capable({"GHOSTTY_RESOURCES_DIR": "/x"})
    assert kitty_capable({"KITTY_WINDOW_ID": "1"})
    assert kitty_capable({"TERM": "xterm-ghostty"})
    assert not kitty_capable({"TERM": "xterm-256color"})
    # an explicit override wins both ways, for a terminal that lies either direction
    assert not kitty_capable({"TERM": "xterm-ghostty", PROTOCOL_ENV: "halfblocks"})
    assert kitty_capable({"TERM": "xterm-256color", PROTOCOL_ENV: "kitty"})


def test_the_capture_is_sized_for_a_graphics_protocol():
    """640x480 at q60 is ~35 KB, so 5 fps is ~175 KB/s — an order below the load that used
    to freeze the Pi, and only spent while the host is stopped."""
    from lekiwi_tui.screens.camera_preview import HEIGHT, QUALITY, WIDTH

    assert (WIDTH, HEIGHT) == (640, 480) and QUALITY == 60


# ── failure shows what the robot actually has ─────────────────────────────────


def test_a_missing_node_reports_the_remote_message_and_asks_what_exists(monkeypatch):
    """The case this replaces the old `f` key with: the yaml says /dev/video4, that node is
    gone, and the answer you need is which nodes exist now."""
    ctx = make_ctx(gpu_name="")
    ctx.doc = _doc(wrist={"index_or_path": "/dev/video4"})

    class _Failing:
        def __init__(self) -> None:
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"\xe2\x9c\x97 could not open /dev/video4 (is the host stopped?)\n")

        def terminate(self): pass

        def wait(self, timeout=None): return 0  # noqa: ANN001

        def kill(self): pass

    screen = CameraPreviewScreen(None, ctx, spawn=_Failing)
    fetched = []
    monkeypatch.setattr(screen, "_fetch_listing", lambda: fetched.append(True))
    screen.on_enter()
    screen._stream._thread.join(timeout=2)

    lines = screen._failure_lines()
    text = "\n".join("".join(sp.content for sp in ln.spans) for ln in lines)
    assert "/dev/video4" in text and "could not open" in text
    assert fetched, "a failure is exactly when the device list is worth fetching"


def test_a_healthy_stream_never_fetches_the_listing(monkeypatch):
    screen, _ = _screen(monkeypatch)
    monkeypatch.setattr(screen, "_fetch_listing",
                        lambda: pytest.fail("must not ask while frames are arriving"))
    screen.on_enter()
    screen._stream._thread.join(timeout=2)
    assert screen._failure_lines() is None


def test_the_hint_row_keeps_keycaps_when_labels_do_not_fit():
    """padded_line drops the right side wholesale on overflow, which is how `p` disappeared
    on a ~120-column terminal. Labels must go before keycaps do."""
    from lekiwi_tui.screens.chrome import hint_slot_row

    keys = (("e", "edit"), ("r", "reload"), ("p", "preview"), ("q", "back"))
    hint = "read-only — e edits lekiwi.yaml, r reloads, p previews the robot cameras"

    def rendered(width):
        return "".join(sp.content for sp in hint_slot_row(hint, width, keys).spans)

    wide = rendered(160)
    assert "preview" in wide and " p " in wide

    tight = rendered(96)          # labels no longer fit
    assert " p " in tight, "the keycap survives"
    assert "preview" not in tight.split("cameras")[-1], "its label does not"
