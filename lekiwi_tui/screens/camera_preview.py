"""camera_preview.py — CameraPreviewScreen: see what a robot camera sees, ~5 fps.

`f` in Robot config lists the Pi's camera nodes; a list cannot say WHICH LENS is which,
which is the actual question after a replug renumbers them. This screen answers it by
previewing one camera at a time as terminal half-blocks, and `tab` walks the configured
cameras so front/wrist/top can be identified in a few seconds.

Design notes
------------
* The frames come from a capture the Pi runs (see scripts/find_cameras.sh emit-stream),
  NOT from the running host: the host's observation socket is PUSH, so subscribing would
  steal every other frame from teleop/record. Consequently this needs the host stopped,
  the same precondition `f` has, and the caller enforces it.
* Nothing decodes or renders on the draw path unless a NEW frame arrived: the cell grid is
  rebuilt only when the reader thread's frame counter moves, and cached otherwise. At 5 fps
  against a ~30 fps redraw that is a 6x saving on thousands of spans.
* Leaving the screen stops the remote capture (:meth:`on_exit`). A preview that keeps a
  camera open after you walk away is a bug on the robot, not a cosmetic issue.
"""
from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, Any

from pyratatui import Color, Constraint, Direction, Layout, Line, Paragraph, Span, Style, Text

try:                                            # pyratatui >= 0.2.9 ships ratatui-image
    from pyratatui import ImagePicker, ImageWidget
except ImportError:                             # older build: the half-block path still works
    ImagePicker = ImageWidget = None

from .. import ROOT
from ..camstream import CameraStream, decode_jpeg, half_blocks
from ..framework import theme
from ..framework.events import ESC, LEFT, RIGHT, TAB, Key
from ..framework.screen import Nothing, Pop, ScreenState
from .chrome import draw_slim_header, hint_slot_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The launcher that owns the remote capture bash (the SOLE argv source, as everywhere).
FIND_CAMERAS_SCRIPT = ROOT / "scripts" / "find_cameras.sh"

#: Preview geometry. 640x480 at q60 is ~35 KB, so 5 fps is ~175 KB/s — still an order below
#: the USB+WiFi load that used to freeze the Pi, and it is only spent while the host is
#: stopped and a single camera streams. The old 320x240 was sized for half-blocks, where the
#: cell grid (120x90 px) was the real ceiling; a graphics protocol shows the frame as-is.
FPS = 5
WIDTH = 640
HEIGHT = 480
QUALITY = 60

#: Cell budget for the image. Beyond this the span count starts to cost more than the
#: extra detail is worth, and a camera is identifiable long before that.
MAX_COLS = 120
MAX_ROWS = 56

#: A half-block cell is one pixel wide and two tall, and the frame is 4:3, so the cell grid
#: has to be ~2.67x wider than it is tall or the picture comes out stretched.
CELL_ASPECT = (WIDTH / HEIGHT) * 2

_ROT_DEGREES = {"NO_ROTATION": 0, "ROTATE_90": 90, "ROTATE_180": 180, "ROTATE_270": 270}

#: Force a renderer: `kitty` | `halfblocks`. Unset = decide from the environment.
PROTOCOL_ENV = "LEKIWI_IMAGE_PROTOCOL"


def kitty_capable(environ: Any = None) -> bool:
    """Whether this terminal speaks the kitty graphics protocol, decided from env vars.

    NOT by querying the terminal: a query writes an escape and reads the reply on the very
    fd the TUI is reading keys from, mid-run, in raw mode. Env detection cannot deadlock or
    swallow a keystroke, and the two terminals that matter here announce themselves.
    """
    import os

    env = os.environ if environ is None else environ
    forced = str(env.get(PROTOCOL_ENV, "")).strip().lower()
    if forced in ("kitty", "halfblocks"):
        return forced == "kitty"
    if env.get("KITTY_WINDOW_ID") or env.get("GHOSTTY_RESOURCES_DIR"):
        return True
    term = f"{env.get('TERM', '')} {env.get('TERM_PROGRAM', '')}".lower()
    return "kitty" in term or "ghostty" in term


def make_picker() -> tuple[Any, str]:
    """``(picker, label)`` — a kitty picker where supported, else half-blocks.

    Returns ``(None, ...)`` if this pyratatui has no image support, which the screen then
    renders with its own half-block path instead of failing.
    """
    if ImagePicker is None:
        return None, "halfblocks (built-in)"
    try:
        if kitty_capable():
            return ImagePicker.kitty(), "kitty graphics"
        return ImagePicker.halfblocks(), "half-blocks"
    except Exception:
        return None, "halfblocks (built-in)"


def rotation_degrees(value: Any) -> int:
    """A yaml rotation (``ROTATE_180``) or a number -> degrees the Pi should apply."""
    text = str(value or "NO_ROTATION").strip().upper()
    if text in _ROT_DEGREES:
        return _ROT_DEGREES[text]
    try:
        deg = int(text)
    except ValueError:
        return 0
    return deg if deg in (0, 90, 180, 270) else 0


def configured_cameras(doc: Any) -> list[dict[str, Any]]:
    """The ``_cameras`` block as ``[{name, device, rotation}]``, in yaml order."""
    from ..config import cfg_get

    cams = cfg_get("_cameras", doc=doc) or {}
    if not isinstance(cams, dict):
        return []
    out = []
    for name, spec in cams.items():
        spec = spec if isinstance(spec, dict) else {}
        out.append({
            "name": str(name),
            "device": str(spec.get("index_or_path", "")),
            "rotation": rotation_degrees(spec.get("rotation")),
        })
    return [c for c in out if c["device"]]


def build_stream_argv(ctx: "Context", camera: dict[str, Any]) -> list[str]:
    """`ssh -n <host> "<emit-stream remote bash>"` for one camera.

    ``-n`` is load-bearing, not hygiene: without it ssh reads the TERMINAL's stdin — the
    same fd the TUI reads keys from — and forwards those bytes to the remote shell. The
    screen then looks frozen, because every keypress goes to the robot instead (observed:
    a live preview that would not answer q, tab or Ctrl+C). The spawn also passes
    stdin=DEVNULL, so neither layer can claim the keyboard.

    No ``-t`` either: a PTY would translate the raw JPEG bytes this reads off stdout.
    """
    from ..remote import validate_remote_name, validate_ssh_host

    host = validate_ssh_host(ctx.cfg["LEKIWI_HOST"])
    conda_env = validate_remote_name(ctx.cfg["CONDA_ENV"], "conda env")
    remote = subprocess.check_output(
        ["bash", str(FIND_CAMERAS_SCRIPT), "emit-stream",
         "--conda-env", conda_env, "--device", str(camera["device"]),
         "--fps", str(FPS), "--width", str(WIDTH), "--height", str(HEIGHT),
         "--quality", str(QUALITY), "--rotation", str(camera["rotation"])],
        text=True,
    )
    return ["ssh", "-n", "-o", "ConnectTimeout=5", host, remote]


def fit_grid(width: int, height: int) -> tuple[int, int]:
    """The largest 4:3-correct cell grid that fits *width* x *height*, within the caps."""
    rows = max(1, min(MAX_ROWS, height))
    cols = max(1, min(MAX_COLS, width, int(rows * CELL_ASPECT)))
    rows = max(1, min(rows, int(cols / CELL_ASPECT)))
    return cols, rows


class CameraPreviewScreen(ScreenState):
    """One camera at a time, ~5 fps, as half-block cells. ``tab``/←→ switch camera."""

    title = "camera preview"

    def __init__(self, app: "App", ctx: "Context", *, index: int = 0,
                 spawn: Any = None) -> None:
        self.app = app
        self.ctx = ctx
        self.cameras = configured_cameras(getattr(ctx, "doc", None))
        self.index = index if 0 <= index < len(self.cameras) else 0
        self._spawn = spawn                 # injected in tests; None = real ssh
        self._stream: CameraStream | None = None
        self._cells: list[list[tuple[str, tuple[int, int, int], tuple[int, int, int]]]] = []
        self._cells_from = -1               # frame_count the cached cells were built from
        self._decode_failed = False
        self._picker, self._protocol = make_picker()
        self._image_state: Any = None
        self._image_from = -1               # frame_count the loaded image was built from
        self._frame_path: Any = None        # the file the protocol loads each frame from
        self._listing = ""                  # sysfs device list, fetched only on failure
        self._listing_started = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self) -> None:
        self._restart()

    def on_exit(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def close(self) -> None:
        """Drop the scratch frame file. Separate from on_exit, which also fires when a
        screen is pushed OVER this one and the preview should survive."""
        import contextlib
        import os

        if self._frame_path:
            with contextlib.suppress(OSError):
                os.unlink(self._frame_path)
            self._frame_path = None

    def _restart(self) -> None:
        self.on_exit()
        self._cells, self._cells_from, self._decode_failed = [], -1, False
        if not self.cameras:
            return
        camera = self.cameras[self.index]
        spawn = self._spawn or (lambda: subprocess.Popen(
            build_stream_argv(self.ctx, camera),
            stdin=subprocess.DEVNULL,          # never let ssh read the TUI's keyboard
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
        self._stream = CameraStream(spawn=spawn)
        self._stream.start(now=time.monotonic())

    # ── keys ──────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in ("q", ESC):
            self.on_exit()
            self.close()
            return Pop()
        if name in (TAB, RIGHT, "l") and self.cameras:
            self.index = (self.index + 1) % len(self.cameras)
            self._restart()
            return Nothing
        if name in (LEFT, "h") and self.cameras:
            self.index = (self.index - 1) % len(self.cameras)
            self._restart()
            return Nothing
        if name == "r":
            self._restart()
            return Nothing
        return Nothing

    # ── the device list, fetched only when the configured node lets us down ────
    def _fetch_listing(self) -> None:
        """Ask the robot what capture nodes it HAS, in a thread.

        Only on failure, and only once: this is the answer to "the yaml says /dev/video4
        and that is gone, so what is there now", which is precisely when a plain error
        message is not enough. The payload needs no lerobot env on the Pi.
        """
        import threading

        if self._listing_started:
            return
        self._listing_started = True

        def run() -> None:
            try:
                from .robot_config import build_find_cameras_argv

                out = subprocess.run(build_find_cameras_argv(self.ctx), capture_output=True,
                                     text=True, timeout=20, check=False)
                self._listing = (out.stdout or out.stderr or "").strip()
            except Exception as exc:
                self._listing = f"(could not list the robot's cameras: {exc})"

        threading.Thread(target=run, daemon=True).start()

    # ── rendering ─────────────────────────────────────────────────────────────
    def _refresh_cells(self, cols: int, rows: int) -> None:
        """Rebuild the cell grid, but only when a new frame has arrived."""
        stream = self._stream
        if stream is None or stream.latest is None or stream.frame_count == self._cells_from:
            return
        pixels = decode_jpeg(stream.latest)
        self._cells_from = stream.frame_count
        if pixels is None:
            self._decode_failed = True
            return
        self._decode_failed = False
        # Nearest-neighbour downsample to the cell grid: two pixel rows per cell row.
        target_h, target_w = max(1, rows * 2), max(1, cols)
        src_h, src_w = len(pixels), len(pixels[0]) if pixels else 0
        if not src_h or not src_w:
            return
        picked = [[pixels[min(src_h - 1, y * src_h // target_h)][min(src_w - 1, x * src_w // target_w)]
                   for x in range(target_w)] for y in range(target_h)]
        self._cells = half_blocks(picked)

    def _failure_lines(self) -> list[Line] | None:
        """The screen when this camera did not come up: what the robot said, plus what it
        actually has. Returns None while things are fine."""
        stream = self._stream
        if stream is None or not stream.error:
            return None
        self._fetch_listing()
        camera = self.cameras[self.index] if self.cameras else {"device": "?"}
        out = [Line([Span(f"✗ {camera['device']}: {stream.error}", theme.ERR_STYLE)]),
               Line([])]
        if not self._listing:
            out.append(Line([Span("asking the robot which capture nodes it has…",
                                  theme.FAINT_STYLE)]))
        else:
            out += [Line([Span(ln, theme.TEXT_STYLE if "/dev/" in ln else theme.MUTED_STYLE)])
                    for ln in self._listing.splitlines()]
            out += [Line([]),
                    Line([Span("q back, then e to point lekiwi.yaml at the right node",
                               theme.FAINT_STYLE)])]
        return out

    def _refresh_image(self) -> bool:
        """Load the latest frame through the graphics protocol. True when there is an image
        to render. The frame is written to a file because that is what the protocol loads,
        and only when the counter moves — 5 loads a second, not one per redraw."""
        import tempfile

        stream = self._stream
        if self._picker is None or stream is None or stream.latest is None:
            return self._image_state is not None
        if stream.frame_count == self._image_from:
            return self._image_state is not None
        if self._frame_path is None:
            fd, path = tempfile.mkstemp(prefix="lekiwi-preview-", suffix=".jpg")
            import os

            os.close(fd)
            self._frame_path = path
        try:
            with open(self._frame_path, "wb") as fh:
                fh.write(stream.latest)
            self._image_state = self._picker.load(self._frame_path)
            self._image_from = stream.frame_count
        except Exception:
            self._picker = None           # fall back to half-blocks for the rest of the run
            self._protocol = "half-blocks (image protocol failed)"
            return False
        return self._image_state is not None

    def _image_lines(self, cols: int, rows: int) -> list[Line]:
        """The half-block fallback: only used when there is no image protocol."""
        self._refresh_cells(cols, rows)
        if self._decode_failed:
            return [Line([Span("frames arrive but cannot be decoded here — install opencv "
                               "or pillow in this env", theme.WARN_STYLE)])]
        if not self._cells:
            return [Line([Span("waiting for the first frame…", theme.FAINT_STYLE)])]
        return [Line([Span(glyph, Style().fg(Color.rgb(*fg)).bg(Color.rgb(*bg)))
                      for glyph, fg, bg in row])
                for row in self._cells]

    def draw(self, frame: Any, area: Any) -> None:
        bands = (Layout().direction(Direction.Vertical)
                 .constraints([Constraint.length(1), Constraint.length(1),
                               Constraint.fill(1), Constraint.length(1)])
                 .split(area))
        camera = self.cameras[self.index] if self.cameras else {"name": "none", "device": "-",
                                                                "rotation": 0}
        rot = f" · rot {camera['rotation']}°" if camera["rotation"] else ""
        draw_slim_header(frame, bands[0], self.ctx, "camera preview",
                         right=[Span(f"{camera['name']} · {camera['device']}{rot}",
                                     theme.TEXT_STYLE)])
        frame.render_widget(
            Paragraph.from_string(theme.rule(bands[1].width)).style(theme.RULE_HEAVY_STYLE),
            bands[1])

        failed = self._failure_lines()
        if failed is not None:
            frame.render_widget(Paragraph(Text(failed)), bands[2])
        elif self._refresh_image():
            # the protocol keeps the frame's real resolution; the widget fits it to the area
            frame.render_stateful_image(ImageWidget(), bands[2], self._image_state)
        else:
            cols, rows = fit_grid(bands[2].width, bands[2].height)
            frame.render_widget(Paragraph(Text(self._image_lines(cols, rows))), bands[2])

        fps = self._stream.fps(now=time.monotonic()) if self._stream else 0.0
        which = f"{self.index + 1}/{len(self.cameras)}" if self.cameras else "0/0"
        frame.render_widget(hint_slot_line(
            f"{which} · {fps:.1f} fps · {WIDTH}x{HEIGHT} · {self._protocol} · host must be stopped",
            bands[3].width,
            keys=(("tab", "next camera"), ("r", "restart"), ("q", "back"))),
            bands[3])


__all__ = ["CameraPreviewScreen", "build_stream_argv", "configured_cameras", "rotation_degrees"]
