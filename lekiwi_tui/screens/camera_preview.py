"""camera_preview.py — CameraPreviewScreen: every robot camera at once, ~5 fps, with
live rotation.

The question this answers is "which lens is which, and is any of them upside down", so it
shows ALL configured cameras as tiles side by side rather than one at a time: three tiles
compared in one glance beats three sequential previews held in memory. Each tile carries its
name, device node and current rotation.

Rotation is interactive because that is the other half of the job. `[` and `]` rotate the
selected camera, the remote capture restarts with the new value, and what you see is exactly
what the robot will send (the degrees map to cv2 the same way lerobot's get_cv2_rotation
does). `w` writes the set into lekiwi.yaml losslessly, keeping a timestamped backup, so the
answer does not have to be retyped by hand.

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
from ..camstream import CameraStream, camera_error, decode_jpeg, half_blocks
from ..framework import theme
from ..framework.events import ESC, LEFT, RIGHT, TAB, Key
from ..framework.screen import Nothing, Pop, ScreenState
from .chrome import draw_slim_header, hint_slot_line

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The launcher that owns the remote capture bash (the SOLE argv source, as everywhere).
FIND_CAMERAS_SCRIPT = ROOT / "scripts" / "find_cameras.sh"

#: Preview geometry, sized for TILES: 320x240 at q50 is ~12 KB, so three cameras at 5 fps is
#: ~180 KB/s — the same budget one full-size single preview used, and an order below the
#: USB+WiFi load that used to freeze the Pi. Only spent while the host is stopped.
FPS = 5
WIDTH = 320
HEIGHT = 240
QUALITY = 50

#: Rotations, in the order `[` and `]` walk them.
ROTATIONS = (0, 90, 180, 270)

#: yaml spelling of each rotation (lerobot's Cv2Rotation names).
ROT_NAMES = {0: "NO_ROTATION", 90: "ROTATE_90", 180: "ROTATE_180", 270: "ROTATE_270"}

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


def build_all_streams_argv(ctx: "Context", cameras: list[dict[str, Any]]) -> list[str]:
    """`ssh -n <host> "<emit-stream-all bash>"` — every camera down ONE channel.

    One session rather than one per camera: three logins would mean three pythons and three
    chances to leave a device open. ``-n`` and stdin=DEVNULL keep ssh off the TUI's keyboard.
    """
    from ..remote import validate_remote_name, validate_ssh_host

    host = validate_ssh_host(ctx.cfg["LEKIWI_HOST"])
    conda_env = validate_remote_name(ctx.cfg["CONDA_ENV"], "conda env")
    args = ["bash", str(FIND_CAMERAS_SCRIPT), "emit-stream-all", "--conda-env", conda_env,
            "--fps", str(FPS), "--width", str(WIDTH), "--height", str(HEIGHT),
            "--quality", str(QUALITY)]
    for cam in cameras:
        args += ["--camera", f"{cam['name']}={cam['device']}@{cam['rotation']}"]
    remote = subprocess.check_output(args, text=True)
    return ["ssh", "-n", "-o", "ConnectTimeout=5", host, remote]


def write_rotations(cameras: list[dict[str, Any]], path: Any = None) -> str:
    """Save each camera's rotation into lekiwi.yaml, losslessly. Returns the backup path.

    ruamel round-trip, so the anchors (&cameras), the `<<:` merges and every comment survive
    — a PyYAML dump would rename anchors to &id001 and drop the lot. A timestamped backup
    goes next to the file first: lekiwi.yaml is git-ignored, so there is no other undo.
    """
    import shutil
    import time as _time

    from .. import CFG_FILE
    from ..config import dump_yaml_rt, load_yaml_rt

    target = CFG_FILE if path is None else path
    doc = load_yaml_rt(target)
    if doc is None:
        raise OSError(f"could not read {target}")
    backup = f"{target}.bak-{_time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(target, backup)
    block = doc.get("_cameras") or {}
    for cam in cameras:
        spec = block.get(cam["name"])
        if spec is not None:
            spec["rotation"] = ROT_NAMES.get(int(cam["rotation"]), "NO_ROTATION")
    dump_yaml_rt(doc, target)
    return backup


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
    """Every configured camera as a tile, ~5 fps, with `[`/`]` rotating the selected one."""

    title = "camera preview"

    def __init__(self, app: "App", ctx: "Context", *, index: int = 0,
                 spawn: Any = None) -> None:
        self.app = app
        self.ctx = ctx
        self.cameras = configured_cameras(getattr(ctx, "doc", None))
        self.index = index if 0 <= index < len(self.cameras) else 0
        self._spawn = spawn                 # injected in tests; None = real ssh
        self._stream: CameraStream | None = None
        self._picker, self._protocol = make_picker()
        self._images: dict[str, Any] = {}   # camera name -> (path, state, frame_count)
        self._cells: dict[str, list] = {}   # camera name -> half-block grid (fallback path)
        self._cells_from: dict[str, int] = {}
        self._decode_failed = False
        self._listing = ""                  # sysfs device list, fetched only on failure
        self._listing_started = False
        self._dirty = False                 # a rotation changed but is not saved yet
        self._saved_note = ""

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def on_enter(self) -> None:
        self._restart()

    def on_exit(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def close(self) -> None:
        """Drop the scratch frame files. Separate from on_exit, which also fires when a
        screen is pushed OVER this one and the preview should survive."""
        import contextlib
        import os

        for path, _state, _n in self._images.values():
            with contextlib.suppress(OSError):
                os.unlink(path)
        self._images.clear()

    def _restart(self) -> None:
        """(Re)start the single multiplexed capture for the current rotation set."""
        self.on_exit()
        self._cells, self._cells_from = {}, {}
        self._decode_failed = False
        if not self.cameras:
            return
        cameras = [dict(c) for c in self.cameras]
        spawn = self._spawn or (lambda: subprocess.Popen(
            build_all_streams_argv(self.ctx, cameras),
            stdin=subprocess.DEVNULL,          # never let ssh read the TUI's keyboard
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        self._stream = CameraStream(spawn=spawn)
        self._stream.start(now=time.monotonic())

    # ── keys ──────────────────────────────────────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in ("q", ESC):
            self.on_exit()
            self.close()
            return Pop()
        if not self.cameras:
            return Nothing
        if name in (TAB, RIGHT, "l"):
            self.index = (self.index + 1) % len(self.cameras)
            return Nothing
        if name in (LEFT, "h"):
            self.index = (self.index - 1) % len(self.cameras)
            return Nothing
        if name in ("]", "["):
            step = 1 if name == "]" else -1
            cam = self.cameras[self.index]
            cam["rotation"] = ROTATIONS[(ROTATIONS.index(cam["rotation"]) + step) % len(ROTATIONS)]
            self._dirty, self._saved_note = True, ""
            self._restart()                 # so the tile shows what the robot would send
            return Nothing
        if name == "w":
            return self._save()
        if name == "r":
            self._restart()
            return Nothing
        return Nothing

    def _save(self) -> Any:
        """`w`: write the rotations into lekiwi.yaml and refresh the doc this screen reads."""
        from ..config import load_yaml

        try:
            backup = write_rotations(self.cameras)
        except Exception as exc:
            if self.app is not None:
                self.app.notify(f"could not write lekiwi.yaml: {exc}", "error")
            return Nothing
        self.ctx.doc = load_yaml()
        self._dirty = False
        self._saved_note = f"saved · backup {backup.rsplit('/', 1)[-1]}"
        if self.app is not None:
            self.app.notify(f"rotations written to lekiwi.yaml (backup {backup})", "info")
        return Nothing

    # ── the device list, fetched only when every camera lets us down ───────────
    def _fetch_listing(self) -> None:
        """Ask the robot what capture nodes it HAS, in a thread.

        Only on total failure, and only once: this is the answer to "the yaml says
        /dev/video4 and that is gone, so what is there now". Needs no lerobot env on the Pi.
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

    # ── tiles ─────────────────────────────────────────────────────────────────
    def _tile_grid(self, area: Any) -> list[Any]:
        """One rect per camera: two across when there is room, stacked when there is not."""
        n = max(1, len(self.cameras))
        per_row = 2 if area.width >= 80 and n > 1 else 1
        row_count = (n + per_row - 1) // per_row
        rows = (Layout().direction(Direction.Vertical)
                .constraints([Constraint.ratio(1, row_count)] * row_count).split(area))
        rects = []
        for r in range(row_count):
            in_row = min(per_row, n - r * per_row)
            cols = (Layout().direction(Direction.Horizontal)
                    .constraints([Constraint.ratio(1, in_row)] * in_row).split(rows[r]))
            rects += [cols[c] for c in range(in_row)]
        return rects

    def _tile_header(self, cam: dict[str, Any], *, selected: bool, width: int) -> Line:
        rot = f"{cam['rotation']}°" if cam["rotation"] else "no rot"
        label = f"{'▸ ' if selected else '  '}{cam['name']}  {cam['device']}  {rot}"
        style = theme.HIGHLIGHT_STYLE if selected else theme.MUTED_STYLE
        return Line([Span(label.ljust(max(0, width)), style)])

    def _image_state(self, cam: dict[str, Any]) -> Any:
        """The protocol's state for this camera's latest frame, reloaded only when it moves."""
        import os
        import tempfile

        stream = self._stream
        if self._picker is None or stream is None:
            return None
        payload = stream.per_camera.get(cam["name"])
        count = stream.counts.get(cam["name"], 0)
        if payload is None:
            return None
        path, state, loaded = self._images.get(cam["name"], (None, None, -1))
        if loaded == count and state is not None:
            return state
        if path is None:
            fd, path = tempfile.mkstemp(prefix=f"lekiwi-preview-{cam['name']}-", suffix=".jpg")
            os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(payload)
            state = self._picker.load(path)
        except Exception:
            self._picker = None            # fall back to half-blocks for the rest of the run
            self._protocol = "half-blocks (image protocol failed)"
            return None
        self._images[cam["name"]] = (path, state, count)
        return state

    def _tile_body_lines(self, cam: dict[str, Any], cols: int, rows: int) -> list[Line]:
        """The fallback (and message) content for one tile."""
        stream = self._stream
        failure = camera_error(stream.remote_errors, cam["name"]) if stream else ""
        if failure:
            return [Line([Span(f"✗ {failure}", theme.ERR_STYLE)])]
        if stream is None or cam["name"] not in stream.per_camera:
            if stream is not None and stream.error:
                return [Line([Span(f"✗ {stream.error}", theme.ERR_STYLE)])]
            return [Line([Span("waiting…", theme.FAINT_STYLE)])]
        pixels = decode_jpeg(stream.per_camera[cam["name"]])
        if pixels is None:
            self._decode_failed = True
            return [Line([Span("no decoder here (install opencv or pillow)", theme.WARN_STYLE)])]
        target_h, target_w = max(1, rows * 2), max(1, cols)
        src_h, src_w = len(pixels), len(pixels[0]) if pixels else 0
        if not src_h or not src_w:
            return [Line([Span("waiting…", theme.FAINT_STYLE)])]
        picked = [[pixels[min(src_h - 1, y * src_h // target_h)][min(src_w - 1, x * src_w // target_w)]
                   for x in range(target_w)] for y in range(target_h)]
        return [Line([Span(glyph, Style().fg(Color.rgb(*fg)).bg(Color.rgb(*bg)))
                      for glyph, fg, bg in row])
                for row in half_blocks(picked)]

    def _all_failed(self) -> bool:
        stream = self._stream
        return bool(stream and not stream.per_camera and stream.error)

    def draw(self, frame: Any, area: Any) -> None:
        bands = (Layout().direction(Direction.Vertical)
                 .constraints([Constraint.length(1), Constraint.length(1),
                               Constraint.fill(1), Constraint.length(1)])
                 .split(area))
        dirty = " · unsaved rotation" if self._dirty else ""
        draw_slim_header(frame, bands[0], self.ctx, "camera preview",
                         right=[Span(f"{len(self.cameras)} cameras{dirty}",
                                     theme.WARN_STYLE if self._dirty else theme.TEXT_STYLE)])
        frame.render_widget(
            Paragraph.from_string(theme.rule(bands[1].width)).style(theme.RULE_HEAVY_STYLE),
            bands[1])

        if self._all_failed():
            self._fetch_listing()
            frame.render_widget(Paragraph(Text(self._failure_lines())), bands[2])
        else:
            for i, (cam, rect) in enumerate(zip(self.cameras, self._tile_grid(bands[2]))):
                inner = (Layout().direction(Direction.Vertical)
                         .constraints([Constraint.length(1), Constraint.fill(1)]).split(rect))
                frame.render_widget(
                    Paragraph(Text([self._tile_header(cam, selected=i == self.index,
                                                      width=inner[0].width)])), inner[0])
                state = self._image_state(cam)
                if state is not None:
                    frame.render_stateful_image(ImageWidget(), inner[1], state)
                else:
                    cols, rows = fit_grid(inner[1].width, inner[1].height)
                    frame.render_widget(
                        Paragraph(Text(self._tile_body_lines(cam, cols, rows))), inner[1])

        fps = self._stream.fps(now=time.monotonic()) if self._stream else 0.0
        note = self._saved_note or f"{self._protocol} · host must be stopped"
        frame.render_widget(hint_slot_line(
            f"[ ] rotate the selected camera, w writes it to lekiwi.yaml · "
            f"{fps:.1f} fps · {note}",
            bands[3].width,
            keys=(("tab", "select"), ("[ ]", "rotate"), ("w", "save"), ("r", "restart"),
                  ("q", "back"))),
            bands[3])

    def _failure_lines(self) -> list[Line]:
        """Every camera failed: say what the robot said, then what it actually has."""
        stream = self._stream
        out = [Line([Span(f"✗ {stream.error if stream else 'no capture'}", theme.ERR_STYLE)]),
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


__all__ = ["CameraPreviewScreen", "build_all_streams_argv", "build_stream_argv",
           "configured_cameras", "fit_grid", "kitty_capable", "make_picker",
           "rotation_degrees", "write_rotations"]
