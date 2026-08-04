"""camstream.py — a low-rate camera preview from the ROBOT, as terminal half-blocks.

Why this exists: `f` in Robot config lists the Pi's camera device nodes, but a list cannot
tell you WHICH LENS is which. That is the actual question after a replug renumbers the
devices, and one look answers it.

Why it is not a tap on the running host: the host publishes observations on a **PUSH**
socket (lekiwi_host.py), and PUSH round-robins across connected peers — a preview would
steal every other frame from the teleop/record client. So the frames come from a capture
the Pi runs only while the host is stopped, which is the same state `f` already requires.

Three pieces, split so the only untestable one is tiny:

* :func:`frames` parses the wire protocol (``FRAME <n>\\n`` + n raw JPEG bytes) off a
  binary stream. Pure, no decoding.
* :func:`decode_jpeg` turns those bytes into rows of RGB tuples. The one seam that needs
  a real image library; imported lazily and reported, never fatal.
* :func:`half_blocks` turns rows of RGB into ``(char, fg, bg)`` cells. Pure and exact, so
  the geometry is unit-tested rather than eyeballed.

Bandwidth is deliberately small. 320x240 at JPEG q50 is ~12 KB a frame, so 5 fps is
~60 KB/s: two orders below the USB+WiFi load that used to freeze the Pi, and the reason
the capture downscales ON the Pi rather than shipping full frames.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import IO, Any

#: Wire framing. A length-prefixed header line keeps the JPEG bytes raw (base64 would add
#: a third to every frame) and needs no PTY, which would translate those bytes.
HEADER = b"FRAME "

#: One cell shows two vertically-stacked pixels: the upper half is painted with the glyph's
#: foreground, the lower half with its background.
UPPER_HALF = "▀"


def frames(stream: IO[bytes], *, max_bytes: int = 4 << 20) -> Iterator[bytes]:
    """Yield JPEG payloads from *stream* until EOF.

    Tolerates junk before a header (the remote shell can print warnings on the same fd)
    by resynchronising on the next ``FRAME`` line, and stops rather than trusting a
    preposterous length, so a corrupted stream cannot make us allocate the machine away.
    """
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.startswith(HEADER):
            continue                      # remote noise; wait for the next header
        try:
            size = int(line[len(HEADER):].strip())
        except ValueError:
            continue
        if size <= 0 or size > max_bytes:
            return
        payload = stream.read(size)
        if not payload or len(payload) < size:
            return                        # truncated tail: the capture went away
        yield payload


def decode_jpeg(payload: bytes) -> list[list[tuple[int, int, int]]] | None:
    """JPEG bytes -> rows of (r, g, b), or None when no decoder is installed.

    cv2 first (it is already in any env that has lerobot), then Pillow. None is not an
    error worth crashing a config screen over: the caller says so on screen instead.
    """
    try:
        import numpy as np

        try:
            import cv2

            arr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            return [[(int(r), int(g), int(b)) for b, g, r in row] for row in arr]
        except ImportError:
            pass

        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(payload)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            px = list(rgb.getdata())
            return [px[y * w:(y + 1) * w] for y in range(h)]
    except Exception:
        return None


def half_blocks(rows: list[list[tuple[int, int, int]]]) -> list[list[tuple[str, tuple[int, int, int], tuple[int, int, int]]]]:
    """Rows of RGB -> rows of ``(glyph, fg, bg)``, two source pixels per cell.

    A terminal cell is about twice as tall as it is wide, so pairing rows this way keeps
    the aspect ratio roughly honest. An odd final row pairs with itself rather than
    inventing black.
    """
    out = []
    for y in range(0, len(rows), 2):
        upper = rows[y]
        lower = rows[y + 1] if y + 1 < len(rows) else rows[y]
        out.append([(UPPER_HALF, upper[x], lower[x] if x < len(lower) else upper[x])
                    for x in range(len(upper))])
    return out


@dataclass
class CameraStream:
    """A running preview: a reader thread keeps only the LATEST frame.

    Same contract as :mod:`~lekiwi_tui.hostprobe` and :mod:`~lekiwi_tui.sysstat`: ``draw``
    runs every frame, so it may only ever look at :attr:`latest`. Frames are dropped rather
    than queued, because a preview that lags is worse than one that skips.
    """

    spawn: Callable[[], Any]              # () -> subprocess.Popen, injected for tests
    latest: bytes | None = None
    frame_count: int = 0
    started_at: float = 0.0
    error: str = ""
    remote_error: str = ""                # first line the remote wrote to stderr
    _proc: Any = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, *, now: float) -> None:
        self.started_at = now
        try:
            self._proc = self.spawn()
        except OSError as exc:
            self.error = f"could not start the remote capture: {exc}"
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        if getattr(self._proc, "stderr", None) is not None:
            threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stderr(self) -> None:
        """Keep the remote's first complaint. 'could not open /dev/video4' is the whole
        answer when a node has renumbered, and it is worth more than a blank screen."""
        try:
            for raw in self._proc.stderr:
                line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if line and not self.remote_error:
                    self.remote_error = line
                    break
        except Exception:
            pass

    def _read(self) -> None:
        stdout = getattr(self._proc, "stdout", None)
        if stdout is None:
            return
        for payload in frames(stdout):
            if self._stop.is_set():
                break
            with self._lock:
                self.latest = payload
                self.frame_count += 1
        if not self._stop.is_set() and self.frame_count == 0:
            # give the stderr reader a moment: its message is more specific than ours
            self._stop.wait(0.4)
            self.error = self.error or self.remote_error or "no frames arrived — is the host stopped?"

    def fps(self, *, now: float) -> float:
        elapsed = now - self.started_at
        return (self.frame_count / elapsed) if elapsed > 0.5 else 0.0

    def stop(self) -> None:
        """Kill the remote capture. Called on leaving the screen, and idempotent: a
        preview that keeps a camera open after you walk away is a bug on the robot."""
        self._stop.set()
        proc = self._proc
        if proc is None:
            return
        for finish in (proc.terminate, proc.kill):
            try:
                finish()
                proc.wait(timeout=2)
                break
            except Exception:
                continue
        self._proc = None


__all__ = ["CameraStream", "UPPER_HALF", "decode_jpeg", "frames", "half_blocks"]
