"""robot_config.py — RobotConfigScreen, the PAINT-screen exemplar (port of the Textual
RobotConfigScreen).

A read-only at-a-glance view of the core ``lekiwi.yaml`` values, grouped into sections
(CONNECTION / ARM & CONTROL / DATASET / CAMERAS / TASK). The lerobot-side config (cameras,
remote_ip, use_degrees, fps, task, dataset) lives in lekiwi.yaml with YAML anchors
(``&cameras``, ``*robot``, ``<<:``) + comments — nested, free-form blocks with no per-field
form — so this screen DISPLAYS the values and opens ``$EDITOR`` to change them; a hand edit
is the right tool. (Settings writes back the flat ``_launcher:`` knobs losslessly; this
screen stays read-only by design — there is no Save.)

Keys follow the bash ``do_robot_config`` case: ``e`` suspends the app and runs
``$EDITOR lekiwi.yaml``, then re-reads from disk and repaints; ``r`` does the same reload
without editing; ``q`` / ``Esc`` pop back. No motion keys — this is a static panel. ``f``
is the one addition: it lists the ROBOT's cameras over ssh, because the device nodes shown
here are Pi-side and renumber themselves (see :func:`build_find_cameras_argv`).

Immediate-mode notes
--------------------
* The values are re-read LIVE from disk (``load_yaml``), never from the launch-time
  ``ctx.cfg`` cache — after an external ``$EDITOR`` edit that cache is stale (bash re-reads
  each loop). CRUCIALLY every row here is ``cfg_get(dotted, doc=...)`` over NESTED paths
  (``_robot.*``, ``host.*``, ``_leader_arm.*``, ``_dataset.*``, ``rollout.task``,
  ``_cameras``) — NONE are flat ``_launcher`` keys — so refreshing ``ctx.cfg`` alone would
  change nothing on screen. We therefore refresh ``ctx.doc`` (what ``draw`` reads) on every
  edit/reload, and also ``ctx.cfg`` so the freshness propagates app-wide. ``draw`` reads
  ``self.ctx.doc`` directly.
* ``e`` / ``r`` return :class:`Invoke` (an async flow) rather than a bare :class:`Suspend`:
  the editor suspend needs a POST-exit hook to re-read the doc, which a fire-and-forget
  ``Suspend`` cannot give. The async ``_edit`` does ``await app.suspend(...)`` then reloads.
* The Textual original's ``call_after_refresh(self._paint)`` mount dance is DROPPED: there is
  no mount-paint race in immediate mode (``draw`` runs every frame from the doc we seed in
  ``__init__``), so the body never starts blank.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text

from .. import CFG_FILE, ROOT
from ..config import Config, cameras_summary, cfg_get, collapse_home, load_yaml, resolve_editor
from ..framework import theme
from ..framework.widgets import wrap_words
from ..framework.events import ESC, Key
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from .chrome import draw_slim_header, hint_slot_line, mode_chip_spans

if TYPE_CHECKING:
    from ..context import Context
    from ..framework.app import App

#: The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

# A thin rule under the header (the Nordic look; sections carry their own hairline).

# Rotation enum → short tag (ported VERBATIM from the Textual original's local _ROT, which
# mirrors config.cameras_summary). Anything unrecognized (incl. a missing rotation) → NO_ROT.
_ROT = {
    "NO_ROTATION": "NO_ROT",
    "ROTATE_90": "ROT90",
    "ROTATE_180": "ROT180",
    "ROTATE_270": "ROT270",
}

# The at-a-glance overview, grouped into sections. Each row is (label, dotted, suffix): the
# value is read live via cfg_get(dotted); suffix is appended (e.g. " ms"). Ported VERBATIM
# from the Textual original — CONNECTION / ARM & CONTROL / DATASET (CAMERAS + TASK rendered
# specially below). All read-only.
_SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("CONNECTION", [
        ("remote ip", "_robot.remote_ip", ""),
        ("robot id", "_robot.id", ""),
        ("loop freq", "host.host.max_loop_freq_hz", " Hz"),
        ("watchdog", "host.host.watchdog_timeout_ms", " ms"),
        ("poll timeout", "_robot.polling_timeout_ms", " ms"),
    ]),
    ("ARM & CONTROL", [
        ("use degrees", "host.robot.use_degrees", ""),
        ("leader port", "_leader_arm.port", ""),
        ("leader id", "_leader_arm.id", ""),
    ]),
    ("DATASET", [
        ("repo id", "_dataset.repo_id", ""),
        ("root", "_dataset.root", ""),
    ]),
]

# Label column width so values align across all sections ("poll timeout" is longest).
_LABEL_W = 15


#: The launcher that owns the remote camera-probe bash — the SOLE argv source, fronted
#: and never re-translated (same seam as host.sh for the host commands).
FIND_CAMERAS_SCRIPT = ROOT / "scripts" / "find_cameras.sh"


def build_find_cameras_argv(ctx: "Context") -> list[str]:
    """`ssh <host> "<emit-detect remote bash>"`, list-only.

    ConnectTimeout keeps a powered-down robot from hanging the suspend; no `-t`, because
    nothing here reads keys — the value is the printed device list, which the operator
    copies into lekiwi.yaml with `e`.
    """
    import subprocess

    from ..remote import validate_remote_name, validate_ssh_host

    host = validate_ssh_host(ctx.cfg["LEKIWI_HOST"])
    conda_env = validate_remote_name(ctx.cfg["CONDA_ENV"], "conda env")
    remote = subprocess.check_output(
        ["bash", str(FIND_CAMERAS_SCRIPT), "emit-detect", "--conda-env", conda_env],
        text=True,
    )
    return ["ssh", "-o", "ConnectTimeout=5", host, remote]


class RobotConfigScreen(ScreenState):
    """Read-only view of lekiwi.yaml core values. ``e`` edits in ``$EDITOR``, ``r`` reloads,
    ``q``/``Esc`` pops back. Values are re-read live from disk on construction and after each
    edit/reload (bash re-reads each loop), never from the launch-time ``ctx.cfg`` cache."""

    title = "robot config"

    def __init__(self, app: "App", ctx: "Context", *, extra: list[str] | None = None) -> None:
        # Store app/ctx; do NOT use app in __init__ (it can be None at root construction).
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        # Seed the doc we render from. The original loaded fresh in on_mount; here we read
        # live once at construction so draw paints from real disk state on frame 1 (no
        # mount-paint race → call_after_refresh is dropped). Re-read on every edit/reload.
        self.ctx.doc = load_yaml()

    # ── interaction (bash do_robot_config case; static panel — e/r/q only) ─────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name == "q" or name == ESC:
            return Pop()
        if name == "e":
            # Suspend into $EDITOR on lekiwi.yaml, then reload — needs a post-exit hook, so
            # an async flow (Invoke), not a fire-and-forget Suspend.
            return Invoke(self._edit)
        if name == "r":
            return Invoke(self._reload)
        if name == "f":
            return Invoke(self._detect_cameras)
        return Nothing

    async def _detect_cameras(self) -> None:
        """``f``: list the ROBOT's cameras, over ssh.

        The `index_or_path` values on this screen are Pi-side device nodes, and a bare
        /dev/videoN is not reboot/replug-stable — adding a camera renumbers the others.
        Probing locally would enumerate the laptop's webcam and answer a question nobody
        asked, so this runs `lerobot-find-cameras` on the robot.

        Refused while the host session is up: that process has every camera open, so the
        probe would report failures for exactly the devices that are working. `None`
        (probe still in flight, or no host configured) is allowed through — a maybe is not
        a reason to block, and the remote output will say what happened.
        """
        from ..hostprobe import host_alive

        if host_alive(self.ctx) is True:
            self.app.notify(
                "host session is running and holds the cameras — stop it first, then f again",
                "warn")
            return
        try:
            argv = build_find_cameras_argv(self.ctx)
        except SystemExit as exc:              # die_usage in the emitter / validators
            self.app.notify(f"cannot probe cameras: {exc}", "error")
            return
        await self.app.suspend(argv)

    async def _edit(self) -> None:
        """``e``: suspend the app and open ``$EDITOR`` on lekiwi.yaml, then reload. The CLI
        owns the real TTY while the editor runs, exactly like bash ``"$ed" "$CFG_FILE"``.
        ``runner`` appends no ``--dry-run`` here: this is an editor on a local file, not a
        lerobot CLI (the original shells ``$EDITOR lekiwi.yaml`` directly too)."""
        await self.app.suspend([resolve_editor(), str(CFG_FILE)])
        self._reload_sync()

    async def _reload(self) -> None:
        """``r``: re-read lekiwi.yaml from disk and repaint (no editor)."""
        self._reload_sync()

    def _reload_sync(self) -> None:
        """Re-read lekiwi.yaml from disk into the context. After an external $EDITOR edit the
        launch-time ctx is stale, so we always read fresh (bash re-reads live each loop). We
        refresh ctx.doc — what draw() actually reads (every visible row is a NESTED cfg_get,
        none are flat _launcher keys) — AND ctx.cfg so the freshness propagates app-wide."""
        self.ctx.doc = load_yaml()
        self.ctx.cfg = Config.load()

    # ── view (rebuilt fresh each frame from ctx.doc) ──────────────────────────
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1)]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "robot config",
                         [Span(f"{collapse_home(str(CFG_FILE))} · read-only",
                               theme.FAINT_STYLE),
                          Span("   ", theme.BASE_STYLE), *mode_chip_spans()])
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(self._body(), rows[2])
        frame.render_widget(hint_slot_line(
            "read-only view — e edits lekiwi.yaml, r reloads after editing",
            rows[3].width,
            keys=(("e", f"edit in {resolve_editor()}"), ("r", "reload"),
                  ("f", "detect cameras on the robot"), ("q", "back"))),
            rows[3])

    # ── body rendering (sections → lines, ported from the original's Text builders) ──
    def _section_line(self, label: str, *, first: bool) -> list[Line]:
        """A dim section label + hairline rule, with a blank line above (except the first).
        Mirrors the original ``_section_text`` (which prefixed a "\\n")."""
        out: list[Line] = []
        if not first:
            out.append(Line([Span("", theme.BASE_STYLE)]))
        rule = theme.rule(max(4, 48 - len(label)))
        out.append(Line([
            Span(label, theme.SECTION_STYLE),
            Span(" ", theme.BASE_STYLE),
            Span(rule, theme.BORDER_STYLE),
        ]))
        return out

    def _kv(self, label: str, value: str, *, dim: bool = False) -> Line:
        """A ``label   value`` row: muted label, bone value (dim/muted for the task line).
        Mirrors the original ``_kv`` (two-space indent + padded label)."""
        val_style = theme.MUTED_STYLE if dim else theme.STATUS_VALUE_STYLE
        return Line([
            Span("  ", theme.BASE_STYLE),
            Span(f"{label:<{_LABEL_W}}", theme.MUTED_STYLE),
            Span(value, val_style),
        ])

    def _camera_lines(self) -> list[Line]:
        """One row per camera: ``front   640×480 @30  NO_ROT``. Empty list if none.
        Ported VERBATIM from the original ``_camera_lines`` (reads the shared ``_cameras``
        anchor; the bash summary's one-liner is expanded to a row each)."""
        cams = cfg_get("_cameras", doc=self.ctx.doc) or {}
        if not isinstance(cams, dict):
            return []
        lines: list[Line] = []
        for name, cam in cams.items():
            cam = cam or {}
            w, h, fps = cam.get("width", "?"), cam.get("height", "?"), cam.get("fps", "?")
            rot = _ROT.get(str(cam.get("rotation")), "NO_ROT")
            lines.append(self._kv(name, f"{w}×{h} @{fps}  {rot}"))
        return lines

    def _wrap(self, text: str, width: int) -> list[str]:
        """Word-wrap the TASK instruction to *width*. Was a char-wrapper, which rendered
        "second" as "se" / "cond"."""
        return wrap_words(text, width)

    def _body(self) -> Paragraph:
        doc = self.ctx.doc
        lines: list[Line] = []
        first = True
        for title, section_rows in _SECTIONS:
            lines.extend(self._section_line(title, first=first))
            first = False
            for label, dotted, suffix in section_rows:
                val = cfg_get(dotted, doc=doc)
                disp = "—" if val is None else f"{val}{suffix}"
                lines.append(self._kv(label, disp))
        # CAMERAS: one row per camera (richer than the bash one-line summary).
        cam_lines = self._camera_lines()
        if cam_lines:
            lines.extend(self._section_line("CAMERAS", first=False))
            lines.extend(cam_lines)
        # TASK: the rollout instruction, dim + wrapped (language-conditioned policies).
        task = cfg_get("rollout.task", doc=doc)
        if task is not None:
            lines.extend(self._section_line("TASK", first=False))
            # Indent matches _kv's two-space gutter; wrap to the indented width.
            for wln in self._wrap(str(task), max(8, 52 - 2)):
                lines.append(Line([
                    Span("  ", theme.BASE_STYLE),
                    Span(wln, theme.MUTED_STYLE),
                ]))
        return Paragraph(Text(lines)).style(theme.BASE_STYLE)


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY dump of the core lekiwi.yaml values this screen shows, as plain text.

    The interactive screen paints the ``_SECTIONS`` rows (CONNECTION / ARM & CONTROL /
    DATASET), a CAMERAS block, and the TASK instruction; this prints the SAME data as flat
    ``label  value`` rows (reusing the verbatim ``_SECTIONS`` labels + suffixes and the
    ``_ROT`` tags). Read-only; never edits.

    Signature is ``(ctx, extra)`` per the port convention (mirrors sync.py / provision.py):
    config comes from ``ctx.doc`` — for a one-shot no-TTY run that doc was just loaded by
    ``load_context``, so there is no staleness reason to re-read from disk (the screen
    re-reads only to pick up an in-app ``$EDITOR`` edit, which can't happen here). ``extra``
    is accepted for hook parity but unused (this action takes no args). Returns 0.

    NOTE: this DIVERGES from the Textual original's ``run_headless`` (which mirrors the bash
    no-TTY branch: only a path line + remote_ip / robot.id / use_degrees / fps / cameras,
    and deliberately NO task). The gap spec asks for the screen's full ``_SECTIONS`` dump
    incl. the TASK row, so we dump everything the screen shows. ``__main__`` calls this hook
    for direct no-TTY robot-config dispatch.
    """
    doc = ctx.doc
    print(f"lekiwi.yaml: {CFG_FILE}")
    # The grouped _SECTIONS rows (same labels / suffixes / "—"-for-None as _body/_kv).
    for title, section_rows in _SECTIONS:
        print(title)
        for label, dotted, suffix in section_rows:
            val = cfg_get(dotted, doc=doc)
            disp = "—" if val is None else f"{val}{suffix}"
            print(f"  {label:<{_LABEL_W}}{disp}")
    # CAMERAS: per-camera rows (same w×h @fps + _ROT tag as the screen's _camera_lines),
    # plus the one-line cameras_summary the bash panel/headless branch printed.
    cams = cfg_get("_cameras", doc=doc)
    if isinstance(cams, dict) and cams:
        print("CAMERAS")
        for name, cam in cams.items():
            cam = cam or {}
            w, h, fps = cam.get("width", "?"), cam.get("height", "?"), cam.get("fps", "?")
            rot = _ROT.get(str(cam.get("rotation")), "NO_ROT")
            print(f"  {name:<{_LABEL_W}}{w}×{h} @{fps}  {rot}")
        print(f"  {'summary':<{_LABEL_W}}{cameras_summary(doc)}")
    # TASK: the rollout instruction the screen renders (language-conditioned policies).
    task = cfg_get("rollout.task", doc=doc)
    if task is not None:
        print("TASK")
        print(f"  {task}")
    return 0


__all__ = ["RobotConfigScreen", "run_headless", "HEADLESS_HOOK"]
