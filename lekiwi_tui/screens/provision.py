"""provision.py — ProvisionScreen ("Set up Pi"), the immediate-mode port of the Textual
ProvisionScreen.

A single SETUP entry that fronts ``scripts/pi_provision.sh``: a stage picker (system /
conda / lerobot, all ON by default) + a Run row. Run SUSPENDS into the script so it owns
the real terminal — that is how the ``system`` stage's ``sudo`` password prompt (over
``ssh -t``) works; the TUI never sees the password. ``pause=True`` holds the screen after
the script exits so the (long, debuggable) output stays visible, then control returns here
and a one-line result (``✓ … finished`` / ``setup exited``) is shown.

This is the PAINT + SUSPEND archetype: the `_fpos` paint over a FIELDS list
(``[*_STAGE_IDS, "run"]``) is the screen navigation idiom; the stage rows are
``rich.Text``→``Line``/``Span``. Launch needs the child's exit code to build ``_msg``, so a
``Suspend`` Action (which discards the rc) won't do — ENTER on Run returns
``Invoke(self._run)`` and ``_run`` is an ``async def`` that does
``await self.app.suspend(provision_argv, env=…, pause=True)`` and reads the rc back (the
host.py pattern). ``runner.suspend_run`` (which ``app.suspend`` delegates to) makes the argv
demo-safe (R8): it appends ``--dry-run`` to a ``scripts/*.sh`` wrapper when ``DRY_RUN`` is on.

The argv/env builders (:func:`build_provision_argv` / :func:`provision_env`), the STAGES
list, the ``_fpos`` paint, and :func:`run_headless` are ported VERBATIM from the Textual
original (only the framework glue around them is new). ``handle_key`` is pure → unit-testable
with synthetic ``Key`` (no ``Terminal``).

Stage notes (pi_provision.sh):
  system   apt deps + dialout/video groups        [sudo, one-time]
  conda    Miniforge + the python env + uv        [no sudo]
  lerobot  delegate to sync.sh --install + smoke  [no sudo; first install / env rebuild]

PI_HOST / PI_ENV / PI_REPO are passed from the TUI config (LEKIWI_HOST / CONDA_ENV /
PI_REPO) so both sides stay consistent.
"""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from pyratatui import Constraint, Direction, Layout, Line, Paragraph, Span, Text


from .. import ROOT
from ..config import resolve_workspace_path
from ..workspace import checkout_provenance, local_checkout, pyproject_version
from ..framework import runner, theme
from ..framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..remote import RemoteValueError, validate_remote_name, validate_ssh_host
from .chrome import draw_slim_header, hint_slot_line, plan_row, section_line, seg

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context import Context
    from ..framework.app import App

HEADLESS_HOOK = "run_headless"

PROVISION_SCRIPT = ROOT / "scripts" / "pi_provision.sh"


# (stage id, one-line what, note). Order = pi_provision.sh STAGES (system→conda→lerobot).
# Ported VERBATIM from the Textual original.
STAGES: list[tuple[str, str, str]] = [
    ("system", "system packages + device permissions", "requires sudo; usually one-time"),
    ("conda", "Miniforge + Python environment + uv", ""),
    ("lerobot", "sync.sh --install + smoke-test", "first install / after an env rebuild; day-to-day changes need only Sync"),
]
_STAGE_IDS = [s[0] for s in STAGES]


def build_provision_argv(
    stages: "Sequence[str]", *, script=PROVISION_SCRIPT
) -> list[str]:
    """`bash pi_provision.sh <stages…>`. Run via bash (not the exec bit) so it works
    regardless of the file's mode. No stages == the script's default (all, in order).

    Ported VERBATIM from the Textual original.
    """
    return ["bash", str(script), *stages]


def provision_env(cfg, *, py_ver="", recreate=False) -> dict[str, str]:  # noqa: ANN001
    """pi_provision.sh reads its knobs from the environment; drive them from the TUI
    config so the script targets the same Pi + env the rest of the TUI uses. LOCAL_REPO
    / LOCAL_PLUGIN (which laptop checkouts get shipped) pass through resolved to
    absolute paths when configured — empty keeps the script's sibling defaults.
    py_ver / recreate come from the screen's form fields (empty/False = script default).
    """
    env = {
        **os.environ,
        "PI_HOST": validate_ssh_host(cfg["LEKIWI_HOST"]),
        "PI_ENV": validate_remote_name(cfg["CONDA_ENV"], "conda env"),
        "PI_REPO": cfg["PI_REPO"],
    }
    local_repo = resolve_workspace_path(str(cfg["LOCAL_REPO"]))
    if local_repo:
        env["LOCAL_REPO"] = local_repo
    local_plugin = resolve_workspace_path(str(cfg["LOCAL_PLUGIN"]))
    if local_plugin:
        env["LOCAL_PLUGIN"] = local_plugin
    if py_ver:
        env["PY_VER"] = validate_remote_name(str(py_ver), "python version")
    if recreate:
        env["RECREATE_ENV"] = "1"
    return env


def shipping_summary(cfg) -> str:  # noqa: ANN001
    """One muted line saying exactly WHAT a run would ship: lerobot version + branch +
    commit from the configured LOCAL_REPO (sibling default when empty) plus the plugin
    version. Wrong checkout on the robot is the failure this catches by eye."""
    repo = local_checkout(cfg, "LOCAL_REPO", "lerobot")
    plugin = local_checkout(cfg, "LOCAL_PLUGIN", "lerobot_robot_lekiwi_pincopen")
    return (
        f"ships lerobot {checkout_provenance(repo)} "
        f"+ plugin {pyproject_version(plugin)}"
    )


class ProvisionScreen(ScreenState):
    """Stage picker → Run. Toggles default ON (the one-time full bring-up); uncheck to
    re-run a subset. Run suspends into pi_provision.sh (sudo prompt works in the real
    terminal), pauses on exit so the output is readable, then returns here.

    FIELDS = the stage ids then ``"run"``; ``_fpos`` indexes into it (the carried-over
    navigation idiom). ↑↓/jk move, ←→/hl toggle the focused stage (a stage only — sign
    ignored, like the original ``action_adjust``), ⏎ toggles a stage / runs on the Run
    row, q/Esc backs out.
    """

    title = "set up Pi"

    FIELDS = [*_STAGE_IDS, "python", "recreate", "run"]

    #: Pi env python versions the conda stage may build (lerobot 0.6 needs >=3.12).
    PY_CHOICES = ["3.12", "3.13"]

    def __init__(
        self, app: "App", ctx: "Context", *, extra: list[str] | None = None
    ) -> None:
        self.app = app
        self.ctx = ctx
        self._extra = list(extra or [])
        # All stages selected by default (the one-time full bring-up). Mirrors the
        # Textual on_mount: {s: True for s in _STAGE_IDS}.
        self._on: dict[str, bool] = {s: True for s in _STAGE_IDS}
        self._py_idx = 0          # index into PY_CHOICES (3.12 = the script default)
        self._recreate = False    # RECREATE_ENV=1: rebuild the env on a python mismatch
        self._fpos = 0
        self._msg = ""
        # Computed once per screen entry (2 git calls); says what Run would ship.
        try:
            self._shipping = shipping_summary(ctx.cfg)
        except Exception:  # provenance is best-effort; never block the screen
            self._shipping = ""

    # ── navigation (pure → returns an Action) ──────────────────────────────────
    def handle_key(self, key: "Key") -> Any:
        name = key.name
        if name in (ESC, "q"):
            return Pop()
        if name in (UP, "k"):
            self._move(-1)
            return Nothing
        if name in (DOWN, "j"):
            self._move(1)
            return Nothing
        if name in (LEFT, "h", RIGHT, "l"):
            # Stages/recreate toggle on both directions (the original action_adjust
            # ignored the sign); the python enum cycles WITH the sign.
            self._adjust(1 if name in (RIGHT, "l") else -1)
            return Nothing
        if name == ENTER:
            return self._activate()
        return Nothing

    def _move(self, delta: int) -> None:
        self._fpos = (self._fpos + delta) % len(self.FIELDS)
        self._msg = ""

    def _adjust(self, delta: int = 1) -> None:
        cur = self.FIELDS[self._fpos]
        if cur in self._on:
            self._on[cur] = not self._on[cur]
            self._msg = ""
        elif cur == "python":
            self._py_idx = (self._py_idx + delta) % len(self.PY_CHOICES)
            self._msg = ""
        elif cur == "recreate":
            self._recreate = not self._recreate
            self._msg = ""

    def _activate(self) -> Any:
        """⏎: toggle a focused stage, or (on the Run row) kick off the suspend flow.

        On a stage row this is identical to ←→ (toggle). On the Run row it returns an
        :class:`Invoke` so the App awaits :meth:`_run` on its loop — needed because the
        flow must read the child's exit code back to build :attr:`_msg` (a ``Suspend``
        Action would discard it)."""
        cur = self.FIELDS[self._fpos]
        if cur in self._on or cur in ("python", "recreate"):
            self._adjust()
            return Nothing
        return Invoke(self._run)

    @property
    def chosen(self) -> list[str]:
        """The selected stage ids, in STAGES order (used by the Run label + tests)."""
        return [s for s in _STAGE_IDS if self._on[s]]

    # ── the suspend flow (async; awaited by the App via Invoke) ────────────────
    async def _run(self) -> None:
        """Suspend into pi_provision.sh for the chosen stages, then surface the rc.

        Suspending hands the child the real TTY so the ``system`` stage's ``sudo`` prompt
        works; ``pause=True`` keeps the output up after it exits. ``app.suspend`` →
        ``runner.suspend_run`` makes the argv demo-safe (R8: ``--dry-run`` appended to the
        ``scripts/*.sh`` wrapper while ``DRY_RUN`` is on). Empty selection / a missing
        script short-circuit with a message (no launch)."""
        stages = self.chosen
        if not stages:
            self._msg = "✗ select at least one stage"
            return
        if not PROVISION_SCRIPT.exists():
            self._msg = f"✗ {PROVISION_SCRIPT} not found"
            return
        try:
            env = provision_env(
                self.ctx.cfg,
                py_ver=self.PY_CHOICES[self._py_idx],
                recreate=self._recreate,
            )
        except RemoteValueError as exc:
            self._msg = f"✗ invalid remote setting: {exc} — fix it in Settings"
            return
        rc = await self.app.suspend(build_provision_argv(stages), env=env, pause=True)
        self._msg = (
            f"✓ {' · '.join(stages)} finished"
            if rc == 0
            else f"setup exited with code {rc}; see the output above"
        )

    # ── view (rebuilt fresh each frame) ────────────────────────────────────────
    _LABEL_W = 12

    def _lab(self, text: str, focused: bool) -> Span:
        return Span(f"{text:<{self._LABEL_W}}",
                    theme.TITLE_STYLE if focused else theme.MUTED_STYLE)

    def _gutter(self, *fields: str) -> Span:
        on = self.FIELDS[self._fpos] in fields
        return Span(theme.selector(on), theme.TITLE_STYLE if on else theme.BASE_STYLE)

    def _focused_hint(self) -> str:
        cur = self.FIELDS[self._fpos]
        for sid, what, note in STAGES:
            if cur == sid:
                return f"{what}{' · ' + note if note else ''} · ←→/⏎ toggle"
        if cur == "python":
            return "Pi env python (conda stage) · ←→ cycle"
        if cur == "recreate":
            return "rebuild the env if its python does not match · ←→/⏎ toggle"
        return "steps run in order over SSH; rerun any subset safely (idempotent)"

    def _body_lines(self, width: int = 100) -> list[Line]:
        cur = self.FIELDS[self._fpos]
        host = self.ctx.cfg["LEKIWI_HOST"]
        lines: list[Line] = [section_line("STAGES")]
        # The three stages as one multi-select pill row (filled = will run). The
        # focused stage's pill gets the label treatment via the gutter + hint slot.
        stage_spans: list[Span] = [self._gutter(*_STAGE_IDS),
                                   self._lab("stages", cur in _STAGE_IDS)]
        for sid in _STAGE_IDS:
            stage_spans.append(seg(f"{sid}{'*' if cur == sid else ''}", self._on[sid]))
            stage_spans.append(Span(" ", theme.BASE_STYLE))
        stage_spans.append(Span("   all three = first-time bring-up", theme.FAINT_STYLE))
        lines.append(Line(stage_spans))
        lines.append(Line([]))
        lines.append(section_line("PI ENVIRONMENT"))
        lines.append(Line([
            self._gutter("python", "recreate"),
            self._lab("python", cur == "python"),
            Span(f"{self.PY_CHOICES[self._py_idx]:<8}",
                 theme.HIGHLIGHT_TEXT_STYLE if cur == "python" else theme.TEXT_STYLE),
            Span("  ", theme.BASE_STYLE),
            self._lab("recreate", cur == "recreate"),
            seg("yes" if self._recreate else "no", self._recreate),
            Span("   conda env " + str(self.ctx.cfg["CONDA_ENV"]), theme.FAINT_STYLE),
        ]))
        lines.append(Line([]))
        focused = cur == "run"
        chosen = self.chosen
        if chosen:
            plan = (f"{' + '.join(chosen)} on {host} · idempotent · "
                    "system may ask the Pi password")
        else:
            plan = "select at least one stage"
        lines.append(plan_row("Run setup", plan, focused=focused))
        if self._shipping:
            lines.append(Line([]))
            lines.append(Line([Span(f"  {self._shipping}", theme.FAINT_STYLE)]))
        return lines

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (Layout().direction(Direction.Vertical).constraints(
            [Constraint.length(1), Constraint.length(1), Constraint.fill(1),
             Constraint.length(1), Constraint.length(1)]).split(area))
        draw_slim_header(frame, rows[0], self.ctx, "set up Pi")
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1])
        frame.render_widget(Paragraph(Text(self._body_lines(rows[2].width))
                                      ).style(theme.BASE_STYLE), rows[2])
        if self._msg:
            frame.render_widget(
                Paragraph(Text([Line([Span(f"  {self._msg}", theme.MUTED_STYLE)])]))
                .style(theme.BASE_STYLE), rows[3])
        frame.render_widget(hint_slot_line(self._focused_hint(), rows[4].width,
                                           keys=(("↑↓/jk", "move"), ("←→", "toggle"),
                                                 ("⏎", "toggle·run"), ("q", "back"))), rows[4])


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY ``python -m lekiwi_tui setup-pi [stages…]``: run pi_provision.sh
    directly (no app loop). Stages come from extra; none = the script default (all). The
    system stage's sudo needs a TTY, so the no-TTY path is for the conda/lerobot stages.

    Ported from the Textual original, with config threaded through ``ctx``.
    """
    try:
        env = provision_env(ctx.cfg)
    except RemoteValueError as exc:
        print(f"Invalid remote setting: {exc}", file=sys.stderr)
        return 2
    return runner.headless_run(build_provision_argv(extra), env=dict(env))


__all__ = [
    "ProvisionScreen",
    "build_provision_argv",
    "provision_env",
    "shipping_summary",
    "run_headless",
    "STAGES",
    "PROVISION_SCRIPT",
    "HEADLESS_HOOK",
]
