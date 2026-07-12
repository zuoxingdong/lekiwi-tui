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

from pathlib import Path

from .. import ROOT
from ..config import resolve_workspace_path
from ..framework import runner, theme
from ..framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from ..framework.screen import Invoke, Nothing, Pop, ScreenState
from ..remote import RemoteValueError, validate_remote_name, validate_ssh_host
from .chrome import LABEL_W_FIELD, keycap_hint_line, option_line, runtime_chips

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..context import Context
    from ..framework.app import App

HEADLESS_HOOK = "run_headless"

PROVISION_SCRIPT = ROOT / "scripts" / "pi_provision.sh"

RULE = "─" * 54

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


def local_checkout(cfg, key: str, fallback: str) -> "Path":  # noqa: ANN001
    """The laptop dir a LOCAL_REPO / LOCAL_PLUGIN config value points at (resolved;
    empty = the sibling-of-this-checkout default the scripts also use)."""
    return Path(resolve_workspace_path(str(cfg[key])) or str(ROOT.parent / fallback))


def pyproject_version(project_dir: "Path") -> str:
    """The `version = "…"` of a checkout's pyproject.toml, or '?' when unreadable."""
    try:
        for line in (project_dir / "pyproject.toml").read_text().splitlines():
            if line.startswith("version"):
                return line.split('"')[1]
    except (OSError, IndexError):
        pass
    return "?"


def checkout_provenance(project_dir: "Path") -> str:
    """`<version> · <branch> @ <sha>` for a checkout — the at-a-glance answer to "what
    exactly would land on the robot?". '✗ not found' when the dir is missing; the git
    fields degrade to '?' for non-repos (a plain export still shows its version)."""
    import subprocess

    if not project_dir.is_dir():
        return "✗ not found"

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(project_dir), *args],
                capture_output=True, text=True, check=False, timeout=5,
            )
            return out.stdout.strip() or "?"
        except (OSError, subprocess.TimeoutExpired):
            return "?"

    return (
        f"{pyproject_version(project_dir)} · "
        f"{_git('rev-parse', '--abbrev-ref', 'HEAD')} @ {_git('rev-parse', '--short', 'HEAD')}"
    )


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
    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),                 # 0 header
                Constraint.length(1),                 # 1 heavy rule
                Constraint.length(1),                 # 2 runtime chips
                Constraint.length(2),                 # 3 info (2 lines)
                Constraint.length(1),                 # 4 gap
                Constraint.length(len(_STAGE_IDS)),   # 5 stage rows
                Constraint.length(1),                 # 6 gap
                Constraint.length(2),                 # 7 python + recreate rows
                Constraint.length(1),                 # 8 gap
                Constraint.length(1),                 # 9 shipping provenance
                Constraint.length(1),                 # 10 run row
                Constraint.length(1),                 # 11 msg
                Constraint.fill(1),                    # 12 spacer
                Constraint.length(1),                 # 13 hint
            ])
            .split(area)
        )

        frame.render_widget(
            Paragraph(Text([Line([
                Span(f"{theme.title_mark()} LEKIWI", theme.TITLE_STYLE),
                Span("  set up Pi", theme.SUBTITLE_STYLE),
            ])])).style(theme.BASE_STYLE),
            rows[0],
        )
        frame.render_widget(
            Paragraph.from_string(theme.rule(rows[1].width)).style(theme.RULE_HEAVY_STYLE), rows[1]
        )
        frame.render_widget(runtime_chips(self.ctx), rows[2])
        frame.render_widget(self._info(), rows[3])
        self._stage_rows(frame, rows[5])
        self._option_rows(frame, rows[7])
        if self._shipping:
            frame.render_widget(
                Paragraph(Text([Line([Span(self._shipping, theme.MUTED_STYLE)])]))
                .style(theme.BASE_STYLE),
                rows[9],
            )
        self._run_row(frame, rows[10])
        if self._msg:
            frame.render_widget(
                Paragraph(Text([Line([Span(self._msg, theme.MUTED_STYLE)])]))
                .style(theme.BASE_STYLE),
                rows[11],
            )
        self._hint(frame, rows[13])

    def _info(self) -> Paragraph:
        host = self.ctx.cfg["LEKIWI_HOST"]
        return Paragraph(Text([
            Line([Span(
                f"prepare {host} for LeKiwi control (laptop → Pi over SSH)",
                theme.MUTED_STYLE,
            )]),
            Line([Span(
                "⚠ system setup may ask for the Pi password in the terminal",
                theme.STATUS_VALUE_STYLE,
            )]),
        ])).style(theme.BASE_STYLE)

    def _stage_rows(self, frame: Any, area: Any) -> None:
        """Render one row per stage into *area* (a multi-line Paragraph). The focused row
        gets the accent ``▌`` left-bar + bold accent label/value (the menu highlight idiom);
        the value is ``‹ on ›`` / ``‹ off ›`` and the stage's one-line ``what`` trails."""
        cur = self.FIELDS[self._fpos]
        lines: list[Line] = []
        for sid, what, _note in STAGES:
            val = theme.choice("on") if self._on[sid] else theme.choice("off")
            focused = cur == sid
            lines.append(option_line(
                sid,
                val,
                what,
                focused=focused,
                label_width=LABEL_W_FIELD,
                width=area.width,
            ))
        frame.render_widget(
            Paragraph(Text(lines)).style(theme.BASE_STYLE), area
        )

    def _option_rows(self, frame: Any, area: Any) -> None:
        """The two conda-stage knobs, rendered like the stage rows: python version
        (‹ 3.12 ›, cycles) and recreate-env (‹ yes/no ›, rebuilds on python mismatch)."""
        cur = self.FIELDS[self._fpos]
        lines = [
            option_line(
                "python",
                theme.choice(self.PY_CHOICES[self._py_idx]),
                "Pi env python (conda stage)",
                focused=cur == "python",
                label_width=LABEL_W_FIELD,
                width=area.width,
            ),
            option_line(
                "recreate",
                theme.choice("yes" if self._recreate else "no"),
                "rebuild the env if its python does not match",
                focused=cur == "recreate",
                label_width=LABEL_W_FIELD,
                width=area.width,
            ),
        ]
        frame.render_widget(Paragraph(Text(lines)).style(theme.BASE_STYLE), area)

    def _run_row(self, frame: Any, area: Any) -> None:
        """The Run row: ``▶ Run  (chosen · stages)`` (or ``select a stage`` when empty),
        bold accent when focused."""
        focused = self.FIELDS[self._fpos] == "run"
        chosen = self.chosen
        label = f"{theme.play_mark()} Run setup  ({' · '.join(chosen) if chosen else 'select a stage'})"
        frame.render_widget(
            Paragraph(Text([option_line(
                label,
                "prepare the Pi over SSH",
                focused=focused,
                label_width=42,
                width=area.width,
                label_unfocused_style=theme.TEXT_STYLE,
            )])).style(theme.BASE_STYLE),
            area,
        )

    def _hint(self, frame: Any, area: Any) -> None:
        frame.render_widget(
            keycap_hint_line([
                ("↑↓/jk", "move"),
                ("←→/hl", "toggle"),
                ("⏎", "toggle/run"),
                ("q", "back"),
            ]),
            area,
        )


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
