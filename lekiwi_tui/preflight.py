"""Advisory preflight checks before launching robot/runtime actions.

These checks catch common operator mistakes without taking control away from the operator:
warnings open a Continue anyway / Cancel modal, and no warning blocks by itself. Safety
guards that must block belong in the owning action (for example Record delete containment).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Clear, Constraint, Direction, Layout, Line, Paragraph, Span, Text

from . import ROOT
from .framework import runner, theme
from .framework.events import DOWN, ENTER, ESC, LEFT, RIGHT, UP, Key
from .framework.screen import Pop, ScreenState

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .context import Context
    from .framework.app import App


@dataclass(frozen=True)
class PreflightIssue:
    """One warning shown before a launch."""

    text: str


def _workspace_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ROOT / p


def _existing_ancestor(path: Path) -> Path:
    cur = path
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def _disk_issue(path: str | Path, label: str, *, min_gb: float) -> PreflightIssue | None:
    target = _workspace_path(path)
    base = _existing_ancestor(target)
    if not base.exists():
        return PreflightIssue(f"{label} path does not exist yet: {target}")
    try:
        free = shutil.disk_usage(base).free / (1024 ** 3)
    except OSError as exc:
        return PreflightIssue(f"could not check free space for {label}: {exc}")
    if free < min_gb:
        return PreflightIssue(f"{label} has only {free:.1f} GB free at {base}; recording/training may fail")
    return None


def _ssh_issue(host: str) -> PreflightIssue | None:
    host = str(host or "").strip()
    if not host:
        return PreflightIssue("Pi host is empty in LEKIWI_HOST")
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2", host, "true"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return PreflightIssue(f"could not run SSH check for {host}: {exc}")
    if proc.returncode != 0:
        reason = (proc.stderr or proc.stdout).strip().splitlines()
        suffix = f" ({reason[0]})" if reason else ""
        return PreflightIssue(f"SSH key check failed for {host}{suffix}")
    return None


def _leader_port_issue(ctx: "Context") -> PreflightIssue | None:
    port = str(ctx.cfg["LEADER_PORT"] or "").strip()
    if not port:
        return PreflightIssue("leader arm serial port is empty")
    if not Path(port).exists():
        return PreflightIssue(f"leader arm serial port is not present: {port}")
    return None


def robot_runtime_issues(ctx: "Context", *, check_leader: bool) -> list[PreflightIssue]:
    """Common robot-runtime checks for teleop/record/eval."""
    issues: list[PreflightIssue] = []
    ssh = _ssh_issue(ctx.cfg["LEKIWI_HOST"])
    if ssh is not None:
        issues.append(ssh)
    if check_leader:
        leader = _leader_port_issue(ctx)
        if leader is not None:
            issues.append(leader)
    return issues


def record_issues(ctx: "Context", *, root: str, parent: str) -> list[PreflightIssue]:
    issues = robot_runtime_issues(ctx, check_leader=True)
    disk = _disk_issue(parent or Path(root).parent or ".", "dataset parent", min_gb=10.0)
    if disk is not None:
        issues.append(disk)
    return issues


def train_issues(ctx: "Context", *, dataset_root: str, policy_root: str) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    ds = _workspace_path(dataset_root)
    if not ds.is_dir():
        issues.append(PreflightIssue(f"training dataset folder is missing: {ds}"))
    disk = _disk_issue(policy_root, "policy output folder", min_gb=10.0)
    if disk is not None:
        issues.append(disk)
    if not ctx.gpu_name:
        issues.append(PreflightIssue("no NVIDIA GPU detected; training on CPU is usually impractical"))
    return issues


def eval_issues(ctx: "Context", *, policy: str) -> list[PreflightIssue]:
    issues = robot_runtime_issues(ctx, check_leader=False)
    if not ctx.gpu_name:
        issues.append(PreflightIssue("no NVIDIA GPU detected; rollout may run slowly on CPU"))
    if policy and not Path(policy).expanduser().exists() and "/" not in policy:
        issues.append(PreflightIssue(f"policy looks like a model id, not a local checkpoint: {policy}"))
    return issues


def _centered_rect(area: Any, width: int, height: int) -> Any:
    w = min(width, max(1, area.width))
    h = min(height, max(1, area.height))
    vbands = (
        Layout()
        .direction(Direction.Vertical)
        .constraints([Constraint.fill(1), Constraint.length(h), Constraint.fill(1)])
        .split(area)
    )
    hbands = (
        Layout()
        .direction(Direction.Horizontal)
        .constraints([Constraint.fill(1), Constraint.length(w), Constraint.fill(1)])
        .split(vbands[1])
    )
    return hbands[1]


class PreflightModalState(ScreenState):
    """Full-screen warning modal returning ``True`` for Continue anyway."""

    def __init__(self, title: str, issues: "Iterable[PreflightIssue]") -> None:
        self.title = title
        self.issues = list(issues)
        self.index = 0

    def handle_key(self, key: Key):
        name = key.name
        if name in (ESC, "q"):
            return Pop(False)
        if name in (UP, DOWN, LEFT, RIGHT, "h", "j", "k", "l"):
            self.index = 1 - self.index
            return None
        if name == ENTER:
            return Pop(self.index == 0)
        return None

    def draw(self, frame: Any, area: Any) -> None:
        frame.render_widget(Paragraph.from_string("").style(theme.BASE_STYLE), area)
        card_w = min(86, max(1, area.width))
        text_w = max(20, card_w - 8)
        issue_lines: list[Line] = []
        max_issue_rows = max(1, int(area.height) - 10)
        truncated = False
        for issue in self.issues:
            wrapped = textwrap.wrap(issue.text, width=max(10, text_w - 2)) or [issue.text]
            for i, part in enumerate(wrapped):
                if len(issue_lines) >= max_issue_rows:
                    truncated = True
                    break
                prefix = "! " if i == 0 else "  "
                issue_lines.append(Line([Span(prefix + part, theme.WARN_STYLE)]))
            if truncated:
                break
        if truncated:
            issue_lines[-1] = Line([Span("... more warnings", theme.MUTED_STYLE)])

        body_h = max(1, len(issue_lines))
        card_h = body_h + 8
        card = _centered_rect(area, card_w, card_h)
        frame.render_widget(Clear(), card)
        block = theme.block("preflight", bordered=True).style(theme.surface_style()).padding(2, 2, 1, 1)
        inner = block.inner(card)
        frame.render_widget(block, card)
        rows = (
            Layout()
            .direction(Direction.Vertical)
            .constraints([
                Constraint.length(1),
                Constraint.length(1),
                Constraint.length(body_h),
                Constraint.fill(1),
                Constraint.length(2),
                Constraint.length(1),
            ])
            .split(inner)
        )

        frame.render_widget(
            Paragraph(Text([Line([Span(self.title, theme.TITLE_STYLE)])])).style(theme.surface_style()),
            rows[0],
        )
        frame.render_widget(
            Paragraph(Text([Line([Span("Review warnings before launch.", theme.MUTED_STYLE)])])).style(
                theme.surface_style()
            ),
            rows[1],
        )
        frame.render_widget(Paragraph(Text(issue_lines)).style(theme.surface_style()), rows[2])

        choices = ["Continue anyway", "Cancel"]
        choice_lines = []
        for i, label in enumerate(choices):
            focused = i == self.index
            choice_lines.append(Line([
                Span(theme.selector(focused), theme.HIGHLIGHT_LABEL_STYLE),
                Span(label, theme.HIGHLIGHT_LABEL_STYLE if focused else theme.TEXT_STYLE),
            ]))
        frame.render_widget(Paragraph(Text(choice_lines)).style(theme.surface_style()), rows[4])
        frame.render_widget(
            Paragraph(Text([Line([
                Span(f" {theme.key_label('↑↓')} ", theme.KEYCAP_STYLE),
                Span(" choose  ", theme.HINT_STYLE),
                Span(f" {theme.key_label('⏎')} ", theme.KEYCAP_STYLE),
                Span(" apply  ", theme.HINT_STYLE),
                Span(" q ", theme.KEYCAP_STYLE),
                Span(" cancel", theme.HINT_STYLE),
            ])])).style(theme.surface_style()),
            rows[5],
        )


async def confirm_preflight(app: "App", title: str, issues: "Iterable[PreflightIssue]") -> bool:
    """Return True if launch should continue."""
    issue_list = list(issues)
    if runner.DRY_RUN or not issue_list:
        return True
    if app.terminal is None:
        for issue in issue_list:
            print(f"! preflight: {issue.text}", file=sys.stderr)
        return True
    return bool(await app.run_modal(PreflightModalState(title, issue_list)))


__all__ = [
    "PreflightIssue",
    "PreflightModalState",
    "confirm_preflight",
    "robot_runtime_issues",
    "record_issues",
    "train_issues",
    "eval_issues",
]
