"""workspace.py — laptop-checkout introspection (which code would ship to the robot).

`local_checkout` resolves a LOCAL_REPO / LOCAL_PLUGIN config value to a directory;
`checkout_provenance` answers "what exactly would land on the Pi?" (version · branch
@ sha). Used by the provision + sync screens; a plain domain module so screens never
import from each other.
"""
from __future__ import annotations

from pathlib import Path

from . import ROOT
from .config import resolve_workspace_path


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
