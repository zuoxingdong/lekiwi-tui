"""lekiwi_tui — TUI control center for the LeKiwi robot.

A standalone immediate-mode control center built on pyratatui's draw-loop model. The
framework-free substrate (config / datasets / policies / kbd_listener / scripts), the app
shell, screens, and widgets all live in this package directory.

Layout mirrors the bash SCRIPT_DIR convention: the package lives under a workspace root,
and the app chdir()s to that root on startup so relative dataset paths resolve there.
Set LEKIWI_ROOT or LEKIWI_TUI_ROOT to override the auto-detected workspace.
"""
from __future__ import annotations

import os
from pathlib import Path

# PKG_DIR = the lekiwi_tui/ package dir. ROOT = the workspace dir the app cd's into.
# Editable installs auto-detect the checkout root as PKG_DIR.parent; users can override it
# from any shell with LEKIWI_ROOT=/path/to/lekiwi-tui.
PKG_DIR: Path = Path(__file__).resolve().parent


def _resolve_root() -> Path:
    for key in ("LEKIWI_ROOT", "LEKIWI_TUI_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return PKG_DIR.parent


ROOT: Path = _resolve_root()

CFG_FILE: Path = ROOT / "lekiwi.yaml"      # single source of truth (all commands +
                                           # the _launcher: ops knobs, env > yaml > default)
CFG_CACHE: Path = ROOT / ".lekiwi-cache"   # per-command blocks sliced from lekiwi.yaml

__all__ = ["PKG_DIR", "ROOT", "CFG_FILE", "CFG_CACHE"]
