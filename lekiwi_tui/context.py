"""context.py — the lekiwi runtime context (cfg / doc / gpu), shared by screens.

The framework :class:`~lekiwi_tui.framework.app.App` is deliberately lekiwi-agnostic
(it knows nothing about config or the GPU). This module holds the lekiwi-specific state the
original ``LekiwiApp.__init__`` carried — the loaded :class:`Config`, the parsed
``lekiwi.yaml`` doc, and the detected GPU name — built ONCE in ``__main__`` and passed to
each screen + the dispatcher. Screens read ``ctx.cfg[...]`` / ``ctx.doc`` / ``ctx.gpu_name``.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

from .config import Config, load_yaml


def gpu_name() -> str:
    """Best-effort GPU name (Textual app's ``gpu_name``). "" if no NVIDIA GPU / nvidia-smi.
    The status line shows it; eval uses it for device auto-detect."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return ""
    name = out[0].strip() if out else ""
    for prefix in ("NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


@dataclass
class Context:
    """Lekiwi runtime state shared across screens (the LekiwiApp cfg/doc/gpu trio)."""

    cfg: Config
    doc: dict[str, Any]
    gpu_name: str
    is_tty: bool = True
    ui_state: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_context(*, gpu: str | None = None, is_tty: bool = True) -> Context:
    """Build the context: load the config + yaml doc + detect the GPU. ``gpu`` is
    injectable for deterministic tests/headless runs (default = real detection)."""
    return Context(
        cfg=Config.load(),
        doc=load_yaml(),
        gpu_name=gpu if gpu is not None else gpu_name(),
        is_tty=is_tty,
    )


__all__ = ["Context", "load_context", "gpu_name"]
