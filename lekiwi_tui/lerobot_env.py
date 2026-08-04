"""lerobot_env.py — which lerobot am I about to drive, and is it new enough?

The TUI never imports lerobot (it launches CLIs), so nothing in the app ever noticed
which lerobot is installed. That is a gap worth closing on the status card, because a
too-old one does not fail at startup: it fails deep inside draccus, mid-launch, after
the robot is already involved.

Two facts, and they are NOT the same question:

* the version, from package metadata (no import: ``lerobot`` costs seconds);
* a capability marker, because the version can lie. A checkout of a released tag
  reports the RELEASE version while still missing fields that release shipped —
  measured on a 0.6.1 tree that has no ``no_stamp``. So ``status()`` reports "looks
  like a pre-release checkout" when the version passes the floor but the marker for
  that floor is absent, which is exactly the case where a flag we emit would be
  rejected by draccus.

Everything is answered for the interpreter running the TUI, which is the same env the
launcher scripts use (both resolve ``python`` from PATH). Facts are cached for the
process: an env cannot change under a running TUI, and ``draw`` runs every frame.
"""
from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Oldest lerobot this TUI targets. Below it, the launchers emit flags it cannot parse
#: and the LeKiwi rollout/replay CLIs do not exist yet.
FLOOR = (0, 6, 1)

#: A field that landed in the floor release, used to tell a real 0.6.1 from a checkout
#: that merely calls itself 0.6.1. Keep it in sync with FLOOR.
FLOOR_MARKER = ("no_stamp", "configs/dataset.py")

OK = "ok"
PRERELEASE = "prerelease"
TOO_OLD = "too_old"
MISSING = "missing"


@dataclass(frozen=True)
class LerobotEnv:
    state: str                      # OK | PRERELEASE | TOO_OLD | MISSING
    version: str | None             # as reported by package metadata
    path: str | None                # the installed package dir
    note: str                       # short human phrase, empty when state is OK


def version_tuple(text: str) -> tuple[int, ...]:
    """``"0.6.1.dev3"`` -> ``(0, 6, 1)``. Trailing non-numeric parts are dropped, so a
    dev/rc suffix compares equal to its release rather than sorting unpredictably."""
    parts: list[int] = []
    for chunk in re.split(r"[.\-+]", text.strip()):
        m = re.match(r"^(\d+)", chunk)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def _package_dir() -> Path | None:
    """The installed lerobot directory, located WITHOUT executing the package."""
    try:
        spec = importlib.util.find_spec("lerobot")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _metadata_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("lerobot")
    except PackageNotFoundError:
        return None


def declares(name: str, rel: str, pkg_dir: Path | None = None) -> bool:
    """True when the installed lerobot's source at *rel* mentions *name*.

    The Python twin of ``lerobot_declares`` in scripts/lib.sh, and textual for the same
    reason: importing the module to ask a yes/no question costs ~3s.
    """
    pkg = pkg_dir if pkg_dir is not None else _package_dir()
    if pkg is None:
        return False
    try:
        return name in (pkg / rel).read_text()
    except OSError:
        return False


def _classify(version: str | None, pkg: Path | None) -> tuple[str, str]:
    if version is None and pkg is None:
        return MISSING, f"not installed (needs {floor_text()}+)"
    if version is None:
        return PRERELEASE, "installed but unversioned"
    if version_tuple(version) < FLOOR:
        return TOO_OLD, f"needs {floor_text()}+"
    if not declares(*FLOOR_MARKER, pkg_dir=pkg):
        return PRERELEASE, f"pre-release, no {FLOOR_MARKER[0]}"
    return OK, ""


def floor_text() -> str:
    return ".".join(str(n) for n in FLOOR)


@lru_cache(maxsize=1)
def status() -> LerobotEnv:
    """The cached verdict for this process."""
    pkg = _package_dir()
    version = _metadata_version()
    state, note = _classify(version, pkg)
    return LerobotEnv(state=state, version=version, path=str(pkg) if pkg else None, note=note)


def summary() -> tuple[str, str, str]:
    """``(value, suffix, level)`` for one status-card cell.

    Three levels rather than a bool, because the states differ in how loud they deserve
    to be. A version that clears the floor is just a fact ("0.6.1"). A checkout claiming
    a release it does not fully carry gets a quiet marker ("0.6.1 · pre") — worth knowing,
    since it decides whether the launchers can pass newer flags, but it is not broken.
    Only genuinely too-old or absent earns a warning, because that one WILL fail a launch.
    """
    env = status()
    if env.state == MISSING:
        return "not found", "", "warn"
    value = env.version or "?"
    if env.state == OK:
        return value, "", "ok"
    if env.state == PRERELEASE:
        return value, " · pre", "note"
    return value, f" · needs {floor_text()}+", "warn"


__all__ = ["FLOOR", "LerobotEnv", "declares", "floor_text", "status", "summary", "version_tuple"]
