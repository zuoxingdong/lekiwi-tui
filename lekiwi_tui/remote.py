"""Remote command validation helpers.

Remote paths that may contain spaces are shell-quoted in the Bash launchers.
SSH destinations, conda environment names, and robot ids are intentionally kept
to simple, unambiguous character sets so they cannot be parsed as shell syntax or
command-line options.
"""
from __future__ import annotations

import re
from typing import Any


class RemoteValueError(ValueError):
    """Raised when a remote command setting is unsafe or malformed."""


_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _text(value: Any, field: str) -> str:
    s = str(value)
    if not s:
        raise RemoteValueError(f"{field} must not be empty")
    if any(ord(ch) < 32 for ch in s):
        raise RemoteValueError(f"{field} must not contain control characters")
    if any(ch.isspace() for ch in s):
        raise RemoteValueError(f"{field} must not contain whitespace")
    return s


def validate_ssh_host(value: Any, field: str = "SSH host") -> str:
    """Validate an SSH destination such as ``lekiwi`` or ``pi@lekiwi.local``."""
    s = _text(value, field)
    if s.startswith("-"):
        raise RemoteValueError(f"{field} must not start with '-'")
    if not _HOST_RE.fullmatch(s) or ".." in s:
        raise RemoteValueError(
            f"{field} contains unsupported characters; use a simple host or user@host"
        )
    return s


def validate_remote_name(value: Any, field: str) -> str:
    """Validate identifier-like remote values: conda env names and robot ids."""
    s = _text(value, field)
    if s.startswith("-") or not _NAME_RE.fullmatch(s):
        raise RemoteValueError(f"{field} must use only letters, numbers, '.', '_', or '-'")
    return s


def validate_positive_int(value: Any, field: str) -> str:
    """Validate a positive integer and return its normalized string form."""
    s = _text(value, field)
    if not s.isdigit() or int(s) <= 0:
        raise RemoteValueError(f"{field} must be a positive integer")
    return str(int(s))
