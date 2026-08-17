"""Task strings offered by a base dataset — shared by the DAgger and Run-policy forms.

A policy is conditioned on a language instruction, and for it to behave it must be one
the policy was TRAINED on. A near-duplicate is a silently different prompt ("into the
second slot" vs "in to the second slot"), and nothing downstream complains: the policy
just performs worse. So both forms offer the strings recorded in a base dataset's
``meta/tasks.parquet`` and cycle them with ←→, keeping free text as the escape hatch
for a genuinely new instruction.

The base dataset differs slightly in meaning per screen — for DAgger it is also the
merge target for the corrections, for Run policy it is just the string source — but the
cycling behavior is identical, which is why it lives here rather than in either screen.
"""
from __future__ import annotations

from pathlib import Path

from ..datasets import dataset_tasks


class TaskChoices:
    """The task strings of a base dataset, cached per root, with ←→ cycling."""

    def __init__(self, base: str) -> None:
        self._base = str(base)
        self._cache: tuple[str, list[str]] | None = None

    @property
    def base(self) -> str:
        return self._base

    @base.setter
    def base(self, root: str) -> None:
        # Assigning a new root implicitly invalidates the cache (it is keyed by root).
        self._base = str(root)

    @property
    def name(self) -> str:
        """The base dataset's folder name — what the forms show on the Base row."""
        return Path(self._base).name

    def tasks(self) -> list[str]:
        """The base's task strings, read once per root (the parquet read is not
        per-frame material; ``draw`` calls this every frame)."""
        if self._cache is not None and self._cache[0] == self._base:
            return self._cache[1]
        tasks = dataset_tasks(self._base)
        self._cache = (self._base, tasks)
        return tasks

    def position(self, current: str) -> int | None:
        """1-based index of *current* among the base's strings; None when the text is
        custom (or the base carries none) — the forms render that as ``–/N`` plus a
        warning rather than passing it off as a valid pick."""
        tasks = self.tasks()
        return tasks.index(current) + 1 if current in tasks else None

    def cycle(self, current: str, delta: int) -> str:
        """The next (delta=+1) or previous (-1) string; *current* unchanged when the
        base has none. Custom text lands on an END of the list, never past it."""
        tasks = self.tasks()
        if not tasks:
            return current
        try:
            i = tasks.index(current)
        except ValueError:
            i = -delta if delta > 0 else 0
        return tasks[(i + delta) % len(tasks)]


__all__ = ["TaskChoices"]
