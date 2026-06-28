"""Dataset helpers — replaces bash record_root / dataset_repo_id / dataset_present /
dataset_episodes / args_have_resume. Behavior ported verbatim from lekiwi.sh.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .config import cfg_get


def record_root(doc: dict | None = None, extra: Sequence[str] = ()) -> str:
    """Where will this record land? A --dataset.root= override in `extra` wins;
    otherwise `record.dataset.root` from the yaml; otherwise the documented default.
    The path is left as-is (relative resolves against cwd, which is ROOT). Bash
    record_root (402-407)."""
    for a in extra:
        if a.startswith("--dataset.root="):
            return a[len("--dataset.root="):]
    r = cfg_get("record.dataset.root", doc=doc)
    return str(r) if r else "../datasets/lekiwi_dataset"


def dataset_repo_id(doc: dict | None = None) -> str:
    """The dataset repo_id (HF namespace/name) from the record yaml; used by `view`
    to point lerobot-dataset-viz at the right dataset. Bash dataset_repo_id (410-414)."""
    r = cfg_get("record.dataset.repo_id", doc=doc)
    return str(r) if r else "local/lekiwi_dataset"


def dataset_present(root: str | Path) -> bool:
    """True on a NON-EMPTY directory, not a valid info.json: a crashed/partial
    recording leaves a dir lerobot still refuses to overwrite, and that's exactly
    when Delete is most useful. Bash dataset_present (418)."""
    p = Path(root)
    return p.is_dir() and any(p.iterdir())


def discover_datasets(parent: str | Path) -> list[tuple[str, str, str]]:
    """Present datasets directly under *parent*, NEWEST FIRST (mtime desc, name tie-break —
    same convention as policies.discover_policies). Each entry is ``(name, root, episodes)``
    where ``root`` is the dir path AS-IS (kept RELATIVE — the app's cwd=ROOT convention means
    "../datasets/foo" must flow unchanged into the lerobot argv; do NOT resolve()) and
    ``episodes`` is dataset_episodes()'s string ("N" / "?"). A dir counts only when
    dataset_present() (non-empty); a missing/!dir *parent* → []. Used by the DatasetPicker
    (replay/view) to let the user pick which recording to play back / browse."""
    p = Path(parent)
    if not p.is_dir():
        return []
    dirs = [d for d in p.iterdir() if d.is_dir() and dataset_present(d)]
    dirs.sort(key=lambda d: (d.stat().st_mtime, d.name), reverse=True)
    return [(d.name, str(d), dataset_episodes(d)) for d in dirs]


def dataset_episodes(root: str | Path) -> str:
    """Best-effort episode count; "?" when meta/info.json is missing or unparseable.
    Bash dataset_episodes (421-424) read .total_episodes via jq."""
    info = Path(root) / "meta" / "info.json"
    try:
        n = json.loads(info.read_text()).get("total_episodes")
    except Exception:
        return "?"
    return "?" if n is None else str(n)


def args_have_resume(extra: Sequence[str]) -> bool:
    """True if --resume or --resume=… is already in the passthrough args. Bash
    args_have_resume (425)."""
    return any(a == "--resume" or a.startswith("--resume=") for a in extra)
