"""Policy discovery — replaces bash discover_policies / resolve_policy and the
checkpoint validity checks. Behavior ported verbatim from lekiwi.sh (633-662).

Note: the do_eval "configured POLICY_PATH is gone → fall back to newest" logic
(bash 831-840) lives in the eval SCREEN (phase 3), not here. resolve_policy is the
three clean cases the architecture pins (empty→newest / relative-under-root→prefixed
/ else as-is); the dead-path fallback is a screen concern.
"""
from __future__ import annotations

from pathlib import Path


def is_valid_checkpoint(p: Path) -> bool:
    """A dir counts as a loadable checkpoint only if it holds config.json +
    model.safetensors (what from_pretrained needs locally). Bash 643/773-775."""
    return (p / "config.json").is_file() and (p / "model.safetensors").is_file()


def discover_policies(root: Path) -> list[Path]:
    """Find loadable checkpoints under `root`, NEWEST FIRST. A dir named
    pretrained_model counts only if is_valid_checkpoint. Sort is by mtime DESC
    (bash `find -printf '%T@\\t%p' | sort -rn`), NOT name — a name sort would put
    040000 before last and silently pick the wrong default. Missing root → []."""
    if not root.is_dir():
        return []
    cands = [
        d for d in root.rglob("pretrained_model")
        if d.is_dir() and is_valid_checkpoint(d)
    ]
    # mtime descending; tie-break on path so equal-mtime seeds order deterministically.
    cands.sort(key=lambda d: (d.stat().st_mtime, str(d)), reverse=True)
    return cands


def resolve_policy(policy_path: str, root: Path) -> str:
    """Resolve POLICY_PATH to something lerobot can load. Bash resolve_policy
    (652-662):
      empty                              → newest checkpoint discovered under root ("auto")
      relative path existing under root  → prefixed with root
      anything else (absolute, HF repo)  → as-is
    Returns '' if auto finds nothing."""
    p = policy_path
    if not p:
        found = discover_policies(root)
        return str(found[0]) if found else ""
    # relative (no leading '/') AND exists under root → prefix it.
    if not p.startswith("/") and (root / p).is_dir():
        return str(root / p)
    return p
