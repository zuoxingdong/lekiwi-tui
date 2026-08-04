"""scoreboard.py — the eval scoreboard sidecar (policies/scoreboard.jsonl).

One JSONL line per judged rollout: {ts, label, task, success}. The eval screen's
post-run G/B verdict modal appends here and its form renders per-task tallies for
the selected checkpoint. Same sidecar pattern as the dataset quality.jsonl — a
plain data layer, importable by any screen (menu/dashboard) without touching eval.
"""
from __future__ import annotations

import json
from pathlib import Path

SCOREBOARD_NAME = "scoreboard.jsonl"


def ckpt_label(policy: str, root: "Path | str") -> str:
    """A short scoreboard label for a checkpoint path:
    <root>/shop_05/checkpoints/060000/pretrained_model → `shop_05-60k`."""
    try:
        parts = Path(policy).relative_to(Path(root)).parts
    except ValueError:
        parts = Path(policy).parts
    run = parts[0] if parts else str(policy)
    step = ""
    for p in parts:
        if p.isdigit():
            n = int(p)
            step = f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)
    return f"{run}-{step}" if step else run


def load_scores(root: "Path | str") -> list[dict]:
    out: list[dict] = []
    try:
        for ln in (Path(root) / SCOREBOARD_NAME).read_text().splitlines():
            try:
                e = json.loads(ln)
                if isinstance(e, dict) and "success" in e:
                    out.append(e)
            except ValueError:
                continue
    except OSError:
        return []
    return out


def append_score(root: "Path | str", entry: dict) -> bool:
    try:
        with open(Path(root) / SCOREBOARD_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except OSError:
        return False


def score_tally(scores: list[dict], *, label: str, task: str | None = None) -> tuple[int, int]:
    """(successes, total) for one checkpoint label, optionally narrowed to a task."""
    hits = [e for e in scores if e.get("label") == label
            and (task is None or e.get("task") == task)]
    return sum(1 for e in hits if e.get("success")), len(hits)


