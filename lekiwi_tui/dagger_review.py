"""Post-session review for DAgger datasets: per-episode stats + junk detection.

A corrections-only dagger session saves one episode per takeover, and two kinds of
junk episode are routine: the tab-reposition-tab arm reset (recorded by necessity —
there is no non-recording way to move the arm) and a fat-fingered tab-tab. Both have
a data signature a manipulation correction cannot have: the GRIPPER NEVER CLOSES
(a reposition moves an empty gripper) and/or the episode is a few seconds long.

The verdicts are delivered through the dataset editor's existing sidecar contract
(``meta/quality.jsonl``): ``write_quality_flags`` appends ``flagged`` lines, and
``DatasetEditScreen.reload`` PRE-MARKS flagged episodes, so the post-session flow is
"open the editor → the junk is already marked → D deletes it" with no new triage UI.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Normalized gripper position is 0..100 with 100 = fully open. A correction that
#: grasps an object must close well below this; a reposition never does.
GRIP_CLOSED_BELOW = 60.0
#: Corrections shorter than this are almost always fumbled tab-tabs; real
#: recover-and-finish arcs run tens of seconds.
MIN_SECONDS = 5.0


def dagger_episode_report(root: str | Path) -> list[dict]:
    """Per-episode stats for the dataset at *root*, with a junk verdict.

    Each entry: ``{"index", "seconds", "grip_min", "grip_max", "arm_travel", "junk"}``
    where ``junk`` is "" or the human-readable reason. Best-effort: any unreadable
    piece → [] (the caller then just skips the review).
    """
    p = Path(root)
    try:
        info = json.loads((p / "meta" / "info.json").read_text())
        fps = float(info.get("fps") or 30)
        names = list(info["features"]["action"]["names"])
    except Exception:
        return []
    files = sorted(p.glob("data/chunk-*/file-*.parquet"))
    if not files:
        return []
    try:
        import pyarrow.parquet as pq  # deferred: heavy, and not per-frame material

        columns = ["episode_index", "action"]
        rows: list[tuple[int, list[float]]] = []
        for f in files:
            table = pq.read_table(f, columns=columns)
            rows.extend(zip(table.column("episode_index").to_pylist(),
                            table.column("action").to_pylist(), strict=True))
    except Exception:
        return []

    grip_i = names.index("arm_gripper.pos") if "arm_gripper.pos" in names else None
    arm_i = [i for i, n in enumerate(names) if n.endswith(".pos") and i != grip_i]
    by_episode: dict[int, list[list[float]]] = {}
    for episode, action in rows:
        if action is not None:
            by_episode.setdefault(int(episode), []).append(list(action))

    out: list[dict] = []
    for episode, actions in by_episode.items():
        seconds = round(len(actions) / fps, 1)
        grips = [a[grip_i] for a in actions] if grip_i is not None else []
        grip_min = float(min(grips)) if grips else float("nan")
        grip_max = float(max(grips)) if grips else float("nan")
        # Total joint travel: the per-tick absolute change summed over the arm joints.
        # Plain Python on purpose — a correction is a few thousand rows, and numpy is
        # not a dependency of this package (it arrives with lerobot, which lives in a
        # different env).
        travel = sum(
            abs(nxt[i] - cur[i])
            for cur, nxt in zip(actions, actions[1:], strict=False)
            for i in arm_i
        )
        reasons = []
        if grip_i is not None and grip_min >= GRIP_CLOSED_BELOW:
            reasons.append("gripper never closes")
        if seconds < MIN_SECONDS:
            reasons.append("very short")
        out.append({
            "index": episode, "seconds": seconds, "grip_min": grip_min,
            "grip_max": grip_max, "arm_travel": round(travel, 1),
            "junk": " · ".join(reasons),
        })
    return sorted(out, key=lambda r: r["index"])


def write_quality_flags(root: str | Path, flags: dict[int, str]) -> bool:
    """Append ``flagged`` verdicts to ``meta/quality.jsonl`` (the dataset editor's
    triage sidecar — flagged episodes arrive PRE-MARKED there). Latest line per
    episode wins in the editor, so appending is the whole contract. Returns False
    when the write fails (the review still shows; only the pre-marking is lost)."""
    if not flags:
        return True
    path = Path(root) / "meta" / "quality.jsonl"
    try:
        with path.open("a", encoding="utf-8") as fh:
            for idx, why in sorted(flags.items()):
                fh.write(json.dumps(
                    {"episode": idx, "verdict": "flagged", "why": why,
                     "source": "dagger-review"}) + "\n")
    except OSError:
        return False
    return True


def session_summary(report: list[dict], *, max_lines: int = 6) -> str:
    """One compact line per episode for the post-session modal, junk called out."""
    parts = []
    for r in report[:max_lines]:
        grip = (f"grip {r['grip_min']:.0f}→{r['grip_max']:.0f}"
                if r["grip_min"] == r["grip_min"] else "no gripper data")  # NaN-safe
        line = f"ep{r['index']} {r['seconds']}s · {grip}"
        if r["junk"]:
            line += f" ⚠ {r['junk']}"
        parts.append(line)
    if len(report) > max_lines:
        parts.append(f"… {len(report) - max_lines} more")
    return "   ".join(parts)


__all__ = ["dagger_episode_report", "write_quality_flags", "session_summary",
           "GRIP_CLOSED_BELOW", "MIN_SECONDS"]
