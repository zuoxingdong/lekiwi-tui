"""Post-session dagger review: the junk heuristic, the quality.jsonl handoff to the
dataset editor (flagged episodes arrive pre-marked), and the review modal flow."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import lekiwi_tui.screens.dagger as dagger_mod
from lekiwi_tui.dagger_review import (
    dagger_episode_report, session_summary, write_quality_flags,
)
from lekiwi_tui.screens.dataset_edit import DatasetEditScreen, load_verdicts

from conftest import make_ctx

ACTION_NAMES = [
    "arm_shoulder_pan.pos", "arm_shoulder_lift.pos", "arm_elbow_flex.pos",
    "arm_wrist_flex.pos", "arm_wrist_roll.pos", "arm_gripper.pos",
    "x.vel", "y.vel", "theta.vel",
]
FPS = 30


def _frame(grip: float, sweep: float) -> list[float]:
    return [sweep, -sweep, sweep, 0.0, 0.0, grip, 0.0, 0.0, 0.0]


def _session_dataset(tmp_path: Path) -> Path:
    """Two episodes: ep0 = a real 20s correction (gripper closes to 2), ep1 = a 3s
    reposition (gripper pinned open) — the exact junk signature from the field."""
    root = tmp_path / "rollout_dagger_20200101_000000"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({
        "total_episodes": 2, "fps": FPS,
        "features": {"action": {"names": ACTION_NAMES}},
    }))
    ep0 = [_frame(grip=100 - i % 99, sweep=float(i % 30)) for i in range(20 * FPS)]
    ep1 = [_frame(grip=98.0, sweep=float(i % 30)) for i in range(3 * FPS)]
    pq.write_table(
        pa.table({"episode_index": [0] * len(ep0) + [1] * len(ep1),
                  "action": ep0 + ep1}),
        root / "data" / "chunk-000" / "file-000.parquet")
    # the episodes-metadata parquet the editor lists rows (and pre-marks) from
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 1], "length": [len(ep0), len(ep1)],
                  "tasks": [["Pick up the red cube."], ["Pick up the red cube."]]}),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root


def test_report_flags_the_reposition_and_clears_the_real_correction(tmp_path):
    report = dagger_episode_report(_session_dataset(tmp_path))
    assert [r["index"] for r in report] == [0, 1]
    real, junk = report
    assert real["junk"] == ""
    assert real["seconds"] == 20.0 and real["grip_min"] < 60
    assert "gripper never closes" in junk["junk"] and "very short" in junk["junk"]


def test_report_is_empty_on_a_missing_or_unreadable_dataset(tmp_path):
    assert dagger_episode_report(tmp_path / "nope") == []
    bare = tmp_path / "bare"
    (bare / "meta").mkdir(parents=True)
    (bare / "meta" / "info.json").write_text("{}")
    assert dagger_episode_report(bare) == []


def test_quality_flags_arrive_pre_marked_in_the_editor(tmp_path):
    root = _session_dataset(tmp_path)
    assert write_quality_flags(root, {1: "gripper never closes · very short"})
    verdicts = load_verdicts(root)
    assert verdicts == {1: "flagged"}


def test_session_summary_caps_and_calls_out_junk():
    report = [{"index": i, "seconds": 10.0, "grip_min": 1.0, "grip_max": 99.0,
               "arm_travel": 100.0, "junk": ""} for i in range(8)]
    report[2]["junk"] = "very short"
    text = session_summary(report, max_lines=6)
    assert "ep2 10.0s" in text and "⚠ very short" in text
    assert "… 2 more" in text and "ep7" not in text


def test_editor_accepts_a_direct_root(tmp_path):
    root = _session_dataset(tmp_path)
    screen = DatasetEditScreen(None, make_ctx(), root=str(root))
    assert screen._root == str(root)
    assert screen._repo_id == f"local/{root.name}"


def test_review_flow_pre_marks_junk_and_opens_the_editor(tmp_path):
    root = _session_dataset(tmp_path)

    class _App:
        def __init__(self) -> None:
            self.pushed = None
            self.notices = []

        async def run_modal(self, modal):  # noqa: ANN001
            return "Review in dataset editor"

        def push(self, screen):  # noqa: ANN001
            self.pushed = screen

        def notify(self, msg, *a, **k):  # noqa: ANN001, ANN002, ANN003
            self.notices.append(str(msg))

    app = _App()
    screen = dagger_mod.DaggerScreen(app, make_ctx())
    asyncio.run(screen._review_session(str(root)))

    assert load_verdicts(root) == {1: "flagged"}       # heuristic wrote the sidecar
    assert isinstance(app.pushed, DatasetEditScreen)    # one keypress into the editor
    assert app.pushed._root == str(root)
    assert 1 in app.pushed._marks                       # junk arrives pre-marked
    assert 0 not in app.pushed._marks
