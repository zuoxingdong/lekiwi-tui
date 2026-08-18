"""scripts/merge.sh + DatasetSelectScreen (the Edit-dataset entry page).

merge.sh is the sole argv source for the N-way merge; the golden test pins its
dry-run token stream. Execution runs against a fake lerobot-edit-dataset to prove
the selective strip (only inputs carrying `intervention`), the episode-stats
normalize, the cleanup, and that sources survive untouched.

The selection screen: Space toggles (order = merge order), ⏎ opens the episode
editor for a single pick and the merge flow for several.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lekiwi_tui.screens.dataset_edit import DatasetEditScreen
from lekiwi_tui.screens.dataset_select import DatasetSelectScreen, merge_out_default
from lekiwi_tui.framework.events import DOWN, ENTER, SPACE, Key
from lekiwi_tui.framework.screen import Invoke

ROOT = Path(__file__).resolve().parents[2]


def _script_workspace(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "lekiwi.example.yaml", tmp_path / "lekiwi.example.yaml")
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    return tmp_path


def _run(ws: Path, *flags: str, dry: bool = True, env: dict | None = None):
    return subprocess.run(
        ["bash", str(ws / "scripts" / "merge.sh"),
         *(["--dry-run"] if dry else []), *flags],
        env={**os.environ, **(env or {})}, capture_output=True, text=True, check=False,
    )


def _dataset(parent: Path, name: str, features: list[str]) -> Path:
    root = parent / name
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        {"total_episodes": 1, "features": {f: {} for f in features}}))
    return root


# ── the golden argv stream (fake paths → every input planned as stripped) ──────


def test_dry_run_golden(tmp_path):
    ws = _script_workspace(tmp_path)
    proc = _run(
        ws,
        "--dataset", "../../datasets/lekiwi-demo",
        "--dataset", "../../datasets/rollout_dagger_a",
        "--out-name", "lekiwi-demo-dagger-v1",
    )
    assert proc.returncode == 0, proc.stderr
    wip = "../../datasets/.lekiwi-demo-dagger-v1.wip"
    assert proc.stdout.splitlines() == [
        "lerobot-edit-dataset",
        "--repo_id", "local/lekiwi-demo",
        "--root", "../../datasets/lekiwi-demo",
        "--new_repo_id", "local/lekiwi-demo__noint",
        "--new_root", f"{wip}/src-0",
        "--operation.type", "remove_feature",
        "--operation.feature_names=['intervention']",
        "---",
        "normalize-episode-stats",
        f"{wip}/src-0",
        "---",
        "lerobot-edit-dataset",
        "--repo_id", "local/rollout_dagger_a",
        "--root", "../../datasets/rollout_dagger_a",
        "--new_repo_id", "local/rollout_dagger_a__noint",
        "--new_root", f"{wip}/src-1",
        "--operation.type", "remove_feature",
        "--operation.feature_names=['intervention']",
        "---",
        "normalize-episode-stats",
        f"{wip}/src-1",
        "---",
        "lerobot-edit-dataset",
        "--new_repo_id", "local/lekiwi-demo-dagger-v1",
        "--new_root", "../../datasets/lekiwi-demo-dagger-v1",
        "--operation.type", "merge",
        "--operation.repo_ids=['local/lekiwi-demo__noint', 'local/rollout_dagger_a__noint']",
        f"--operation.roots=['{wip}/src-0', '{wip}/src-1']",
    ]


def test_flag_validation(tmp_path):
    ws = _script_workspace(tmp_path)
    assert _run(ws, "--dataset", "a", "--out-name", "x").returncode == 2      # <2 inputs
    assert _run(ws, "--dataset", "a", "--dataset", "b").returncode == 2      # no out-name
    proc = _run(ws, "--dataset", "a", "--dataset", "b", "--out-name", "a/b")
    assert proc.returncode == 2 and "folder name" in proc.stderr


# ── execution plumbing (fake lerobot-edit-dataset) ────────────────────────────


def _fake_tool(ws: Path) -> Path:
    """A lerobot-edit-dataset stub: log the argv, create --new_root (as the real
    tool does), so the strip/merge/cleanup logic runs for real."""
    bin_dir = ws / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "lerobot-edit-dataset"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$@\" >> {ws}/calls.log\n"
        "new_root=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ $1 == --new_root ]]; then new_root=$2; shift 2; else shift; fi\n"
        "done\n"
        "mkdir -p \"$new_root/meta\"\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return bin_dir


def test_execution_strips_only_dagger_inputs_and_cleans_up(tmp_path):
    ws = _script_workspace(tmp_path)
    data = tmp_path / "datasets"
    base = _dataset(data, "lekiwi-demo", ["action", "observation.state"])
    sess = _dataset(data, "rollout_dagger_a", ["action", "observation.state", "intervention"])
    bin_dir = _fake_tool(ws)

    proc = _run(ws, "--dataset", str(base), "--dataset", str(sess),
                "--out-name", "plate-v2", dry=False,
                env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert proc.returncode == 0, proc.stderr

    calls = (ws / "calls.log").read_text().splitlines()
    assert len(calls) == 2                                  # ONE strip (session only) + merge
    assert "remove_feature" in calls[0] and str(sess) in calls[0]
    assert "merge" in calls[1]
    assert str(base) in calls[1]                            # base merged AS-IS, no copy
    assert (data / "plate-v2").is_dir()
    assert not (data / ".plate-v2.wip").exists()            # temp copies cleaned up
    assert base.is_dir() and sess.is_dir()                  # sources untouched


def test_feature_mismatch_fails_fast_with_the_diff(tmp_path):
    ws = _script_workspace(tmp_path)
    data = tmp_path / "datasets"
    base = _dataset(data, "lekiwi-demo", ["action", "observation.state"])
    other = _dataset(data, "rollout_shelf", ["action", "intervention"])  # missing state
    bin_dir = _fake_tool(ws)

    proc = _run(ws, "--dataset", str(base), "--dataset", str(other),
                "--out-name", "plate-v2", dry=False,
                env={"PATH": f"{bin_dir}:{os.environ['PATH']}"})
    assert proc.returncode != 0
    assert "features differ" in proc.stderr and "observation.state" in proc.stderr
    assert not (ws / "calls.log").exists()                  # failed BEFORE any tool run


def test_normalize_drops_stale_intervention_stats_columns(tmp_path):
    """The correctness step the pipeline exists for: remove_feature copytree's
    meta/episodes verbatim, and aggregate APPENDS episode rows into shared parquets,
    so a stale stats/intervention/* column would poison the merged metadata."""
    ws = _script_workspace(tmp_path)
    ds = tmp_path / "copy"
    (ds / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    f = ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    pq.write_table(pa.table({
        "episode_index": [0, 1],
        "length": [600, 300],
        "stats/action/min": [0.1, 0.2],
        "stats/intervention/min": [1.0, 1.0],
        "stats/intervention/max": [1.0, 1.0],
    }), f)

    proc = _run(ws, "--normalize-only", str(ds), dry=False)
    assert proc.returncode == 0, proc.stderr
    cols = list(pq.read_table(f).schema.names)
    assert "stats/action/min" in cols                       # real stats survive
    assert not [c for c in cols if c.startswith("stats/intervention")]


# ── DatasetSelectScreen ───────────────────────────────────────────────────────


def _select_ctx(parent: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        doc={"record": {"dataset": {"root": str(parent / "lekiwi-demo"),
                                    "repo_id": "local/lekiwi-demo"}}},
        cfg={}, gpu_name="", ui_state={})


def _episodeful(root: Path) -> None:
    (root / "meta").parent.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)


class _App:
    def __init__(self, modal_answers: list) -> None:
        self.pushed = None
        self.suspended = None
        self.notices = []
        self._answers = list(modal_answers)

    async def run_modal(self, modal):  # noqa: ANN001
        return self._answers.pop(0)

    async def suspend(self, argv, **kwargs):  # noqa: ANN001
        self.suspended = list(argv)
        return 0

    def push(self, screen):  # noqa: ANN001
        self.pushed = screen

    def notify(self, msg, *a, **k):  # noqa: ANN001, ANN002, ANN003
        self.notices.append(str(msg))


def _three_datasets(tmp_path: Path) -> Path:
    parent = tmp_path / "datasets"
    _dataset(parent, "lekiwi-demo", ["action", "observation.state"])
    _dataset(parent, "rollout_dagger_a", ["action", "observation.state", "intervention"])
    _dataset(parent, "rollout_dagger_b", ["action", "observation.state", "intervention"])
    return parent


def test_rows_carry_the_dagger_tag(tmp_path):
    parent = _three_datasets(tmp_path)
    screen = DatasetSelectScreen(None, _select_ctx(parent))
    by_name = {name: root for name, root, _eps in screen._rows}
    assert set(by_name) == {"lekiwi-demo", "rollout_dagger_a", "rollout_dagger_b"}
    assert screen._is_dagger(by_name["rollout_dagger_a"])
    assert not screen._is_dagger(by_name["lekiwi-demo"])


def test_enter_with_no_toggle_opens_the_editor_on_the_cursor_row(tmp_path):
    parent = _three_datasets(tmp_path)
    app = _App([])
    screen = DatasetSelectScreen(app, _select_ctx(parent))
    screen.handle_key(Key(DOWN))
    action = screen.handle_key(Key(ENTER))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())
    assert isinstance(app.pushed, DatasetEditScreen)
    assert app.pushed._root == screen._rows[1][1]


def test_toggling_several_merges_in_toggle_order(tmp_path):
    parent = _three_datasets(tmp_path)
    # toggle row 1 FIRST, then row 0 — the merge must respect that order
    app = _App(["my-merge", "Merge"])
    screen = DatasetSelectScreen(app, _select_ctx(parent))
    screen.handle_key(Key(DOWN))
    screen.handle_key(Key(SPACE))     # rows[1]
    screen.handle_key(Key("k"))
    screen.handle_key(Key(SPACE))     # rows[0]
    action = screen.handle_key(Key(ENTER))
    asyncio.run(action.thunk())

    argv = app.suspended
    assert argv is not None and argv[1].endswith("scripts/merge.sh")
    roots = [argv[i + 1] for i, t in enumerate(argv) if t == "--dataset"]
    assert roots == [screen._rows[1][1], screen._rows[0][1]]   # toggle order, not list order
    assert argv[argv.index("--out-name") + 1] == "my-merge"
    assert app.pushed is None                                  # merge, not the editor


def test_cancelling_the_confirm_keeps_everything(tmp_path):
    parent = _three_datasets(tmp_path)
    app = _App(["my-merge", None])    # Esc on the confirm
    screen = DatasetSelectScreen(app, _select_ctx(parent))
    screen.handle_key(Key(SPACE))
    screen.handle_key(Key(DOWN))
    screen.handle_key(Key(SPACE))
    action = screen.handle_key(Key(ENTER))
    asyncio.run(action.thunk())
    assert app.suspended is None


def test_merge_out_default_versions_and_tags_dagger(tmp_path):
    parent = tmp_path
    (parent / "lekiwi-demo-dagger-v1").mkdir()
    (parent / "lekiwi-demo-dagger-v3").mkdir()
    assert merge_out_default(["lekiwi-demo"], True, parent) == "lekiwi-demo-dagger-v4"
    assert merge_out_default(["lekiwi-demo"], False, parent) == "lekiwi-demo-merged-v1"
