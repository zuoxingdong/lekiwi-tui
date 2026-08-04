"""Dataset editor: edit.sh in-place swap semantics (golden + fake-tool run) and the
DatasetEditScreen view-model (loaders, anomaly flags, pre-marking, delete argv)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import types
from pathlib import Path

from lekiwi_tui import ROOT
from lekiwi_tui.screens.dataset_edit import (
    DatasetEditScreen,
    anomalies,
    delete_argv,
    load_episodes,
    load_verdicts,
)
from conftest import make_ctx

EDIT_SH = ROOT / "scripts" / "edit.sh"


def _key(name: str, **mods) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, ctrl=False, alt=False, shift=False, **mods)


def _make_dataset(root: Path, lengths: list[int], fps: int = 30) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(
        {"total_episodes": len(lengths), "total_frames": sum(lengths), "fps": fps}))
    table = pa.table({
        "episode_index": list(range(len(lengths))),
        "length": lengths,
        "tasks": [["pick the thing"] for _ in lengths],
    })
    pq.write_table(table, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")


# ── edit.sh ───────────────────────────────────────────────────────────────────


def test_edit_sh_dry_run_golden(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds, [100, 110])
    out = subprocess.run(
        ["bash", str(EDIT_SH), "--repo-id", "local/ds", "--root", str(ds),
         "--episodes", "[1]", "--dry-run"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == "lerobot-edit-dataset"
    assert "--new_root" in out and out[out.index("--new_root") + 1] == f"{ds}.edit-tmp"
    assert out[out.index("--operation.episode_indices") + 1] == "[1]"
    # the swap plan is printed so the golden pins the in-place semantics too
    assert any(ln.startswith(f"## backup: {ds} -> {ds}.bak-") for ln in out)
    assert f"## swap:   {ds}.edit-tmp -> {ds}" in out


def test_edit_sh_refuses_non_dataset_and_missing_episodes(tmp_path):
    r = subprocess.run(["bash", str(EDIT_SH), "--root", str(tmp_path / "nope"),
                        "--episodes", "[0]"], capture_output=True, text=True)
    assert r.returncode == 2 and "meta/info.json" in r.stderr
    ds = tmp_path / "ds"
    _make_dataset(ds, [100])
    r = subprocess.run(["bash", str(EDIT_SH), "--root", str(ds)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--episodes" in r.stderr


def test_edit_sh_swaps_in_place_and_keeps_backup(tmp_path):
    """Run edit.sh against a FAKE lerobot-edit-dataset that writes --new_root; the
    dataset path must hold the edited result and the original must survive as .bak."""
    ds = tmp_path / "ds"
    _make_dataset(ds, [100, 110])
    (ds / "ORIGINAL").write_text("v1")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "lerobot-edit-dataset"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'while [[ $# -gt 0 ]]; do [[ "$1" == "--new_root" ]] && new="$2"; shift; done\n'
        'mkdir -p "$new/meta"\n'
        'echo "{}" > "$new/meta/info.json"\n'
        'echo edited > "$new/EDITED"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(EDIT_SH), "--root", str(ds), "--episodes", "[0]"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (ds / "EDITED").exists()                      # edited result AT THE SAME PATH
    assert not (ds / "ORIGINAL").exists()
    baks = list(tmp_path.glob("ds.bak-*"))
    assert len(baks) == 1 and (baks[0] / "ORIGINAL").read_text() == "v1"
    assert not (tmp_path / "ds.edit-tmp").exists()


def test_edit_sh_failure_leaves_dataset_untouched(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds, [100])
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "lerobot-edit-dataset"
    fake.write_text("#!/usr/bin/env bash\nexit 3\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(EDIT_SH), "--root", str(ds), "--episodes", "[0]"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 3
    assert (ds / "meta" / "info.json").exists()          # original untouched
    assert not list(tmp_path.glob("ds.bak-*"))
    assert not (tmp_path / "ds.edit-tmp").exists()


# ── loaders + view-model ──────────────────────────────────────────────────────


def test_load_episodes_verdicts_anomalies(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds, [900, 860, 880, 400, 890])
    (ds / "meta" / "quality.jsonl").write_text(
        json.dumps({"episode": 3, "verdict": "flagged"}) + "\n"
        + json.dumps({"episode": 0, "verdict": "flagged"}) + "\n"
        + json.dumps({"episode": 0, "verdict": "good"}) + "\n")  # latest line wins

    rows = load_episodes(ds)
    assert [r.index for r in rows] == [0, 1, 2, 3, 4]
    assert rows[3].seconds == round(400 / 30, 1)
    assert load_verdicts(ds) == {0: "good", 3: "flagged"}
    assert anomalies(rows) == {3: "short"}


def test_delete_argv_tokens():
    argv = delete_argv("../datasets/ds", "local/ds", [4, 1])
    assert argv[0] == "bash" and argv[1].endswith("edit.sh")
    assert argv[argv.index("--episodes") + 1] == "[1, 4]"
    assert argv[argv.index("--root") + 1] == "../datasets/ds"


def test_screen_premarks_flagged_and_toggles(tmp_path, monkeypatch):
    ds = tmp_path / "ds"
    _make_dataset(ds, [900, 860, 400])
    (ds / "meta" / "quality.jsonl").write_text(
        json.dumps({"episode": 2, "verdict": "flagged"}) + "\n")


    ctx = make_ctx(gpu_name="")
    scr = DatasetEditScreen(None, ctx)
    scr._root = str(ds)
    scr._repo_id = "local/ds"
    scr.reload()

    assert scr._marks == {2}                 # triage flag arrives pre-marked
    scr._cursor = 0
    scr.handle_key(_key(" "))                # Space marks the highlighted episode
    assert scr._marks == {0, 2}
    scr.handle_key(_key(" "))
    assert scr._marks == {2}

    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(100, 30))
    assert "✗ flagged" in body and "⚠ short" in body and "1 marked" in body


def test_screen_switch_dataset_via_picker(tmp_path):
    """The `d` key retargets the editor through the shared pick_dataset flow."""
    import asyncio

    from lekiwi_tui.framework.screen import Invoke

    other = tmp_path / "other-ds"
    _make_dataset(other, [300, 320])

    ctx = make_ctx(gpu_name="")
    scr = DatasetEditScreen(None, ctx)

    action = scr.handle_key(_key("d"))
    assert isinstance(action, Invoke)

    # Drive the flow with a fake app whose picker "chooses" the other dataset.
    class _App:
        terminal = object()

        async def run_modal(self, modal):
            from lekiwi_tui.widgets.pickers import DatasetPicker
            assert isinstance(modal, DatasetPicker)
            return str(other)

    scr.app = _App()
    asyncio.run(scr._choose_dataset())
    assert scr._root == str(other)
    assert scr._repo_id.endswith("/other-ds")
    assert [r.frames for r in scr._rows] == [300, 320]

    # Esc (picker returns None) keeps the current dataset.
    class _AppCancel:
        terminal = object()

        async def run_modal(self, modal):
            return None

    scr.app = _AppCancel()
    asyncio.run(scr._choose_dataset())
    assert scr._root == str(other)


# ── retag (modify_tasks with backup+swap) ─────────────────────────────────────


def test_retag_argv_tokens():
    from lekiwi_tui.screens.dataset_edit import retag_argv

    argv = retag_argv("../datasets/ds", "local/ds", [7, 3], 'Pick "it" up · fast')
    assert argv[argv.index("--op") + 1] == "retag"
    assert argv[argv.index("--episodes") + 1] == "[3, 7]"
    assert argv[argv.index("--task") + 1] == 'Pick "it" up · fast'


def test_edit_sh_retag_dry_run_golden(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds, [100, 110])
    out = subprocess.run(
        ["bash", str(EDIT_SH), "--op", "retag", "--repo-id", "local/ds",
         "--root", str(ds), "--episodes", "[0, 1]",
         "--task", 'Grab the "tin" → basket', "--dry-run"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    assert out[0] == "lerobot-edit-dataset"
    # the tool must only ever see the TEMP COPY (modify_tasks edits in place)
    assert out[out.index("--root") + 1] == f"{ds}.edit-tmp"
    assert out[out.index("--operation.type") + 1] == "modify_tasks"
    tasks = json.loads(out[out.index("--operation.episode_tasks") + 1])
    assert tasks == {"0": 'Grab the "tin" → basket', "1": 'Grab the "tin" → basket'}
    assert f"## copy:   {ds} -> {ds}.edit-tmp" in out
    assert any(ln.startswith(f"## backup: {ds} -> {ds}.bak-") for ln in out)


def test_edit_sh_retag_requires_task(tmp_path):
    ds = tmp_path / "ds"
    _make_dataset(ds, [100])
    r = subprocess.run(["bash", str(EDIT_SH), "--op", "retag", "--root", str(ds),
                        "--episodes", "[0]"], capture_output=True, text=True)
    assert r.returncode == 2 and "--task" in r.stderr


def test_edit_sh_retag_copies_edits_and_swaps(tmp_path):
    """Retag flow: the fake tool edits the COPY it is pointed at; edit.sh must swap
    the edited copy into place and keep the untouched original as .bak."""
    ds = tmp_path / "ds"
    _make_dataset(ds, [100, 110])
    (ds / "MARKER").write_text("original")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "lerobot-edit-dataset"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'while [[ $# -gt 0 ]]; do [[ "$1" == "--root" ]] && root="$2"; shift; done\n'
        '[[ -f "$root/meta/info.json" ]] || exit 9\n'   # must be a real dataset copy
        'echo retagged > "$root/MARKER"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    r = subprocess.run(["bash", str(EDIT_SH), "--op", "retag", "--root", str(ds),
                        "--episodes", "[0]", "--task", "new"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert (ds / "MARKER").read_text().strip() == "retagged"   # edited copy in place
    baks = list(tmp_path.glob("ds.bak-*"))
    assert len(baks) == 1 and (baks[0] / "MARKER").read_text() == "original"
    assert not (tmp_path / "ds.edit-tmp").exists()


def test_screen_retag_flow(tmp_path):
    import asyncio

    from lekiwi_tui.framework.screen import Invoke

    ds = tmp_path / "ds"
    _make_dataset(ds, [500, 510, 520])

    ctx = make_ctx(gpu_name="")
    scr = DatasetEditScreen(None, ctx)
    scr._root = str(ds)
    scr._repo_id = "local/ds"
    scr.reload()

    assert scr.handle_key(_key("T")).__class__.__name__ != "Invoke"  # no marks → msg
    assert "nothing marked" in scr._msg
    scr._marks = {1, 2}
    assert isinstance(scr.handle_key(_key("T")), Invoke)

    ran: list[list[str]] = []

    class _App:
        terminal = object()

        async def run_modal(self, modal):
            return "Pick up the Handcream and place into the basket."

        async def suspend(self, argv, **kw):
            ran.append(list(argv))
            return 0

        def notify(self, *a, **k):
            pass

    scr.app = _App()
    asyncio.run(scr._retag_marked())
    assert ran and ran[0][ran[0].index("--op") + 1] == "retag"
    assert ran[0][ran[0].index("--episodes") + 1] == "[1, 2]"
    assert ran[0][ran[0].index("--task") + 1].startswith("Pick up the Handcream")
