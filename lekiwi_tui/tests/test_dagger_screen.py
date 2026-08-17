"""DaggerScreen: the task picker fed from the base dataset, the dynamic field list,
and the launch argv (dagger.sh is fronted with the resolved flags + a per-session
stamped --dataset-root). Mirrors the EvalScreen test idiom (fake checkpoint dirs,
FakeApp capturing the suspended argv, confirm_preflight monkeypatched)."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import lekiwi_tui.screens.dagger as dagger_mod
from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.datasets import dataset_tasks
from lekiwi_tui.framework.events import LEFT, RIGHT, Key
from lekiwi_tui.framework.screen import Invoke
from lekiwi_tui.screens.dagger import _CHEATSHEET, _GO, DaggerScreen, session_root

TASKS = [
    "Pick up the red cube and place it in the right box.",
    "Pick up the red cube and place it in the left box.",
    "Pick up the red cube and place it in the near box.",
]

RENAME_MAP = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.top": "observation.images.camera2",
    "observation.images.wrist": "observation.images.camera3",
}


def _write_tasks_parquet(path: Path, tasks: list[str]) -> None:
    """A v3 meta/tasks.parquet: the strings live in a `task` column (v3 writes them as
    the frame index, which parquet materializes exactly this way). pyarrow, not pandas —
    it is what the package itself reads parquet with, and the only one CI installs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "task": tasks}), path)


def _base_dataset(tmp_path: Path, name: str = "lekiwi-demo", tasks: list[str] = TASKS) -> Path:
    root = tmp_path / "datasets" / name
    (root / "meta").mkdir(parents=True)
    _write_tasks_parquet(root / "meta" / "tasks.parquet", tasks)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 40}))
    return root


def _checkpoint(tmp_path: Path, name: str, cams: list[str]) -> Path:
    ckpt = tmp_path / "models" / name / "pretrained_model"
    ckpt.mkdir(parents=True)
    features = {"observation.state": {"type": "STATE", "shape": [9]}}
    features.update({c: {"type": "VISUAL", "shape": [3, 480, 640]} for c in cams})
    (ckpt / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": features}))
    (ckpt / "model.safetensors").write_text("")
    return ckpt


def _ctx(tmp_path: Path, base: Path) -> Context:
    (tmp_path / "models").mkdir(exist_ok=True)
    return Context(
        cfg=Config(values={
            "POLICY_ROOT": str(tmp_path / "models"),
            "POLICY_PATH": "",
            "INFERENCE": "sync",
            "EXECUTION_HORIZON": "20",
            "DISPLAY_DATA": "off",
        }),
        doc={
            "record": {"dataset": {"root": str(base), "repo_id": "local/lekiwi-demo"}},
            "dagger": {
                "task": "",
                "inference": {"type": "sync"},
                "strategy": {"type": "dagger"},
                "teleop": {"type": "lekiwi_pincopen_leader"},
                "rename_map": dict(RENAME_MAP),
            },
        },
        gpu_name="RTX 2050",
        is_tty=True,
    )


class _App:
    """Captures the suspended argv; answers the cheat-sheet modal."""

    def __init__(self, cheatsheet_answer: str = _GO) -> None:
        self.suspended = None
        self.notices: list[str] = []
        self._answer = cheatsheet_answer

    async def suspend(self, argv, **kwargs):  # noqa: ANN001
        self.suspended = list(argv)
        return 0

    async def run_modal(self, modal):  # noqa: ANN001
        return self._answer

    def notify(self, msg, *a, **k):  # noqa: ANN001, ANN002, ANN003
        self.notices.append(str(msg))


async def _preflight_ok(*args, **kwargs):  # noqa: ANN002, ANN003
    return True


def _patch_preflight(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(dagger_mod, "confirm_preflight", _preflight_ok)
    monkeypatch.setattr(dagger_mod, "dagger_issues", lambda *a, **k: [])


def _run_start(screen) -> None:  # noqa: ANN001
    action = screen.handle_key(Key("s"))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())


# ── dataset_tasks ─────────────────────────────────────────────────────────────


def test_dataset_tasks_reads_the_v3_index(tmp_path):
    base = _base_dataset(tmp_path)
    assert dataset_tasks(base) == TASKS


def test_dataset_tasks_is_empty_on_missing_or_garbage(tmp_path):
    assert dataset_tasks(tmp_path / "nope") == []
    bad = tmp_path / "bad" / "meta"
    bad.mkdir(parents=True)
    (bad / "tasks.parquet").write_text("not parquet")
    assert dataset_tasks(tmp_path / "bad") == []


# ── the task picker ───────────────────────────────────────────────────────────


def test_task_row_cycles_the_base_datasets_strings(tmp_path):
    base = _base_dataset(tmp_path)
    screen = DaggerScreen(None, _ctx(tmp_path, base))
    screen._fpos = screen._fields().index("task")
    screen.handle_key(Key(RIGHT))
    assert screen._task_text == TASKS[0]  # first press lands on the first string
    screen.handle_key(Key(RIGHT))
    assert screen._task_text == TASKS[1]
    screen.handle_key(Key(LEFT))
    assert screen._task_text == TASKS[0]
    # and the choice survives reopening (session memory)
    ctx = screen.ctx
    assert DaggerScreen(None, ctx)._task_text == TASKS[0]


def test_task_cycling_is_inert_without_strings(tmp_path):
    base = tmp_path / "datasets" / "empty"
    base.mkdir(parents=True)
    (base / "data").mkdir()
    screen = DaggerScreen(None, _ctx(tmp_path, base))
    screen._fpos = screen._fields().index("task")
    screen.handle_key(Key(RIGHT))
    assert screen._task_text == ""  # nothing to cycle; ⏎ free text is the path


def _row_text(lines) -> list[str]:  # noqa: ANN001 — flatten Lines to plain strings
    return ["".join(s.content for s in ln.spans) for ln in lines]


def test_task_row_shows_the_stepper_cell_and_flags_custom_text(tmp_path):
    """The task is adjustable, so it must LOOK adjustable: the same ‹ › guillemets
    every stepper row carries, with the position in the base's string list. A string
    the base does not contain is the silent divergence the picker exists to prevent — it
    renders ‹ –/N › plus a visible warning instead of passing silently."""
    base = _base_dataset(tmp_path)
    screen = DaggerScreen(None, _ctx(tmp_path, base))
    screen._task_text = TASKS[1]
    rows = _row_text(screen._body_lines(120))
    task_row = next(r for r in rows if "Task" in r)
    assert "‹ 2/3 ›" in task_row and TASKS[1][:20] in task_row

    screen._task_text = "a hand-typed instruction"
    rows = _row_text(screen._body_lines(120))
    task_row = next(r for r in rows if "Task" in r)
    assert "‹ –/3 ›" in task_row
    assert any("not one of" in r and "⚠" in r for r in rows)  # the divergence warning

    # no strings in the base -> no stepper cell, just the free-text invitation
    empty = tmp_path / "datasets" / "empty"
    (empty / "data").mkdir(parents=True)
    screen._base = str(empty)
    screen._task_text = ""
    rows = _row_text(screen._body_lines(120))
    task_row = next(r for r in rows if "Task" in r)
    assert "‹" not in task_row and "(⏎ to type)" in task_row


# ── the dynamic field list ────────────────────────────────────────────────────


def test_fields_fold_advanced_and_swap_exec_with_backend(tmp_path):
    base = _base_dataset(tmp_path)
    screen = DaggerScreen(None, _ctx(tmp_path, base))
    fields = screen._fields()
    assert fields == ["policy", "base", "task", "backend", "cameras", "steps", "flow",
                      "target", "advanced", "start"]
    screen._advanced = True
    assert screen._fields() == ["policy", "base", "task", "backend", "cameras", "steps",
                                "flow", "target", "advanced", "mode", "input", "duration",
                                "display", "extra", "start"]
    # rtc paces via exec-horizon instead of n_action_steps; flow stays for both
    screen._backend = "rtc"
    assert "exec" in screen._fields() and "steps" not in screen._fields()
    assert "flow" in screen._fields()


def test_zero_sentinel_labels_show_the_checkpoint_own_values(tmp_path):
    base = _base_dataset(tmp_path)
    ctx = _ctx(tmp_path, base)
    ckpt = tmp_path / "models" / "labeled" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps(
        {"type": "smolvla", "input_features": {}, "n_action_steps": 50, "num_steps": 10}))
    (ckpt / "model.safetensors").write_text("")
    screen = DaggerScreen(None, ctx)
    screen._policy = str(ckpt)
    screen._refresh_ckpt_defaults()
    assert screen._steps.zero_label == "checkpoint default (50)"
    assert screen._flow.zero_label == "checkpoint default (10)"


# ── launch ────────────────────────────────────────────────────────────────────


def test_start_fronts_dagger_sh_with_the_resolved_flags(tmp_path, monkeypatch):
    _patch_preflight(monkeypatch)
    base = _base_dataset(tmp_path)
    ckpt = _checkpoint(tmp_path, "native", [f"observation.images.{c}" for c in ("front", "wrist", "top")])
    app = _App()
    screen = DaggerScreen(app, _ctx(tmp_path, base))
    screen._policy = str(ckpt)
    screen._task_text = TASKS[1]
    screen._target.set_text("15")
    screen._steps.set_text("10")
    screen._flow.set_text("8")
    _run_start(screen)

    argv = app.suspended
    assert argv is not None
    assert argv[0] == "bash" and argv[1].endswith("scripts/dagger.sh")
    assert argv[argv.index("--task") + 1] == TASKS[1]
    assert argv[argv.index("--target") + 1] == "15"
    assert argv[argv.index("--action-steps") + 1] == "10"
    assert argv[argv.index("--flow-steps") + 1] == "8"
    assert argv[argv.index("--record-autonomous") + 1] == "off"
    assert argv[argv.index("--input") + 1] == "keyboard"
    assert argv[argv.index("--cam-slots") + 1] == "native"  # native-key checkpoint
    root = argv[argv.index("--dataset-root") + 1]
    # stamped session dir under the datasets parent — the screen must know it
    # afterwards (post-session review), so it is computed here, not in the script
    assert re.fullmatch(
        re.escape(str(tmp_path / "datasets")) + r"/rollout_dagger_\d{8}_\d{6}", root)
    assert dagger_mod._state(screen.ctx)["last_root"] == root


def test_start_blocks_without_a_task_string(tmp_path, monkeypatch):
    _patch_preflight(monkeypatch)
    base = _base_dataset(tmp_path)
    ckpt = _checkpoint(tmp_path, "any", ["observation.images.front"])
    app = _App()
    screen = DaggerScreen(app, _ctx(tmp_path, base))
    screen._policy = str(ckpt)
    screen._task_text = ""
    _run_start(screen)
    assert app.suspended is None
    assert "task" in screen._err.lower()


def test_declining_the_cheatsheet_cancels_the_launch(tmp_path, monkeypatch):
    _patch_preflight(monkeypatch)
    base = _base_dataset(tmp_path)
    ckpt = _checkpoint(tmp_path, "any", ["observation.images.front"])
    app = _App(cheatsheet_answer=None)  # Esc on the cheat-sheet modal
    screen = DaggerScreen(app, _ctx(tmp_path, base))
    screen._policy = str(ckpt)
    screen._task_text = TASKS[0]
    _run_start(screen)
    assert app.suspended is None


def test_cheatsheet_is_structured_and_correct():
    # the three rules that cost data to learn live
    assert "Squeeze the trigger" in _CHEATSHEET
    assert "never teleop with tab" in _CHEATSHEET
    assert "Stop at a stable point" in _CHEATSHEET
    # structured: one key per line + a gap before the tips (wrap_label honors \n)
    assert _CHEATSHEET.count("\n") >= 4 and "\n\n" in _CHEATSHEET
    # no hub-push advertising: this setup is local-only (push_to_hub off)
    assert "push" not in _CHEATSHEET.lower()
    # it must fit the confirm card without truncation at its inner width
    from lekiwi_tui.framework.modals import _CARD_WIDTH, _MAX_LABEL_ROWS, wrap_label

    rows = wrap_label(_CHEATSHEET, _CARD_WIDTH - 2 - 2 * 2)
    assert len(rows) <= _MAX_LABEL_ROWS
    assert not rows[-1].endswith("…")


def test_session_root_shape(tmp_path):
    import time

    now = time.localtime(0)
    assert session_root(str(tmp_path), now=now).endswith(
        "rollout_dagger_" + time.strftime("%Y%m%d_%H%M%S", now))
