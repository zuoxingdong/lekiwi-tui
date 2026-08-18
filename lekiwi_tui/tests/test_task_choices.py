"""The shared task-string picker: TaskChoices logic + the Run-policy form's Base/Task rows.

A policy is conditioned on a language instruction and only behaves on the ones it was
trained on, so both forms offer the strings recorded in a base dataset rather than free
text alone. Run policy additionally DERIVES its default base from the checkpoint's own
train_config.json, which is the only authoritative answer to "what strings does this
policy know".
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.framework.events import ENTER, LEFT, RIGHT, Key
from lekiwi_tui.framework.screen import Invoke
from lekiwi_tui.screens.eval import EvalScreen, resolve_base_dataset, training_dataset_name
from lekiwi_tui.widgets.task_choices import TaskChoices

TASKS = ["Pick up the red cube and place it in the right box.",
         "Pick up the red cube and place it in the left box.",
         "Pick up the red cube and place it in the near box."]


def _write_tasks_parquet(path: Path, tasks: list[str]) -> None:
    """A v3 meta/tasks.parquet: the strings live in a `task` column (v3 writes them as
    the frame index, which parquet materializes exactly this way). pyarrow, not pandas —
    it is what the package itself reads parquet with, and the only one CI installs."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"task_index": list(range(len(tasks))), "task": tasks}), path)


def _dataset(parent: Path, name: str, tasks: list[str] = TASKS) -> Path:
    root = parent / name
    (root / "meta").mkdir(parents=True)
    _write_tasks_parquet(root / "meta" / "tasks.parquet", tasks)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 40}))
    return root


def _checkpoint(tmp_path: Path, name: str, trained_on: str | None = None) -> Path:
    ckpt = tmp_path / "models" / name / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": {}}))
    (ckpt / "model.safetensors").write_text("")
    if trained_on is not None:
        (ckpt / "train_config.json").write_text(
            json.dumps({"dataset": {"repo_id": f"local/{trained_on}"}}))
    return ckpt


def _ctx(tmp_path: Path, base: Path) -> Context:
    (tmp_path / "models").mkdir(exist_ok=True)
    return Context(
        cfg=Config(values={"POLICY_ROOT": str(tmp_path / "models"), "POLICY_PATH": "",
                           "INFERENCE": "sync", "EXECUTION_HORIZON": "20",
                           "DISPLAY_DATA": "off", "LEKIWI_HOST": "lekiwi"}),
        doc={"record": {"dataset": {"root": str(base), "repo_id": "local/base"}},
             "rollout": {"task": "a stale yaml string", "rename_map": {}}},
        gpu_name="", is_tty=True,
    )


def _rows(screen) -> list[str]:  # noqa: ANN001
    return ["".join(s.content for s in ln.spans) for ln in screen._body_lines(120)]


# ── TaskChoices ───────────────────────────────────────────────────────────────


def test_cycle_walks_the_list_and_wraps(tmp_path):
    tc = TaskChoices(str(_dataset(tmp_path, "plate")))
    assert tc.tasks() == TASKS and tc.name == "plate"
    assert tc.cycle(TASKS[0], 1) == TASKS[1]
    assert tc.cycle(TASKS[2], 1) == TASKS[0]      # wraps forward
    assert tc.cycle(TASKS[0], -1) == TASKS[2]     # wraps backward


def test_custom_text_lands_on_an_end_not_past_it(tmp_path):
    tc = TaskChoices(str(_dataset(tmp_path, "plate")))
    assert tc.cycle("hand typed", 1) == TASKS[0]    # → first
    assert tc.cycle("hand typed", -1) == TASKS[-1]  # ← last
    assert tc.position("hand typed") is None
    assert tc.position(TASKS[1]) == 2


def test_no_strings_makes_cycling_inert(tmp_path):
    empty = tmp_path / "empty"
    (empty / "data").mkdir(parents=True)
    tc = TaskChoices(str(empty))
    assert tc.tasks() == []
    assert tc.cycle("keep me", 1) == "keep me"
    assert tc.position("keep me") is None


def test_changing_the_base_invalidates_the_cache(tmp_path):
    a, b = _dataset(tmp_path, "a"), _dataset(tmp_path, "b", ["only one"])
    tc = TaskChoices(str(a))
    assert len(tc.tasks()) == 3
    tc.base = str(b)
    assert tc.tasks() == ["only one"] and tc.name == "b"


# ── checkpoint -> training dataset ────────────────────────────────────────────


def test_training_dataset_is_read_from_the_checkpoint(tmp_path):
    ckpt = _checkpoint(tmp_path, "ft", trained_on="lekiwi-demo")
    assert training_dataset_name(str(ckpt)) == "lekiwi-demo"          # namespace stripped
    assert training_dataset_name(str(_checkpoint(tmp_path, "bare"))) is None

    parent = tmp_path / "datasets"
    _dataset(parent, "lekiwi-demo")
    assert resolve_base_dataset(str(ckpt), parent) == str(parent / "lekiwi-demo")
    # records a dataset that is not on this machine -> no guess
    other = _checkpoint(tmp_path, "elsewhere", trained_on="not-here")
    assert resolve_base_dataset(str(other), parent) is None


# ── the Run-policy form ───────────────────────────────────────────────────────


def test_base_defaults_to_the_checkpoints_training_dataset(tmp_path):
    parent = tmp_path / "datasets"
    yaml_base = _dataset(parent, "yaml-default", ["yaml string"])
    _dataset(parent, "lekiwi-demo")
    ctx = _ctx(tmp_path, yaml_base)
    ctx.cfg.values["POLICY_PATH"] = str(_checkpoint(tmp_path, "ft", trained_on="lekiwi-demo"))

    screen = EvalScreen(None, ctx)
    assert screen._task_choices.name == "lekiwi-demo"   # not the yaml's dataset
    assert screen._base_source == "from checkpoint"
    assert screen._task_choices.tasks() == TASKS


def test_base_falls_back_to_the_yaml_dataset(tmp_path):
    parent = tmp_path / "datasets"
    yaml_base = _dataset(parent, "yaml-default", ["yaml string"])
    ctx = _ctx(tmp_path, yaml_base)
    ctx.cfg.values["POLICY_PATH"] = str(_checkpoint(tmp_path, "bare"))  # records nothing

    screen = EvalScreen(None, ctx)
    assert screen._task_choices.name == "yaml-default"
    assert screen._base_source == "yaml default"


def test_task_row_cycles_and_shows_the_stepper_cell(tmp_path):
    parent = tmp_path / "datasets"
    base = _dataset(parent, "plate")
    screen = EvalScreen(None, _ctx(tmp_path, base))
    assert screen._fields()[:3] == ["policy", "base", "task"]

    # the seeded yaml string is NOT one of the base's -> ‹ –/3 › + a visible warning
    rows = _rows(screen)
    assert any("‹ –/3 ›" in r for r in rows)
    assert any("not one of plate's strings" in r for r in rows)

    screen._fpos = screen._fields().index("task")
    screen.handle_key(Key(RIGHT))
    assert screen._task_text == TASKS[0]
    rows = _rows(screen)
    assert any("‹ 1/3 ›" in r for r in rows)
    assert not any("not one of" in r for r in rows)   # warning clears on a valid pick
    screen.handle_key(Key(LEFT))
    assert screen._task_text == TASKS[-1]   # wraps backward past the first entry

    # the pick survives reopening the form
    assert EvalScreen(None, screen.ctx)._task_text == screen._task_text


def test_blank_task_is_never_flagged(tmp_path):
    """Blank means "use the saved yaml default", which is a legitimate choice."""
    screen = EvalScreen(None, _ctx(tmp_path, _dataset(tmp_path / "datasets", "plate")))
    screen._task_text = ""
    rows = _rows(screen)
    assert any("(saved default)" in r for r in rows)
    assert not any("not one of" in r for r in rows)


class _PickerApp:
    """Answers whatever the next modal is with `answer` (PolicyPicker returns a path)."""

    def __init__(self, answer=None) -> None:  # noqa: ANN001
        self.answer = answer

    async def run_modal(self, modal):  # noqa: ANN001
        return self.answer


def test_picking_a_base_pins_it_against_checkpoint_switches(tmp_path, monkeypatch):
    parent = tmp_path / "datasets"
    yaml_base = _dataset(parent, "yaml-default", ["yaml string"])
    _dataset(parent, "lekiwi-demo")
    picked = _dataset(parent, "hand-picked", ["hand picked string"])
    ckpt = _checkpoint(tmp_path, "ft", trained_on="lekiwi-demo")

    app = _PickerApp()
    screen = EvalScreen(app, _ctx(tmp_path, yaml_base))

    async def _pick(app_, doc, extra=None, *, title):  # noqa: ANN001
        return "local/hand-picked", str(picked)

    monkeypatch.setattr("lekiwi_tui.dispatch.pick_dataset", _pick)
    screen._fpos = screen._fields().index("base")
    action = screen.handle_key(Key(ENTER))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())

    assert screen._task_choices.name == "hand-picked" and screen._base_pinned
    assert screen._base_source == "pinned"
    assert screen._task_text == "hand picked string"  # a stale string snaps to the new base

    # A real checkpoint switch through _pick_policy must NOT move a pinned base.
    app.answer = str(ckpt)
    asyncio.run(screen._pick_policy())
    assert screen._policy == str(ckpt)
    assert screen._task_choices.name == "hand-picked"


def test_an_unpinned_base_follows_the_new_checkpoint(tmp_path):
    parent = tmp_path / "datasets"
    yaml_base = _dataset(parent, "yaml-default", ["yaml string"])
    _dataset(parent, "lekiwi-demo")
    ctx = _ctx(tmp_path, yaml_base)
    # Start on a checkpoint that records no training dataset, so the base really does
    # begin at the yaml fallback (an empty POLICY_PATH auto-discovers the newest
    # checkpoint under POLICY_ROOT, which would already derive one).
    ctx.cfg.values["POLICY_PATH"] = str(_checkpoint(tmp_path, "bare"))
    screen = EvalScreen(_PickerApp(), ctx)
    assert screen._base_source == "yaml default"

    ckpt = _checkpoint(tmp_path, "ft", trained_on="lekiwi-demo")
    screen.app.answer = str(ckpt)
    asyncio.run(screen._pick_policy())        # the real path, not a re-implementation
    assert screen._policy == str(ckpt)
    assert screen._task_choices.name == "lekiwi-demo"
    assert screen._base_source == "from checkpoint"
    assert screen._task_choices.tasks() == TASKS
