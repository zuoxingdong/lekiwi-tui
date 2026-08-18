"""The two policy forms share their configuration half.

Run policy and DAgger are the same ``lerobot-rollout`` invocation with a different
strategy, so everything about HOW the policy runs is configured once and remembered
once. These tests pin the two user-visible consequences: settings carry across the two
screens, and `c` on Run policy opens DAgger already configured — the natural next step
the moment a rollout starts failing.

They also pin the boundary: the rows that belong to only one screen must NOT leak into
the other's memory, or a correction target would silently follow you into a form that
cannot use it.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.framework.events import Key
from lekiwi_tui.framework.screen import Invoke
from lekiwi_tui.screens.dagger import DaggerScreen
from lekiwi_tui.screens.eval import EvalScreen
from lekiwi_tui.screens.policy_form import SHARED_STATE_KEY, PolicyFormScreen

TASKS = ["Pick up the red cube and place it in the right box.",
         "Pick up the red cube and place it in the left box."]


def _dataset(parent: Path, name: str) -> Path:
    root = parent / name
    (root / "meta").mkdir(parents=True)
    pq.write_table(pa.table({"task_index": list(range(len(TASKS))), "task": TASKS}),
                   root / "meta" / "tasks.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 12}))
    return root


def _checkpoint(tmp_path: Path, name: str) -> Path:
    ckpt = tmp_path / "models" / name / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text(json.dumps({"type": "smolvla", "input_features": {}}))
    (ckpt / "model.safetensors").write_text("")
    return ckpt


def _ctx(tmp_path: Path) -> Context:
    (tmp_path / "models").mkdir(exist_ok=True)
    base = _dataset(tmp_path / "datasets", "lekiwi-demo")
    return Context(
        cfg=Config(values={"POLICY_ROOT": str(tmp_path / "models"), "POLICY_PATH": "",
                           "INFERENCE": "sync", "EXECUTION_HORIZON": "20",
                           "DISPLAY_DATA": "off", "LEKIWI_HOST": "lekiwi"}),
        doc={"record": {"dataset": {"root": str(base), "repo_id": "local/lekiwi-demo"}},
             "rollout": {"task": "", "rename_map": {}},
             "dagger": {"task": "", "rename_map": {}, "strategy": {"type": "dagger"},
                        "teleop": {"type": "lekiwi_pincopen_leader"}}},
        gpu_name="", is_tty=True,
    )


# ── the settings carry across ─────────────────────────────────────────────────


def test_configuration_carries_from_run_policy_to_dagger(tmp_path):
    ctx = _ctx(tmp_path)
    ckpt = _checkpoint(tmp_path, "ft")

    ev = EvalScreen(None, ctx)
    ev._policy = str(ckpt)
    ev._task_text = TASKS[1]
    ev._backend = "rtc"
    ev._cam_mode = "native"
    ev._steps.set_text("12")
    ev._flow.set_text("8")
    ev._extra_text = "--policy.vlm_dtype=bfloat16"
    ev._remember()

    dg = DaggerScreen(None, ctx)
    assert dg._policy == str(ckpt)          # the checkpoint you were just watching
    assert dg._task_text == TASKS[1]        # on the instruction you were watching it on
    assert dg._backend == "rtc"
    assert dg._cam_mode == "native"
    assert dg._steps.value == 12 and dg._flow.value == 8
    assert dg._extra_text == "--policy.vlm_dtype=bfloat16"


def test_it_carries_back_the_other_way_too(tmp_path):
    ctx = _ctx(tmp_path)
    ckpt = _checkpoint(tmp_path, "ft")
    dg = DaggerScreen(None, ctx)
    dg._policy = str(ckpt)
    dg._task_text = TASKS[0]
    dg._remember()

    ev = EvalScreen(None, ctx)
    assert ev._policy == str(ckpt) and ev._task_text == TASKS[0]


def test_each_screens_own_rows_stay_its_own(tmp_path):
    """A corrections target has no meaning in Run policy, so it must not follow you
    there — only the shared half is shared."""
    ctx = _ctx(tmp_path)
    dg = DaggerScreen(None, ctx)
    dg._target.set_text("25")
    dg._advanced = True
    dg._record_all = True
    dg._remember()

    shared = ctx.ui_state[SHARED_STATE_KEY]
    assert "target" not in shared and "record_all" not in shared
    assert ctx.ui_state[DaggerScreen.STATE_KEY]["target"] == 25

    # and they survive reopening DAgger itself
    again = DaggerScreen(None, ctx)
    assert again._target.value == 25 and again._advanced and again._record_all


# ── the handoff ───────────────────────────────────────────────────────────────


class _PushApp:
    def __init__(self) -> None:
        self.pushed = None

    def push(self, screen):  # noqa: ANN001
        self.pushed = screen


def test_c_on_run_policy_opens_dagger_already_configured(tmp_path):
    ctx = _ctx(tmp_path)
    ckpt = _checkpoint(tmp_path, "ft")
    app = _PushApp()
    ev = EvalScreen(app, ctx)
    ev._policy = str(ckpt)
    ev._task_text = TASKS[0]
    ev._backend = "rtc"

    action = ev.handle_key(Key("c"))
    assert isinstance(action, Invoke)
    asyncio.run(action.thunk())

    assert isinstance(app.pushed, DaggerScreen)
    assert app.pushed._policy == str(ckpt)      # no re-picking the checkpoint
    assert app.pushed._task_text == TASKS[0]    # no re-picking the task
    assert app.pushed._backend == "rtc"


def test_dagger_does_not_hand_back_with_c(tmp_path):
    """`c` is Run policy's shortcut; in DAgger the same key must stay free (it is a
    plain character, and a stray screen push mid-form would be surprising)."""
    ctx = _ctx(tmp_path)
    dg = DaggerScreen(_PushApp(), ctx)
    assert dg.handle_key(Key("c")) is not None  # handled as "nothing", not a push
    assert dg.app.pushed is None


# ── the shape of the split ────────────────────────────────────────────────────


def test_both_screens_are_policy_forms_and_agree_on_the_shared_rows(tmp_path):
    ctx = _ctx(tmp_path)
    ev, dg = EvalScreen(None, ctx), DaggerScreen(None, ctx)
    assert isinstance(ev, PolicyFormScreen) and isinstance(dg, PolicyFormScreen)
    # the shared half is identical and in the same order on both
    assert ev._common_fields() == dg._common_fields()
    # Run policy's rows are a strict SUBSET of DAgger's — which is the whole reason the
    # config half is shared. (DAgger folds duration/display/extra behind Advanced, so
    # compare with the fold open.)
    dg._advanced = True
    assert set(ev._fields()) < set(dg._fields())
    assert set(dg._fields()) - set(ev._fields()) == {"target", "advanced", "mode", "input"}


def test_screens_read_their_own_yaml_section(tmp_path):
    """Defaults come from each screen's own block, so a dagger-only setting (e.g. a
    different camera map while testing) cannot leak into Run policy."""
    ctx = _ctx(tmp_path)
    ctx.doc["rollout"]["task"] = "rollout string"
    ctx.doc["dagger"]["task"] = "dagger string"
    assert EvalScreen(None, ctx)._task_text == "rollout string"
    ctx.ui_state.clear()  # a remembered task would win over both
    assert DaggerScreen(None, ctx)._task_text == "dagger string"
