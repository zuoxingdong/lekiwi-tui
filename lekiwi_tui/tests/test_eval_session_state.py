from __future__ import annotations

import asyncio

import pyratatui as pr

from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.screens.eval import EvalScreen


def _ctx(tmp_path) -> Context:  # noqa: ANN001
    root = tmp_path / "models"
    (root / "policy-a").mkdir(parents=True)
    return Context(
        cfg=Config(values={
            "POLICY_ROOT": str(root),
            "POLICY_PATH": "policy-a",
            "INFERENCE": "sync",
            "EXECUTION_HORIZON": "20",
            "DISPLAY_DATA": "off",
        }),
        doc={"rollout": {"task": "pick up the cube"}},
        gpu_name="",
        is_tty=True,
    )


class _PromptApp:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    async def run_modal(self, modal):  # noqa: ANN001
        return self.answer


def test_eval_screen_remembers_form_values_for_the_session(tmp_path):
    ctx = _ctx(tmp_path)
    first = EvalScreen(None, ctx)

    first._policy = "/tmp/custom-policy"
    first._task_text = "place the block in the tray"
    first._backend = "rtc"
    first._exec.set_text("37")
    first._dur.set_text("45")
    first._show = True
    first._remember()

    reopened = EvalScreen(None, ctx)

    assert reopened._policy == "/tmp/custom-policy"
    assert reopened._task_text == "place the block in the tray"
    assert reopened._backend == "rtc"
    assert reopened._exec.value == 37
    assert reopened._dur.value == 45
    assert reopened._show is True
    assert ctx.doc["rollout"]["task"] == "pick up the cube"
    assert ctx.cfg["INFERENCE"] == "sync"


def test_eval_task_prompt_updates_session_memory(tmp_path):
    ctx = _ctx(tmp_path)
    screen = EvalScreen(_PromptApp("sort the red cube"), ctx)

    asyncio.run(screen._edit_task())

    reopened = EvalScreen(None, ctx)
    assert reopened._task_text == "sort the red cube"


def test_eval_screen_shortens_policy_row_but_keeps_summary_detail(tmp_path):
    rel = (
        "lekiwi_policy/"
        "checkpoints/latest/pretrained_model"
    )
    root = tmp_path / "models"
    (root / rel).mkdir(parents=True)
    ctx = Context(
        cfg=Config(values={
            "POLICY_ROOT": str(root),
            "POLICY_PATH": rel,
            "INFERENCE": "sync",
            "EXECUTION_HORIZON": "20",
            "DISPLAY_DATA": "off",
        }),
        doc={"rollout": {"task": "Pick up the object and place it in the tray"}},
        gpu_name="CUDA",
        is_tty=True,
    )
    screen = EvalScreen(None, ctx)

    policy_row = screen._value("policy", width=42)

    assert len(policy_row) <= 42
    assert "pretrained_model" in policy_row
    assert policy_row.endswith("default")
    assert rel in screen._summary_value("policy")

    class Frame:
        def __init__(self) -> None:
            self.area = pr.Rect(0, 0, 120, 34)
            self.n = 0

        def render_widget(self, widget, rect):  # noqa: ANN001
            self.n += 1

    frame = Frame()
    screen.draw(frame, frame.area)
    assert frame.n >= 6
