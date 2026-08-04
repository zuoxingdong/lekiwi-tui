"""The redesigned eval page: grouped view-model, backend row swap, always-visible
cam note, and the post-run scoreboard (label / append / tally / verdict flow)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lekiwi_tui.config import Config
from lekiwi_tui.context import Context
from lekiwi_tui.scoreboard import append_score, ckpt_label, load_scores, score_tally
from lekiwi_tui.screens.eval import EvalScreen


def _ckpt(tmp_path: Path, cams=("front", "wrist", "top")) -> Path:
    p = tmp_path / "models" / "shop_05" / "checkpoints" / "060000" / "pretrained_model"
    p.mkdir(parents=True)
    (p / "config.json").write_text(json.dumps({
        "n_action_steps": 50, "num_steps": 10,
        "input_features": {f"observation.images.{c}": {} for c in cams},
    }))
    (p / "model.safetensors").write_bytes(b"x")
    return p


def _screen(tmp_path: Path) -> EvalScreen:
    ckpt = _ckpt(tmp_path)
    ctx = Context(
        cfg=Config(values={
            "POLICY_ROOT": str(tmp_path / "models"),
            "POLICY_PATH": "auto",
            "INFERENCE": "sync",
            "EXECUTION_HORIZON": "20",
            "DISPLAY_DATA": "off",
        }),
        doc={"rollout": {"task": "pick the gum",
                         "rename_map": {"front": "camera1", "wrist": "camera2", "top": "camera3"}}},
        gpu_name="RTX 2050",
        is_tty=True,
    )
    scr = EvalScreen(None, ctx)
    scr._policy = str(ckpt)
    return scr


def test_ckpt_label_and_scoreboard_roundtrip(tmp_path):
    assert ckpt_label(str(tmp_path / "shop_05/checkpoints/060000/pretrained_model"),
                      tmp_path) == "shop_05-60k"
    assert ckpt_label("lerobot/smolvla_base", tmp_path) == "lerobot"
    for ok in (True, True, False):
        assert append_score(tmp_path, {"label": "shop_05-60k", "task": "pick gum",
                                       "success": ok})
    scores = load_scores(tmp_path)
    assert score_tally(scores, label="shop_05-60k", task="pick gum") == (2, 3)
    assert score_tally(scores, label="shop_05-60k") == (2, 3)
    assert score_tally(scores, label="other") == (0, 0)


def test_eval_body_groups_cam_note_and_backend_swap(tmp_path):
    scr = _screen(tmp_path)
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "POLICY" in body and "RUN" in body
    # the resolved camera mapping is ALWAYS visible (safety fact)
    assert "auto → " in body
    assert "Action steps" in body and "Action horizon" not in body
    scr._backend = "rtc"
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(96))
    assert "Action horizon" in body and "Action steps" not in body


def test_eval_scoreboard_lines_render_for_selected_ckpt(tmp_path):
    scr = _screen(tmp_path)
    root = scr._root
    assert scr._scoreboard_lines(96) == []          # no verdicts yet → invisible
    append_score(root, {"label": "shop_05-60k", "task": "pick the gum", "success": True})
    append_score(root, {"label": "shop_05-60k", "task": "pick the gum", "success": False})
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._scoreboard_lines(96))
    assert "SCOREBOARD" in body and "1/2" in body


def test_eval_verdict_modal_appends_score(tmp_path):
    scr = _screen(tmp_path)

    class _App:
        def __init__(self, answer):
            self.answer = answer
            self.toasts = []

        async def run_modal(self, modal):
            return self.answer

        def notify(self, msg, lvl="info"):
            self.toasts.append(msg)

    scr.app = _App("Success")
    asyncio.run(scr._ask_verdict(0))
    scr.app = _App("Failure")
    asyncio.run(scr._ask_verdict(130))              # Ctrl+C end still judged
    scr.app = _App("Success")
    asyncio.run(scr._ask_verdict(1))                # crashed run → NOT judged
    scr.app = _App("Skip — no verdict")
    asyncio.run(scr._ask_verdict(0))                # skip appends nothing

    scores = load_scores(scr._root)
    assert [e["success"] for e in scores] == [True, False]
    assert all(e["label"] == "shop_05-60k" for e in scores)
