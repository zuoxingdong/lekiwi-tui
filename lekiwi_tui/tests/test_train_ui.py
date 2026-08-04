"""The redesigned train page: step/loss parsing (incl. the K-suffix fix), the
telemetry view-model, and the grouped form."""
from __future__ import annotations

import time

from lekiwi_tui.screens.train import TrainScreen, _fmt_k, parse_step
from conftest import make_ctx


def _train_screen(ui_state=None):

    ctx = make_ctx(gpu_name="RTX 2050", ui_state=ui_state or {})
    return TrainScreen(None, ctx)


def test_parse_step_honors_big_number_suffixes():
    # lerobot logs `step:12K` via format_big_number — 12 THOUSAND, not step 12.
    assert parse_step("INFO step:12K smpl:99K loss:0.041") == 12000
    assert parse_step("step:1.2M x") == 1_200_000
    assert parse_step("Checkpoint policy after step 12400") == 12400
    assert parse_step("no step here") is None
    assert _fmt_k(12400) == "12.4k" and _fmt_k(20000) == "20k" and _fmt_k(800) == "800"


def test_train_telemetry_view_model():
    scr = _train_screen()
    scr._target_steps = 20000
    assert scr._telemetry_lines(90) == []          # silent until the first tracker line
    scr._rate_t0 = (time.monotonic() - 1000, 2000)  # fake an earlier sample for rate/eta
    for ln in (
        "INFO step:4K smpl:32K loss:0.120 grdn:1.021 lr:1.0e-05",
        "INFO Checkpoint policy after step 5000",
        "INFO step:12K smpl:99K loss:0.041 grdn:0.821 lr:1.0e-05",
    ):
        scr._on_line(ln)
    tele = ["".join(sp.content for sp in tl.spans) for tl in scr._telemetry_lines(90)]
    assert "12k/20k" in tele[0] and "last ckpt 5k" in tele[0] and "eta" in tele[0]
    assert "0.041" in tele[1] and "grdn 0.821" in tele[1]
    assert scr._loss_hist == [0.120, 0.041]
    assert scr._rate_sps > 0


def test_train_form_view_model_groups_plan_and_result():
    scr = _train_screen(ui_state={"train_rate_sps": 1.6})
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(95))
    assert "RUN" in body and "OPTIMIZATION" in body
    assert "new run" in body                       # run-status on the name row
    # cost-estimate plan: steps · batch · dataset · ETA from the cached rate
    assert "20k steps" in body and "~3.5 h on RTX 2050" in body
    # post-run result line renders once set
    scr._result = "✓ finished · reached step 20k · loss 0.028 · last ckpt 20k"
    body = "\n".join("".join(sp.content for sp in ln.spans)
                     for ln in scr._body_lines(95))
    assert "✓ finished" in body and "Run policy" in body
