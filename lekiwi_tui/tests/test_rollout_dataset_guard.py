"""The rollout strategy/dataset agreement check.

lerobot builds a rollout dataset only for a non-`base` strategy, and then refuses any
name whose basename is not `rollout_*`. It refuses AFTER connecting the robot, so the
form is a much better place to say it than the launch is.
"""
from __future__ import annotations

from lekiwi_tui.preflight import eval_issues, rollout_dataset_issues

from conftest import make_ctx


def _doc(strategy: str = "base", repo_id: str | None = None, resume: bool | None = None) -> dict:
    rollout: dict = {"strategy": {"type": strategy}}
    if repo_id is not None:
        rollout["dataset"] = {"repo_id": repo_id, "root": "../../datasets/x"}
    if resume is not None:
        rollout["resume"] = resume
    return {"rollout": rollout}


def _texts(doc: dict) -> str:
    return " | ".join(i.text for i in rollout_dataset_issues(doc))


# ── the configurations that are fine ──────────────────────────────────────────


def test_the_default_shape_is_silent():
    """base + no dataset: autonomous eval, nothing recorded. The common case."""
    assert rollout_dataset_issues(_doc()) == []


def test_a_properly_named_recording_dataset_is_silent():
    assert rollout_dataset_issues(_doc("episodic", "local/rollout_plate")) == []
    assert rollout_dataset_issues(_doc("dagger", "zuoxingdong/rollout_shelf")) == []


def test_resume_skips_the_name_check_like_lerobot_does():
    """Resume reuses an existing dataset, so lerobot never re-validates the name."""
    assert rollout_dataset_issues(_doc("episodic", "local/plate", resume=True)) == []


def test_a_missing_rollout_block_says_nothing():
    assert rollout_dataset_issues({}) == []
    assert rollout_dataset_issues(None) == []


# ── the configurations that cost a startup ────────────────────────────────────


def test_a_recording_strategy_without_a_dataset_is_flagged():
    text = _texts(_doc("episodic"))
    assert "records episodes" in text and "save nothing" in text


def test_a_wrongly_named_recording_dataset_names_the_rule_and_the_cost():
    text = _texts(_doc("sentry", "local/plate-run7"))
    assert "rollout_<name>" in text
    assert "local/plate-run7" in text
    assert "after the robot is connected" in text, "the cost is the point of catching it here"


def test_the_prefix_is_checked_on_the_basename_not_the_namespace():
    """`rollout_x/plate` must NOT pass: lerobot splits the namespace off first."""
    assert _texts(_doc("episodic", "rollout_x/plate")) != ""
    assert rollout_dataset_issues(_doc("episodic", "rollout_x/rollout_plate")) == []


def test_base_with_a_dataset_is_flagged_as_ignored():
    text = _texts(_doc("base", "local/rollout_plate"))
    assert "records nothing" in text and "ignored" in text


# ── wiring ────────────────────────────────────────────────────────────────────


def test_eval_issues_carries_the_check(monkeypatch):
    import lekiwi_tui.preflight as pf

    monkeypatch.setattr(pf, "robot_runtime_issues", lambda *a, **k: [])
    ctx = make_ctx(gpu_name="RTX 4090")
    ctx.doc = _doc("episodic", "local/plate-run7")
    texts = [i.text for i in eval_issues(ctx, policy="/some/local/ckpt")]
    assert any("rollout_<name>" in t for t in texts)
