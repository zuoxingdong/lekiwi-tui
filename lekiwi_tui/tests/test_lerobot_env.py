"""The lerobot floor: version parsing, the four verdicts, and the status-card cell.

The verdict that matters most is PRERELEASE, because it is the one a version comparison
alone gets wrong: a checkout of a released tag reports the release version while missing
fields that release shipped, and emitting a flag against it fails inside draccus.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import lekiwi_tui.lerobot_env as le
from lekiwi_tui.screens.menu import MenuScreen

from conftest import make_ctx


def _pkg(tmp_path: Path, *, marker: bool, with_file: bool = True) -> Path:
    """A fake installed lerobot tree: only the file the marker lives in matters."""
    pkg = tmp_path / "lerobot"
    (pkg / "configs").mkdir(parents=True, exist_ok=True)
    if with_file:
        body = "class DatasetRecordConfig:\n    fps: int = 30\n"
        if marker:
            body += "    no_stamp: bool = False\n"
        (pkg / "configs" / "dataset.py").write_text(body)
    return pkg


# ── version parsing ───────────────────────────────────────────────────────────


def test_version_tuple_ignores_dev_and_rc_suffixes():
    assert le.version_tuple("0.6.1") == (0, 6, 1)
    assert le.version_tuple("0.6.1.dev3") == (0, 6, 1)
    assert le.version_tuple("0.7.0rc1") == (0, 7, 0)
    assert le.version_tuple("0.6.1+local.7") == (0, 6, 1)
    assert le.version_tuple("weird") == ()


def test_the_floor_orders_as_expected():
    assert le.version_tuple("0.5.1") < le.FLOOR <= le.version_tuple("0.6.1")
    assert le.FLOOR < le.version_tuple("0.7.0")


# ── the four verdicts ─────────────────────────────────────────────────────────


def test_a_real_floor_release_is_ok(tmp_path):
    state, note = le._classify("0.6.1", _pkg(tmp_path, marker=True))
    assert (state, note) == (le.OK, "")


def test_a_checkout_claiming_the_floor_without_its_fields_is_flagged(tmp_path):
    """The measured case: a 0.6.1 tree with no no_stamp. Version says fine, it is not."""
    state, note = le._classify("0.6.1", _pkg(tmp_path, marker=False))
    assert state == le.PRERELEASE
    assert le.FLOOR_MARKER[0] in note


def test_an_older_release_is_too_old(tmp_path):
    state, note = le._classify("0.5.1", _pkg(tmp_path, marker=True))
    assert state == le.TOO_OLD and "0.6.1" in note


def test_no_lerobot_at_all(tmp_path):
    state, note = le._classify(None, None)
    assert state == le.MISSING and "0.6.1" in note


def test_installed_but_unversioned_is_flagged_not_fatal(tmp_path):
    state, _ = le._classify(None, _pkg(tmp_path, marker=True))
    assert state == le.PRERELEASE


def test_declares_is_false_when_the_module_is_absent(tmp_path):
    pkg = _pkg(tmp_path, marker=True, with_file=False)
    assert not le.declares("no_stamp", "configs/dataset.py", pkg_dir=pkg)
    assert not le.declares("no_stamp", "configs/dataset.py", pkg_dir=tmp_path / "nope")


def test_the_declared_floor_matches_the_packaging_extra():
    """Two places state the floor; a reader who trusts one must not be misled."""
    pyproject = (Path(le.__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert f'"lerobot>={le.floor_text()}"' in pyproject


# ── the status-card cell ──────────────────────────────────────────────────────


def _fake_status(monkeypatch, state, version):
    monkeypatch.setattr(le, "status",
                        lambda: le.LerobotEnv(state=state, version=version, path="/p",
                                              note="needs 0.6.1+" if state != le.OK else ""))


def test_summary_is_a_bare_version_when_all_is_well(monkeypatch):
    _fake_status(monkeypatch, le.OK, "0.6.1")
    assert le.summary() == ("0.6.1", "", "ok")


def test_summary_marks_a_prerelease_quietly(monkeypatch):
    """Worth knowing (it decides which flags the launchers may pass) but not broken,
    so it must not read like a failure on the card."""
    _fake_status(monkeypatch, le.PRERELEASE, "0.6.1")
    assert le.summary() == ("0.6.1", " · pre", "note")


def test_summary_warns_only_when_a_launch_would_fail(monkeypatch):
    _fake_status(monkeypatch, le.TOO_OLD, "0.5.1")
    value, suffix, level = le.summary()
    assert (value, level) == ("0.5.1", "warn") and "0.6.1" in suffix

    _fake_status(monkeypatch, le.MISSING, None)
    assert le.summary() == ("not found", "", "warn")


def test_the_status_card_shows_lerobot_and_marks_a_problem(monkeypatch):
    """The whole point is that it is visible before you launch anything."""
    def card(*summary):
        monkeypatch.setattr(le, "summary", lambda: summary)
        return "".join(sp.content for sp in
                       MenuScreen(MagicMock(), make_ctx(gpu_name="RTX 4090"))._identity_spans())

    assert "lerobot 0.6.1" in card("0.6.1", "", "ok")
    assert "⚠" not in card("0.6.1", "", "ok")
    # a pre-release is a footnote, not an alarm
    assert "lerobot 0.6.1 · pre" in card("0.6.1", " · pre", "note")
    assert "⚠" not in card("0.6.1", " · pre", "note")
    # too old is an alarm
    assert "lerobot ⚠ 0.5.1 · needs 0.6.1+" in card("0.5.1", " · needs 0.6.1+", "warn")
