"""Tests for Replay + View: dataset selection AND the episode-picker type-contract fix.

Two things are pinned here:

1. DATASET SELECTION (feature): replay/view now open a DatasetPicker (discovered datasets
   under the datasets dir + a Custom path row), then the episode picker, then front the
   launcher with the CHOSEN dataset — replay via passthrough --dataset.repo_id=/--dataset.root=
   overrides (the record.sh-proven config_path-then-override mechanism), view via its
   --repo-id/--root flags. Roots stay RELATIVE ("../datasets/foo"), per the cwd=ROOT convention.

2. THE EPISODE BUG (regression): dataset_episodes() returns the count as a STRING ("24"/"?")
   but EpisodeScreen wants int|None; _ask_episode converts at the boundary. Skipping it raised
   "'>=' not supported between instances of 'str' and 'int'" on every replay/view.

Run from the lerobot env: conda run -n lekiwi pytest <thisfile> -q
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lekiwi_tui import dispatch as dispatch_mod
from lekiwi_tui.datasets import discover_datasets
from lekiwi_tui.dispatch import Dispatcher
from lekiwi_tui.framework import runner
from lekiwi_tui.framework.events import DOWN, ENTER, ESC, Key
from lekiwi_tui.framework.screen import Pop
from lekiwi_tui.screens.episode import EpisodeScreen
from lekiwi_tui.widgets import pickers
from lekiwi_tui.widgets.pickers import CUSTOM, DatasetPicker


# ── EpisodeScreen contract (the str->int episode-count fix) ───────────────────
def test_episode_screen_accepts_int_count_and_none():
    known = EpisodeScreen(None, None, title="t", repo_id="r", root="x", episodes=24)
    assert known._known is True and known._n == 24

    unknown = EpisodeScreen(None, None, title="t", repo_id="r", root="x", episodes=None)
    assert unknown._known is False and unknown._n == 0


def test_episode_screen_raises_on_raw_string_count():
    # The exact failure the dispatcher used to trigger: the screen's `episodes >= 0` guard
    # cannot compare a str. _ask_episode is responsible for converting before this point.
    with pytest.raises(TypeError):
        EpisodeScreen(None, None, title="t", repo_id="r", root="x", episodes="24")


# ── discover_datasets ─────────────────────────────────────────────────────────
def test_discover_datasets_lists_present_dirs_and_skips_empty(tmp_path):
    (tmp_path / "a" / "meta").mkdir(parents=True)
    (tmp_path / "a" / "meta" / "info.json").write_text('{"total_episodes": 3}')
    (tmp_path / "b" / "data").mkdir(parents=True)
    (tmp_path / "empty").mkdir()  # present()? no — empty dir, must be skipped

    found = discover_datasets(tmp_path)

    names = {n for n, _, _ in found}
    assert names == {"a", "b"}  # "empty" excluded
    # roots are returned AS-IS under the given parent (not resolved to absolute elsewhere)
    assert all(r.startswith(str(tmp_path)) for _, r, _ in found)
    eps = {n: e for n, _, e in found}
    assert eps["a"] == "3" and eps["b"] == "?"  # b has no info.json -> "?"


def test_discover_datasets_missing_parent_is_empty(tmp_path):
    assert discover_datasets(tmp_path / "nope") == []


# ── DatasetPicker ─────────────────────────────────────────────────────────────
def _make_ds(parent, name, n):
    (parent / name / "meta").mkdir(parents=True)
    (parent / name / "meta" / "info.json").write_text(f'{{"total_episodes": {n}}}')
    return str(parent / name)


def test_dataset_picker_preselects_default_and_appends_custom(tmp_path):
    _make_ds(tmp_path, "ds1", 5)
    default_root = _make_ds(tmp_path, "ds2", 9)

    pk = DatasetPicker(tmp_path, default_root=default_root, title="pick")

    assert pk.entries[-1][1] == CUSTOM                  # Custom row is last
    assert pk.entries[pk._sel][1] == default_root       # default pre-selected
    assert any("← default" in label for label, _ in pk.entries)


def test_dataset_picker_keys(tmp_path):
    r1 = _make_ds(tmp_path, "ds1", 5)
    r2 = _make_ds(tmp_path, "ds2", 9)
    pk = DatasetPicker(tmp_path, default_root="", title="pick")
    roots = [v for _, v in pk.entries]

    start = pk._sel
    pk.handle_key(Key(name=DOWN))
    assert pk._sel == (start + 1) % len(pk.entries)      # DOWN moves selection

    pk._sel = roots.index(r2)
    assert pk.handle_key(Key(name=ENTER)).result == r2   # Enter -> chosen root
    assert pk.handle_key(Key(name=ESC)).result is None   # Esc -> cancel
    assert r1 in roots and CUSTOM in roots


# ── full Replay/View dispatch path (dataset picker -> episode -> argv) ─────────
class _FakeApp:
    """App stand-in driving the two-modal flow. run_modal dispatches by modal TYPE:
      DatasetPicker  -> returns self.dataset_choice (a root str, CUSTOM, or None)
      PromptModalState -> returns self.custom_path (the typed custom dataset path)
      EpisodeScreen  -> feeds self.episode_keys to handle_key until it returns Pop."""

    def __init__(self, *, dataset_choice, episode_keys=(), custom_path=None):
        self.terminal = object()  # non-None -> the flow opens the real pickers
        self.dataset_choice = dataset_choice
        self.episode_keys = list(episode_keys)
        self.custom_path = custom_path
        self.built = []
        self.notes = []

    def notify(self, msg, level="info"):
        self.notes.append((level, msg))

    async def run_modal(self, modal):
        self.built.append(modal)
        cls = type(modal).__name__
        if cls == "DatasetPicker":
            return self.dataset_choice
        if cls == "PromptModalState":
            return self.custom_path
        if cls == "EpisodeScreen":
            for key in self.episode_keys:
                action = modal.handle_key(key)
                if isinstance(action, Pop):
                    return action.result
            return None
        raise AssertionError(f"unexpected modal: {cls}")

    def episode_screen(self):
        return next(m for m in self.built if type(m).__name__ == "EpisodeScreen")


@pytest.fixture
def patched(monkeypatch):
    """Controlled dataset coordinates + a captured suspend_run. discover_datasets is stubbed
    empty so the real DatasetPicker constructs without touching the filesystem (the choice is
    injected via _FakeApp.run_modal); dataset_episodes returns the STRING the helper really
    returns, so the str->int conversion is exercised end to end."""
    monkeypatch.setattr(dispatch_mod, "record_root", lambda doc, extra=(): "../datasets/lekiwi-finger")
    monkeypatch.setattr(dispatch_mod, "dataset_repo_id", lambda doc=None: "local/lekiwi-finger")
    monkeypatch.setattr(dispatch_mod, "dataset_present", lambda root: True)
    monkeypatch.setattr(dispatch_mod, "dataset_episodes", lambda root: "24")
    monkeypatch.setattr(pickers, "discover_datasets", lambda parent: [])

    captured = {}

    def fake_suspend(app, argv, **kw):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(runner, "suspend_run", fake_suspend)
    return captured


def _type_episode(s: str):
    """Keystrokes that type *s* into the episode field then press Enter."""
    return [Key(name=c) for c in s] + [Key(name=ENTER)]


def _disp(app):
    d = Dispatcher(SimpleNamespace(doc={}))
    d.bind(app)
    return d


def test_replay_uses_chosen_dataset_with_relative_root(patched):
    app = _FakeApp(dataset_choice="../datasets/lekiwi-demo", episode_keys=_type_episode("3"))
    rc = asyncio.run(_disp(app)._replay(app, []))

    assert rc == 0
    argv = patched["argv"]
    assert argv[0] == "bash" and argv[1].endswith("replay.sh")
    assert argv[argv.index("--episode") + 1] == "3"
    # chosen dataset overrides the slice; root stays RELATIVE (cwd=ROOT convention)
    assert "--dataset.repo_id=local/lekiwi-demo" in argv
    assert "--dataset.root=../datasets/lekiwi-demo" in argv
    # episode count was converted str("24") -> int before reaching the picker
    assert app.episode_screen()._known and app.episode_screen()._n == 24


def test_view_uses_chosen_dataset_with_relative_root(patched):
    app = _FakeApp(dataset_choice="../datasets/lekiwi-demo", episode_keys=_type_episode("5"))
    rc = asyncio.run(_disp(app)._view(app, []))

    assert rc == 0
    argv = patched["argv"]
    assert argv[1].endswith("view.sh")
    assert argv[argv.index("--repo-id") + 1] == "local/lekiwi-demo"
    assert argv[argv.index("--root") + 1] == "../datasets/lekiwi-demo"  # relative preserved
    assert argv[argv.index("--episode-index") + 1] == "5"


def test_custom_path_resolves_repo_id_from_basename(patched):
    app = _FakeApp(dataset_choice=CUSTOM, custom_path="/tmp/my-ds", episode_keys=_type_episode("0"))
    rc = asyncio.run(_disp(app)._replay(app, []))

    assert rc == 0
    argv = patched["argv"]
    assert "--dataset.root=/tmp/my-ds" in argv
    assert "--dataset.repo_id=local/my-ds" in argv  # ns from config + basename of the path


def test_cancel_at_dataset_picker_runs_nothing(patched):
    app = _FakeApp(dataset_choice=None)
    rc = asyncio.run(_disp(app)._replay(app, []))

    assert rc == 0
    assert "argv" not in patched  # cancelled before any suspend
    assert not any(type(m).__name__ == "EpisodeScreen" for m in app.built)  # never reached


def test_unknown_episode_count_becomes_none(patched, monkeypatch):
    monkeypatch.setattr(dispatch_mod, "dataset_episodes", lambda root: "?")
    app = _FakeApp(dataset_choice="../datasets/lekiwi-demo", episode_keys=_type_episode("0"))
    rc = asyncio.run(_disp(app)._replay(app, []))

    assert rc == 0
    assert app.episode_screen()._known is False  # "?" -> None -> unknown (no crash)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
