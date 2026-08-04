"""Shared fixtures: the real-config ctx stub every screen test was hand-rolling."""
from __future__ import annotations

import types

import pytest


def make_ctx(gpu_name: str = "", ui_state: dict | None = None):
    """A SimpleNamespace ctx over the real lekiwi.yaml (what 8 tests duplicated)."""
    from lekiwi_tui import CFG_FILE
    from lekiwi_tui.config import Config, load_yaml

    return types.SimpleNamespace(cfg=Config.load(CFG_FILE), doc=load_yaml(),
                                 gpu_name=gpu_name, ui_state=ui_state or {})


@pytest.fixture
def ctx():
    return make_ctx()
