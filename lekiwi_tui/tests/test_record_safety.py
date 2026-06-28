from __future__ import annotations

from lekiwi_tui.screens.record import safe_delete_target, valid_dataset_name


def test_dataset_name_rejects_dot_directories():
    assert valid_dataset_name("good_dataset-01")
    assert not valid_dataset_name(".")
    assert not valid_dataset_name("..")
    assert not valid_dataset_name("../outside")


def test_safe_delete_target_allows_child_under_dataset_parent(tmp_path):
    parent = tmp_path / "datasets"
    target = parent / "partial_run"

    result = safe_delete_target(target, parent)

    assert result.ok
    assert result.path == target.resolve(strict=False)


def test_safe_delete_target_rejects_parent_and_outside_paths(tmp_path):
    parent = tmp_path / "datasets"

    parent_result = safe_delete_target(parent, parent)
    outside_result = safe_delete_target(parent / ".." / "other", parent)

    assert not parent_result.ok
    assert "parent" in parent_result.reason
    assert not outside_result.ok
    assert "outside" in outside_result.reason
