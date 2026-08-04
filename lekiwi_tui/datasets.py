"""Dataset helpers — replaces bash record_root / dataset_repo_id / dataset_present /
dataset_episodes / args_have_resume. Behavior ported verbatim from lekiwi.sh.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from collections.abc import Sequence
from pathlib import Path

from . import CFG_FILE, ROOT
from .config import cfg_get, dump_yaml_rt, load_yaml_rt

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def record_root(doc: dict | None = None, extra: Sequence[str] = ()) -> str:
    """Where will this record land? A --dataset.root= override in `extra` wins;
    otherwise `record.dataset.root` from the yaml; otherwise the documented default.
    The path is left as-is (relative resolves against cwd, which is ROOT). Bash
    record_root (402-407)."""
    for a in extra:
        if a.startswith("--dataset.root="):
            return a[len("--dataset.root="):]
    r = cfg_get("record.dataset.root", doc=doc)
    return str(r) if r else "../datasets/lekiwi_dataset"


def dataset_repo_id(doc: dict | None = None) -> str:
    """The dataset repo_id (HF namespace/name) from the record yaml; used by `view`
    to point lerobot-dataset-viz at the right dataset. Bash dataset_repo_id (410-414)."""
    r = cfg_get("record.dataset.repo_id", doc=doc)
    return str(r) if r else "local/lekiwi_dataset"


def dataset_present(root: str | Path) -> bool:
    """True on a NON-EMPTY directory, not a valid info.json: a crashed/partial
    recording leaves a dir lerobot still refuses to overwrite, and that's exactly
    when Delete is most useful. Bash dataset_present (418)."""
    p = Path(root)
    return p.is_dir() and any(p.iterdir())


def discover_datasets(parent: str | Path) -> list[tuple[str, str, str]]:
    """Present datasets directly under *parent*, NEWEST FIRST (mtime desc, name tie-break —
    same convention as policies.discover_policies). Each entry is ``(name, root, episodes)``
    where ``root`` is the dir path AS-IS (kept RELATIVE — the app's cwd=ROOT convention means
    "../datasets/foo" must flow unchanged into the lerobot argv; do NOT resolve()) and
    ``episodes`` is dataset_episodes()'s string ("N" / "?"). A dir counts only when
    dataset_present() (non-empty); a missing/!dir *parent* → []. Used by the DatasetPicker
    (replay/view) to let the user pick which recording to play back / browse."""
    p = Path(parent)
    if not p.is_dir():
        return []
    dirs = [d for d in p.iterdir() if d.is_dir() and dataset_present(d)]
    dirs.sort(key=lambda d: (d.stat().st_mtime, d.name), reverse=True)
    return [(d.name, str(d), dataset_episodes(d)) for d in dirs]


def dataset_episodes(root: str | Path) -> str:
    """Best-effort episode count; "?" when meta/info.json is missing or unparseable.
    Bash dataset_episodes (421-424) read .total_episodes via jq."""
    info = Path(root) / "meta" / "info.json"
    try:
        n = json.loads(info.read_text()).get("total_episodes")
    except Exception:
        return "?"
    return "?" if n is None else str(n)


def args_have_resume(extra: Sequence[str]) -> bool:
    """True if --resume or --resume=… is already in the passthrough args. Bash
    args_have_resume (425)."""
    return any(a == "--resume" or a.startswith("--resume=") for a in extra)


# ── dataset stats (record screen panel) ───────────────────────────────────────
_STATS_CACHE: dict[str, tuple[float, "dict | None"]] = {}
_STATS_TTL = 5.0


def dataset_stats_parts(root: str | Path) -> "dict | None":
    """One informative line about the dataset at *root* for the record panel:
    "34 episodes · 21.3 min · 812 MB · updated 14:02" — or "" when nothing exists yet.

    Called from ``draw`` every frame, so results are cached ~5s per root (the disk walk
    over a video dataset is NOT per-frame material). Best-effort: any unreadable piece
    degrades to '?' rather than raising."""
    import time as _time

    key = str(root)
    now = _time.monotonic()
    hit = _STATS_CACHE.get(key)
    if hit and (now - hit[0]) < _STATS_TTL:
        return hit[1]

    parts = None
    p = Path(root)
    if dataset_present(p):
        eps = dataset_episodes(p)
        minutes = "?"
        try:
            info = json.loads((p / "meta" / "info.json").read_text())
            frames, fps = info.get("total_frames"), info.get("fps")
            if isinstance(frames, (int, float)) and isinstance(fps, (int, float)) and fps:
                minutes = f"{frames / fps / 60:.1f}"
        except Exception:
            pass
        size_b = 0
        newest = 0.0
        try:
            for dirpath, _dirs, files in os.walk(p):
                for f in files:
                    try:
                        st = os.stat(os.path.join(dirpath, f))
                    except OSError:
                        continue
                    size_b += st.st_size
                    newest = max(newest, st.st_mtime)
        except OSError:
            pass
        size = (f"{size_b / 1e9:.1f} GB" if size_b >= 1e9
                else f"{size_b / 1e6:.0f} MB" if size_b else "?")
        updated = (_time.strftime("%H:%M", _time.localtime(newest)) if newest else "?")
        parts = {"episodes": eps, "minutes": minutes, "size": size, "updated": updated}
    _STATS_CACHE[key] = (now, parts)
    return parts


def dataset_stats(root: str | Path) -> str:
    """The one-line rendering of :func:`dataset_stats_parts` ("" when no dataset):
    "34 episodes · 21.3 min · 812 MB · updated 14:02"."""
    parts = dataset_stats_parts(root)
    if not parts:
        return ""
    return (f"{parts['episodes']} episodes · {parts['minutes']} min · "
            f"{parts['size']} · updated {parts['updated']}")


# ── the record-dataset domain (moved from screens/record.py: screens must not
#    import from screens; this module owns dataset logic) ────────────────────
@dataclass(frozen=True)
class DeleteTarget:
    ok: bool
    path: Path
    reason: str = ""


def dataset_defaults(doc: dict | None = None) -> dict:
    """Per-run record defaults from the yaml `record` block (ported verbatim)."""
    repo_id = str(cfg_get("record.dataset.repo_id", doc=doc) or "local/lekiwi_dataset")
    root = str(cfg_get("record.dataset.root", doc=doc) or "../../datasets/lekiwi_dataset")
    ns = repo_id.rsplit("/", 1)[0] if "/" in repo_id else "local"
    parent = str(Path(root).parent) if str(Path(root).parent) != "." else ""

    def _int(key: str, fallback: int) -> int:
        v = cfg_get(f"record.dataset.{key}", doc=doc)
        if isinstance(v, bool):
            return fallback
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return fallback

    def _bool(key: str, fallback: bool) -> bool:
        v = cfg_get(f"record.dataset.{key}", doc=doc)
        if v is None:
            return fallback
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("1", "true", "on", "yes")

    return {
        "name": Path(root).name or "lekiwi_dataset",
        "ns": ns, "parent": parent,
        "task": str(cfg_get("record.dataset.single_task", doc=doc) or ""),
        "episodes": _int("num_episodes", 5), "ep_time": _int("episode_time_s", 40),
        "reset_time": _int("reset_time_s", 5), "fps": _int("fps", 30),
        "img_threads": _int("num_image_writer_threads_per_camera", 3),
        "display": bool(cfg_get("record.display_data", doc=doc)),
        "streaming": _bool("streaming_encoding", True),
        "resume": bool(cfg_get("record.resume", doc=doc)),
    }


def resolve_repo_root(name: str, ns: str, parent: str) -> tuple[str, str]:
    """A dataset name -> (repo_id, root), preserving the yaml namespace + parent dir."""
    return f"{ns}/{name}", (f"{parent}/{name}" if parent else name)


def valid_dataset_name(name: str) -> bool:
    """Return True for a single safe dataset folder name."""
    return bool(_NAME_RE.fullmatch(name)) and name not in (".", "..")


def workspace_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return ROOT / p


def safe_delete_target(root: str | Path, parent: str | Path) -> DeleteTarget:
    """Validate a record delete target against its configured dataset parent."""
    raw = str(root).strip()
    if raw in ("", "/", os.path.expanduser("~")):
        return DeleteTarget(False, workspace_path(root), "empty, root, or home path")
    target = workspace_path(root).resolve(strict=False)
    parent_path = workspace_path(parent or ".").resolve(strict=False)
    if target == parent_path:
        return DeleteTarget(False, target, "target is the dataset parent")
    if target.name in ("", ".", ".."):
        return DeleteTarget(False, target, "target has no safe folder name")
    try:
        target.relative_to(parent_path)
    except ValueError:
        return DeleteTarget(False, target, f"target is outside dataset parent {parent_path}")
    return DeleteTarget(True, target)


def persist_record_defaults(*, repo_id: str, root: str, task: str, episodes: int,
                            ep_time: int, reset_time: int, fps: int, img_threads: int,
                            display: bool, streaming: bool, cfg_path: Path = CFG_FILE) -> None:
    """Write the form values back into lekiwi.yaml (lossless ruamel round-trip). Ported
    verbatim from the Textual app (anchors/merges/comments survive; shared &task anchor)."""
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    doc = load_yaml_rt(cfg_path)
    rd = doc["record"]["dataset"]
    rd["num_episodes"] = int(episodes)
    rd["episode_time_s"] = int(ep_time)
    rd["reset_time_s"] = int(reset_time)
    rd["fps"] = int(fps)
    rd["num_image_writer_threads_per_camera"] = int(img_threads)
    rd["streaming_encoding"] = bool(streaming)
    doc["record"]["display_data"] = bool(display)
    doc["_dataset"]["repo_id"] = repo_id
    doc["_dataset"]["root"] = root
    ns = DoubleQuotedScalarString(task)
    ns.yaml_set_anchor("task", always_dump=True)
    doc["_task"] = ns
    doc["record"]["dataset"]["single_task"] = ns
    doc["rollout"]["task"] = ns
    dump_yaml_rt(doc, cfg_path)


