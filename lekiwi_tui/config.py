"""Config layer — replaces bash CONFIG_SPEC / config_default / config_load /
config_parse_file / config_save / cfg_for / cfg_get / _py / .lekiwi-cache slicing.

The ops/launcher knobs (CONFIG_SPEC) now live in a single `_launcher:` mapping inside
lekiwi.yaml (the old standalone lekiwi.conf is retired). The key is `_`-prefixed, so
cfg_for never slices it into a per-command CLI config. Every value there is stored
QUOTED, so `off`/`on`/`600` load as the strings the enum/int knobs expect (an unquoted
scalar would `safe_load` to a bool/int and corrupt those knobs).

PyYAML resolves anchors and `<<:` merges natively, so the python-subprocess dance the
bash script needed (it had no in-process YAML) is gone here.

Precedence is identical to bash: env var present at launch > `_launcher` in the yaml >
built-in default. "Present" means set-ness, not truthiness: an env var exported empty
still wins over the yaml, matching bash `[ -n "${!key+x}" ]`.

Settings SAVE round-trips lekiwi.yaml via ruamel.yaml (not PyYAML): it preserves the
anchors (&robot / &cameras), the `<<:` merges, and the comments, rewriting only the
`_launcher` values (each as a SingleQuotedScalarString, so they stay quoted strings).
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import SingleQuotedScalarString

from . import CFG_CACHE, CFG_FILE, EXAMPLE_CFG_FILE, ROOT

# kind strings, mirroring the bash registry's third axis:
#   text | path | int | enum:<a>,<b>,...   (path defaults expand $HOME/$ROOT)
KIND = str


@dataclass(frozen=True)
class Field:
    """One config knob. Mirrors a bash CONFIG_SPEC "key|kind|hint" entry plus its
    config_default. `default` already has $HOME / $ROOT expanded at build time, so
    Field is the single place those expand (same contract as bash config_default)."""

    key: str
    kind: KIND
    hint: str
    default: str


# Built-in defaults (bash config_default). $HOME / $ROOT (== bash $SCRIPT_DIR) are
# expanded HERE and only here. POLICY_PATH default is "" = auto (newest under root).
_DEFAULTS: dict[str, str] = {
    "LAPTOP_ENV": "lekiwi",
    "MAMBA_ROOT": str(Path.home() / "miniforge3"),
    "LEADER_PORT": "/dev/ttyACM0",
    "LEADER_ID": "lekiwi_leader",
    "LEKIWI_HOST": "lekiwi",
    "ROBOT_ID": "lekiwi",
    "ROBOT_TYPE": "lekiwi_pincopen",
    "CONNECTION_TIME": "600",
    "CONDA_ENV": "lekiwi",
    "PI_REPO": "lekiwi/lerobot",
    "LOCAL_REPO": "",
    "LOCAL_PLUGIN": "",
    "POLICY_PATH": "",
    "POLICY_ROOT": str(ROOT.parent.parent / "models"),  # <workspace>/models; checkout at <workspace>/projects/lekiwi-tui
    "INFERENCE": "sync",
    "EXECUTION_HORIZON": "20",
    "DISPLAY_DATA": "off",
}

# CONFIG_SPEC — same keys/kinds/hints/defaults and ORDER as the bash registry. The
# hints are copied verbatim (terse bash voice; no em-dashes). Order matters: it is
# the order config_save writes lines and the Settings form renders rows.
CONFIG_SPEC: list[Field] = [
    Field("LAPTOP_ENV", "text", "local conda env with LeRobot installed", _DEFAULTS["LAPTOP_ENV"]),
    Field("MAMBA_ROOT", "path", "conda installation root", _DEFAULTS["MAMBA_ROOT"]),
    Field("LEADER_PORT", "text", "leader arm serial port for calibration", _DEFAULTS["LEADER_PORT"]),
    Field("LEADER_ID", "text", "leader arm calibration id", _DEFAULTS["LEADER_ID"]),
    Field("LEKIWI_HOST", "text", "SSH host name or IP for the Pi", _DEFAULTS["LEKIWI_HOST"]),
    Field("ROBOT_ID", "text", "robot id used by the Pi host", _DEFAULTS["ROBOT_ID"]),
    Field("ROBOT_TYPE", "enum:lekiwi_pincopen,lekiwi", "follower robot; lekiwi_pincopen is the STS3250+PincOpen plugin, lekiwi is stock", _DEFAULTS["ROBOT_TYPE"]),
    Field("CONNECTION_TIME", "int", "default host session length in seconds", _DEFAULTS["CONNECTION_TIME"]),
    Field("CONDA_ENV", "text", "Pi conda env for host and setup commands", _DEFAULTS["CONDA_ENV"]),
    Field("PI_REPO", "text", "LeRobot checkout path on the Pi, relative to Pi home", _DEFAULTS["PI_REPO"]),
    Field("LOCAL_REPO", "path", "laptop lerobot checkout shipped to the Pi; empty = sibling of this checkout", _DEFAULTS["LOCAL_REPO"]),
    Field("LOCAL_PLUGIN", "path", "laptop lerobot_robot_lekiwi_pincopen dir shipped to the Pi; empty = sibling", _DEFAULTS["LOCAL_PLUGIN"]),
    Field("POLICY_PATH", "path", "default policy checkpoint; empty selects newest under POLICY_ROOT", _DEFAULTS["POLICY_PATH"]),
    Field("POLICY_ROOT", "path", "folder scanned for policy checkpoints", _DEFAULTS["POLICY_ROOT"]),
    Field("INFERENCE", "enum:sync,rtc", "policy runner: sync per tick, rtc for smoother slow policies", _DEFAULTS["INFERENCE"]),
    Field("EXECUTION_HORIZON", "int", "RTC action horizon; keep above policy inference delay", _DEFAULTS["EXECUTION_HORIZON"]),
    Field("DISPLAY_DATA", "enum:off,on", "default Rerun display for non-interactive policy runs", _DEFAULTS["DISPLAY_DATA"]),
]

_SPEC_KEYS: list[str] = [f.key for f in CONFIG_SPEC]
_SPEC_BY_KEY: dict[str, Field] = {f.key: f for f in CONFIG_SPEC}


def is_config_key(key: str) -> bool:
    """Whitelist check (bash config_is_key) — only these keys are read from `_launcher`."""
    return key in _SPEC_BY_KEY


def collapse_home(path) -> str:  # noqa: ANN001 - str or Path
    """~-collapse an absolute path for display (the one shared implementation —
    seven screens used to hand-roll this)."""
    s = str(path)
    home = os.path.expanduser("~")
    return "~" + s[len(home):] if s.startswith(home) else s


def as_int(v, default: int) -> int:  # noqa: ANN001
    """Coerce a config value (str digits or int; bool is NOT an int here) to int."""
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v)
    return default


def resolve_workspace_path(value: str) -> str:
    """Resolve a LOCAL_REPO / LOCAL_PLUGIN style config value to an absolute path.

    Empty stays empty (the consumer's "auto: sibling of this checkout" default); a
    relative value resolves against ROOT (the workspace dir the app cd's into), so
    lekiwi.yaml can carry portable entries like '../lerobot' that mean the same thing
    no matter where a launcher script is invoked from. sync.sh applies the same rule
    in bash — keep the two in step."""
    value = str(value or "").strip()
    if not value:
        return ""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def resolve_editor() -> str:
    """The editor `e` opens: $VISUAL, else $EDITOR, else nvim if it is on PATH, else vi.

    Shared by the Settings and Robot-config screens (one impl). Mirrors bash's
    `${VISUAL:-${EDITOR:-...}}` (empty counts as unset, hence `or`), but upgrades the
    final fallback from plain vi to nvim when installed — so `e` opens nvim out of the
    box. Set $EDITOR/$VISUAL to override (those still win). Uses the `shutil` module
    attribute (not a bound `which`) so tests can monkeypatch `shutil.which`."""
    return (
        os.environ.get("VISUAL")
        or os.environ.get("EDITOR")
        or ("nvim" if shutil.which("nvim") else "vi")
    )


@dataclass
class Config:
    """Effective config. `values` holds every CONFIG_SPEC key (resolved value);
    `env_set` is the set of keys whose value came from the environment at launch
    (drives precedence AND the Settings form's [env override] provenance tag)."""

    values: dict[str, str] = field(default_factory=dict)
    env_set: set[str] = field(default_factory=set)

    @classmethod
    def load(
        cls,
        cfg_path: Path = CFG_FILE,
        env: Mapping[str, str] = os.environ,
    ) -> "Config":
        """Resolve effective config: env present at launch > `_launcher` > default.

        The knobs come from the `_launcher:` mapping in lekiwi.yaml (only whitelisted
        keys; load_yaml never sources code). Values there are already strings (the yaml
        quotes them, task 1) and are read RAW — we deliberately do NOT `str()`-coerce,
        because an unquoted `600`/`off` would have loaded as int/bool and a `str()` would
        mask that as "600"/"False" (the quoting in the yaml is the real fix). Env
        precedence is by SET-NESS (key in env), not value, so an exported-but-empty var
        still beats the yaml — bash `[ -n "${!key+x}" ]`.
        """
        # Snapshot which keys are env-set FIRST: an env value beats both yaml + default.
        env_set = {key for key in _SPEC_KEYS if key in env}
        values: dict[str, str] = {}

        # 1) `_launcher:` block (only whitelisted keys; env-set keys skipped — env wins).
        launcher = load_yaml(cfg_path).get("_launcher") or {}
        if isinstance(launcher, Mapping):
            for key, val in launcher.items():
                if not is_config_key(key) or key in env_set:
                    continue
                values[key] = val

        # 2) env-set keys take their launch value.
        for key in env_set:
            values[key] = env[key]

        # 3) built-in defaults fill whatever is still unset.
        for f in CONFIG_SPEC:
            values.setdefault(f.key, f.default)

        # Empty DISPLAY_DATA (a hand-blanked or legacy value) normalizes to off so the
        # enum knob always round-trips a real value (bash line 116).
        if not values.get("DISPLAY_DATA"):
            values["DISPLAY_DATA"] = "off"

        return cls(values=values, env_set=env_set)

    def save(self, cfg_path: Path = CFG_FILE) -> None:
        """Round-trip lekiwi.yaml via ruamel, rewriting ONLY the `_launcher` values,
        then atomically replace (tmp + os.replace).

        ruamel.yaml (round-trip mode) preserves the anchors (&robot / &cameras), the
        `<<:` merges, and the comments — a PyYAML dump would rename anchors to &id001
        and drop every comment. Each spec value is wrapped in a SingleQuotedScalarString
        so it is re-emitted QUOTED: a freshly-assigned bare `off`/`600` would reload as a
        bool/int and reintroduce the type bug. If `_launcher` is absent it is inserted at
        the top of the document (so the bootstrap shim can still find it).

        load_yaml_rt / dump_yaml_rt are the shared (one place) safe load+write for this
        file; record's persist_record_defaults goes through the same pair."""
        doc = load_yaml_rt(cfg_path)
        if doc is None:
            doc = CommentedMap()  # empty doc / missing file

        launcher = doc.get("_launcher")
        if not isinstance(launcher, Mapping):
            launcher = CommentedMap()
            doc.insert(0, "_launcher", launcher)  # near the top (after nothing = first)

        # Rewrite every spec key as a quoted scalar, in spec order (matches the form).
        for key in _SPEC_KEYS:
            launcher[key] = SingleQuotedScalarString(str(self.values[key]))

        dump_yaml_rt(doc, cfg_path)

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    def copy(self) -> dict[str, str]:
        """Working copy for the Settings form (edits land here, reach the yaml on Save)."""
        return dict(self.values)


def _readable_yaml_path(path: Path) -> Path | None:
    """Return the config path to read.

    The private ``lekiwi.yaml`` remains the real-run source of truth. In a fresh public
    checkout, preview commands and CI can read ``lekiwi.example.yaml`` instead so dry-run
    paths still exercise argv construction without requiring private robot config.
    """
    if path.exists():
        return path
    if path == CFG_FILE and EXAMPLE_CFG_FILE.exists():
        return EXAMPLE_CFG_FILE
    return None


def load_yaml(path: Path = CFG_FILE) -> dict:
    """yaml.safe_load of lekiwi.yaml; anchors + `<<:` merges resolved into a plain
    dict. Returns {} for a missing/empty file so callers can fail soft like bash."""
    readable = _readable_yaml_path(path)
    if readable is None:
        return {}
    data = yaml.safe_load(readable.read_text())
    return data or {}


def _ruamel() -> YAML:
    """The ONE ruamel.yaml configuration the whole package round-trips lekiwi.yaml
    with: round-trip mode (preserves anchors / `<<:` merges / comments) + preserved
    quotes. Both load_yaml_rt and dump_yaml_rt build from this so a load→edit→dump is
    lossless and symmetric (a config drift between the two would reflow the file)."""
    ruamel = YAML()
    ruamel.preserve_quotes = True
    return ruamel


def load_yaml_rt(path: Path = CFG_FILE):  # -> ruamel CommentedMap | None
    """Round-trip load of lekiwi.yaml (ruamel, NOT PyYAML): keeps the anchors, the
    `<<:` merges, and the comments so an edit + dump_yaml_rt rewrites only what changed.
    Returns a CommentedMap (None for a missing/empty file). Callers that mutate then
    dump_yaml_rt: the single safe writer for this file (config.save + record persist)."""
    if not path.exists():
        return None
    with path.open("r") as fh:
        return _ruamel().load(fh)


def dump_yaml_rt(doc, path: Path = CFG_FILE) -> None:  # noqa: ANN001
    """Atomically write a round-trip doc back to lekiwi.yaml (tmp + os.replace), using
    the SAME ruamel settings as load_yaml_rt so the round-trip is byte-minimal. The
    atomic replace means a crash mid-write never leaves a truncated config."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w") as fh:
        _ruamel().dump(doc, fh)
    os.replace(tmp, path)


def cfg_for(
    cmd: str,
    *,
    doc: dict | None = None,
    cache_dir: Path = CFG_CACHE,
) -> Path | None:
    """Slice the top-level block `cmd` out of lekiwi.yaml, dump it (anchors resolved)
    to cache_dir/<cmd>.yaml, and return that path. None if the block is absent.

    The dump MUST mirror the bash pipeline byte-for-byte:
        yaml.safe_dump(block, sort_keys=False)
    with NO extra kwargs — any default_flow_style/width/indent override would change
    the output and break --config_path consumers (and the round-trip test that diffs
    against the committed .lekiwi-cache/<cmd>.yaml). The slice keeps each block a
    valid standalone draccus config (the whole file would fail: foreign keys)."""
    if doc is None:
        doc = load_yaml()
    block = doc.get(cmd)
    if block is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{cmd}.yaml"
    out.write_text(yaml.safe_dump(block, sort_keys=False))
    return out


def cfg_get(dotted: str, *, doc: dict | None = None) -> object | None:
    """One scalar by dotted path, e.g. cfg_get("record.dataset.root"). None on any
    miss (missing key, non-dict on the way down) — bash cfg_get printed nothing."""
    if doc is None:
        doc = load_yaml()
    cur: object = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# Rotation enum → short label, mirroring bash cfg_cameras_summary (lekiwi.sh.legacy
# 431-441). Anything unrecognized (incl. a missing rotation) reads as NO_ROT.
_ROT_ABBR: dict[str, str] = {
    "NO_ROTATION": "NO_ROT",
    "ROTATE_90": "ROT90",
    "ROTATE_180": "ROT180",
    "ROTATE_270": "ROT270",
}


def cameras_summary(doc: dict | None = None) -> str:
    """One-line camera summary from the shared `_cameras` anchor, e.g.
    "front·NO_ROT wrist·ROT180 top·NO_ROT  3×640x480" (bash cfg_cameras_summary).

    Byte-for-byte with bash: middot-joined "name·ABBR" per camera, then TWO spaces
    and "N×WxH" using the FIRST camera's width/height. Empty/missing `_cameras` → ""
    (bash printed nothing). Read-only; never rounds or rewrites the yaml."""
    if doc is None:
        doc = load_yaml()
    cams = doc.get("_cameras") or {}
    if not isinstance(cams, dict) or not cams:
        return ""
    parts = [
        f"{name}·{_ROT_ABBR.get(str((c or {}).get('rotation')), 'NO_ROT')}"
        for name, c in cams.items()
    ]
    line = " ".join(parts)
    first = next(iter(cams.values())) or {}
    line += f"  {len(cams)}×{first.get('width')}x{first.get('height')}"
    return line
