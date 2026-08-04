#!/usr/bin/env bash
# ============================================================================
# lib.sh — shared bash helpers for the modular scripts/*.sh launchers.
#
# THE PATTERN (this is the reference every scripts/<cmd>.sh follows)
# ----------------------------------------------------------------------------
# lekiwi.yaml is the single source of truth (one block per lerobot command, plus
# the `_`-prefixed shared anchors + the `_launcher:` ops knobs). Each command's
# launcher script is a thin, FLAG-DRIVEN, NON-INTERACTIVE wrapper that:
#   1. parses long flags (e.g. --episode 7) into shell variables,
#   2. slices its config block out of lekiwi.yaml with `cfg_slice <cmd>` and
#      reads any extra scalars with `cfg_get` / `launcher_get`,
#   3. assembles the EXACT lerobot-* argv as a bash ARRAY (this script is the
#      SINGLE source of truth for that argv, pinned by static golden tests),
#   4. on --dry-run (or DRY=1) prints that argv one token per line and exits 0;
#      otherwise `exec`s it (so the lerobot CLI inherits this script's real TTY).
# The TUI owns ALL interactivity (pickers / modals); the scripts never prompt.
# This keeps each script standalone-runnable AND frontable by the TUI, which
# just gathers inputs and calls `bash scripts/<cmd>.sh <flags>`.
#
# FLAG-SURFACE CONVENTION (kept stable so the TUI and humans agree)
# ----------------------------------------------------------------------------
#   --dry-run            print the argv (one token per line), do not exec.  Also
#                        triggered by the env var DRY=1 (handy in tests / CI).
#   --<knob> <value>     one space-separated long flag per per-run knob, e.g.
#                        replay: --episode <N>;  teleop: --display on|off
#                        --fps <n> --duration <n>.  Unknown trailing flags are
#                        tolerated and forwarded verbatim into the lerobot argv
#                        (passthrough), so power users can add --dataset.root=…
#                        etc. without the wrapper needing to know every override.
#   --config_path stays the SPACE (two-token) form `--config_path <slice>`, NOT
#                        the `=` form, matching the Python builders.
#
# WHY a python -c instead of importing the TUI package? The scripts must run on hosts
# where the package may not be importable (a bare Pi, a fresh checkout) and must not
# depend on the TUI being installed. They use ONLY pyyaml (guaranteed on the env PATH),
# so the launcher layer stays standalone.
# ============================================================================
set -euo pipefail

# ROOT = repo root (this file lives in scripts/). Resolve symlinks so the
# directory matches Python's `Path(__file__).resolve().parent.parent` exactly.
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$LIB_DIR/.." && pwd -P)"
# Run fronted commands FROM the repo root so the lerobot CLIs resolve relative config
# paths (dataset root ../datasets, POLICY_ROOT ../models) against ROOT, not the caller's
# cwd. cfg_slice/cfg_get pass ROOT to python explicitly, so they stay unaffected.
cd "$ROOT"

# The interpreter the slice/scalar helpers use. Use the env `python` on PATH (what the
# tests use), falling back to python3.
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

# cfg_slice <cmd>
#   Slice the top-level <cmd> block out of lekiwi.yaml, dump it with PyYAML
#   safe_dump(block, sort_keys=False) to <ROOT>/.lekiwi-cache/<cmd>.yaml, and echo
#   that file's path. The echoed path is printed BY PYTHON from
#   `Path(ROOT).resolve()/".lekiwi-cache"/<cmd>.yaml`, so it matches the package
#   cfg_for helper's absolute path. The write uses NO extra safe_dump kwargs, so the
#   bytes equal the committed slice (the parity + byte-equality gates compare a real
#   diff, not a path-form diff).
#   Absent block → no file, empty output, exit 0 (soft-miss, like cfg_for → None).
cfg_slice() {
  local cmd="$1"
  "$PY" - "$ROOT" "$cmd" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
cmd = sys.argv[2]
cfg = root / "lekiwi.yaml"
if not cfg.exists():
    cfg = root / "lekiwi.example.yaml"
if not cfg.exists():
    sys.exit(0)
doc = yaml.safe_load(cfg.read_text()) or {}
block = doc.get(cmd)
if block is None:
    sys.exit(0)                      # absent block -> empty output (cfg_for None)
cache = root / ".lekiwi-cache"
cache.mkdir(parents=True, exist_ok=True)
out = cache / f"{cmd}.yaml"
out.write_text(yaml.safe_dump(block, sort_keys=False))   # byte-identical to cfg_for
print(out)
PY
}

# cfg_get <dotted>
#   One scalar from lekiwi.yaml by dotted path, e.g. cfg_get record.dataset.root.
#   Echoes nothing on any miss (missing key / non-mapping on the way down). Anchors +
#   `<<:` merges are resolved by safe_load, so a merged-in value reads through.
#   Read-only.
cfg_get() {
  local dotted="$1"
  "$PY" - "$ROOT" "$dotted" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
dotted = sys.argv[2]
cfg = root / "lekiwi.yaml"
if not cfg.exists():
    cfg = root / "lekiwi.example.yaml"
if not cfg.exists():
    sys.exit(0)
cur = yaml.safe_load(cfg.read_text()) or {}
for part in dotted.split("."):
    if not isinstance(cur, dict) or part not in cur:
        sys.exit(0)                  # miss -> empty output
    cur = cur[part]
if cur is not None:
    print(cur)
PY
}

# launcher_get <KEY>
#   One ops/launcher knob: the value of _launcher.<KEY> from lekiwi.yaml (the
#   block that replaced lekiwi.conf). Thin wrapper over cfg_get so there is one
#   reader. Empty on miss. NOTE: this is the RAW yaml value only — it does NOT
#   apply the env-var override / built-in default precedence the TUI's Config.load
#   does; the scripts read knobs straight from the yaml.
launcher_get() {
  cfg_get "_launcher.$1"
}

# lerobot_declares <name> <module-relative-path>
#   True when the INSTALLED lerobot's source at <path> mentions <name> — a feature
#   probe for config fields whose absence would make draccus reject our own flag.
#
#   WHY textual, and WHY not the version: `import lerobot.configs.dataset` costs ~3s,
#   which is too much to spend per launch on a yes/no question, and the version cannot
#   answer it at all. A dev checkout mid-release reports the RELEASE version while
#   still missing fields that release has (measured: a 0.6.1 tree with no `no_stamp`),
#   so a version comparison would confidently emit a flag that then fails to parse.
#   find_spec locates the package WITHOUT executing it (~0.02s).
#
#   False when lerobot is not importable at all: callers then omit the flag, which is
#   the safe direction — the launch still runs, it just keeps lerobot's default.
lerobot_declares() {
  local name="${1:?name}" rel="${2:?module path}" pkg
  pkg="$("$PY" - <<'PY' 2>/dev/null
import importlib.util
import pathlib
spec = importlib.util.find_spec("lerobot")
print(pathlib.Path(spec.origin).parent if spec and spec.origin else "")
PY
)"
  [[ -n "$pkg" && -f "$pkg/$rel" ]] || return 1
  grep -q -- "$name" "$pkg/$rel"
}

# shell_quote <value>
#   Emit a shell-escaped token suitable for embedding inside a remote bash command.
#   Callers still pass SSH/scp locally as argv arrays; this is only for the remote
#   command string after `ssh <host> "<remote bash>"`.
shell_quote() {
  printf '%q' "$1"
}

die_usage() {
  printf '%s\n' "$*" >&2
  exit 2
}

validate_ssh_host() {
  local value="${1:-}" label="${2:-SSH host}"
  [[ -n "$value" ]] || die_usage "$label must not be empty"
  [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
  [[ "$value" != *[$'\n\r\t ']* ]] || die_usage "$label must not contain whitespace"
  [[ "$value" =~ ^[A-Za-z0-9_.@-]+$ ]] || die_usage "$label contains unsupported characters; use a simple host or user@host"
  [[ "$value" != *..* ]] || die_usage "$label must not contain '..'"
}

validate_remote_name() {
  local value="${1:-}" label="${2:-remote name}"
  [[ -n "$value" ]] || die_usage "$label must not be empty"
  [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
  [[ "$value" != *[$'\n\r\t ']* ]] || die_usage "$label must not contain whitespace"
  [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || die_usage "$label must use only letters, numbers, '.', '_', or '-'"
}

validate_positive_int() {
  local value="${1:-}" label="${2:-integer}"
  [[ "$value" =~ ^[0-9]+$ ]] && (( value > 0 )) || die_usage "$label must be a positive integer"
}

validate_remote_path() {
  local value="${1:-}" label="${2:-remote path}"
  [[ -n "$value" ]] || die_usage "$label must not be empty"
  [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
  [[ "$value" != *[$'\n\r']* ]] || die_usage "$label must not contain control characters"
}

validate_optional_cli_flag() {
  local value="${1:-}" label="${2:-CLI flag}"
  [[ -z "$value" ]] && return 0
  [[ "$value" =~ ^--[A-Za-z0-9_.-]+=[A-Za-z0-9_./:+-]+$ ]] || die_usage "$label is not a safe single CLI flag"
}
