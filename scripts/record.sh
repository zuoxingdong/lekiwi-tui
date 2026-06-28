#!/usr/bin/env bash
# ============================================================================
# record.sh — flag-driven launcher for `lerobot-record`.
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the full
# pattern + flag-surface convention). Record is the richest command: it has a
# FRESH-vs-RESUME branch (the resume case drops two tokens) AND it derives the
# dataset's repo_id + root from a single --name.
#
# This script is the SINGLE source of truth for the lerobot-record argv (the TUI's
# RecordScreen + headless path only front it). Its --dry-run output is pinned by local
# regression tests. It builds, token-for-token:
#
#   lerobot-record --config_path <slice record>
#       [--resume=true (RESUME only)]
#       --dataset.repo_id=<ns>/<name> --dataset.root=<parent>/<name>
#       [--dataset.single_task=<task> --dataset.fps=<fps> (FRESH only)]
#       --dataset.num_episodes=<n> --dataset.episode_time_s=<n>
#       --dataset.reset_time_s=<n> --display_data=<true|false>
#       --dataset.streaming_encoding=<true|false>
#       --dataset.num_image_writer_threads_per_camera=<n> [passthrough...]
#
#   * --config_path is the SPACE (two-token) form, NOT --config_path=...
#   * <slice record> is `cfg_slice record` (lekiwi.yaml's `record:` block, sliced to
#     .lekiwi-cache/record.yaml; the echoed path string == config.cfg_for("record")).
#   * --name maps to BOTH --dataset.repo_id (<ns>/<name>) and --dataset.root
#     (<parent>/<name>), reusing the yaml namespace + parent dir — the same split
#     dataset_defaults() + resolve_repo_root() do in Python (ns = repo_id before the
#     last "/", parent = the root's parent dir, "" when the root has no parent).
#   * --resume true emits --resume=true and DROPS --dataset.single_task + .fps (the
#     existing dataset fixes them); --resume false emits NEITHER a resume token nor
#     drops anything (FRESH passes every field). There is no --resume=false token.
#   * --display on|off maps to the lowercase bool --display_data=true|false.
#   * --streaming-encoding on|off maps to --dataset.streaming_encoding=true|false
#     (real-time video encode for near-instant saves; off = encode at session end).
#   * --image-writer-threads N maps to
#     --dataset.num_image_writer_threads_per_camera=N (writer threads per camera, x3
#     cameras; raise if frames drop). Defaults to the yaml value (3) when omitted.
#
# Usage:
#   scripts/record.sh --name cubes --task "grab the cube" --episodes 12 \
#       --episode-time 30 --reset-time 4 --fps 30 --display on    # fresh record
#   scripts/record.sh --name cubes --resume true                 # append to it
#   scripts/record.sh --dry-run --name cubes --task t            # print argv, no run
#   scripts/record.sh --name cubes --task t --dataset.root=/tmp/ds  # passthrough
#   DRY=1 scripts/record.sh --name cubes --task t                # env-var dry-run
#
# NON-INTERACTIVE: this never prompts and never deletes. The TUI's RecordScreen
# gathers the fields, runs the existing-dataset Resume/Delete safety modal, then
# fronts this script with --name/--task/.../--resume.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# norm_bool <value> -> "true"|"false"
#   Normalize a flag value to the lowercase bool token --display_data takes.
#   Accepts the display flag form (on/off) and the yaml/cfg_get form (true/false/
#   True/False/1/0) so a bare standalone run seeded from cfg_get renders the same.
norm_bool() {
  case "${1,,}" in
    on|true|1|yes) printf 'true' ;;
    *)             printf 'false' ;;
  esac
}

# is_true <value> -> exit 0 when truthy (true/1/yes/on), else 1
#   Used for --resume: only --resume true takes the resume branch.
is_true() {
  case "${1,,}" in
    true|1|yes|on) return 0 ;;
    *)             return 1 ;;
  esac
}

# ── defaults (seed a bare standalone run from the yaml, like RecordScreen.on_mount
#    via dataset_defaults) ─────────────────────────────────────────────────────
# The TUI always passes every flag, so these defaults only matter for a human
# running the script with no knobs. repo_id/root are read once to derive the
# namespace + parent dir + default name; the per-run scalars mirror the `record`
# block. Kept as STRINGS so the conditional branches below are set -e safe.
_repo_id="$(cfg_get record.dataset.repo_id)"; _repo_id="${_repo_id:-local/lekiwi_dataset}"
_root="$(cfg_get record.dataset.root)";       _root="${_root:-../datasets/lekiwi_dataset}"

# Split the namespace + parent the SAME way resolve_repo_root / dataset_defaults do:
#   ns     = repo_id before the last "/" (else "local", matching the Python guard).
#   parent = the root's parent dir; bash `dirname` returns "." for a bare name, which
#            Python's dataset_defaults maps to "" (root then == name, no parent prefix).
if [[ "$_repo_id" == */* ]]; then ns="${_repo_id%/*}"; else ns="local"; fi
parent="$(dirname "$_root")"
[[ "$parent" == "." ]] && parent=""

name="$(basename "$_root")"; name="${name:-lekiwi_dataset}"
task="$(cfg_get record.dataset.single_task)"
episodes="$(cfg_get record.dataset.num_episodes)";  episodes="${episodes:-5}"
ep_time="$(cfg_get record.dataset.episode_time_s)"; ep_time="${ep_time:-40}"
reset_time="$(cfg_get record.dataset.reset_time_s)"; reset_time="${reset_time:-5}"
fps="$(cfg_get record.dataset.fps)";                fps="${fps:-30}"
img_threads="$(cfg_get record.dataset.num_image_writer_threads_per_camera)"; img_threads="${img_threads:-3}"  # writer threads per camera (x3 cams)
display="$(norm_bool "$(cfg_get record.display_data)")"   # true|false
streaming="$(cfg_get record.dataset.streaming_encoding)"; streaming="$(norm_bool "${streaming:-true}")"  # true|false (real-time video encode)
resume="false"         # FRESH by default; --resume true switches to the resume branch
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

# ── parse flags ─────────────────────────────────────────────────────────────
# The named knobs map 1:1 to the form fields RecordScreen gathers. --display takes
# on|off and --resume takes true|false (NOT the lerobot `=` token form — that is
# emitted below). --dry-run flips DRY. Anything else is forwarded verbatim into the
# lerobot argv (passthrough), e.g. --dataset.root=... .
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)         name="$2"; shift 2 ;;
    --task)         task="$2"; shift 2 ;;
    --episodes)     episodes="$2"; shift 2 ;;
    --episode-time) ep_time="$2"; shift 2 ;;
    --reset-time)   reset_time="$2"; shift 2 ;;
    --fps)          fps="$2"; shift 2 ;;
    --display)      display="$(norm_bool "$2")"; shift 2 ;;
    --streaming-encoding) streaming="$(norm_bool "$2")"; shift 2 ;;
    --image-writer-threads) img_threads="$2"; shift 2 ;;
    --resume)       resume="$2"; shift 2 ;;
    --dry-run)      dry=1; shift ;;
    *)              extra+=("$1"); shift ;;   # tolerate + forward passthrough flags
  esac
done

# ── resolve --name -> repo_id + root (matches resolve_repo_root) ─────────────
#   repo_id = <ns>/<name>;  root = <parent>/<name>  (or just <name> when parent="").
repo_id="${ns}/${name}"
if [[ -n "$parent" ]]; then root="${parent}/${name}"; else root="$name"; fi

# ── assemble argv (the single source of truth, pinned by the golden test) ────
# cfg_slice writes the slice and echoes its absolute path; the $() strips the
# trailing newline so the path is a single clean token.
slice="$(cfg_slice record)"
# We front lerobot-record through lerobot_record_kbd.py, a thin shim that swaps
# lerobot's pynput keyboard listener (a silent no-op on Wayland: arrow keys / ESC
# never reach the record loop, only Ctrl+C does) for a stdin reader, then calls
# lerobot's own main() with this exact argv. lerobot's repo is untouched; the only
# change vs `lerobot-record` is the first two tokens (`python <shim>`).
argv=(
  python "$SCRIPT_DIR/lerobot_record_kbd.py"
  --config_path "$slice"
)
# RESUME emits --resume=true BEFORE repo_id/root; FRESH emits no resume token.
if is_true "$resume"; then
  argv+=("--resume=true")
fi
argv+=(
  "--dataset.repo_id=${repo_id}"
  "--dataset.root=${root}"
)
# single_task + fps are FRESH-only (the existing dataset fixes them on resume).
if ! is_true "$resume"; then
  argv+=(
    "--dataset.single_task=${task}"
    "--dataset.fps=${fps}"
  )
fi
argv+=(
  "--dataset.num_episodes=${episodes}"
  "--dataset.episode_time_s=${ep_time}"
  "--dataset.reset_time_s=${reset_time}"
  "--display_data=${display}"
  "--dataset.streaming_encoding=${streaming}"
  "--dataset.num_image_writer_threads_per_camera=${img_threads}"
)
# Append any passthrough flags (empty in the TUI path, so the launch test still
# matches; populated only when a human adds extra lerobot overrides).
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the parity
# gate captures these lines). Otherwise exec, so the record shim inherits this
# script's real TTY (the TUI suspends into it; the shim reads arrow keys / ESC
# straight from that stdin — see lerobot_record_kbd.py).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

exec "${argv[@]}"
