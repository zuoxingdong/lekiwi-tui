#!/usr/bin/env bash
# ============================================================================
# find_cameras.sh — emitter for the Pi-side bash that lists the robot's cameras.
#
# WHY REMOTE: LeKiwi's cameras hang off the Pi, and the `index_or_path` values in
# lekiwi.yaml are Pi-side device nodes. Running `lerobot-find-cameras` on the laptop
# enumerates the laptop's own webcam and says nothing about front/wrist/top, so the
# question "which /dev/videoN is the wrist today" can only be answered on the robot.
# It needs answering more often than one would like: a bare /dev/videoN is not
# reboot/replug-stable, and adding a camera renumbers the others.
#
# Same seam as host.sh: this script only PRINTS the remote bash (one argv token), and
# the Python side (screens/robot_config.py) builds the `ssh <host> "<that token>"`
# invocation. Nothing is shipped to the Pi and nothing here spawns ssh.
#
# LIST-ONLY BY DEFAULT. `lerobot-find-cameras` prints the detected cameras and then
# captures frames from each into --output-dir; --record-time 0 keeps the printout and
# writes no files, which is all the yaml needs. Pass a positive --record-time when you
# actually want sample frames on the Pi to eyeball which lens is which.
#
# Usage:
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi --backend all --record-time 2
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi --dry-run   # accepted, no-op
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# Where sample frames land ON THE PI when --record-time > 0. Under /tmp so a probe can
# never fill the Pi's home or leave anything the next run has to clean up.
REMOTE_OUT_DIR="/tmp/lekiwi-find-cameras"

# emit_detect <conda_env> <backend> <record_time> <warmup> <out_dir>
#   Print the remote DETECT bash. The mamba hook + activate lines mirror
#   host.sh emit_launch, so both remote payloads fail the same way on a Pi whose env
#   is missing rather than in some new way.
emit_detect() {
  local conda_env="$1" backend="$2" record_time="$3" warmup="$4" out_dir="$5"
  local type_arg=""
  [[ "$backend" == "all" ]] || type_arg="$backend"
  printf '
        eval "$(~/miniforge3/bin/mamba shell hook --shell bash)" || exit 1
        mamba activate %s || { echo '\''✗ could not activate %s'\'' >&2; exit 1; }

        echo "▸ probing cameras on $(hostname) (record-time %ss)"
        lerobot-find-cameras %s \
            --output-dir %s \
            --record-time-s %s \
            --warmup-s %s
    ' "$conda_env" "$conda_env" "$record_time" "$type_arg" "$out_dir" "$record_time" "$warmup"
}

# ── parse ───────────────────────────────────────────────────────────────────
# First positional is the subcommand (emit-detect). Long flags only; unknown flags are
# an error, since the emitted bash is fixed (no passthrough).
subcmd="${1:-}"; shift || true
conda_env=""
backend="opencv"       # the LeKiwi cameras are plain UVC; `all` also probes RealSense
record_time="0"        # 0 = print the list, write no frames
warmup="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)   conda_env="${2:-}"; shift 2 ;;
    --backend)     backend="${2:-}"; shift 2 ;;
    --record-time) record_time="${2:-}"; shift 2 ;;
    --warmup)      warmup="${2:-}"; shift 2 ;;
    --dry-run)     shift ;;   # accepted and ignored: this script only ever prints
    *)             die_usage "unknown flag: $1" ;;
  esac
done

case "$backend" in
  opencv|realsense|all) ;;
  *) die_usage "--backend must be 'opencv', 'realsense' or 'all' (got '$backend')" ;;
esac
[[ "$record_time" =~ ^[0-9]+$ ]] || die_usage "--record-time must be a whole number of seconds"
[[ "$warmup" =~ ^[0-9]+$ ]] || die_usage "--warmup must be a whole number of seconds"

case "$subcmd" in
  emit-detect)
    validate_remote_name "$conda_env" "conda env"
    emit_detect "$conda_env" "$backend" "$record_time" "$warmup" "$REMOTE_OUT_DIR" ;;
  ""|-h|--help) die_usage "usage: find_cameras.sh emit-detect --conda-env <env> [--backend opencv|realsense|all] [--record-time N] [--warmup N]" ;;
  *) die_usage "unknown subcommand: $subcmd (expected emit-detect)" ;;
esac
