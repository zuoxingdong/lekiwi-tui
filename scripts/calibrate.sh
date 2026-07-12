#!/usr/bin/env bash
# ============================================================================
# calibrate.sh — flag-driven launcher for `lerobot-calibrate` (two targets).
#
# WHY TWO TARGETS
# ----------------------------------------------------------------------------
# Calibration is per-arm. The SO101 LEADER's motors hang off the laptop, so it
# calibrates LOCALLY (a plain `lerobot-calibrate --teleop.type=so101_leader …`,
# `exec`'d like replay.sh). The LeKiwi FOLLOWER's motors live on the Pi (the
# laptop client's calibrate() is a no-op), so it calibrates REMOTELY over
# `ssh -t <PI> "<bash>"`, where "<bash>" is a single argv token run on the Pi
# (the mamba-hook activate + the interactive lekiwi calibration). That split
# mirrors how the TUI's CalibrateScreen calibrates each arm (leader local vs
# follower over ssh).
#
# This script is the SOLE source of BOTH argvs; the static argv goldens in
# test_calibrate.py pin the exact tokens it must print:
#
#   leader  → lerobot-calibrate --teleop.type=so101_leader
#                 --teleop.port=<LEADER_PORT> --teleop.id=<LEADER_ID> [passthrough…]
#   follower→ ssh -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -t <host> "<remote>"
#               where <remote> is the 3-line emit-follower-remote bash below.
#
#   • The leader knobs come straight from the TUI (it always passes --leader-port /
#     --leader-id), so there is no cfg_slice here — calibrate has no lekiwi.yaml
#     block; the values are env-config scalars the screen reads, not a config_path.
#   • The follower remote script is the riskiest piece (it activates the Pi env via
#     the SAME mamba hook host-launch uses, then runs the INTERACTIVE calibration),
#     so — exactly like host.sh's remote bash — it is EMITTED via printf and reused
#     as the single ssh argv token. The byte-equality test (against an independent
#     _golden_follower_remote in test_calibrate.py) gates it, like host.sh emit-kill.
#
# SUBCOMMANDS / FLAGS
# ----------------------------------------------------------------------------
#   calibrate.sh --target leader  --leader-port P --leader-id I [passthrough…]
#       Local SO101 leader calibration. Unknown trailing flags pass through into
#       the lerobot-calibrate argv (e.g. --teleop.foo=bar), like the other launchers.
#
#   calibrate.sh --target follower --host H --conda-env E --robot-id R [--robot-type T]
#       `ssh -t` the Pi for the interactive follower calibration. (No passthrough:
#       the remote bash is fixed, so the follower target takes no extra.)
#
#   calibrate.sh emit-follower-remote --conda-env E --robot-id R [--robot-type T]
#       Print ONLY the remote follower bash. Like host.sh emit-kill, this keeps the
#       script the single source of that bash; the byte-equality test asserts it
#       against an independent golden (test_calibrate.py _golden_follower_remote).
#
#   A leading --dry-run (or DRY=1) prints the assembled argv one token per line and
#   exits 0 (for follower, the full ssh argv with the remote script embedded as the
#   last token — exactly what the follower ssh golden expects). emit-follower-remote
#   ignores --dry-run (it only ever prints, like host.sh's emitters).
#
# Usage:
#   scripts/calibrate.sh --target leader --leader-port /dev/ttyACM0 --leader-id lekiwi_leader
#   scripts/calibrate.sh --target follower --host lekiwi --conda-env lekiwi --robot-id lekiwi
#   scripts/calibrate.sh --target leader --leader-port /dev/ttyACM0 --leader-id l --dry-run
#   scripts/calibrate.sh emit-follower-remote --conda-env lekiwi --robot-id lekiwi
#
# NON-INTERACTIVE: this never prompts. The TUI's CalibrateScreen picks the arm and
# passes the resolved cfg values as flags here; the live prompts ("move the arm,
# press ENTER") come from lerobot-calibrate itself over the inherited TTY (leader)
# or the remote PTY `ssh -t` allocates (follower).
# ============================================================================
set -euo pipefail

# Load the shared helpers (same as replay.sh / host.sh) relative to THIS file, so
# the script is runnable from any working directory. calibrate has no lekiwi.yaml
# block (no cfg_slice), but sourcing lib.sh keeps the launcher family uniform and
# leaves cfg_get/cfg_slice available to forks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# emit_follower_remote <conda_env> <robot_id> <robot_type>
#   Print the remote FOLLOWER bash. robot_type picks the follower --robot.type:
#   lekiwi_pincopen (the default) is the PincOpen robot plugin (lerobot_robot_lekiwi_pincopen,
#   installed on the Pi by pi_provision.sh); lerobot-calibrate's plugin discovery
#   registers it, and its calibrate() skips the gripper (fixed EPROM calibration)
#   unlike stock lekiwi. The TUI passes the ROBOT_TYPE config value here.
#   This is its single source; the byte-equality test
#   pins it to test_calibrate.py's _golden_follower_remote. printf reproduces the exact
#   bytes (the gate depends on them): three lines, each
#   ending in a newline (so the string ends with a trailing newline). The remote-side
#   $(...) mamba hook is LITERAL here (it is evaluated on the Pi, not by this printf),
#   so it sits inside the single-quoted format and is emitted as-is. conda_env fills
#   the first two %s (the activate target + the error message), robot_type then
#   robot_id the last two.
emit_follower_remote() {
  local conda_env="$1" robot_id="$2" robot_type="$3"
  printf 'eval "$(~/miniforge3/bin/mamba shell hook --shell bash)" || exit 1
mamba activate %s || { echo '\''✗ could not activate %s'\'' >&2; exit 1; }
lerobot-calibrate --robot.type=%s --robot.id=%s
' "$conda_env" "$conda_env" "$robot_type" "$robot_id"
}

# ── subcommand split ──────────────────────────────────────────────────────────
# emit-follower-remote is a pure emitter (like host.sh emit-kill): peel it off the
# front and handle it before the launch flag-parse. Everything else is a launch
# (--target leader|follower) and falls through to the parser below.
if [[ "${1:-}" == "emit-follower-remote" ]]; then
  shift
  conda_env=""
  robot_id=""
  robot_type="lekiwi_pincopen"     # default: the PincOpen plugin robot
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --conda-env)  conda_env="$2"; shift 2 ;;
      --robot-id)   robot_id="$2"; shift 2 ;;
      --robot-type) robot_type="$2"; shift 2 ;;
      --dry-run)    shift ;;          # accepted + ignored (always emits)
      *)
        echo "calibrate.sh: unknown flag '$1' for emit-follower-remote" >&2
        exit 2 ;;
    esac
  done
  : "${DRY:-0}"   # DRY=1 is likewise a no-op; referenced so set -u stays happy
  validate_remote_name "$conda_env" "conda env"
  validate_remote_name "$robot_id" "robot id"
  validate_remote_name "$robot_type" "robot type"
  emit_follower_remote "$conda_env" "$robot_id" "$robot_type"
  exit 0
fi

# ── parse launch flags ────────────────────────────────────────────────────────
# --target leader|follower selects which argv to build. leader knobs:
# --leader-port / --leader-id. follower knobs: --host / --conda-env / --robot-id.
# --dry-run flips DRY. For leader ONLY, unknown trailing flags pass through into the
# lerobot-calibrate argv (the passthrough `extra`); follower takes no extra.
target=""
leader_port=""
leader_id=""
host=""
conda_env=""
robot_id=""
robot_type="lekiwi_pincopen"     # follower --robot.type; the TUI passes ROBOT_TYPE
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # leader passthrough flags appended after the built argv

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)       target="$2"; shift 2 ;;
    --leader-port)  leader_port="$2"; shift 2 ;;
    --leader-id)    leader_id="$2"; shift 2 ;;
    --host)         host="$2"; shift 2 ;;
    --conda-env)    conda_env="$2"; shift 2 ;;
    --robot-id)     robot_id="$2"; shift 2 ;;
    --robot-type)   robot_type="$2"; shift 2 ;;
    --dry-run)      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags (leader)
  esac
done

# ── assemble argv per target ──────────────────────────────────────────────────
case "$target" in
  leader)
    # Local SO101 leader calibration — the leader argv golden pins these tokens.
    argv=(
      lerobot-calibrate
      --teleop.type=so101_leader
      "--teleop.port=${leader_port}"
      "--teleop.id=${leader_id}"
    )
    if [[ ${#extra[@]} -gt 0 ]]; then
      argv+=("${extra[@]}")
    fi
    ;;
  follower)
    validate_ssh_host "$host" "LEKIWI_HOST"
    validate_remote_name "$conda_env" "conda env"
    validate_remote_name "$robot_id" "robot id"
    validate_remote_name "$robot_type" "robot type"
    [[ ${#extra[@]} -eq 0 ]] || die_usage "follower calibration does not accept passthrough flags"
    # `ssh -t` the Pi for the interactive follower calibration — the follower ssh argv
    # golden pins these tokens, with the emit-follower-remote bash as the single last
    # token (so the embedded remote script is byte-identical to the emit golden).
    # The remote ends in a trailing newline (emit_follower_remote prints one); a plain
    # $(...) command-sub would EAT it, making the exec'd ssh token one byte short of
    # the golden. The `printf x` sentinel + `${remote%x}` strip preserves it, so both
    # exec AND --dry-run carry the exact \n-terminated token the golden expects.
    remote="$(emit_follower_remote "$conda_env" "$robot_id" "$robot_type"; printf x)"
    remote="${remote%x}"
    argv=(
      ssh
      -o ServerAliveInterval=5
      -o ServerAliveCountMax=3
      -t "$host"
      "$remote"
    )
    ;;
  ""|*)
    echo "calibrate.sh: --target must be leader or follower (got '${target}')" >&2
    echo "usage: calibrate.sh --target leader  --leader-port P --leader-id I [extra…]" >&2
    echo "       calibrate.sh --target follower --host H --conda-env E --robot-id R [--robot-type T]" >&2
    echo "       calibrate.sh emit-follower-remote --conda-env E --robot-id R [--robot-type T]" >&2
    exit 2 ;;
esac

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the parity gate
# captures these lines; for follower the last line is the multi-line remote script).
# Otherwise exec, so lerobot-calibrate / ssh inherits this script's real TTY — the
# TUI suspends into it, so the interactive prompts (and ssh -t's remote PTY) work and
# Ctrl+C reaches the CLI directly.
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

exec "${argv[@]}"
