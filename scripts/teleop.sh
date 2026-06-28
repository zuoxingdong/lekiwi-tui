#!/usr/bin/env bash
# ============================================================================
# teleop.sh — flag-driven launcher for `lerobot-teleoperate`.
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the full
# pattern + flag-surface convention). Teleop has three per-run knobs instead of
# replay's one, and one CONDITIONAL token, so it shows the two ways a script
# diverges from the template while keeping the same skeleton.
#
# This script is the SOLE source of the lerobot-teleoperate argv. Its --dry-run is
# pinned by local regression tests. The argv it builds:
#
#   lerobot-teleoperate --config_path <slice teleop> --display_data=<true|false>
#       --fps=<n> [--teleop_time_s=<n> when duration>0] [passthrough...]
#
#   * --config_path is the SPACE (two-token) form, NOT --config_path=...
#   * <slice teleop> is `cfg_slice teleop` (lekiwi.yaml's `teleop:` block, sliced to
#     .lekiwi-cache/teleop.yaml; the echoed path string == config.cfg_for("teleop")).
#   * --display on|off maps to the lowercase bool --display_data=true|false.
#   * --duration <n> maps to --teleop_time_s=<n>, but ONLY when n>0 (0 = run until
#     Ctrl+C, so the token is omitted).
#
# Usage:
#   scripts/teleop.sh --display on --fps 30                 # teleop until Ctrl+C
#   scripts/teleop.sh --display off --fps 25 --duration 60  # 60s, headless view
#   scripts/teleop.sh --dry-run --display on --fps 30       # print the argv, do not run
#   scripts/teleop.sh --display on --fps 30 --robot.id=foo  # trailing flags pass through
#   DRY=1 scripts/teleop.sh --display on --fps 30           # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: this never prompts. The TUI's TeleopScreen gathers Display /
# Duration / FPS, then fronts this script with --display/--fps/--duration.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# norm_bool <value> -> "true"|"false"
#   Normalize a display value to the lowercase bool token --display_data wants.
#   Accepts the flag form (on/off) and the yaml/cfg_get form (true/false/True/False/
#   1/0) so a bare standalone run seeded from cfg_get renders identically. Anything
#   not truthy is "false".
norm_bool() {
  case "${1,,}" in
    on|true|1|yes) printf 'true' ;;
    *)             printf 'false' ;;
  esac
}

# ── defaults (seed a bare standalone run from the yaml, like TeleopScreen.on_mount) ──
# The TUI always passes all three flags, so these defaults only matter for a human
# running the script with no knobs. display/fps mirror the `teleop` block; duration
# defaults to 0 (omitted -> until Ctrl+C). Kept as STRINGS so the empty/0 guard below
# is set -e safe.
display="$(norm_bool "$(cfg_get teleop.display_data)")"   # true|false
fps="$(cfg_get teleop.fps)"; fps="${fps:-30}"
duration="0"           # teleop_time_s; 0/blank -> omit the token (run until Ctrl+C)
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

# ── parse flags ─────────────────────────────────────────────────────────────
# --display on|off, --fps <n>, --duration <n> are the three real knobs. --dry-run
# flips DRY. Anything else is an unrecognized trailing flag: collect it into `extra`
# and forward it verbatim into the lerobot argv (passthrough), e.g. --robot.id=...
while [[ $# -gt 0 ]]; do
  case "$1" in
    --display)
      display="$(norm_bool "$2")"; shift 2 ;;
    --fps)
      fps="$2"; shift 2 ;;
    --duration)
      duration="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

# ── assemble argv (the static golden in test_teleop_argv.py pins this) ────────
# cfg_slice writes the slice and echoes its absolute path; the $() strips the
# trailing newline so the path is a single clean token.
slice="$(cfg_slice teleop)"
# We front lerobot-teleoperate through lerobot_teleop_kbd.py, a thin shim that swaps
# pynput's keyboard Listener (a no-op on Wayland: the auto-attached base keyboard never
# moves the base) for an evdev reader that gets real press/release below the compositor.
# lerobot's repo is untouched; the only change vs `lerobot-teleoperate` is the first two
# tokens (`python <shim>`). Requires the user in the `input` group (see the shim header).
argv=(
  python "$SCRIPT_DIR/lerobot_teleop_kbd.py"
  --config_path "$slice"
  "--display_data=${display}"
  "--fps=${fps}"
)
# Conditional token: --teleop_time_s only when duration>0 (0/blank -> omit it, run
# until Ctrl+C). The -n guard keeps -gt from tripping set -e on an empty value, and
# this token sits BEFORE the passthrough extras.
if [[ -n "$duration" && "$duration" -gt 0 ]]; then
  argv+=("--teleop_time_s=${duration}")
fi
# Append any passthrough flags (empty in the TUI launch path, so the launch test's
# expected argv has none; populated only when a human adds extra lerobot overrides).
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the parity
# gate captures these lines). Otherwise exec, so lerobot-teleoperate inherits this
# script's real TTY (the TUI suspends into it; teleop reads the keyboard directly).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

exec "${argv[@]}"
