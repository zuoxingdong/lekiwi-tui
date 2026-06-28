#!/usr/bin/env bash
# ============================================================================
# replay.sh — flag-driven launcher for `lerobot-replay` (the PILOT TEMPLATE).
#
# This is the reference shape every other scripts/<cmd>.sh mirrors. Replay is the
# simplest command (one per-run knob, --episode), so it shows the pattern with the
# least noise. See lib.sh's header for the full pattern + flag-surface convention.
#
# This script is the SOLE source of the lerobot-replay argv (the TUI fronts it; there
# is no Python builder). The dry-run golden test (test_replay_argv.py) pins the argv:
#
#   lerobot-replay --config_path <slice replay> --dataset.episode=<N> [passthrough…]
#
#   • --config_path is the SPACE (two-token) form, NOT `--config_path=…`.
#   • <slice replay> is `cfg_slice replay` (lekiwi.yaml's `replay:` block, sliced to
#     .lekiwi-cache/replay.yaml; the echoed path string == config.cfg_for("replay")).
#   • --episode <N> maps to the `=` scalar override --dataset.episode=<N>.
#
# Usage:
#   scripts/replay.sh --episode 3                 # replay episode 3 on the robot
#   scripts/replay.sh --dry-run --episode 3       # print the argv, do not run
#   scripts/replay.sh --episode 3 --dataset.root=/tmp/ds   # trailing flags pass through
#   DRY=1 scripts/replay.sh --episode 3           # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: this never prompts for the episode. The TUI's EpisodeScreen
# gathers + range-checks the index, then fronts this script with --episode <N>.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# ── parse flags ─────────────────────────────────────────────────────────────
# --episode <N> is the one real knob. --dry-run flips DRY. Anything else is an
# unrecognized trailing flag: collect it into `extra` and forward it verbatim
# into the lerobot argv (passthrough), so e.g. --dataset.root=… still works
# without this wrapper having to know every lerobot override.
episode="0"            # blank/omitted -> 0, matching the EpisodeScreen default
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episode)
      episode="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

# ── assemble argv (pinned by the test_replay_argv.py dry-run golden) ─────────
# cfg_slice writes the slice and echoes its absolute path; the $() strips the
# trailing newline so the path is a single clean token.
slice="$(cfg_slice replay)"
argv=(
  lerobot-replay
  --config_path "$slice"
  "--dataset.episode=${episode}"
)
# Append any passthrough flags (empty in the TUI path, so the launch test still
# matches; populated only when a human adds extra lerobot overrides).
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the golden
# gate captures these lines). Otherwise exec, so lerobot-replay inherits this
# script's real TTY (the TUI suspends into it; Ctrl+C reaches the CLI directly).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

exec "${argv[@]}"
