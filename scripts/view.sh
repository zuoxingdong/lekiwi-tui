#!/usr/bin/env bash
# ============================================================================
# view.sh — flag-driven launcher for `lerobot-dataset-viz` (browse a recording).
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the full
# pattern + flag-surface convention). View is the one command that does NOT slice a
# lekiwi.yaml block: it reads the dataset straight off disk, so there is no
# --config_path and no cfg_slice — only the three HYPHENATED dataset-viz flags.
#
# This script is the SOLE source of the lerobot-dataset-viz argv (the TUI fronts it;
# there is no Python builder). The dry-run golden test (the view test in
# test_actions.py) pins the argv:
#
#   lerobot-dataset-viz --repo-id <R> --root <ROOT> --episode-index <N> [passthrough…]
#
#   * Flags are HYPHENATED (--repo-id / --root / --episode-index), NOT the
#     --dataset.episode= underscore style replay/teleop use.
#   * --repo-id / --root default to the `record:` dataset's repo_id / root in
#     lekiwi.yaml (the same values the package dataset helpers resolve), so a bare
#     standalone run targets the recorded dataset.
#   * --episode-index defaults to 0 (the EpisodeScreen / bash default).
#
# Usage:
#   scripts/view.sh --episode-index 2                       # view episode 2 in Rerun
#   scripts/view.sh --root /tmp/ds --episode-index 0        # view a dataset elsewhere
#   scripts/view.sh --dry-run --episode-index 2             # print the argv, do not run
#   scripts/view.sh --episode-index 2 --dataset.root=/tmp/ds  # trailing flags pass through
#   DRY=1 scripts/view.sh --episode-index 2                 # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: this never prompts. The TUI's _run_view checks the dataset is
# present, the EpisodeScreen gathers + range-checks the index, then this script is
# fronted with --repo-id / --root / --episode-index.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# ── defaults (seed a bare standalone run from the yaml, like _run_view does) ──
# repo_id / root mirror datasets.dataset_repo_id / record_root: the `record:` block's
# values, falling back to the documented defaults when the keys are absent. episode is
# 0 (blank/omitted -> 0). Kept as STRINGS. The TUI always passes all three flags, so
# these defaults only matter for a human running the script with no knobs.
repo_id="$(cfg_get record.dataset.repo_id)"; repo_id="${repo_id:-local/lekiwi_dataset}"
root="$(cfg_get record.dataset.root)"; root="${root:-../datasets/lekiwi_dataset}"
episode="0"            # --episode-index; blank/omitted -> 0
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

# ── parse flags ─────────────────────────────────────────────────────────────
# --repo-id <R>, --root <ROOT>, --episode-index <N> are the three real knobs.
# --dry-run flips DRY. Anything else is an unrecognized trailing flag: collect it
# into `extra` and forward it verbatim into the lerobot argv (passthrough).
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      repo_id="$2"; shift 2 ;;
    --root)
      root="$2"; shift 2 ;;
    --episode-index)
      episode="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

# ── assemble argv (pinned by the view dry-run golden in test_actions.py) ─────
# No --config_path: view reads the dataset off disk, it slices no lekiwi.yaml block.
argv=(
  lerobot-dataset-viz
  --repo-id "$repo_id"
  --root "$root"
  --episode-index "$episode"
)
# Append any passthrough flags (empty in the TUI path, so the launch test still
# matches; populated only when a human adds extra lerobot-dataset-viz overrides).
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the golden
# gate captures these lines). Otherwise exec, so lerobot-dataset-viz inherits this
# script's real TTY (the TUI suspends into it; Rerun opens its own window).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

exec "${argv[@]}"
