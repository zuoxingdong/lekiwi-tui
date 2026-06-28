#!/usr/bin/env bash
# ============================================================================
# sync.sh — rsync the local lerobot clone up to the Pi (the "Sync to Pi" action).
#
# A NEW launcher (no lerobot-* CLI): it pushes a routine source change to the Pi so
# the editable install there picks it up, WITHOUT re-running the whole `lerobot`
# provision stage (no env rebuild, no reinstall, no smoke-test). Use it after editing
# the clone; use `setup-pi` (pi_provision.sh) for first-time bring-up or a dep change.
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the pattern).
# It reuses pi_provision.sh's `lerobot` stage rsync recipe VERBATIM (same flags, same
# excludes) so the two stay consistent — change the recipe in BOTH if it ever moves:
#
#   ssh <host> "mkdir -p '<repo>'"        # ensure the target dir exists (first sync)
#   rsync -az --exclude '.git/' …  <LOCAL_REPO>/  <host>:<repo>/
#
#   * -az = archive + compress; NO --delete (would wipe Pi-only files; calibration
#     lives in ~/.cache, outside the tree). Excludes match pi_provision.sh exactly.
#   * <host> / <repo> default to _launcher.LEKIWI_HOST / _launcher.PI_REPO in
#     lekiwi.yaml (the SyncScreen passes no flags, so these defaults are what runs);
#     an exported LEKIWI_HOST / PI_REPO env var wins (TUI config precedence), and
#     --host / --repo flags win over everything (standalone use).
#   * <LOCAL_REPO> is the lerobot clone shipped to the Pi. By default the script
#     uses a sibling of this checkout, with a fallback for older parent-workspace
#     layouts. Override with LOCAL_REPO=… for a non-standard tree.
#
# Usage:
#   scripts/sync.sh                         # rsync the clone to the configured Pi
#   scripts/sync.sh --host my-pi --repo bob/lerobot   # override host / repo
#   scripts/sync.sh --dry-run               # print the rsync argv, touch nothing
#   scripts/sync.sh --delete                # trailing rsync flags pass through
#   DRY=1 scripts/sync.sh                    # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: this never prompts. The TUI's SyncScreen shows an info/confirm
# page, then fronts this script (suspend, pause-on-exit so the rsync log stays up).
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) + ROOT, relative to
# THIS file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# ── defaults (host/repo: env var > _launcher > built-in, like Config.load) ──
# launcher_get reads _launcher.<KEY> from lekiwi.yaml; an exported env var wins over it
# (TUI precedence: env at launch > yaml > default). Built-in defaults match config.py.
host="${LEKIWI_HOST:-$(launcher_get LEKIWI_HOST)}"; host="${host:-lekiwi}"
repo="${PI_REPO:-$(launcher_get PI_REPO)}"; repo="${repo:-lekiwi/lerobot}"
# The clone shipped to the Pi. Prefer a sibling of this checkout for public clones,
# but keep a parent-workspace fallback for older local layouts. Override with
# LOCAL_REPO=… for a non-standard tree.
default_local_repo() {
  local sibling parent_workspace
  sibling="$(cd "$ROOT/.." && pwd -P)/lerobot"
  parent_workspace="$(cd "$ROOT/../.." && pwd -P)/lerobot"
  if [[ -d "$sibling" || ! -d "$parent_workspace" ]]; then
    printf '%s\n' "$sibling"
  else
    printf '%s\n' "$parent_workspace"
  fi
}
local_repo="${LOCAL_REPO:-$(default_local_repo)}"
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough rsync flags appended after the recipe flags

# ── parse flags ─────────────────────────────────────────────────────────────
# --host <h>, --repo <r> override the destination. --dry-run flips DRY. Anything else
# is an unrecognized trailing flag: collect it into `extra` and forward it verbatim
# into the rsync argv (passthrough), e.g. --delete or --exclude '…'.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="$2"; shift 2 ;;
    --repo)
      repo="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

validate_ssh_host "$host" "LEKIWI_HOST"
validate_remote_path "$repo" "PI_REPO"
remote_repo="$(shell_quote "${repo%/}/")"

# ── assemble the rsync argv (pi_provision.sh `lerobot` stage recipe, verbatim) ──
# trailing slash on the source copies its CONTENTS into the dest dir (not a nested dir).
argv=(
  rsync -az
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc'
  --exclude '*.egg-info/' --exclude 'outputs/' --exclude 'wandb/' --exclude 'logs/'
)
# Passthrough flags sit with the other rsync options, BEFORE the source/dest operands.
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi
argv+=("${local_repo}/" "${host}:${remote_repo}")

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the rsync argv one token per line and exit 0 (the parity
# gate captures these lines); touch NOTHING (no ssh, no network). Otherwise ensure the
# target dir exists, then exec rsync so it inherits this script's real TTY (progress +
# any ssh auth prompt reach the user; the TUI suspends into it).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

[ -d "$local_repo" ] || { printf 'sync.sh: local clone not found: %s\n' "$local_repo" >&2; exit 1; }
ssh "$host" "mkdir -p -- $remote_repo"
exec "${argv[@]}"
