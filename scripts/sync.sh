#!/usr/bin/env bash
# ============================================================================
# sync.sh — rsync the local lerobot clone up to the Pi (the "Sync to Pi" action).
#
# A NEW launcher (no lerobot-* CLI): the everyday "laptop is the source of truth"
# button. It MIRRORS the clone + the PincOpen plugin to the Pi (editable installs there
# pick code changes up with no build step), and — because it fingerprints the two
# pyproject.toml files on the Pi before/after the rsync — it re-runs the editable
# install AUTOMATICALLY when the sync changed dependencies (a version bump like
# 0.5.2→0.6). So "whenever something changed, just sync again" is literally the whole
# workflow; `setup-pi` (pi_provision.sh) is only for first bring-up / python changes.
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the pattern).
# It reuses pi_provision.sh's `lerobot` stage rsync recipe VERBATIM (same flags, same
# excludes) so the two stay consistent — change the recipe in BOTH if it ever moves:
#
#   ssh <host> "mkdir -p '<repo>'"        # ensure the target dir exists (first sync)
#   rsync -az --delete --exclude '.git' …  <LOCAL_REPO>/  <host>:<repo>/
#
#   * -az = archive + compress. --delete makes the Pi tree an exact mirror so a
#     version jump leaves no stale modules behind; it is SAFE here because the only
#     Pi-local artifacts in the tree (*.egg-info/, __pycache__/) are on the exclude
#     list, which also protects them from deletion (rsync never deletes excluded
#     paths without --delete-excluded), and calibration lives in ~/.cache, outside
#     the tree. Excludes match pi_provision.sh exactly.
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
#   scripts/sync.sh --install               # force the editable installs (fresh env)
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
# Pi conda env, needed only for the conditional dep reinstall after a fingerprint change.
conda_env="${CONDA_ENV:-$(launcher_get CONDA_ENV)}"; conda_env="${conda_env:-lekiwi}"
# The clone shipped to the Pi: env var > _launcher.LOCAL_REPO > sibling default (with
# a parent-workspace fallback for older local layouts). A relative value resolves
# against ROOT, the same rule as config.resolve_workspace_path — keep the two in step.
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
resolve_workspace_path() {           # '' stays ''; relative resolves against $ROOT
  local value="${1:-}"
  [[ -z "$value" ]] && { printf '%s\n' ""; return; }
  case "$value" in
    "~"*) value="${HOME}${value#\~}" ;;
  esac
  [[ "$value" == /* ]] && { printf '%s\n' "$value"; return; }
  printf '%s\n' "$ROOT/$value"
}
local_repo="$(resolve_workspace_path "${LOCAL_REPO:-$(launcher_get LOCAL_REPO)}")"
local_repo="${local_repo:-$(default_local_repo)}"
# The PincOpen robot plugin rides along: env var > _launcher.LOCAL_PLUGIN > sibling
# default. Shipped next to the clone on the Pi (its editable install there picks
# changes up the same way). PI_PLUGIN overrides the Pi-side destination.
local_plugin="$(resolve_workspace_path "${LOCAL_PLUGIN:-$(launcher_get LOCAL_PLUGIN)}")"
local_plugin="${local_plugin:-$(cd "$ROOT/.." && pwd -P)/lerobot_robot_lekiwi_pincopen}"
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough rsync flags appended after the recipe flags

# ── parse flags ─────────────────────────────────────────────────────────────
# --host <h>, --repo <r> override the destination. --dry-run flips DRY. Anything else
# is an unrecognized trailing flag: collect it into `extra` and forward it verbatim
# into the rsync argv (passthrough), e.g. --delete or --exclude '…'.
force_install=0          # --install: run the editable installs even if unchanged
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="$2"; shift 2 ;;
    --repo)
      repo="$2"; shift 2 ;;
    --install)
      force_install=1; shift ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

validate_ssh_host "$host" "LEKIWI_HOST"
validate_remote_path "$repo" "PI_REPO"
remote_repo="$(shell_quote "${repo%/}/")"
# Plugin destination: next to the repo on the Pi (lekiwi/lerobot ->
# lekiwi/lerobot_robot_lekiwi_pincopen), matching pi_provision.sh's PI_PLUGIN default.
plugin="${PI_PLUGIN:-$(dirname "$repo")/lerobot_robot_lekiwi_pincopen}"
validate_remote_path "$plugin" "PI_PLUGIN"
remote_plugin="$(shell_quote "${plugin%/}/")"

# ── assemble the rsync argv (pi_provision.sh `lerobot` stage recipe, verbatim) ──
# trailing slash on the source copies its CONTENTS into the dest dir (not a nested dir).
argv=(
  rsync -az --delete
  --exclude '.git' --exclude '__pycache__/' --exclude '*.pyc'
  --exclude '*.egg-info/' --exclude 'outputs/' --exclude 'wandb/' --exclude 'logs/'
)
# Passthrough flags sit with the other rsync options, BEFORE the source/dest operands.
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi
argv+=("${local_repo}/" "${host}:${remote_repo}")

# Second push: the PincOpen plugin, fixed recipe (no passthrough — the extras target
# the clone; the plugin is tiny).
plugin_argv=(
  rsync -az --delete
  --exclude '.git' --exclude '__pycache__/' --exclude '*.pyc'
  --exclude '*.egg-info/'
  "${local_plugin}/" "${host}:${remote_plugin}"
)

# ── dry-run vs run ───────────────────────────────────────────────────────────
# --dry-run / DRY=1: print both rsync argvs one token per line and exit 0 (the parity
# gate captures these lines); touch NOTHING (no ssh, no network). Otherwise: print the
# shipping provenance, fingerprint the two remote pyproject.toml files, run both rsyncs
# in the foreground (they inherit this script's real TTY: progress + any ssh auth
# prompt reach the user; the TUI suspends into it), then re-fingerprint — if a
# pyproject changed, the sync ALSO changed dependencies, so re-run the editable
# installs on the Pi (uv; same flags as pi_provision.sh's lerobot stage). A plain code
# sync leaves the fingerprints identical and skips the install entirely.
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  printf '%s\n' "${plugin_argv[@]}"
  exit 0
fi

[ -d "$local_repo" ] || { printf 'sync.sh: local clone not found: %s\n' "$local_repo" >&2; exit 1; }
[ -d "$local_plugin" ] || { printf 'sync.sh: PincOpen plugin not found: %s\n' "$local_plugin" >&2; exit 1; }

# Provenance: say exactly what is being shipped (version + branch + commit), so a wrong
# checkout is caught by eye before it lands on the robot.
repo_ver="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$local_repo/pyproject.toml" 2>/dev/null | head -1)"
repo_ref="$(git -C "$local_repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
repo_sha="$(git -C "$local_repo" rev-parse --short HEAD 2>/dev/null || true)"
plugin_ver="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$local_plugin/pyproject.toml" 2>/dev/null | head -1)"
printf 'shipping lerobot %s (%s @ %s) + lerobot_robot_lekiwi_pincopen %s  ->  %s\n' \
  "${repo_ver:-?}" "${repo_ref:-?}" "${repo_sha:-?}" "${plugin_ver:-?}" "$host"

# Dependency fingerprint BEFORE the sync (missing files hash to nothing — first sync
# then registers as "changed" and triggers the install, which is what we want).
fingerprint() {
  ssh "$host" "sha256sum -- ${remote_repo}pyproject.toml ${remote_plugin}pyproject.toml 2>/dev/null" || true
}
pre_fp="$(fingerprint)"

ssh "$host" "mkdir -p -- $remote_repo $remote_plugin"
"${argv[@]}"
"${plugin_argv[@]}"

post_fp="$(fingerprint)"
if [[ "$force_install" != "1" && "$pre_fp" == "$post_fp" ]]; then
  echo "sync.sh: dependencies unchanged — editable installs already current."
  exit 0
fi

if [[ "$force_install" == "1" ]]; then
  echo "sync.sh: --install — running the editable installs on the Pi (env '$conda_env')…"
else
  echo "sync.sh: pyproject changed — re-running the editable installs on the Pi (env '$conda_env')…"
fi
# Same install recipe as pi_provision.sh's lerobot stage (CPU torch, no CUDA source pins).
ssh "$host" "PI_ENV=$(shell_quote "$conda_env") PI_REPO=$(shell_quote "${repo%/}") PI_PLUGIN=$(shell_quote "${plugin%/}") MAMBA_PREFIX=$(shell_quote "${MAMBA_PREFIX:-}") bash -s" <<'SH'
set -euo pipefail
MAMBA="${MAMBA_PREFIX:-$HOME/miniforge3}"
uv="$MAMBA/envs/$PI_ENV/bin/uv"
py="$MAMBA/envs/$PI_ENV/bin/python"
if [ ! -x "$uv" ]; then
    echo "sync.sh: env '$PI_ENV' has no uv — run Set up Pi (conda + lerobot stages) first." >&2
    exit 1
fi
"$uv" pip install --python "$py" --no-sources --torch-backend=cpu -e "${PI_REPO}[lekiwi]"
"$uv" pip install --python "$py" -e "$PI_PLUGIN"
# One-time calibration seed: the plugin robot is named lekiwi_pincopen, so its
# calibration lives in calibration/robots/lekiwi_pincopen/. Seed it from the old
# robots/lekiwi/ files so an already-calibrated Pi keeps working (idempotent).
CAL="${HF_LEROBOT_CALIBRATION:-$HOME/.cache/huggingface/lerobot/calibration}/robots"
if [ -d "$CAL/lekiwi" ] && [ ! -d "$CAL/lekiwi_pincopen" ]; then
    cp -r "$CAL/lekiwi" "$CAL/lekiwi_pincopen"
    echo "  calibration: seeded robots/lekiwi_pincopen from robots/lekiwi"
fi
"$py" -c 'import lerobot, lerobot_robot_lekiwi_pincopen.lekiwi_host
print("install OK  lerobot", lerobot.__version__, "+ plugin registered")'
SH
