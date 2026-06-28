#!/usr/bin/env bash
# ============================================================================
# train.sh — flag-driven launcher for `lerobot-train` (the CAREFUL one).
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the full
# pattern + flag-surface convention), but train has three traits the others do
# not. This script is the SINGLE SOURCE of the lerobot-train argv; its validated
# tokens are pinned by the static golden in test_train.py:
#
#   1. --config_path is the SINGLE-token `=` form  (--config_path=<path>), NOT the
#      space form the other scripts use. lerobot's parser.parse_arg only matches
#      --key=value, and both the yaml policy-block extraction (fresh) and the resume
#      train_config lookup depend on it (the space form silently skips both).
#   2. OFFLINE ENV: train exports HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 so the run
#      stays fully local — EXCEPT a FRESH init from a hub id (a non-directory init,
#      e.g. lerobot/smolvla_base) which must go online ONCE to download the base.
#      This script OWNS that env: it exports the keys for the right case before
#      exec, so it is self-contained standalone AND the TUI can front it with a
#      plain inherited environment (RunScreen passes env=None).
#   3. FRESH vs RESUME branch:
#        fresh : --config_path=<slice train> --policy.path=<init> --output_dir=…
#                --job_name=<run> --steps --batch_size --save_freq
#                [--policy.device=cuda when a GPU is named]
#                [--policy.use_amp=true when AMP is on]            (fresh only)
#        resume: --resume=true --config_path=<run>/…/train_config.json --output_dir=…
#                --dataset.root --dataset.repo_id --batch_size --num_workers
#                [--steps only when a new TOTAL is given]
#      A SERVER-trained run's saved train_config carries remote paths + a server
#      batch size; resume overrides output_dir / dataset.root / dataset.repo_id /
#      batch_size / num_workers back to LOCAL values so it runs on this machine too.
#
# --dry-run prints the EXPORTED ENV first (e.g. HF_HUB_OFFLINE=1), one KEY=value
# per line, THEN the argv starting at the `lerobot-train` token (one token per
# line). The golden test splits the output at `lerobot-train`: lines before it are
# the offline env, lines from it on are the argv it asserts against the golden.
#
# Usage:
#   scripts/train.sh --mode fresh --run myrun --init lerobot/smolvla_base \
#       --steps 20000 --batch 8 --save 5000 --amp on --gpu CUDA \
#       --policy-root /home/me/models
#   scripts/train.sh --mode resume --run myrun --batch 4 --policy-root /home/me/models
#   scripts/train.sh --dry-run --mode fresh --run r --init /ckpt --policy-root /m ...
#   DRY=1 scripts/train.sh --mode fresh ...                 # env-var dry-run (tests)
#
# NON-INTERACTIVE: this never prompts. The TUI's TrainScreen gathers the run name,
# init checkpoint, steps/batch/save, AMP, detects resume (Resume/Cancel + new total
# steps), then fronts this script with the resolved flags. --policy-root and --gpu
# are passed by the TUI (app.cfg POLICY_ROOT, app.gpu_name) so the script does not
# have to re-resolve env-var/default precedence or probe the GPU itself.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# norm_bool <value> -> "on"|"off"
#   Normalize an AMP value to a stable on/off the --policy.use_amp guard reads.
#   Accepts the flag form (on/off) and the yaml/cfg_get form (true/false/True/
#   False/1/0) so a bare standalone run seeded from cfg_get behaves identically.
norm_bool() {
  case "${1,,}" in
    on|true|1|yes) printf 'on' ;;
    *)             printf 'off' ;;
  esac
}

# ── defaults (seed a bare standalone run from the yaml, like TrainScreen.on_mount) ──
# The TUI always passes every flag, so these defaults only matter for a human running
# the script with no knobs. Numeric defaults mirror the train.* yaml block (the same
# fallbacks train_yaml_int uses in the TUI); amp seeds from train.policy.use_amp.
# Kept as STRINGS so the empty-value guards below are set -e safe.
mode="fresh"
run=""
init="lerobot/smolvla_base"
steps="$(cfg_get train.steps)";       steps="${steps:-20000}"
batch="$(cfg_get train.batch_size)";  batch="${batch:-8}"
save="$(cfg_get train.save_freq)";    save="${save:-5000}"
amp="$(norm_bool "$(cfg_get train.policy.use_amp)")"   # on|off
gpu=""                 # GPU name; nonempty -> --policy.device=cuda (FRESH only)
policy_root=""         # output root; the TUI passes app.cfg POLICY_ROOT (resolved)
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

# ── parse flags ─────────────────────────────────────────────────────────────
# Long flags map to the knobs above. --dry-run flips DRY. Anything else is an
# unrecognized trailing flag: collect it into `extra` and forward it verbatim into
# the lerobot argv (passthrough), e.g. --policy.foo=bar.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        mode="$2"; shift 2 ;;
    --run)         run="$2"; shift 2 ;;
    --init)        init="$2"; shift 2 ;;
    --steps)       steps="$2"; shift 2 ;;
    --batch)       batch="$2"; shift 2 ;;
    --save)        save="$2"; shift 2 ;;
    --amp)         amp="$(norm_bool "$2")"; shift 2 ;;
    --gpu)         gpu="$2"; shift 2 ;;
    --policy-root) policy_root="$2"; shift 2 ;;
    --dry-run)     dry=1; shift ;;
    *)             extra+=("$1"); shift ;;   # tolerate + forward passthrough flags
  esac
done

# outdir = POLICY_ROOT/run. The TUI passes str(Path(app.cfg POLICY_ROOT)) which is
# already trailing-slash-normalized, so a plain "$policy_root/$run" join matches
# str(Path(policy_root) / run) token-for-token.
outdir="${policy_root}/${run}"

# ── offline decision (offline True unless a FRESH init from a hub id) ──
# A FRESH init that is NOT an existing directory is a hub id (smolvla_base) and goes
# online ONCE; every other case (fresh-from-local-dir, resume) stays offline.
offline=1
if [[ "$mode" == "fresh" && ! -d "$init" ]]; then
  offline=0
fi

# ── assemble argv (the single source of truth, pinned by the test_train.py golden) ──
if [[ "$mode" == "resume" ]]; then
  # Resume: read the LOCAL overrides from the yaml with the same fallbacks the TUI
  # uses (train.dataset.root -> ../datasets/lekiwi_dataset; train.num_workers -> 4).
  ds_root="$(cfg_get train.dataset.root)";   ds_root="${ds_root:-../datasets/lekiwi_dataset}"
  ds_repo="$(cfg_get train.dataset.repo_id)"
  nworkers="$(cfg_get train.num_workers)";   nworkers="${nworkers:-4}"
  argv=(
    lerobot-train
    --resume=true
    # SINGLE-token `=` form; resume reads the saved train_config (NOT the slice).
    "--config_path=${outdir}/checkpoints/last/pretrained_model/train_config.json"
    "--output_dir=${outdir}"
    "--dataset.root=${ds_root}"
    "--dataset.repo_id=${ds_repo}"
    "--batch_size=${batch}"
    "--num_workers=${nworkers}"
  )
  # --steps only when the user gave a new TOTAL (blank rsteps -> omit the token).
  if [[ -n "$steps" ]]; then
    argv+=("--steps=${steps}")
  fi
else
  # Fresh: --config_path is the sliced train block (SINGLE-token `=` form). cfg_slice
  # writes the slice and echoes its absolute path; $() strips the trailing newline.
  slice="$(cfg_slice train)"
  argv=(
    lerobot-train
    "--config_path=${slice}"
    "--policy.path=${init}"
    "--output_dir=${outdir}"
    "--job_name=${run}"
    "--steps=${steps}"
    "--batch_size=${batch}"
    "--save_freq=${save}"
  )
  # --policy.device=cuda only when a GPU is named (FRESH only).
  if [[ -n "$gpu" ]]; then
    argv+=("--policy.device=cuda")
  fi
  # --policy.use_amp=true only when AMP is on (FRESH only; on resume the saved
  # train_config's value wins).
  if [[ "$amp" == "on" ]]; then
    argv+=("--policy.use_amp=true")
  fi
fi
# Append any passthrough flags (empty in the TUI path, so the launch test still
# matches; populated only when a human adds extra lerobot overrides).
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the EXPORTED ENV first (KEY=value, only when offline),
# then the argv one token per line, and exit 0. The parity test splits at the
# `lerobot-train` token. Otherwise export the env for real (offline case) and exec,
# so lerobot-train inherits this script's TTY (the TUI streams it via RunScreen).
if [[ "$dry" == "1" ]]; then
  if [[ "$offline" == "1" ]]; then
    printf '%s\n' "HF_HUB_OFFLINE=1" "HF_DATASETS_OFFLINE=1"
  fi
  printf '%s\n' "${argv[@]}"
  exit 0
fi

if [[ "$offline" == "1" ]]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
exec "${argv[@]}"
