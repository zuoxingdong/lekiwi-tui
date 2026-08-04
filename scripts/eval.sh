#!/usr/bin/env bash
# ============================================================================
# eval.sh — flag-driven launcher for `lerobot-rollout` (run a trained policy on
# the robot and WATCH it, NOT lerobot-eval which is sim-only).
#
# Mirrors the pilot template scripts/replay.sh (see lib.sh's header for the full
# pattern + flag-surface convention). Eval is the busiest command: it has more
# CONDITIONAL tokens than teleop (--device, the rtc exec-horizon, --duration), so
# it shows several "emit only when …" guards stacked behind the fixed argv head.
#
# This script is the SOLE source of the lerobot-rollout argv (the TUI's EvalScreen and
# its headless path both front it). Local regression tests assert this script's
# --dry-run tokens against an explicit literal list, so a regression here turns the
# suite red. The argv shape:
#
#   lerobot-rollout --config_path <slice rollout> --policy.path=<p>
#       --inference.type=<sync|rtc> --display_data=<true|false>
#       [--task=<t>                                 when --task is non-empty]
#       [--device=cuda                              when --gpu is non-empty]
#       [--inference.rtc.execution_horizon=<eh>     rtc backend only]
#       [--duration=<n>                             when n>0]
#       [--rename_map={k: k, ...}                   when --cam-slots native]
#       [--policy.n_action_steps=<n>                when --action-steps n>0]
#       [--policy.num_steps=<n>                     when --flow-steps n>0]
#       [passthrough...]
#
#   * --config_path is the SPACE (two-token) form, NOT --config_path=...
#   * NOTE: rollout's device override is --device=cuda (a TOP-LEVEL flag), NOT
#     --policy.device — matching bash _eval_launch.
#   * <slice rollout> is `cfg_slice rollout` (lekiwi.yaml's `rollout:` block, sliced
#     to .lekiwi-cache/rollout.yaml; the echoed path == config.cfg_for("rollout")).
#     The command is "eval" but the config block + cache file + binary are all
#     "rollout".
#   * --display on|off maps to the lowercase bool --display_data=true|false, exactly
#     as the Python builder renders it. --display_data is ALWAYS present.
#   * --device=cuda is emitted only when --gpu is non-empty. We take GPU presence as
#     a FLAG (not nvidia-smi here) because only its truthiness matters (the name string
#     never lands in argv), and detecting it in the script would make the golden test
#     host-dependent. The TUI passes app.gpu_name.
#     A power user can still force it via a passthrough --device=... (last-wins).
#   * --inference.rtc.execution_horizon is emitted only for the rtc backend; it is
#     still passed (--exec-horizon) for sync, just not emitted.
#   * --duration maps to --duration=<n>, but ONLY when n>0 (0 = use the config
#     default).
#   * --cam-slots map|native picks which camera keys the policy sees. `map` (default)
#     keeps the slice's rename_map untouched — today's behavior, for checkpoints
#     trained on renamed slots (camera1/2/3). `native` emits an IDENTITY rename_map
#     built from the slice map's KEYS (robot-side names), for checkpoints trained on
#     the raw robot keys (front/wrist/top). An identity map
#     — NOT an empty one — because draccus MERGES dict CLI overrides into the config
#     value (`--rename_map={}` is a silent no-op); re-pointing every key to itself is
#     the only CLI-side way to neutralize the slice map. `auto` is resolved by the
#     TUI (it reads the checkpoint's config.json input_features), NOT here: the
#     script stays non-interactive and never inspects the checkpoint.
#   * --action-steps <n> maps to --policy.n_action_steps=<n>, ONLY when n>0
#     (0 = use the checkpoint's own value). Sync backend pacing: how many actions of
#     the predicted chunk execute open-loop before the next policy forward. rtc
#     ignores n_action_steps (it consumes via execution_horizon), so the TUI only
#     surfaces this knob for sync — but the flag is backend-independent here.
#   * --flow-steps <n> maps to --policy.num_steps=<n>, ONLY when n>0 (0 = the
#     checkpoint's own value; smolvla's standard is 10). Flow-matching Euler
#     integration steps per chunk — applies to sync AND rtc (rtc guides through the
#     integrator). A policy that never reads num_steps simply ignores the override
#     (inert), so there is no model-type gating anywhere.
#   * deliberately NO --use_torch_compile: it was rejected for SmolVLA eval (per-shape
#     recompiles are a training opt, not an eval one), so it stays absent here.
#
# Usage:
#   scripts/eval.sh --policy ~/run/checkpoints/last/pretrained_model --display on
#   scripts/eval.sh --policy lerobot/smolvla_base --backend rtc --exec-horizon 22
#   scripts/eval.sh --policy /p --backend sync --gpu CUDA --duration 60
#   scripts/eval.sh --policy /p --cam-slots native --action-steps 10  # native-key ckpt
#   scripts/eval.sh --dry-run --policy /p --backend rtc --gpu CUDA --exec-horizon 22
#   scripts/eval.sh --policy /p --display on --policy.foo=bar   # trailing flags pass through
#   DRY=1 scripts/eval.sh --policy /p --backend sync            # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: this never prompts. The TUI's EvalScreen resolves + validates the
# checkpoint, gathers Backend / Exec-horizon / Duration / Display, detects the GPU,
# then fronts this script with --policy/--backend/--exec-horizon/--duration/--display/--gpu.
# ============================================================================
set -euo pipefail

# Load the shared helpers (cfg_slice / cfg_get / launcher_get) relative to THIS
# file, so the script is runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# norm_bool <value> -> "true"|"false"
#   Normalize a display value to the lowercase bool token --display_data takes.
#   Accepts the flag form (on/off) and the yaml/cfg_get form (true/false/1/0/yes)
#   so a bare standalone run seeded from cfg_get renders identically. Anything not
#   truthy is "false".
norm_bool() {
  case "${1,,}" in
    on|true|1|yes) printf 'true' ;;
    *)             printf 'false' ;;
  esac
}

# ── defaults (seed a bare standalone run from the yaml, like EvalScreen.on_mount) ──
# The TUI always passes every flag, so these defaults only matter for a human running
# the script with no knobs. backend/exec-horizon/display mirror the env-config knobs
# the screen reads; policy has no yaml default (the screen resolves it) so it must be
# given; duration defaults to 0 (omitted -> use the rollout config default). gpu
# defaults to empty (no --device=cuda) — a standalone user opts in with --gpu.
# Kept as STRINGS so the empty/0 guards below are set -e safe.
policy=""                                              # --policy.path=<p> (required)
task=""                # --task=<t> (policy's language instruction) override; blank -> use
                       # the rollout config default (the slice's `task: *task`).
backend="$(cfg_get rollout.inference.type)"; backend="${backend:-sync}"
eh="$(cfg_get rollout.inference.rtc.execution_horizon)"; eh="${eh:-20}"
display="$(norm_bool "$(cfg_get rollout.display_data)")"  # true|false
duration="0"           # 0/blank -> omit --duration (use the config default)
gpu=""                 # GPU name; non-empty -> emit --device=cuda
cam_slots="map"        # map -> keep the slice rename_map; native -> identity override
steps="0"              # 0/blank -> omit --policy.n_action_steps (checkpoint default)
flow="0"               # 0/blank -> omit --policy.num_steps (checkpoint default)
dry="${DRY:-0}"        # DRY=1 in the env is an alias for --dry-run
extra=()               # passthrough flags appended after the built argv

# ── parse flags ─────────────────────────────────────────────────────────────
# --policy <p>, --backend sync|rtc, --exec-horizon <n>, --duration <n>, --display
# on|off, --gpu <name> are the real knobs. --dry-run flips DRY. Anything else is an
# unrecognized trailing flag: collect it into `extra` and forward it verbatim into
# the lerobot argv (passthrough), so e.g. --device=cpu or --policy.foo=bar still work.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy)
      policy="$2"; shift 2 ;;
    --task)
      task="$2"; shift 2 ;;
    --backend)
      backend="$2"; shift 2 ;;
    --exec-horizon)
      eh="$2"; shift 2 ;;
    --duration)
      duration="$2"; shift 2 ;;
    --display)
      display="$(norm_bool "$2")"; shift 2 ;;
    --gpu)
      gpu="$2"; shift 2 ;;
    --cam-slots)
      cam_slots="$2"; shift 2 ;;
    --action-steps)
      steps="$2"; shift 2 ;;
    --flow-steps)
      flow="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

# The TUI resolves its "auto" mode BEFORE fronting this script; here only the two
# concrete values are legal (fail fast on a typo rather than silently keeping map).
case "$cam_slots" in
  map|native|trained) ;;
  *) die_usage "--cam-slots must be 'map', 'native' or 'trained' (got '$cam_slots'); 'auto' is TUI-side" ;;
esac

# rename_identity_token
#   Print the single --rename_map=… token that neutralizes the slice's rename_map:
#   an IDENTITY map over the slice map's KEYS (the robot-side camera names), in yaml
#   flow style. Draccus MERGES dict CLI overrides into the config value, so every
#   key must be re-pointed to itself — an empty {} would be a no-op. Reads
#   lekiwi.yaml (fallback example) like cfg_get; absent/empty map -> no output, and
#   the caller skips the token (native then degrades to today's behavior).
rename_identity_token() {
  "$PY" - "$ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
cfg = root / "lekiwi.yaml"
if not cfg.exists():
    cfg = root / "lekiwi.example.yaml"
if not cfg.exists():
    sys.exit(0)
doc = yaml.safe_load(cfg.read_text()) or {}
rollout = doc.get("rollout")
rmap = rollout.get("rename_map") if isinstance(rollout, dict) else None
if not isinstance(rmap, dict) or not rmap:
    sys.exit(0)                      # nothing to neutralize -> no token
body = ", ".join(f"{k}: {k}" for k in rmap)
print("--rename_map={" + body + "}")
PY
}

# rename_trained_token
#   Print the --rename_map=… token reproducing the mapping the CHECKPOINT WAS TRAINED
#   WITH, read from its own train_config.json (fallback: saved policy_preprocessor.json).
#
#   This exists because a set-based compatibility check cannot see a PERMUTATION: if the
#   yaml sends wrist->camera2 while the policy was trained with top->camera2, both sides
#   still use exactly {camera1,camera2,camera3}, nothing errors, and the robot simply acts
#   on two swapped views. The checkpoint is the authority on what it expects, so send that.
#   Absent/unreadable -> no output, and the caller falls back to the yaml map.
rename_trained_token() {
  "$PY" - "$policy" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])


def from_preprocessor(d):
    for step in d.get("steps") or []:
        m = (step.get("config") or {}).get("rename_map")
        if isinstance(m, dict) and m:
            return m
    return None


for name, pick in (("train_config.json", lambda d: d.get("rename_map")),
                   ("policy_preprocessor.json", from_preprocessor)):
    try:
        data = json.loads((base / name).read_text())
    except Exception:
        continue
    m = pick(data)
    if isinstance(m, dict) and m:
        print("--rename_map={" + ", ".join(f"{k}: {v}" for k, v in m.items()) + "}")
        break
PY
}

# ── assemble argv (the sole source; pinned by the golden test) ────────────────
# cfg_slice writes the slice and echoes its absolute path; the $() strips the
# trailing newline so the path is a single clean token.
slice="$(cfg_slice rollout)"
# We front lerobot-rollout through lerobot_rollout_kbd.py, a thin shim that swaps
# pynput's keyboard Listener (a silent no-op on Wayland: episodic arrows / highlight
# save+push / dagger space+tab+enter and ESC never reach the rollout loop, only Ctrl+C
# does) for a stdin reader emitting real pynput key objects, then calls lerobot's own
# main() with this exact argv. lerobot's repo is untouched; the only change vs
# `lerobot-rollout` is the first two tokens (`python <shim>`). NOTE: the shim only
# delivers keys when the child inherits a real TTY — which now holds on EVERY path:
# direct `bash eval.sh`, direct-mode `python -m lekiwi_tui eval`, AND the
# interactive TUI EvalScreen (which suspends into the child via runner.suspend_run,
# like record).
argv=(
  python "$SCRIPT_DIR/lerobot_rollout_kbd.py"
  --config_path "$slice"
  "--policy.path=${policy}"
  "--inference.type=${backend}"
  "--display_data=${display}"
)
# Conditional tokens, appended in this fixed order (pinned by the golden test):
#   1) the top-level --task (policy's language instruction) only when non-empty. Blank
#      uses the rollout config default (the slice's `task: *task`). The TUI passes the
#      current task every run; a standalone run omits it unless --task is given. One
#      token even with spaces (--task="pick up the cube").
if [[ -n "$task" ]]; then
  argv+=("--task=${task}")
fi
#   2) --device=cuda only when a GPU is present (--gpu non-empty).
if [[ -n "$gpu" ]]; then
  argv+=("--device=cuda")
fi
#   3) the rtc exec-horizon only for the rtc backend.
if [[ "$backend" == "rtc" ]]; then
  argv+=("--inference.rtc.execution_horizon=${eh}")
fi
#   4) --duration only when n>0 (0 = use the config default). The -n guard keeps
#      -gt from tripping set -e on an empty value.
if [[ -n "$duration" && "$duration" -gt 0 ]]; then
  argv+=("--duration=${duration}")
fi
#   5) the identity rename_map only for --cam-slots native (one token even with
#      spaces inside the braces). Skipped when the yaml has no map to neutralize.
if [[ "$cam_slots" == "native" ]]; then
  identity_token="$(rename_identity_token)"
  if [[ -n "$identity_token" ]]; then
    argv+=("$identity_token")
  fi
elif [[ "$cam_slots" == "trained" ]]; then
  trained_token="$(rename_trained_token)"
  if [[ -n "$trained_token" ]]; then
    argv+=("$trained_token")
  fi
fi
#   6) --policy.n_action_steps only when --action-steps n>0 (0 = the checkpoint's
#      own value). Same -n guard as --duration.
if [[ -n "$steps" && "$steps" -gt 0 ]]; then
  argv+=("--policy.n_action_steps=${steps}")
fi
#   7) --policy.num_steps only when --flow-steps n>0 (0 = the checkpoint's own
#      value). Same -n guard as --duration.
if [[ -n "$flow" && "$flow" -gt 0 ]]; then
  argv+=("--policy.num_steps=${flow}")
fi
# Append any passthrough flags last (so a user --device= wins, draccus last-wins).
# Empty in the TUI path, so the launch test still matches.
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
# --dry-run / DRY=1: print the argv one token per line and exit 0 (the parity gate
# captures these lines). Otherwise exec, so the rollout shim inherits this script's
# real TTY (the TUI suspends into it; the shim reads arrow keys / ESC straight from
# that stdin — see lerobot_rollout_kbd.py).
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

# ── preflight: will this checkpoint's config even load? ──────────────────────
# A checkpoint trained on a lerobot with a config field this one lacks (an unreleased
# fix, a local patch, a policy plugin) fails inside draccus with a ~60-line traceback,
# and it fails AFTER the form is filled, the GPU chosen and the robot about to move.
# ckpt_preflight.py answers that question in one paragraph first. It exits non-zero
# ONLY when the config genuinely will not load; "cannot tell" (hub repo id, no
# config.json, lerobot not importable) passes through silently, because a preflight
# that blocks a working launch is worse than no preflight. Verdicts are cached, so
# only the first launch of a new checkpoint pays the import.
# Escape hatch: EVAL_SKIP_PREFLIGHT=1 (dry-run skips it already, one exit above).
if [[ "${EVAL_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$PY" "$SCRIPT_DIR/ckpt_preflight.py" "$policy" || exit $?
fi

exec "${argv[@]}"
