#!/usr/bin/env bash
# ============================================================================
# dagger.sh — flag-driven launcher for `lerobot-rollout --strategy.type=dagger`
# (HIL correction collection: the policy drives, you take over with the leader
# when it goes wrong; each correction saves as one episode).
#
# Mirrors scripts/eval.sh (same lib.sh pattern, same dry-run golden-test gate);
# the differences are exactly dagger's needs: a teleop in the slice, a dataset
# being written (so --dataset.root / --dataset.single_task / the corrections
# target), and the strategy knobs. The argv shape:
#
#   python <scripts>/lerobot_dagger_kbd.py --config_path <slice dagger> --policy.path=<p>
#       --inference.type=<sync|rtc> --display_data=<true|false>
#       --dataset.root=<root>
#       [--task=<t>]                              when --task is non-empty
#       [--dataset.single_task=<t>]               when --task is non-empty
#       [--device=cuda]                           when --gpu is non-empty
#       [--inference.rtc.execution_horizon=<eh>   rtc backend only]
#       [--strategy.num_episodes=<n>              when --target n>0]
#       [--strategy.record_autonomous=true        when --record-autonomous on]
#       [--strategy.input_device=pedal            when --input pedal]
#       [--duration=<n>                           when n>0]
#       [--rename_map={...}                       when --cam-slots native|trained]
#       [--policy.n_action_steps=<n>              when --action-steps n>0]
#       [--policy.num_steps=<n>                   when --flow-steps n>0]
#       [--policy.compile_model=<true|false>      when --compile on|off]
#       [passthrough...]
#
#   * --config_path is the SPACE (two-token) form, NOT --config_path=... .
#   * <slice dagger> is `cfg_slice dagger` (lekiwi.yaml's `dagger:` block, sliced to
#     .lekiwi-cache/dagger.yaml).
#   * --dataset.root is ALWAYS emitted. Blank --dataset-root computes a per-session
#     stamped dir under ../../datasets (rollout stamps only repo_id, never the root,
#     so a reused root fails on the existing directory — the stamp is ours to add).
#   * --task feeds BOTH the policy conditioning (--task) and the dataset stamp
#     (--dataset.single_task): dagger records single_task onto every correction, and
#     the two must agree with the scene or the policy runs the wrong goal.
#   * --target maps to --strategy.num_episodes (session ends after N saved
#     corrections), only when n>0 (0 = the config/dataset default).
#   * --record-autonomous on flips corrections-only into sentry-style record-all
#     (frames tagged intervention true/false). Off is the yaml default; only `on`
#     emits a token.
#   * --input keyboard|pedal — only `pedal` emits a token (keyboard is the default).
#     Pedal device/key codes ride in via passthrough (--strategy.pedal.*).
#   * --cam-slots map|native|trained: same semantics + helpers as eval.sh, reading
#     THIS script's `dagger:` yaml section. `auto` is TUI-side, rejected here.
#   * --compile auto|on|off: same semantics as eval.sh (the checkpoint's OWN
#     policy.compile_model; auto emits no token). A DAgger session is long, so paying
#     the compile once is usually right here — but it is the same knob either way.
#   * --action-steps / --flow-steps: same semantics as eval.sh (sync chunk pacing /
#     FM integration steps; 0 = the checkpoint's own value, no token emitted). The
#     autonomous phase is a live rollout, so its pacing knobs matter here just as
#     much — a long open-loop chunk also delays how fast `space` takes effect.
#
# Usage:
#   scripts/dagger.sh --policy /ckpt --task "Pick up the red cube…" --target 10
#   scripts/dagger.sh --policy /ckpt --backend rtc --exec-horizon 22 --gpu CUDA
#   scripts/dagger.sh --dry-run --policy /p --dataset-root ../../datasets/rollout_x
#   DRY=1 scripts/dagger.sh --policy /p              # env-var dry-run (tests / CI)
#
# NON-INTERACTIVE: never prompts. The TUI's DaggerScreen resolves the checkpoint,
# the base dataset's task string, the corrections target and the GPU, then fronts
# this script with --policy/--task/--target/--backend/--display/--gpu/--dataset-root.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# norm_bool <value> -> "true"|"false"  (same acceptance set as eval.sh)
norm_bool() {
  case "${1,,}" in
    on|true|1|yes) printf 'true' ;;
    *)             printf 'false' ;;
  esac
}

# ── defaults (seed a bare standalone run from the yaml, like the screen does) ──
policy=""                                              # --policy.path=<p> (required)
task=""                # session task string; blank -> the slice's `task:`/`single_task:`
backend="$(cfg_get dagger.inference.type)"; backend="${backend:-sync}"
eh="$(cfg_get dagger.inference.rtc.execution_horizon)"; eh="${eh:-20}"
display="$(norm_bool "$(cfg_get dagger.display_data)")"
target="0"             # 0/blank -> omit --strategy.num_episodes (config default)
record_autonomous="off"
input="keyboard"
duration="0"           # 0/blank -> omit --duration (config default: no limit)
gpu=""                 # GPU name; non-empty -> emit --device=cuda
cam_slots="map"        # map -> keep the slice rename_map (same modes as eval.sh)
steps="0"              # 0/blank -> omit --policy.n_action_steps (checkpoint default)
flow="0"               # 0/blank -> omit --policy.num_steps (checkpoint default)
compile_mode="auto"    # auto -> omit --policy.compile_model (checkpoint's own setting)
dataset_root=""        # blank -> per-session stamped dir (computed below)
dry="${DRY:-0}"
extra=()

# ── parse flags ─────────────────────────────────────────────────────────────
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
    --target)
      target="$2"; shift 2 ;;
    --record-autonomous)
      record_autonomous="$2"; shift 2 ;;
    --input)
      input="$2"; shift 2 ;;
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
    --compile)
      compile_mode="$2"; shift 2 ;;
    --dataset-root)
      dataset_root="$2"; shift 2 ;;
    --dry-run)
      dry=1; shift ;;
    *)
      extra+=("$1"); shift ;;   # tolerate + forward trailing passthrough flags
  esac
done

case "$cam_slots" in
  map|native|trained) ;;
  *) die_usage "--cam-slots must be 'map', 'native' or 'trained' (got '$cam_slots'); 'auto' is TUI-side" ;;
esac
# --compile keeps its `auto` here (it just means "emit no token"); see eval.sh.
case "$compile_mode" in
  auto|on|off) ;;
  *) die_usage "--compile must be 'auto', 'on' or 'off' (got '$compile_mode')" ;;
esac
case "$input" in
  keyboard|pedal) ;;
  *) die_usage "--input must be 'keyboard' or 'pedal' (got '$input')" ;;
esac

# Per-session dataset dir. Rollout stamps repo_id but NEVER the root, so a fixed
# root fails on the second session ("directory exists"); stamping the dir here keeps
# every session collision-free AND visible to the TUI's dataset picker (../../datasets
# = the my_robot data root, same convention as record's dataset.root).
if [[ -z "$dataset_root" ]]; then
  dataset_root="../../datasets/rollout_dagger_$(date +%Y%m%d_%H%M%S)"
fi

# rename_identity_token — identical to eval.sh's, over the `dagger:` section's map.
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
dagger = doc.get("dagger")
rmap = dagger.get("rename_map") if isinstance(dagger, dict) else None
if not isinstance(rmap, dict) or not rmap:
    sys.exit(0)                      # nothing to neutralize -> no token
body = ", ".join(f"{k}: {k}" for k in rmap)
print("--rename_map={" + body + "}")
PY
}

# rename_trained_token — identical to eval.sh's: reproduce the mapping the checkpoint
# was trained with (train_config.json, fallback policy_preprocessor.json).
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
slice="$(cfg_slice dagger)"
# We front lerobot-rollout through lerobot_dagger_kbd.py — the dagger twin of
# record's shim, for the same two reasons record needs one and eval does not:
# the base wasd during corrections is HOLD-to-move (needs key release: kitty
# stdin or evdev, which upstream's Wayland fallback never provides), and the
# session keys (space/tab/enter/ESC) must share ONE stdin reader with it or the
# two threads race for the bytes on fd 0. The only change vs `lerobot-rollout`
# is the first two tokens (`python <shim>`); the child still needs (and gets)
# the real TTY on every path: direct bash, direct-mode CLI, TUI suspend.
argv=(
  python "$SCRIPT_DIR/lerobot_dagger_kbd.py"
  --config_path "$slice"
  "--policy.path=${policy}"
  "--inference.type=${backend}"
  "--display_data=${display}"
  "--dataset.root=${dataset_root}"
)
# Conditional tokens, appended in this fixed order (pinned by the golden test):
#   1) the session task string — policy conditioning AND the dataset stamp. Blank
#      uses the slice defaults (`task:` / `dataset.single_task:`), which the TUI
#      never relies on (it always passes the picked string).
if [[ -n "$task" ]]; then
  argv+=("--task=${task}")
  argv+=("--dataset.single_task=${task}")
fi
#   2) --device=cuda only when a GPU is present (--gpu non-empty).
if [[ -n "$gpu" ]]; then
  argv+=("--device=cuda")
fi
#   3) the rtc exec-horizon only for the rtc backend.
if [[ "$backend" == "rtc" ]]; then
  argv+=("--inference.rtc.execution_horizon=${eh}")
fi
#   4) the corrections target only when n>0 (0 = the config default).
if [[ -n "$target" && "$target" -gt 0 ]]; then
  argv+=("--strategy.num_episodes=${target}")
fi
#   5) record-all mode only when switched on (the yaml default is corrections-only).
if [[ "$(norm_bool "$record_autonomous")" == "true" ]]; then
  argv+=("--strategy.record_autonomous=true")
fi
#   6) the input device only for pedal (keyboard is the yaml default).
if [[ "$input" == "pedal" ]]; then
  argv+=("--strategy.input_device=pedal")
fi
#   7) --duration only when n>0 (0 = the config default: no time limit).
if [[ -n "$duration" && "$duration" -gt 0 ]]; then
  argv+=("--duration=${duration}")
fi
#   8) the rename_map override for native/trained cam-slots (one token even with
#      spaces inside the braces). Skipped when there is nothing to emit.
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
#   9) --policy.n_action_steps only when --action-steps n>0 (0 = the checkpoint's
#      own value). Same -n guard as --duration.
if [[ -n "$steps" && "$steps" -gt 0 ]]; then
  argv+=("--policy.n_action_steps=${steps}")
fi
#  10) --policy.num_steps only when --flow-steps n>0 (0 = the checkpoint's own value).
if [[ -n "$flow" && "$flow" -gt 0 ]]; then
  argv+=("--policy.num_steps=${flow}")
fi
#  11) --policy.compile_model only when --compile is on|off (auto = checkpoint's own).
if [[ "$compile_mode" != "auto" ]]; then
  argv+=("--policy.compile_model=$([[ "$compile_mode" == "on" ]] && echo true || echo false)")
fi
# Append any passthrough flags last (draccus last-wins). Empty in the TUI path.
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

# ── dry-run vs exec ──────────────────────────────────────────────────────────
if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  exit 0
fi

# ── preflight: will this checkpoint's config even load? (same gate as eval.sh) ──
if [[ "${EVAL_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$PY" "$SCRIPT_DIR/ckpt_preflight.py" "$policy" || exit $?
fi

exec "${argv[@]}"
