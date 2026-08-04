#!/usr/bin/env bash
# ============================================================================
# edit.sh — flag-driven launcher for `lerobot-edit-dataset` (delete episodes /
# retag tasks), with GUARANTEED IN-PLACE + BACKUP semantics for BOTH ops.
#
# Follows the lib.sh launcher pattern (flag-driven, non-interactive, dry-run
# golden-testable) with one deliberate extension: lerobot-edit-dataset's own
# path semantics are inconsistent —
#   * delete_episodes: does NOT edit in place; without --new_root it silently
#     writes the result to $HF_LEROBOT_HOME/<repo_id> (the HF cache).
#   * modify_tasks:    ONLY edits in place; --new_root is ignored outright.
# This script normalizes both to one contract:
#
#   delete:  tool writes --new_root <root>.edit-tmp     ─┐
#   retag:   cp -a <root> <root>.edit-tmp, tool edits it ─┤ then SWAP:
#                                                         ├ mv <root> <root>.bak-<ts>
#                                                         └ mv <root>.edit-tmp <root>
#
# The dataset path never changes, every edit leaves a timestamped backup next
# to it, and a tool failure removes the temp dir leaving the original
# byte-identical.
#
# Usage:
#   scripts/edit.sh --episodes "[12]"                          # delete (default op)
#   scripts/edit.sh --op retag --episodes "[3,7]" --task "Pick up the gum …"
#   scripts/edit.sh --root /path/ds --episodes "[0]" --dry-run # print plan only
#   DRY=1 scripts/edit.sh --episodes "[12]"                    # env-var dry-run
#
# NON-INTERACTIVE: this never prompts. The TUI's DatasetEditScreen gathers the
# marked indices (and the new task text / typed confirm), then fronts this script.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# ── defaults (a bare standalone run targets the `record:` dataset) ───────────
repo_id="$(cfg_get record.dataset.repo_id)"; repo_id="${repo_id:-local/lekiwi_dataset}"
root="$(cfg_get record.dataset.root)"; root="${root:-../datasets/lekiwi_dataset}"
op="delete"            # delete | retag
episodes=""            # JSON list of indices, e.g. "[0, 2, 5]"
task=""                # retag only: the new task text for the given episodes
dry="${DRY:-0}"
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --op)       op="$2"; shift 2 ;;
    --repo-id)  repo_id="$2"; shift 2 ;;
    --root)     root="$2"; shift 2 ;;
    --episodes) episodes="$2"; shift 2 ;;
    --task)     task="$2"; shift 2 ;;
    --dry-run)  dry=1; shift ;;
    *)          extra+=("$1"); shift ;;   # forwarded verbatim (passthrough)
  esac
done

[[ "$op" == "delete" || "$op" == "retag" ]] || {
  echo "edit.sh: --op must be delete or retag (got '$op')" >&2; exit 2; }
[[ -n "$episodes" ]] || { echo "edit.sh: --episodes \"[i, j, …]\" is required" >&2; exit 2; }
[[ "$op" != "retag" || -n "$task" ]] || {
  echo "edit.sh: --task \"<new instruction>\" is required for --op retag" >&2; exit 2; }

# ── safety: refuse to edit anything that is not a real dataset folder ────────
root="${root%/}"
case "$root" in
  ""|"/"|"$HOME") echo "edit.sh: refusing unsafe dataset root '$root'" >&2; exit 2 ;;
esac
[[ -f "$root/meta/info.json" ]] || {
  echo "edit.sh: no dataset at '$root' (missing meta/info.json)" >&2; exit 2; }

tmp="${root}.edit-tmp"
bak="${root}.bak-$(date +%Y%m%d-%H%M%S)"

# ── assemble argv (single source of truth, pinned by the dry-run goldens) ────
if [[ "$op" == "delete" ]]; then
  argv=(
    lerobot-edit-dataset
    --repo_id "$repo_id"
    --root "$root"
    --new_repo_id "$repo_id"
    --new_root "$tmp"
    --operation.type delete_episodes
    --operation.episode_indices "$episodes"
  )
else
  # {"<idx>": "<task>", …} for the marked episodes — built in python so arbitrary
  # task text (quotes, unicode, newlines) survives JSON-encoding intact.
  episode_tasks="$(python3 -c '
import json, sys
eps = json.loads(sys.argv[1])
print(json.dumps({str(int(e)): sys.argv[2] for e in eps}))' "$episodes" "$task")"
  # NOTE --root points at the TEMP COPY: modify_tasks edits in place, so the tool
  # must only ever see the copy, never the real dataset.
  argv=(
    lerobot-edit-dataset
    --repo_id "$repo_id"
    --root "$tmp"
    --operation.type modify_tasks
    --operation.episode_tasks "$episode_tasks"
  )
fi
if [[ ${#extra[@]} -gt 0 ]]; then
  argv+=("${extra[@]}")
fi

if [[ "$dry" == "1" ]]; then
  printf '%s\n' "${argv[@]}"
  [[ "$op" == "retag" ]] && echo "## copy:   $root -> $tmp"
  echo "## backup: $root -> $bak"
  echo "## swap:   $tmp -> $root"
  exit 0
fi

rm -rf "$tmp"
if [[ "$op" == "retag" ]]; then
  cp -a "$root" "$tmp"
fi
rc=0
"${argv[@]}" || rc=$?
if [[ "$rc" -ne 0 ]]; then
  rm -rf "$tmp"
  echo "edit.sh: lerobot-edit-dataset failed (rc=$rc); dataset untouched" >&2
  exit "$rc"
fi
[[ -f "$tmp/meta/info.json" ]] || {
  rm -rf "$tmp"
  echo "edit.sh: no dataset at $tmp after the edit; original untouched" >&2; exit 1; }

mv "$root" "$bak"
mv "$tmp" "$root"
echo "edited in place: $root"
echo "backup kept at:  $bak"
