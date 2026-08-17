#!/usr/bin/env bash
# ============================================================================
# merge.sh — merge N datasets into a new one, dagger-aware and stats-correct.
#
# Fronted by the dataset-selection page (Edit dataset → toggle several → ⏎).
# The common case is base demos + dagger correction sessions, so the script
# handles the dagger-specific cleanup itself:
#
#   1. any input carrying the `intervention` feature is stripped into a temp
#      copy (lerobot's remove_feature; the source dataset is untouched) —
#      lerobot's merge requires IDENTICAL features across inputs;
#   2. the temp copy's per-episode stats are NORMALIZED: remove_feature filters
#      the global stats.json but copytree's meta/episodes verbatim, so stale
#      stats/intervention/* columns survive there — and aggregate APPENDS each
#      source's episodes-metadata rows into shared parquet files, so a schema
#      mismatch poisons the merged metadata. The normalize step drops those
#      columns before the merge ever sees them;
#   3. all inputs merge (lerobot aggregate: re-aggregates global stats from the
#      per-source stats.json, re-indexes tasks/episodes) into a NEW dataset
#      next to the FIRST input;
#   4. temp copies are removed (trap'd).
#
# Nothing is modified in place; every input survives byte-identical.
#
#   scripts/merge.sh --dataset <root> --dataset <root> [--dataset <root>…] \
#       --out-name <name> [--ns local] [--dry-run]
#
#   * --dataset    an input dataset dir, repeatable, ≥2. ORDER MATTERS: episodes
#                  concatenate in this order, and the FIRST input is the feature
#                  reference and the output's parent dir.
#   * --out-name   the output dataset's folder/repo name (created next to the
#                  first input; refuses to overwrite).
#   * --ns         repo namespace label (default local; cosmetic).
#   * --dry-run / DRY=1: print each planned step one token per line with `---`
#     separators and exit 0 (the golden test pins these). The internal
#     episode-stats normalize prints as a `normalize-episode-stats` pseudo-command.
#
# Feature pre-check: every input's features minus `intervention` must equal the
# first input's (minus its own `intervention`, if any). Fails in milliseconds
# with the actual key diff instead of after minutes of stripping.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

datasets=()
out_name=""
ns="local"
dry="${DRY:-0}"

normalize_only=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  datasets+=("$2"); shift 2 ;;
    --out-name) out_name="$2"; shift 2 ;;
    --ns)       ns="$2"; shift 2 ;;
    --normalize-only) normalize_only="$2"; shift 2 ;;  # run ONLY the episode-stats
                                                       # normalize on <root> (tests/debug)
    --dry-run)  dry=1; shift ;;
    *) die_usage "unknown flag '$1' (this launcher forwards nothing)" ;;
  esac
done

# normalize_episode_stats <root>
#   Drop stale stats/intervention/* columns from a dataset copy's meta/episodes
#   parquets (see header point 2). Missing episodes dir → nothing to do.
normalize_episode_stats() {
  "$PY" - "$1" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
for f in sorted(root.glob("meta/episodes/**/*.parquet")):
    table = pq.read_table(f)
    drop = [c for c in table.schema.names
            if c == "stats/intervention" or c.startswith("stats/intervention/")]
    if drop:
        pq.write_table(table.drop_columns(drop), f)
        print(f"  dropped {len(drop)} stale stats column(s) from {f.name}")
PY
}

# Reachable standalone so the parquet surgery is testable (and re-runnable by hand
# on a wip copy a crash left behind). Skips the main-flow validation on purpose.
if [[ -n "$normalize_only" ]]; then
  normalize_episode_stats "$normalize_only"
  exit 0
fi

[[ ${#datasets[@]} -ge 2 ]] || die_usage "at least two --dataset <root> inputs are required"
[[ -n "$out_name" ]] || die_usage "--out-name <name> is required"
case "$out_name" in
  */*) die_usage "--out-name is a folder name, not a path (got '$out_name')" ;;
esac

first="${datasets[0]}"
parent="$(dirname "$first")"
out_root="$parent/$out_name"
work="$parent/.$out_name.wip"

# has_intervention <root> -> exit 0 when the dataset's features carry `intervention`
has_intervention() {
  "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

try:
    info = json.loads((Path(sys.argv[1]) / "meta" / "info.json").read_text())
except Exception:
    sys.exit(1)
sys.exit(0 if "intervention" in info.get("features", {}) else 1)
PY
}

if [[ "$dry" != "1" ]]; then
  for d in "${datasets[@]}"; do
    [[ -d "$d" ]] || die_usage "dataset not found: $d"
  done
  [[ ! -e "$out_root" ]] || die_usage "output already exists: $out_root (pick another --out-name)"

  # Feature pre-check: every input's features minus `intervention` must match the first's.
  "$PY" - "${datasets[@]}" <<'PY'
import json
import sys
from pathlib import Path


def feats(root):
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return set(info["features"]) - {"intervention"}


first = sys.argv[1]
want = feats(first)
for d in sys.argv[2:]:
    have = feats(d)
    if have != want:
        extra = sorted(have - want)
        missing = sorted(want - have)
        detail = "; ".join(p for p in (
            f"extra: {extra}" if extra else "",
            f"missing: {missing}" if missing else "") if p)
        sys.exit(f"✗ features differ between {first} and {d} ({detail}) — "
                 "lerobot merge requires identical features")
print("features compatible")
PY
fi

# ── plan the steps (the sole source; pinned by the golden test) ───────────────
strip_cmds=()      # lerobot-edit-dataset remove_feature argvs (newline-joined)
normalize_dsts=()  # wip copies whose meta/episodes needs the stale-stats drop
merge_ids=()
merge_roots=()
for i in "${!datasets[@]}"; do
  d="${datasets[$i]}"
  d_name="$(basename "$d")"
  # Strip when the input carries `intervention`. A dry-run on a path that does not
  # exist (golden tests use fake paths) plans the strip conservatively, so the
  # printed plan shows the fuller pipeline.
  plan_strip=0
  if [[ -d "$d" ]]; then
    has_intervention "$d" && plan_strip=1
  elif [[ "$dry" == "1" ]]; then
    plan_strip=1
  fi
  if [[ "$plan_strip" == "1" ]]; then
    dst="$work/src-$i"
    strip_cmds+=("$(printf '%s\n' \
      lerobot-edit-dataset \
      --repo_id "$ns/$d_name" \
      --root "$d" \
      --new_repo_id "$ns/${d_name}__noint" \
      --new_root "$dst" \
      --operation.type remove_feature \
      "--operation.feature_names=['intervention']")")
    normalize_dsts+=("$dst")
    merge_ids+=("'$ns/${d_name}__noint'")
    merge_roots+=("'$dst'")
  else
    merge_ids+=("'$ns/$d_name'")
    merge_roots+=("'$d'")
  fi
done

# join with ", " (bash's ${arr[*]} joins on IFS's FIRST char only)
_join() { local out="" sep=""; for x in "$@"; do out+="$sep$x"; sep=", "; done; printf '%s' "$out"; }
ids_list="[$(_join "${merge_ids[@]}")]"
roots_list="[$(_join "${merge_roots[@]}")]"
merge_cmd="$(printf '%s\n' \
  lerobot-edit-dataset \
  --new_repo_id "$ns/$out_name" \
  --new_root "$out_root" \
  --operation.type merge \
  "--operation.repo_ids=$ids_list" \
  "--operation.roots=$roots_list")"

if [[ "$dry" == "1" ]]; then
  for i in "${!strip_cmds[@]}"; do
    printf '%s\n---\n' "${strip_cmds[$i]}"
    printf 'normalize-episode-stats\n%s\n---\n' "${normalize_dsts[$i]}"
  done
  printf '%s\n' "$merge_cmd"
  exit 0
fi

# ── execute ───────────────────────────────────────────────────────────────────
cleanup() { rm -rf "$work"; }
trap cleanup EXIT
rm -rf "$work"
mkdir -p "$work"

for i in "${!strip_cmds[@]}"; do
  mapfile -t argv <<< "${strip_cmds[$i]}"
  echo "▸ stripping intervention: ${argv[4]}"
  "${argv[@]}"
  echo "▸ normalizing episode stats: ${normalize_dsts[$i]}"
  normalize_episode_stats "${normalize_dsts[$i]}"
done

mapfile -t argv <<< "$merge_cmd"
echo "▸ merging ${#datasets[@]} dataset(s) into $out_root"
"${argv[@]}"

echo "✓ merged dataset ready: $out_root ($ns/$out_name)"
