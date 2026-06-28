#!/usr/bin/env bash
# lekiwi.sh — one-command launcher for the LeKiwi TUI.
# Activates the lerobot conda env (so the TUI *and* the launcher scripts see lerobot),
# then hands off to the package. Run it from anywhere:  ./lekiwi.sh [action] [args…]
#   ./lekiwi.sh                 # the menu
#   ./lekiwi.sh teleop          # jump straight to one screen
#   ./lekiwi.sh --dry-run       # preview the commands instead of running them
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
printf '\e[H\e[2J\e[3J'                       # pure-ANSI clear (ghostty terminfo-safe)

# MAMBA_ROOT / LAPTOP_ENV: env var at launch ▸ lekiwi.yaml ▸ built-in default.
# Read from the `_launcher:` block in lekiwi.yaml with plain awk (this runs BEFORE the
# conda env exists, so no python). Block-scoped: match `_launcher:`, then read indented
# `KEY:` lines until the next non-indented line; strip surrounding single/double quotes.
MAMBA_ROOT="${MAMBA_ROOT:-$(awk -v key=MAMBA_ROOT -v sq="'" '
  /^_launcher:/    { inb=1; next }
  inb && /^[^ \t]/ { inb=0 }
  inb {
    line=$0; sub(/^[ \t]+/,"",line)
    if (substr(line,1,length(key)+1) == key ":") {
      v=substr(line,length(key)+2); sub(/^[ \t]+/,"",v)
      c=substr(v,1,1)
      if ((c==sq || c=="\"") && substr(v,length(v),1)==c) v=substr(v,2,length(v)-2)
      print v; exit
    }
  }' lekiwi.yaml 2>/dev/null)}"; MAMBA_ROOT="${MAMBA_ROOT:-$HOME/miniforge3}"
LAPTOP_ENV="${LAPTOP_ENV:-$(awk -v key=LAPTOP_ENV -v sq="'" '
  /^_launcher:/    { inb=1; next }
  inb && /^[^ \t]/ { inb=0 }
  inb {
    line=$0; sub(/^[ \t]+/,"",line)
    if (substr(line,1,length(key)+1) == key ":") {
      v=substr(line,length(key)+2); sub(/^[ \t]+/,"",v)
      c=substr(v,1,1)
      if ((c==sq || c=="\"") && substr(v,length(v),1)==c) v=substr(v,2,length(v)-2)
      print v; exit
    }
  }' lekiwi.yaml 2>/dev/null)}"; LAPTOP_ENV="${LAPTOP_ENV:-lekiwi}"

# shellcheck disable=SC1090,SC1091
source "$MAMBA_ROOT/etc/profile.d/conda.sh" || { echo "✗ conda.sh not found under $MAMBA_ROOT — set MAMBA_ROOT." >&2; exit 1; }
conda activate "$LAPTOP_ENV" || { echo "✗ could not activate '$LAPTOP_ENV' — set LAPTOP_ENV." >&2; exit 1; }
exec python -m lekiwi_tui "$@"
