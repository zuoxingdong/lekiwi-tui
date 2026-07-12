#!/usr/bin/env bash
# ============================================================================
# host.sh — emitter for the Pi-side (REMOTE) bash that the host commands run.
#
# WHY THIS SCRIPT IS DIFFERENT FROM replay.sh / teleop.sh
# ----------------------------------------------------------------------------
# replay/teleop assemble a LOCAL lerobot-* argv and `exec` it (see lib.sh's
# header). The host commands are not local: they run `ssh -t <PI> "<bash>"`,
# where "<bash>" is a single argv token executed ON THE PI. That remote bash is
# the riskiest, most-tuned piece of the whole port (the graceful-INT→SIGKILL
# cleanup, the host backgrounded under `wait`, the CONNECTION_TIME+60 overrun
# net for launch; the pgrep/kill-9 sweep for kill). Keeping it here as shell
# makes it inspectable and forkable on its own.
#
# It does NOT build the ssh argv and does NOT spawn ssh. The Python side keeps
# the load-bearing seam explicit: HostLaunchScreen and HostKillScreen build the
# ssh argv, run ssh under a local PTY, and let `ssh -t` allocate the remote PTY.
# Stop writes \x03 to that PTY for the remote trap's graceful stop with a
# SIGKILL fallback. ONLY the SOURCE of the remote bash text lives here: Python's
# builders shell out to `bash host.sh emit-launch …` / `emit-kill …` and use the
# captured stdout verbatim as that single ssh argv token. The emitted bytes MUST
# stay token-for-token stable, which is why emission is `printf` (byte-exact: leading
# newline, the 8-space heredoc indent, the literal `\`-newline continuations,
# and the exact trailing whitespace are all reproduced; `subprocess.check_output`
# on the Python side preserves them, unlike a shell `$(...)` which would strip
# trailing newlines).
#
# The host CONFIG (cameras / use_degrees / loop-freq) is a SEPARATE thing: it is
# scp'd to the Pi as /tmp/lekiwi_host.yaml by Python's ship_host_config and
# passed via --config_path. This script is a LOCAL emitter only — its stdout is
# sent INLINE as the ssh command, so host.sh itself is never shipped to the Pi.
#
# SUBCOMMANDS
# ----------------------------------------------------------------------------
#   host.sh emit-launch --conda-env E --robot-id R --connection-time S \
#                       [--robot-type T] [--cfg-flag F] [--loop-flag F]
#       Print the remote LAUNCH bash (== screens/host.py remote_script). --cfg-flag
#       / --loop-flag default to empty (the host then uses its built-in defaults);
#       empty values reproduce bash's collapsed inner spacing on the launch line.
#       --robot-type picks the host module (whitelist, default lekiwi_pincopen):
#         lekiwi_pincopen → python -m lerobot_robot_lekiwi_pincopen.lekiwi_host  (plugin)
#         lekiwi          → python -m lerobot.robots.lekiwi.lekiwi_host   (stock)
#
#   host.sh emit-kill --robot-id R
#       Print the remote KILL bash: pgrep the lekiwi_host for this robot id,
#       kill -9, then a ps|grep|awk sweep.
#
# A leading --dry-run (or DRY=1) is accepted and IGNORED: this script only ever
# prints the remote bash (it never spawns anything), so --dry-run and a plain run
# are identical. The flag exists for surface-parity with the other launchers and
# so `host.sh emit-launch --dry-run …` works in tests/CI the same way.
#
# Usage:
#   scripts/host.sh emit-launch --conda-env lekiwi --robot-id lekiwi \
#       --connection-time 600 --cfg-flag --config_path=/tmp/lekiwi_host.yaml \
#       --loop-flag --host.max_loop_freq_hz=30
#   scripts/host.sh emit-kill --robot-id lekiwi
#
# NON-INTERACTIVE: never prompts. The TUI's HostLaunchScreen gathers minutes /
# robot id / loop Hz; host.py and host_kill.py pass the resolved values as flags here.
# ============================================================================
set -euo pipefail

# Load the shared helpers (same as replay.sh / teleop.sh) relative to THIS file,
# so the script is runnable from any working directory. host.sh does not read
# lekiwi.yaml itself (config shipping stays in Python), but sourcing lib.sh keeps
# the launcher family uniform and leaves cfg_get/cfg_slice available to forks.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# emit_launch <conda_env> <robot_id> <connection_time> <cfg_flag> <loop_flag> <host_module>
#   Print the remote LAUNCH bash, byte-for-byte equal to screens/host.py
#   remote_script(). printf reproduces the exact bytes: the leading newline, the
#   8-space indent on every line, the `\`-newline line continuations on the
#   lekiwi_host launch line, and the trailing newline + 4 spaces with NO final
#   newline. The remote-side vars ($PID/$WPID/$RC/$GRACE/$i and the $(...) mamba
#   hook) are LITERAL here — they are evaluated on the Pi, not by this printf — so
#   they sit inside the single-quoted format and are emitted as-is. Only the six
#   %s positions are substituted on the local side. cfg_flag/loop_flag splice raw
#   (empty when not shipped), reproducing bash's collapsed inner spacing exactly.
#   host_module comes from the --robot-type whitelist below; the default is the
#   PincOpen plugin wrapper (python -m lerobot_robot_lekiwi_pincopen.lekiwi_host: STS3250
#   arms + gripper tuning, same CLI as the stock host; the wrapper module is
#   deliberately also named lekiwi_host so emit-kill's pgrep patterns keep matching
#   whichever module runs).
emit_launch() {
  local conda_env="$1" robot_id="$2" connection_time="$3" cfg_flag="$4" loop_flag="$5" host_module="$6"
  printf '
        GRACE=5
        cleanup() {
            echo
            echo '\''🛑 Stopping LeKiwi host…'\''
            i=0; while [ "$i" -lt "$GRACE" ]; do kill -0 "$PID" 2>/dev/null || break; sleep 1; i=$(( i + 1 )); done
            if kill -0 "$PID" 2>/dev/null; then
                echo '\''⚠️  host did not exit in time — forcing kill'\''
                kill -s KILL "$PID" 2>/dev/null || true
            fi
            kill "$WPID" 2>/dev/null || true
            exit 130
        }
        trap cleanup INT TERM

        eval "$(~/miniforge3/bin/mamba shell hook --shell bash)" || exit 1
        mamba activate %s || { echo '\''✗ could not activate %s'\'' >&2; exit 1; }

        ( trap - INT TERM; exec python -m %s %s %s \
            --robot.id=%s \
            --host.connection_time_s=%s ) &
        PID=$!

        # Overrun safety net: if the host outlives its session by ~60s (wedged on
        # servo comms), nudge then force-kill so this never hangs forever.
        ( sleep $(( %s + 60 ))
          echo '\''⚠️  session overran — stopping host'\'' >&2
          kill -s INT "$PID" 2>/dev/null
          sleep $GRACE
          kill -s KILL "$PID" 2>/dev/null ) &
        WPID=$!

        wait "$PID"; RC=$?
        kill "$WPID" 2>/dev/null || true
        exit $RC
    ' "$conda_env" "$conda_env" "$host_module" "$cfg_flag" "$loop_flag" "$robot_id" "$connection_time" "$connection_time"
}

# emit_kill <robot_id>
#   Print the remote KILL bash. The pgrep/grep `\.id` regex escapes are literal
#   backslash-dot (doubled in the printf format so a single `\.` is emitted), and
#   awk's `$2` is literal. robot_id substitutes into all four %s positions. This
#   emission DOES end with a trailing newline.
emit_kill() {
  local robot_id="$1"
  printf 'PIDS=$(pgrep -f '\''python.*lekiwi_host.*--robot\\.id=%s'\'' || true)
if [ -n "$PIDS" ]; then
    echo "Found processes: $PIDS"; echo $PIDS | xargs -r kill -9
    echo '\''✅ Killed lekiwi_host for ROBOT_ID=%s'\''
else echo '\''ℹ️  No lekiwi_host processes for ROBOT_ID=%s'\''; fi
ps aux | grep '\''[p]ython.*lekiwi_host.*robot\\.id=%s'\'' | awk '\''{print $2}'\'' | xargs -r kill -9 2>/dev/null || true
' "$robot_id" "$robot_id" "$robot_id" "$robot_id"
}

# ── subcommand + flag parse ──────────────────────────────────────────────────
# First positional is the subcommand (emit-launch | emit-kill). Remaining args
# are long flags. --dry-run / DRY=1 is accepted and ignored (this script only
# ever prints — there is nothing to suppress). Unknown flags are an error (unlike
# the lerobot launchers there is no passthrough: the emitted bash is fixed).
sub="${1:-}"
[[ $# -gt 0 ]] && shift

# launch knobs (empty cfg/loop flags = not shipped -> host uses built-in defaults)
conda_env=""
robot_id=""
robot_type="lekiwi_pincopen"   # the follower to launch; the TUI passes ROBOT_TYPE
connection_time=""
cfg_flag=""
loop_flag=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)        conda_env="$2"; shift 2 ;;
    --robot-id)         robot_id="$2"; shift 2 ;;
    --robot-type)       robot_type="$2"; shift 2 ;;
    --connection-time)  connection_time="$2"; shift 2 ;;
    --cfg-flag)         cfg_flag="$2"; shift 2 ;;
    --loop-flag)        loop_flag="$2"; shift 2 ;;
    --dry-run)          shift ;;          # accepted + ignored (always emits)
    *)
      echo "host.sh: unknown flag '$1'" >&2
      exit 2 ;;
  esac
done
: "${DRY:-0}"   # DRY=1 is likewise a no-op; referenced so set -u stays happy

# robot type -> host module WHITELIST. A case map (not string splicing) so an
# arbitrary --robot-type can never smuggle tokens into the remote python -m line.
host_module_for() {
  case "$1" in
    lekiwi_pincopen) printf '%s\n' "lerobot_robot_lekiwi_pincopen.lekiwi_host" ;;  # PincOpen plugin
    lekiwi)          printf '%s\n' "lerobot.robots.lekiwi.lekiwi_host" ;;   # stock lerobot
    *)
      echo "host.sh: unknown robot type '$1' (expected lekiwi_pincopen | lekiwi)" >&2
      exit 2 ;;
  esac
}

case "$sub" in
  emit-launch)
    validate_remote_name "$conda_env" "conda env"
    validate_remote_name "$robot_id" "robot id"
    validate_positive_int "$connection_time" "connection time"
    validate_optional_cli_flag "$cfg_flag" "config flag"
    validate_optional_cli_flag "$loop_flag" "loop flag"
    host_module="$(host_module_for "$robot_type")"
    emit_launch "$conda_env" "$robot_id" "$connection_time" "$cfg_flag" "$loop_flag" "$host_module" ;;
  emit-kill)
    validate_remote_name "$robot_id" "robot id"
    emit_kill "$robot_id" ;;
  ""|-h|--help|help)
    echo "usage: host.sh emit-launch --conda-env E --robot-id R --connection-time S [--robot-type T] [--cfg-flag F] [--loop-flag F]" >&2
    echo "       host.sh emit-kill --robot-id R" >&2
    [[ "$sub" == "" ]] && exit 2 || exit 0 ;;
  *)
    echo "host.sh: unknown subcommand '$sub' (expected emit-launch | emit-kill)" >&2
    exit 2 ;;
esac
