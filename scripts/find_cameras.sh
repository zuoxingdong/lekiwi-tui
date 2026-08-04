#!/usr/bin/env bash
# ============================================================================
# find_cameras.sh — emitter for the Pi-side bash that lists the robot's cameras.
#
# WHY REMOTE: LeKiwi's cameras hang off the Pi, and the `index_or_path` values in
# lekiwi.yaml are Pi-side device nodes. Running `lerobot-find-cameras` on the laptop
# enumerates the laptop's own webcam and says nothing about front/wrist/top, so the
# question "which /dev/videoN is the wrist today" can only be answered on the robot.
# It needs answering more often than one would like: a bare /dev/videoN is not
# reboot/replug-stable, and adding a camera renumbers the others.
#
# Same seam as host.sh: this script only PRINTS the remote bash (one argv token), and
# the Python side (screens/robot_config.py) builds the `ssh <host> "<that token>"`
# invocation. Nothing is shipped to the Pi and nothing here spawns ssh.
#
# LIST-ONLY BY DEFAULT. `lerobot-find-cameras` prints the detected cameras and then
# captures frames from each into --output-dir; --record-time 0 keeps the printout and
# writes no files, which is all the yaml needs. Pass a positive --record-time when you
# actually want sample frames on the Pi to eyeball which lens is which.
#
# Usage:
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi --backend all --record-time 2
#   scripts/find_cameras.sh emit-detect --conda-env lekiwi --dry-run   # accepted, no-op
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

# Where sample frames land ON THE PI when --record-time > 0. Under /tmp so a probe can
# never fill the Pi's home or leave anything the next run has to clean up.
REMOTE_OUT_DIR="/tmp/lekiwi-find-cameras"

# emit_detect <conda_env> <backend> <record_time> <warmup> <out_dir>
#   Print the remote DETECT bash. The mamba hook + activate lines mirror
#   host.sh emit_launch, so both remote payloads fail the same way on a Pi whose env
#   is missing rather than in some new way.
emit_detect() {
  local conda_env="$1" backend="$2" record_time="$3" warmup="$4" out_dir="$5"
  local type_arg=""
  [[ "$backend" == "all" ]] || type_arg="$backend"
  printf '
        eval "$(~/miniforge3/bin/mamba shell hook --shell bash)" || exit 1
        mamba activate %s || { echo '\''✗ could not activate %s'\'' >&2; exit 1; }

        echo "▸ probing cameras on $(hostname) (record-time %ss)"
        lerobot-find-cameras %s \
            --output-dir %s \
            --record-time-s %s \
            --warmup-s %s
    ' "$conda_env" "$conda_env" "$record_time" "$type_arg" "$out_dir" "$record_time" "$warmup"
}

# emit_stream <conda_env> <device> <fps> <width> <height> <quality> <rotation>
#   Print the remote STREAM bash: capture one device and write length-prefixed JPEG
#   frames to stdout, forever, until the ssh channel closes.
#
#   The list from emit-detect says which device nodes exist; it cannot say which LENS is
#   which, and that is the question after a replug renumbers them. Hence a preview.
#
#   DOWNSCALE AND COMPRESS ON THE PI. A 320x240 q50 frame is ~12 KB, so 5 fps is ~60 KB/s
#   — two orders below the USB+WiFi load that used to hard-freeze the Pi. Shipping full
#   frames to downscale locally would spend the bandwidth we deliberately protect.
#
#   Rotation is applied here so the preview matches what the robot actually sends (a
#   wrist camera configured ROTATE_180 should look upright), which also makes a wrong
#   rotation visible instead of confusing.
emit_stream() {
  local conda_env="$1" device="$2" fps="$3" width="$4" height="$5" quality="$6" rotation="$7"
  printf '
        eval "$(~/miniforge3/bin/mamba shell hook --shell bash)" || exit 1
        mamba activate %s || { echo '\''✗ could not activate %s'\'' >&2; exit 1; }

        python - <<'\''PY'\''
import sys, time
import cv2

cap = cv2.VideoCapture("%s")
if not cap.isOpened():
    sys.stderr.write("✗ could not open %s (is the host stopped?)\\n")
    raise SystemExit(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
period = 1.0 / %s
out = sys.stdout.buffer
try:
    while True:
        start = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            sys.stderr.write("✗ capture ended\\n")
            break
        if %s in rot:
            frame = cv2.rotate(frame, rot[%s])
        frame = cv2.resize(frame, (%s, %s), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), %s])
        if ok:
            payload = jpeg.tobytes()
            out.write(b"FRAME %%d\\n" %% len(payload))
            out.write(payload)
            out.flush()
        time.sleep(max(0.0, period - (time.monotonic() - start)))
finally:
    cap.release()
PY
    ' "$conda_env" "$conda_env" "$device" "$device" "$fps" "$rotation" "$rotation" \
      "$width" "$height" "$quality"
}

# ── parse ───────────────────────────────────────────────────────────────────
# First positional is the subcommand (emit-detect | emit-stream). Long flags only;
# unknown flags are an error, since the emitted bash is fixed (no passthrough).
subcmd="${1:-}"; shift || true
conda_env=""
backend="opencv"       # the LeKiwi cameras are plain UVC; `all` also probes RealSense
record_time="0"        # 0 = print the list, write no frames
warmup="1"
device=""              # emit-stream: which Pi-side node to preview
fps="5"                # a preview, not a viewfinder: enough to identify a lens
width="320"
height="240"
quality="50"
rotation="0"           # 0 | 90 | 180 | 270, applied on the Pi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)   conda_env="${2:-}"; shift 2 ;;
    --backend)     backend="${2:-}"; shift 2 ;;
    --record-time) record_time="${2:-}"; shift 2 ;;
    --warmup)      warmup="${2:-}"; shift 2 ;;
    --device)      device="${2:-}"; shift 2 ;;
    --fps)         fps="${2:-}"; shift 2 ;;
    --width)       width="${2:-}"; shift 2 ;;
    --height)      height="${2:-}"; shift 2 ;;
    --quality)     quality="${2:-}"; shift 2 ;;
    --rotation)    rotation="${2:-}"; shift 2 ;;
    --dry-run)     shift ;;   # accepted and ignored: this script only ever prints
    *)             die_usage "unknown flag: $1" ;;
  esac
done

case "$backend" in
  opencv|realsense|all) ;;
  *) die_usage "--backend must be 'opencv', 'realsense' or 'all' (got '$backend')" ;;
esac
[[ "$record_time" =~ ^[0-9]+$ ]] || die_usage "--record-time must be a whole number of seconds"
[[ "$warmup" =~ ^[0-9]+$ ]] || die_usage "--warmup must be a whole number of seconds"

case "$subcmd" in
  emit-detect)
    validate_remote_name "$conda_env" "conda env"
    emit_detect "$conda_env" "$backend" "$record_time" "$warmup" "$REMOTE_OUT_DIR" ;;
  emit-stream)
    validate_remote_name "$conda_env" "conda env"
    # The device is interpolated into remote python source, so it is validated hard:
    # a plain /dev/... node or a small integer index, nothing else.
    [[ "$device" =~ ^(/dev/[A-Za-z0-9_./-]+|[0-9]{1,3})$ ]] \
      || die_usage "--device must be a /dev path or a small index (got '${device:-}')"
    [[ "$device" != *..* ]] || die_usage "--device must not contain '..'"
    validate_positive_int "$fps" "--fps"
    (( fps <= 30 )) || die_usage "--fps must be 30 or less (this is a preview, not a recording)"
    validate_positive_int "$width" "--width"
    validate_positive_int "$height" "--height"
    validate_positive_int "$quality" "--quality"
    (( quality <= 100 )) || die_usage "--quality must be 1..100"
    case "$rotation" in
      0|90|180|270) ;;
      *) die_usage "--rotation must be 0, 90, 180 or 270 (got '$rotation')" ;;
    esac
    emit_stream "$conda_env" "$device" "$fps" "$width" "$height" "$quality" "$rotation" ;;
  ""|-h|--help) die_usage "usage: find_cameras.sh emit-detect --conda-env <env> [--backend opencv|realsense|all] [--record-time N] [--warmup N]
       find_cameras.sh emit-stream --conda-env <env> --device /dev/videoN [--fps 5] [--width 320] [--height 240] [--quality 50] [--rotation 0|90|180|270]" ;;
  *) die_usage "unknown subcommand: $subcmd (expected emit-detect or emit-stream)" ;;
esac
