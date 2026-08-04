#!/usr/bin/env bash
#
# pi_provision.sh - bring up the LeKiwi host on a Raspberry Pi 5, driven FROM the laptop.
#
# Target: Raspberry Pi 5, aarch64, Debian, CPU-only (no CUDA). The laptop is the source
# of truth; this SSHes in and builds a self-contained conda env running the SAME (latest)
# lerobot the laptop runs, so one repo drives both sides.
#
# Work is split into 4 idempotent STAGES, one per layer of the stack. Run them all
# (default) or pick a subset to re-run after a fix:
#
#   ./pi_provision.sh                 run every stage in order
#   ./pi_provision.sh conda lerobot   run just those stages, in the given order
#   ./pi_provision.sh --dry-run       print config + selected stages, do not ssh
#   ./pi_provision.sh list            list the stages
#
#   system    OS layer:      apt (git, build-essential, ffmpeg, ...) + dialout/video groups  [sudo]
#   network   OS layer:      force WiFi power-save OFF (NM dispatcher + conf) — brcmfmac freeze fix  [sudo]
#   conda     runtime layer: Miniforge3 (aarch64) + mamba-create the python 3.12 env + uv
#   lerobot   app layer:     delegate to sync.sh --install (mirror + editable installs), then smoke-test
#
# Installs use uv (much faster than pip); uv lives in the conda env (mamba-managed). Latest
# lerobot needs python >=3.12, so the env is 3.12. The Pi has no GPU, so torch/torchvision
# are CPU wheels via uv's --torch-backend=cpu; --no-sources is also required because
# lerobot's pyproject pins torch to a CUDA-128 index that would otherwise override that
# (see the lerobot stage). Editable install -> a routine source change only needs `lerobot`.
#
# Config (all env-overridable). Durable tip: add an ~/.ssh/config `Host lekiwi` alias so
# everything below is DHCP-proof.
PI_HOST="${PI_HOST:-lekiwi}"                       # ssh target
PI_REPO="${PI_REPO:-lekiwi/lerobot}"               # repo path on the Pi, home-relative (resolves against the remote $HOME, so any Pi username works)
PI_PLUGIN="${PI_PLUGIN:-lekiwi/lerobot_robot_lekiwi_pincopen}"  # PincOpen robot plugin path on the Pi, home-relative
PI_ENV="${PI_ENV:-lekiwi}"                          # conda env name on the Pi
PY_VER="${PY_VER:-3.12}"                            # env python; latest lerobot needs >=3.12
RECREATE_ENV="${RECREATE_ENV:-}"                    # set to 1 to rebuild the env on a python mismatch
APT_PACKAGES="${APT_PACKAGES:-git build-essential ffmpeg rsync curl}"
MAMBA_PREFIX="${MAMBA_PREFIX:-}"                    # empty -> $HOME/miniforge3 on the Pi
MINIFORGE_URL="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh}"
set -euo pipefail

# Laptop clone to ship. Prefer a sibling of this checkout for public clones, but keep
# a parent-workspace fallback for older local layouts. Override with LOCAL_REPO=….
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"                          # this control-center dir
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
LOCAL_REPO="${LOCAL_REPO:-$(default_local_repo)}"
# The PincOpen robot plugin (STS3250 arms + gripper tuning) — a sibling project of
# this checkout, shipped and editable-installed alongside the clone. It replaces the
# old lekiwi.py fork patch: the host launches via `python -m
# lerobot_robot_lekiwi_pincopen.lekiwi_host`, calibrate uses --robot.type=lekiwi_pincopen.
LOCAL_PLUGIN="${LOCAL_PLUGIN:-$(cd "$ROOT/.." && pwd -P)/lerobot_robot_lekiwi_pincopen}"

STAGES=(system network conda lerobot)
dry="${DRY:-0}"

# ── helpers ─────────────────────────────────────────────────────────────────
banner() { printf '\n\033[1;36m▸ [%s]\033[0m %s\n' "$1" "${2:-}"; }
die()    { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
die_usage() { printf '%s\n' "$*" >&2; exit 2; }
shell_quote() { printf '%q' "$1"; }

validate_ssh_host() {
    local value="${1:-}" label="${2:-SSH host}"
    [[ -n "$value" ]] || die_usage "$label must not be empty"
    [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
    [[ "$value" != *[$'\n\r\t ']* ]] || die_usage "$label must not contain whitespace"
    [[ "$value" =~ ^[A-Za-z0-9_.@-]+$ ]] || die_usage "$label contains unsupported characters; use a simple host or user@host"
    [[ "$value" != *..* ]] || die_usage "$label must not contain '..'"
}

validate_remote_name() {
    local value="${1:-}" label="${2:-remote name}"
    [[ -n "$value" ]] || die_usage "$label must not be empty"
    [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
    [[ "$value" != *[$'\n\r\t ']* ]] || die_usage "$label must not contain whitespace"
    [[ "$value" =~ ^[A-Za-z0-9_.-]+$ ]] || die_usage "$label must use only letters, numbers, '.', '_', or '-'"
}

validate_remote_path() {
    local value="${1:-}" label="${2:-remote path}"
    [[ -n "$value" ]] || die_usage "$label must not be empty"
    [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
    [[ "$value" != *[$'\n\r']* ]] || die_usage "$label must not contain control characters"
}

validate_optional_remote_path() {
    local value="${1:-}" label="${2:-remote path}"
    [[ -z "$value" ]] && return 0
    [[ "$value" != -* ]] || die_usage "$label must not start with '-'"
    [[ "$value" != *[$'\n\r']* ]] || die_usage "$label must not contain control characters"
}

validate_simple_url() {
    local value="${1:-}" label="${2:-URL}"
    [[ "$value" == http://* || "$value" == https://* ]] || die_usage "$label must start with http:// or https://"
    [[ "$value" != *[$'\n\r\t ']* ]] || die_usage "$label must not contain whitespace"
}

validate_package_list() {
    local pkg
    for pkg in $APT_PACKAGES; do
        [[ "$pkg" =~ ^[A-Za-z0-9_.:+-]+$ ]] || die_usage "APT package '$pkg' contains unsupported characters"
    done
}

validate_python_version() {
    [[ "$PY_VER" =~ ^[0-9]+[.][0-9]+$ ]] || die_usage "PY_VER must look like 3.12"
}

# Remote env passed to every `pi` heredoc, so remote bodies stay quote-clean (a quoted
# heredoc means no local interpolation and no \$ escaping; config arrives as env vars).
pi() { ssh "$PI_HOST" "$RENV bash -s"; }   # feed a heredoc on stdin; runs on the Pi

# ── stages ──────────────────────────────────────────────────────────────────
stage_system() {
    banner system "apt: $APT_PACKAGES  +  groups: dialout, video"
    # One sudo session (ssh -t so a password can prompt): packages, then serial+camera groups.
    ssh -t "$PI_HOST" "
        sudo apt-get update &&
        sudo apt-get install -y $APT_PACKAGES &&
        sudo usermod -aG dialout,video \"\$USER\" &&
        echo \"groups now: \$(id -nG)  (log out/in once if dialout/video were just added)\"
    "
}

stage_network() {
    banner network "WiFi power-save OFF (brcmfmac Pi-freeze fix): NM dispatcher + conf"
    # WHY: the Pi's brcmfmac WiFi wedges under concurrent WiFi + USB/DMA load when
    # power-save engages (see my_robot/PI_FREEZE_RUNBOOK.md). A persistent conf alone has
    # silently failed to apply before, so we ALSO install a NetworkManager dispatcher that
    # re-asserts `iw ... set power_save off` on every association — surviving reconnects and
    # roaming, covering teleop/record/rollout alike. Idempotent: both files are overwritten.
    #
    # The remote body is base64-encoded so its nested heredocs survive the ssh argument
    # cleanly; `ssh -t` allocates a TTY so sudo can prompt (same interactive model as
    # stage_system). sudo reads the password from the TTY while bash reads the script on stdin.
    local script b64
    script="$(cat <<'SH'
set -eu
install -d -m 0755 /etc/NetworkManager/dispatcher.d /etc/NetworkManager/conf.d
# Dispatcher: NM calls it with $1=iface $2=action, as root, on every state change.
cat > /etc/NetworkManager/dispatcher.d/50-wifi-powersave-off <<'EOF'
#!/bin/sh
# LeKiwi: force WiFi power-save OFF on every association (brcmfmac freeze fix).
IFACE="$1"; ACTION="$2"
case "$IFACE" in wlan*) ;; *) exit 0 ;; esac
if [ "$ACTION" = "up" ] || [ "$ACTION" = "connectivity-change" ]; then
  IW="$(command -v iw || echo /usr/sbin/iw)"
  "$IW" dev "$IFACE" set power_save off
fi
EOF
# NM ignores dispatcher scripts that are not root-owned or are group/world-writable.
chown root:root /etc/NetworkManager/dispatcher.d/50-wifi-powersave-off
chmod 0755 /etc/NetworkManager/dispatcher.d/50-wifi-powersave-off
# Persistent baseline (takes effect on the next association; the dispatcher guarantees it now).
cat > /etc/NetworkManager/conf.d/wifi-powersave-off.conf <<'EOF'
[connection]
wifi.powersave = 2
EOF
# Apply immediately on any present wifi iface (also re-asserted on the next 'up').
for ifc in $(iw dev 2>/dev/null | awk '/Interface/{print $2}'); do
  iw dev "$ifc" set power_save off 2>/dev/null || true
  printf '  %s power_save -> ' "$ifc"; iw dev "$ifc" get power_save 2>/dev/null | awk '{print $NF}' || echo '?'
done
echo "wifi power-save: dispatcher + conf installed"
SH
)"
    b64="$(printf '%s' "$script" | base64 | tr -d '\n')"
    ssh -t "$PI_HOST" "echo $b64 | base64 -d | sudo bash"
}

stage_conda() {
    banner conda "Miniforge3 aarch64 + mamba env '$PI_ENV' (python $PY_VER) + uv"
    pi <<'SH'
set -euo pipefail
MAMBA="${MAMBA_PREFIX:-$HOME/miniforge3}"

# Miniforge (provides mamba + conda)
if [ -x "$MAMBA/bin/mamba" ]; then
    echo "miniforge present at $MAMBA"
else
    curl -fsSL "$MINIFORGE_URL" -o /tmp/miniforge.sh || wget -qO /tmp/miniforge.sh "$MINIFORGE_URL"
    bash /tmp/miniforge.sh -b -p "$MAMBA"
    rm -f /tmp/miniforge.sh
    echo "miniforge installed at $MAMBA"
fi

# Env via mamba (fast solver). Version-aware: RECREATE_ENV=1 rebuilds a mismatched env
# (e.g. an old py3.11 one) instead of failing.
envpy="$MAMBA/envs/$PI_ENV/bin/python"
create=0
if [ -x "$envpy" ]; then
    cur="$("$envpy" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
    if [ "$cur" = "$PY_VER" ]; then
        echo "env '$PI_ENV' present (python $cur)"
    elif [ -z "$RECREATE_ENV" ]; then
        echo "✗ env '$PI_ENV' is python $cur, not $PY_VER." >&2
        echo "  re-run with RECREATE_ENV=1 to rebuild it (or: $MAMBA/bin/mamba env remove -y -n $PI_ENV)" >&2
        exit 1
    else
        echo "env '$PI_ENV' is python $cur; RECREATE_ENV set -> rebuilding as $PY_VER"
        "$MAMBA/bin/mamba" env remove -y -n "$PI_ENV"
        create=1
    fi
else
    create=1
fi
[ "$create" = 1 ] && "$MAMBA/bin/mamba" create -y -n "$PI_ENV" python="$PY_VER"

# uv lives IN the env (mamba-managed, removed with the env); the `lerobot` stage uses it as
# the installer because it is far faster than pip. Idempotent: skip if already present.
[ -x "$MAMBA/envs/$PI_ENV/bin/uv" ] || "$MAMBA/bin/mamba" install -y -n "$PI_ENV" uv
echo "env '$PI_ENV' ready (python $PY_VER, $("$MAMBA/envs/$PI_ENV/bin/uv" --version))"
SH
}

stage_lerobot() {
    banner lerobot "delegate delivery to sync.sh --install (mirror + editable installs), then smoke-test"
    # DELIVERY = sync.sh, the single source of the rsync + install recipe. --install
    # forces the editable installs even when the dep fingerprints match, which covers
    # the case sync alone cannot see: a freshly (re)built env with an unchanged tree.
    # Env-name mapping: this script's PI_HOST/PI_ENV are sync.sh's LEKIWI_HOST/CONDA_ENV.
    # sync.sh also prints the shipping provenance and seeds the pincopen calibration dir.
    LEKIWI_HOST="$PI_HOST" CONDA_ENV="$PI_ENV" PI_REPO="$PI_REPO" PI_PLUGIN="$PI_PLUGIN" \
        LOCAL_REPO="$LOCAL_REPO" LOCAL_PLUGIN="$LOCAL_PLUGIN" MAMBA_PREFIX="$MAMBA_PREFIX" \
        bash "$SCRIPT_DIR/sync.sh" --install
    # What stays HERE is bring-up validation, not delivery:
    pi <<'SH'
set -euo pipefail
MAMBA="${MAMBA_PREFIX:-$HOME/miniforge3}"
py="$MAMBA/envs/$PI_ENV/bin/python"

# Smoke-test (the gate): import the host-specific deps lerobot-info does NOT exercise --
# cv2, zmq, the lazily-loaded feetech SDK (scservo_sdk), and the host module -- so a missing
# camera/motor dep fails the stage here rather than at first robot connect.
"$py" -c 'import cv2, zmq, scservo_sdk
import lerobot.robots.lekiwi.lekiwi_host
print("host imports OK  (cv2", cv2.__version__, "+ zmq + scservo_sdk + lekiwi_host)")'
# Plugin gate: the host wrapper must import and the robot type must be registered,
# or host launch / calibrate would fail at first use instead of here.
"$py" -c 'import lerobot_robot_lekiwi_pincopen.lekiwi_host
from lerobot.robots import RobotConfig
assert "lekiwi_pincopen" in RobotConfig.get_known_choices(), "lekiwi_pincopen not registered"
print("plugin OK  (robot.type=lekiwi_pincopen + host wrapper importable)")'
# Devices are warn-only: they may simply be unplugged at provision time.
ls /dev/video*  >/dev/null 2>&1 && echo "  cameras: $(ls /dev/video* | wc -l) /dev/video* nodes" || echo "  (no /dev/video*  — cameras unplugged?)"
ls /dev/ttyACM* >/dev/null 2>&1 && echo "  motor bus: $(ls /dev/ttyACM* | tr '\n' ' ')"          || echo "  (no /dev/ttyACM* — motor bus unplugged?)"

# Provenance footer: lerobot's own environment report (versions, torch/CUDA, ffmpeg, CLIs).
echo "──────────────── lerobot-info ────────────────"
"$MAMBA/envs/$PI_ENV/bin/lerobot-info"
SH
}

# ── dispatch ─────────────────────────────────────────────────────────────────
is_stage() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

args=()
for a in "$@"; do
    case "$a" in
        --dry-run) dry=1 ;;
        *) args+=("$a") ;;
    esac
done
set -- "${args[@]}"

case "${1:-__all__}" in
    list) printf '%s\n' "${STAGES[@]}"; exit 0 ;;
    __all__) set -- "${STAGES[@]}" ;;            # no args -> every stage in order
    *) for a in "$@"; do is_stage "$a" || die "unknown stage '$a' (try: $0 list)"; done ;;
esac

validate_ssh_host "$PI_HOST" "PI_HOST"
validate_remote_name "$PI_ENV" "PI_ENV"
validate_python_version
validate_remote_path "$PI_REPO" "PI_REPO"
validate_remote_path "$PI_PLUGIN" "PI_PLUGIN"
validate_optional_remote_path "$MAMBA_PREFIX" "MAMBA_PREFIX"
validate_simple_url "$MINIFORGE_URL" "MINIFORGE_URL"
validate_package_list
[[ -z "$RECREATE_ENV" || "$RECREATE_ENV" == "1" ]] || die_usage "RECREATE_ENV must be empty or 1"

RENV="PI_ENV=$(shell_quote "$PI_ENV") PY_VER=$(shell_quote "$PY_VER") PI_REPO=$(shell_quote "$PI_REPO") PI_PLUGIN=$(shell_quote "$PI_PLUGIN") MAMBA_PREFIX=$(shell_quote "$MAMBA_PREFIX") MINIFORGE_URL=$(shell_quote "$MINIFORGE_URL") RECREATE_ENV=$(shell_quote "$RECREATE_ENV")"
remote_repo="$(shell_quote "${PI_REPO%/}/")"
remote_plugin="$(shell_quote "${PI_PLUGIN%/}/")"

if [ "$dry" = 1 ]; then
    printf 'PI_HOST=%s\n' "$PI_HOST"
    printf 'PI_ENV=%s\n' "$PI_ENV"
    printf 'PI_REPO=%s\n' "$PI_REPO"
    printf 'PI_PLUGIN=%s\n' "$PI_PLUGIN"
    printf 'LOCAL_REPO=%s\n' "$LOCAL_REPO"
    printf 'LOCAL_PLUGIN=%s\n' "$LOCAL_PLUGIN"
    printf 'stages:\n'
    for stage in "$@"; do printf '%s\n' "$stage"; done
    exit 0
fi

for stage in "$@"; do "stage_$stage"; done
