# Launcher Scripts

Each `scripts/*.sh` file is a non-interactive launcher. The TUI gathers inputs;
the script builds the final `lerobot-*` argv.

## Contract

- Parse long flags.
- Slice command config from `lekiwi.yaml` when needed.
- Assemble argv as a Bash array.
- On `--dry-run` or `DRY=1`, print one argv token per line and exit.
- Otherwise `exec` the command so it owns the real TTY.

Run scripts from an environment where `python`, `pyyaml`, and the relevant
`lerobot-*` CLI are on `PATH`.

## Safety

- Unknown trailing flags pass through to `lerobot-*` where supported.
- `host.sh` and follower calibration reject unknown flags because they emit fixed
  remote bash payloads.
- SSH hosts and identifier-like remote values are validated.
- Remote paths such as `PI_REPO` are shell-quoted before use inside remote
  `ssh` commands.

## Scripts

| Script | Purpose |
| --- | --- |
| `teleop.sh` | `lerobot-teleoperate` |
| `record.sh` | `lerobot-record` |
| `replay.sh` | `lerobot-replay` |
| `eval.sh` | `lerobot-rollout` |
| `train.sh` | `lerobot-train` |
| `view.sh` | `lerobot-dataset-viz` |
| `calibrate.sh` | leader local calibration, follower SSH calibration |
| `host.sh` | emits Pi-side host launch/kill bash |
| `sync.sh` | rsync local LeRobot source to the Pi |
| `pi_provision.sh` | first-time Pi setup stages |

## Useful Dry Runs

```bash
bash scripts/teleop.sh --dry-run --display on --fps 30
bash scripts/record.sh --dry-run --name demo --task "pick and place"
bash scripts/eval.sh --dry-run --policy /path/to/checkpoint --backend sync
bash scripts/train.sh --dry-run --mode fresh --run demo --init lerobot/smolvla_base
bash scripts/sync.sh --dry-run
bash scripts/pi_provision.sh --dry-run conda lerobot
```
