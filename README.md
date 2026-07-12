# lerobot_robot_lekiwi_pincopen

My LeKiwi runs STS3250 servos on the four big arm joints and a
[PincOpen](https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen)
gripper. This plugin lets a **stock [LeRobot](https://github.com/huggingface/lerobot)
install (0.5.x / 0.6.x) drive that hardware**, zero source edits.

I wrote up the hardware build in
[Mobile Manipulation with LeKiwi + PincOpen](https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen).
This package is that integration as installable code.

**vs stock `lekiwi`:**

- arm joints 1-4 on **STS3250** (wrist_roll and gripper stay STS3215)
- **PincOpen gripper**: fixed EPROM calibration, skipped during interactive calibration
- **tuned servo params** written on every connect, all exposed as config fields:
  tuning is a yaml/CLI edit (`--robot.heavy_p_coefficient=10`), never a code change

## Install

```bash
pip install -e .
```

## Use

LeRobot auto-discovers the plugin by its package name (the official
[third-party conventions](https://huggingface.co/docs/lerobot/integrate_hardware)):

```bash
# calibrate (gripper is skipped, its EPROM calibration is applied)
lerobot-calibrate --robot.type=lekiwi_pincopen --robot.id=my_lekiwi

# host (stock lekiwi_host skips plugin discovery, hence the wrapper; same CLI, same yaml)
python -m lerobot_robot_lekiwi_pincopen.lekiwi_host --config_path=host.yaml
```

The client side (teleop/record/eval) needs nothing from this package:
`lekiwi_client` never touches motors.

Calibration files live under
`~/.cache/huggingface/lerobot/calibration/robots/lekiwi_pincopen/`.
Coming from stock `lekiwi`: copy that folder once.

## Tuning

| Field | Default |
|---|---|
| `arm_p_coefficient` | 14 |
| `heavy_p_coefficient` | 10 |
| `heavy_acceleration` | 200 |
| `gripper_acceleration` | 200 |
| `gripper_overload_torque` | 65 (percent) |
| `gripper_protective_torque` | 5 (percent) |
| `gripper_protection_time` | 7 (x10 ms) |

**P=10 on the big joints is the load-bearing fix.** Stock writes P=16, which gave me
jitter and servo overload shutdowns on this hardware.

## Tests

```bash
pip install -e .[dev]
python -m pytest tests
```

No hardware needed, everything stops short of `bus.connect()`.

## Related

- [Mobile Manipulation with LeKiwi + PincOpen](https://huggingface.co/blog/zuoxingdong/mobile-manipulation-lekiwi-pincopen):
  the hardware story
- [lekiwi-tui](https://github.com/zuoxingdong/lekiwi-tui): my terminal control center,
  ships and drives this plugin automatically

## License

Apache-2.0. `calibrate()`/`configure()` derive from LeRobot (Apache-2.0,
The HuggingFace Inc. team); see the file headers.
