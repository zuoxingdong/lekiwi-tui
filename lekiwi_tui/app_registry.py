"""app_registry.py — the standalone action registry.

The 13 actions are grouped into HOST/COLLECT/LEARN/SETUP sections and expose aliases for
direct-mode launch. Screen/stream/host actions point at
``lekiwi_tui.screens.<mod>:<Class>``; suspend actions name a handler the dispatcher
(``dispatch.py``) resolves. This module is pure data + lookup — it imports no framework
and no screen, so listing the menu never imports a screen (lazy-import).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["suspend", "stream", "screen", "host"]


@dataclass(frozen=True)
class Action:
    """One menu action: id/icon/label/hint/section plus kind and handler target string.

    handler:
      * screen/stream/host kind -> "lekiwi_tui.screens.<module>:<Class>", lazily
        imported + pushed by the dispatcher (degrades to a 'not ported yet' toast if the
        screen module is absent).
      * suspend kind -> a bare handler id the dispatcher maps to an app-level coroutine
        (currently replay / view).
    """

    id: str
    icon: str
    label: str
    hint: str
    section: str   # HOST | COLLECT | LEARN | SETUP
    kind: Kind
    handler: str


# ── Action registry ───────────────────────────────────────────────────────────
ACTIONS: list[Action] = [
    Action("host-launch", "🚀", "Start host", "start the Pi robot host; choose session length", "HOST", "host", "lekiwi_tui.screens.host:HostLaunchScreen"),
    Action("host-kill", "🛑", "Stop host", "stop the running Pi host server", "HOST", "host", "lekiwi_tui.screens.host_kill:HostKillScreen"),
    Action("teleop", "🎮", "Teleoperate", "drive the arm and mobile base manually", "COLLECT", "screen", "lekiwi_tui.screens.teleop:TeleopScreen"),
    Action("record", "🔴", "Record", "record episodes into a local dataset", "COLLECT", "screen", "lekiwi_tui.screens.record:RecordScreen"),
    Action("replay", "🎞️", "Replay", "run recorded actions on the robot", "COLLECT", "suspend", "replay"),
    Action("view", "🔍", "View", "open a recorded episode in Rerun", "COLLECT", "suspend", "view"),
    Action("train", "🧠", "Train", "fine-tune SmolVLA locally", "LEARN", "stream", "lekiwi_tui.screens.train:TrainScreen"),
    Action("eval", "🤖", "Run policy", "run a trained policy without recording", "LEARN", "stream", "lekiwi_tui.screens.eval:EvalScreen"),
    Action("setup-pi", "🧰", "Set up Pi", "install Pi system, conda, and LeRobot dependencies", "SETUP", "screen", "lekiwi_tui.screens.provision:ProvisionScreen"),
    Action("sync", "🔄", "Sync to Pi", "copy the local LeRobot source to the Pi", "SETUP", "screen", "lekiwi_tui.screens.sync:SyncScreen"),
    Action("calibrate", "🎯", "Calibrate", "leader (local) or follower (on the Pi)", "SETUP", "screen", "lekiwi_tui.screens.calibrate:CalibrateScreen"),
    Action("robot-config", "🤖", "Robot config", "review robot, camera, dataset, and task settings", "SETUP", "screen", "lekiwi_tui.screens.robot_config:RobotConfigScreen"),
    Action("settings", "⚙️", "Settings", "edit launcher settings in lekiwi.yaml", "SETUP", "screen", "lekiwi_tui.screens.settings:SettingsScreen"),
]

ACTIONS_BY_ID: dict[str, Action] = {a.id: a for a in ACTIONS}

# Aliases: id → also reachable as these single letters/words.
ALIASES: dict[str, str] = {
    "launch": "host-launch", "l": "host-launch",
    "kill": "host-kill", "k": "host-kill",
    "t": "teleop",
    "r": "record",
    "p": "replay",
    "viz": "view", "v": "view",
    "c": "calibrate",
    "rollout": "eval", "e": "eval",
    "robot": "robot-config", "y": "robot-config",
    "config": "settings", "s": "settings",
    "pi": "setup-pi", "provision": "setup-pi",
    "rsync": "sync", "push": "sync",
}


def resolve_action(name: str) -> Action | None:
    """Map an action id or alias to an Action; None if unknown."""
    canonical = ALIASES.get(name, name)
    return ACTIONS_BY_ID.get(canonical)


__all__ = ["Action", "Kind", "ACTIONS", "ACTIONS_BY_ID", "ALIASES", "resolve_action"]
