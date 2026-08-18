"""dagger.py — DaggerScreen: configure + launch HIL correction collection
(lerobot-rollout --strategy.type=dagger, fronted by scripts/dagger.sh).

The configuration half — checkpoint, base dataset and its task strings, backend, camera
slots, pacing — lives in :mod:`lekiwi_tui.screens.policy_form`, shared with the Run
policy form and remembered once for both: the settings you tuned while watching a
rollout fail are already in place when you come here to correct it (Run policy's ``c``
lands you here directly).

What is DAgger's own: it RECORDS, so it needs a corrections target, a recording mode,
an input device, a preflight that checks the leader and the disk, and a per-session
dataset root stamped at launch (rollout stamps only the repo id, never the root, so a
second session would otherwise fail on the existing directory). Start also shows the
session cheat-sheet — the rules that cost data to learn live — before handing over the
terminal, and the session ends in a junk review rather than a scoreboard verdict.
"""
from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyratatui import Line, Span

from .. import ROOT
from ..dagger_review import dagger_episode_report, session_summary, write_quality_flags
from ..datasets import dataset_episodes, record_root
from ..framework import theme
from ..framework.modals import ConfirmModalState
from ..framework.screen import Nothing
from ..framework.widgets import NumberField
from ..preflight import confirm_preflight, dagger_issues
from .chrome import clip_end as _clip_end
from .chrome import number_line, plan_row, section_line, seg, setting_line
from .policy_form import PolicyFormScreen, _checkpoint_error, _state_int

if TYPE_CHECKING:
    from ..framework.events import Key

DAGGER_SCRIPT = ROOT / "scripts" / "dagger.sh"
STATE_KEY = "dagger"

#: The session cheat-sheet, shown as the last gate before the terminal is handed
#: over — one key per line, then the three rules that cost data (or a dropped
#: grasped object) to discover live. Structured with \n (wrap_label honors breaks).
#: Deliberately NO `enter = push` line: this setup is local-only (push_to_hub off,
#: `local/` namespace), so advertising the hub push would be a lie.
_CHEATSHEET = (
    "space — pause / resume. Pausing glides the leader onto the follower (~2 s): "
    "hands off until it settles.\n"
    "tab — start / stop a correction (only while paused). Every stop saves one episode.\n"
    "esc — end the session; the arm returns to its start pose.\n"
    "\n"
    "Taking over mid-grasp? Squeeze the trigger as you grab, or the gripper eases open.\n"
    "Stop at a stable point: grasp = steady lifted hold; insertion = seated + released.\n"
    "Reset the scene by hand while paused — never teleop with tab, corrections record."
)
_GO = "Start session"
_STAY = "Cancel"


def session_root(parent: str, *, now: "time.struct_time | None" = None) -> str:
    """The per-session dataset dir: ``<parent>/rollout_dagger_<YYYYMMDD_HHMMSS>``.

    Computed HERE (not left to dagger.sh's identical fallback) so the screen knows
    the exact root afterwards — the post-session review reads it. The ``rollout_``
    prefix is lerobot's own validation rule for deployment datasets.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return str(Path(parent) / f"rollout_dagger_{stamp}")


class DaggerScreen(PolicyFormScreen):
    """DAgger session form; Start gates on preflight + cheat-sheet then suspends."""

    title = "dagger"
    CONFIG_SECTION = "dagger"
    STATE_KEY = STATE_KEY
    DURATION_STEP = 60
    DURATION_ZERO_LABEL = "no time limit"
    _LABEL_W = 14

    # ── the rows that are dagger's own ────────────────────────────────────────
    def _init_own(self, shared: dict[str, Any]) -> None:
        own = self._own_state()
        self._target = NumberField(
            "Corrections", _state_int(own, "target", 10), minimum=1, step=1)
        self._advanced = bool(own.get("advanced", False))
        self._record_all = bool(own.get("record_all", False))
        self._pedal = bool(own.get("pedal", False))

    def _remember_own(self) -> None:
        self._own_state().update({
            "target": self._target.value,
            "advanced": self._advanced,
            "record_all": self._record_all,
            "pedal": self._pedal,
        })

    def _extra_fields(self) -> list[str]:
        f = ["target", "advanced"]
        if self._advanced:
            f += ["mode", "input", *self._tail_fields()]
        return f

    def _num(self, key: str) -> "NumberField | None":
        return self._target if key == "target" else super()._num(key)

    def _blank_task_label(self) -> str:
        # NOT "(saved default)": Start refuses a blank task here, because dagger stamps
        # the string onto every correction it records.
        return "(←→ pick a string · ⏎ type)" if self._task_choices.tasks() else "(⏎ to type)"

    def _handle_own_key(self, key: "Key", cur: str) -> Any | None:
        from ..framework.events import ENTER, LEFT, RIGHT

        name = key.name
        toggles = {"advanced": "_advanced", "mode": "_record_all", "input": "_pedal"}
        if name in (LEFT, "h", RIGHT, "l", ENTER) and cur in toggles:
            setattr(self, toggles[cur], not getattr(self, toggles[cur]))
            if cur == "advanced":
                # Folding can shorten the field list under the cursor.
                self._fpos = min(self._fpos, len(self._fields()) - 1)
            self._remember()
            self._err = ""
            return Nothing
        return None

    # ── launch ────────────────────────────────────────────────────────────────
    async def _start(self) -> None:
        app = self.app
        try:
            extra_tokens = shlex.split(self._extra_text)
        except ValueError as exc:
            self._err = f"extra flags: {exc}"
            return
        if not self._policy:
            self._err = (f"No policy found: POLICY_PATH is auto, and no checkpoint "
                         f"exists under '{self._root}'.")
            return
        err = _checkpoint_error(self._policy)
        if err is not None:
            self._err = err
            return
        if not self._task_text:
            self._err = "Pick a task string first — dagger stamps it on every correction."
            return
        parent = str(Path(record_root(self.ctx.doc)).parent)
        if not await confirm_preflight(
            app, "DAgger preflight",
            dagger_issues(self.ctx, policy=self._policy, parent=parent),
        ):
            return
        # The last gate: the session rules that cost data to learn live.
        if await app.run_modal(ConfirmModalState(_CHEATSHEET, [_GO, _STAY])) != _GO:
            return
        self._remember()
        root = session_root(parent)
        mode, _note, _conf = self._cam_detail()
        argv = [
            "bash", str(DAGGER_SCRIPT),
            "--policy", self._policy, "--task", self._task_text,
            "--backend", self._backend, "--exec-horizon", str(self._exec.value),
            "--action-steps", str(self._steps.value),
            "--flow-steps", str(self._flow.value),
            "--target", str(self._target.value),
            "--record-autonomous", "on" if self._record_all else "off",
            "--input", "pedal" if self._pedal else "keyboard",
            "--duration", str(self._dur.value),
            "--display", "on" if self._show else "off",
            "--gpu", self.ctx.gpu_name, "--cam-slots", mode,
            "--dataset-root", root,
            *extra_tokens, *self._extra,
        ]
        rc = await app.suspend(argv, pause=True)
        # 130 (Ctrl+C) is a normal end for an until-ESC session, not a crash.
        self._own_state()["last_root"] = root
        if rc in (0, 130):
            await self._review_session(root)

    async def _review_session(self, root: str) -> None:
        """The post-session gate: per-episode stats, junk pre-flagged, one keypress
        into the dataset editor. Junk verdicts ride the editor's own quality.jsonl
        sidecar, so flagged episodes arrive there already marked — review is
        "open → D → typed delete", not a second triage UI."""
        app = self.app
        n = dataset_episodes(root)
        if n in ("?", "0"):
            app.notify(f"dagger session ended — nothing recorded ({root} stays empty)", "info")
            return
        report = dagger_episode_report(root)
        flagged = {r["index"]: r["junk"] for r in report if r["junk"]}
        if flagged and not write_quality_flags(root, flagged):
            app.notify("✗ could not write junk flags to meta/quality.jsonl", "warn")
        junk_note = (f" — {len(flagged)} look like junk (pre-marked for delete)"
                     if flagged else "")
        title = (f"Session saved {n} correction(s) → {Path(root).name}{junk_note}.  "
                 + session_summary(report))
        review = "Review in dataset editor"
        if await app.run_modal(ConfirmModalState(title, [review, "Done"])) == review:
            from .dataset_edit import DatasetEditScreen

            app.push(DatasetEditScreen(app, self.ctx, root=root))

    # ── view ──────────────────────────────────────────────────────────────────
    def _body_lines(self, width: int = 100) -> list[Line]:
        cur = self._cur()
        w = int(width)
        lines: list[Line] = [section_line("POLICY")]
        lines += self._policy_rows(w)
        lines.append(Line([]))
        lines.append(section_line("TASK"))
        lines += self._task_rows(w)
        lines.append(Line([]))
        lines.append(section_line("SESSION"))
        lines += self._backend_camera_rows(w)
        lines += self._pacing_rows(w)
        lines.append(number_line(self._target, "Corrections", cur == "target",
                                 "session ends after this many saved corrections",
                                 width=w, label_width=self._LABEL_W))
        # Advanced — collapsed by default; the folded row names what it hides.
        folded = "mode · input · duration · display · extra flags"
        lines.append(Line([
            self._gutter("advanced"), self._lab("Advanced", cur == "advanced"),
            Span("▾ " if self._advanced else "▸ ", theme.TEXT_STYLE),
            Span("" if self._advanced else folded, theme.FAINT_STYLE)]))
        if self._advanced:
            lines.append(setting_line(
                "Mode",
                [seg("corrections-only", not self._record_all), Span(" ", theme.BASE_STYLE),
                 seg("record-all", self._record_all)],
                "record-all also saves autonomous frames (tagged)",
                focused=cur == "mode", label_width=self._LABEL_W, width=w))
            lines.append(setting_line(
                "Input",
                [seg("keyboard", not self._pedal), Span(" ", theme.BASE_STYLE),
                 seg("pedal", self._pedal)],
                "pedal = PCsensor footswitch (device via extra flags)",
                focused=cur == "input", label_width=self._LABEL_W, width=w))
            lines += self._tail_rows(w, duration_note="hard time limit for the whole session")
        lines.append(Line([]))
        focused = cur == "start"
        if self._host_alive() is False:
            plan_span = Span("⚠ host not reachable — Start host first (menu 1)",
                             theme.WARN_STYLE)
        else:
            parent = Path(record_root(self.ctx.doc)).parent
            plan_span = Span(_clip_end(
                f"collect {self._target.value} corrections · {self._backend}"
                f" · → {parent}/rollout_dagger_<launch time>", max(24, w - 14)),
                theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Start", [plan_span], focused=focused))
        return lines

    _OWN_HINTS = {
        "target": "saved corrections that end the session · ←→ ±1 · ⏎ type",
        "advanced": "←→/⏎ fold or unfold the advanced settings",
        "mode": "corrections-only = clean fine-tune data · record-all needs intervention-aware training",
        "input": "who sends pause/correction — keyboard (space/tab) or foot pedal",
        "duration": "0 = no time limit · ←→ ±60 · ⏎ type a number",
    }

    def _focused_hint(self) -> str:
        field = self._cur()
        if field in self._OWN_HINTS:
            return self._OWN_HINTS[field]
        hint = self._common_hint(field)
        if hint:
            return hint
        if self._host_alive() is False:
            return "preflight will stop the launch while the host is down"
        return ("space pauses · tab corrects · ESC ends — the cheat-sheet repeats "
                "before launch · s also starts")


__all__ = ["DaggerScreen", "session_root"]
