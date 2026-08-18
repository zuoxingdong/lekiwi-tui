"""eval.py — EvalScreen: configure + launch a policy rollout on the robot (lerobot-rollout).

The configuration half — checkpoint, base dataset and its task strings, backend, camera
slots, pacing, duration, display, passthrough flags — lives in
:mod:`lekiwi_tui.screens.policy_form`, shared with the DAgger form and remembered once
for both, so settings tuned while watching a rollout are already in place when you
switch to collecting corrections.

What is EvalScreen's own: it records nothing, so there is no dataset and no leader; its
preflight is the eval one; Start SUSPENDS into the rollout (it owns the real TTY for its
keyboard controls) and the run ends with a scoreboard verdict. Fronts scripts/eval.sh (the sole argv source). NO compile toggle
(intentional: per-shape recompiles are a training optimisation, not an eval one).
"""
from __future__ import annotations

import shlex
import time

from pyratatui import Line, Span

from .. import ROOT
from ..config import cfg_get, collapse_home
from ..framework import runner, theme
from ..framework.modals import ConfirmModalState
from ..preflight import confirm_preflight, eval_issues
from ..scoreboard import append_score, ckpt_label, load_scores, score_tally
from .chrome import clip_end as _clip_end
from .chrome import clip_middle as _clip_middle
from .chrome import section_line, slim_status_spans
from .policy_form import (
    CAM_MODES, PolicyFormScreen, _checkpoint_error, _device_note, _policy_root_path,
    cam_map_conflicts, cam_pairs, detect_cam_detail, detect_cam_slots,
    resolve_base_dataset, resolve_eval_policy, shared_state, training_dataset_name,
    training_rename_map,
)

# The run_headless hook name used by direct no-TTY CLI dispatch.
HEADLESS_HOOK = "run_headless"

EVAL_SCRIPT = ROOT / "scripts" / "eval.sh"


class EvalScreen(PolicyFormScreen):
    """Eval/rollout configuration form; Start suspends into lerobot-rollout (real TTY)."""

    title = "run policy"
    CONFIG_SECTION = "rollout"
    STATE_KEY = "eval"
    DURATION_STEP = 5
    DURATION_ZERO_LABEL = "saved default"

    def _extra_fields(self) -> list[str]:
        return self._tail_fields()

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
        if not await confirm_preflight(
            app, "Run policy preflight", eval_issues(self.ctx, policy=self._policy),
        ):
            return
        self._remember()
        cam_slots, _note = self._cam_resolved()  # eval.sh takes only the concrete value
        argv = [
            "bash", str(EVAL_SCRIPT),
            "--policy", self._policy, "--task", self._task_text, "--backend", self._backend,
            "--exec-horizon", str(self._exec.value), "--action-steps", str(self._steps.value),
            "--flow-steps", str(self._flow.value),
            "--cam-slots", cam_slots, "--duration", str(self._dur.value),
            "--display", "on" if self._show else "off", "--gpu", self.ctx.gpu_name,
            *extra_tokens, *self._extra,
        ]
        rc = await app.suspend(argv, pause=True)
        await self._ask_verdict(rc)

    async def _ask_verdict(self, rc: int) -> None:
        """The post-run scoreboard verdict: one modal (Success / Failure / Skip), one
        JSONL append. Only after a REAL run that ended normally (0, or 130 = Ctrl+C,
        the usual end of an until-Ctrl+C rollout) — a crashed run isn't a verdict."""
        if runner.DRY_RUN or rc not in (0, 130):
            return
        label = ckpt_label(self._policy, self._root)
        task = (self._task_text or str(cfg_get("rollout.task", doc=self.ctx.doc) or "")).strip()
        choice = await self.app.run_modal(ConfirmModalState(
            f'Scoreboard — how did {label} do on "{_clip_end(task, 48)}"?',
            ["Success", "Failure", "Skip — no verdict"]))
        if choice not in ("Success", "Failure"):
            return
        ok = choice == "Success"
        if append_score(self._root, {"ts": time.time(), "label": label, "task": task,
                                     "success": ok}):
            s, n = score_tally(load_scores(self._root), label=label, task=task)
            self.app.notify(f"scoreboard: {label} on this task → {s}/{n}", "info")
        else:
            self.app.notify("✗ could not write the scoreboard file", "warn")

    # ── view ──────────────────────────────────────────────────────────────────
    def _header_right(self) -> list[Span]:
        """Eval needs BOTH live facts: the host (robot side) and the GPU (policy side)."""
        spans = slim_status_spans(self.ctx)
        gpu = [Span("GPU ", theme.MUTED_STYLE)]
        if self.ctx.gpu_name:
            gpu += [Span(f"{theme.status_dot()} ", theme.OK_STYLE),
                    Span("cuda", theme.TEXT_STYLE)]
        else:
            gpu.append(Span("cpu — slow", theme.WARN_STYLE))
        # host-dot spans come first, then the GPU, then the mode chip (already in spans).
        return spans[:-2] + gpu + [Span("   ", theme.BASE_STYLE)] + spans[-2:]

    def _plan(self, width: int = 100) -> str:
        return _clip_end(
            f"run {ckpt_label(self._policy, self._root)} · {self._backend}"
            f" · {self._dur.display()} · hands the terminal to the policy",
            max(24, int(width) - 14))

    def _scoreboard_lines(self, width: int) -> list[Line]:
        """Per-task tallies for the SELECTED checkpoint (up to 3 tasks, most-judged
        first). Empty when the checkpoint has no verdicts yet — zero cost to ignore."""
        scores = load_scores(self._root)
        label = ckpt_label(self._policy, self._root)
        tasks: dict[str, tuple[int, int]] = {}
        for e in scores:
            if e.get("label") == label:
                t = str(e.get("task", ""))
                s, n = tasks.get(t, (0, 0))
                tasks[t] = (s + (1 if e.get("success") else 0), n + 1)
        if not tasks:
            return []
        out = [Line([]), Line([Span("  SCOREBOARD", theme.FAINT_STYLE),
                               Span(f"  {label}", theme.FAINT_STYLE)])]
        for t, (s, n) in sorted(tasks.items(), key=lambda kv: -kv[1][1])[:3]:
            frac_ok = n > 0 and s / n >= 0.5
            bar = "▓" * s + "░" * (n - s) if not theme.ASCII_MODE else "#" * s + "-" * (n - s)
            out.append(Line([
                Span(f"  {_clip_end(t, max(20, width - 40)):<{max(20, width - 40)}}",
                     theme.MUTED_STYLE),
                Span(f"  {s}/{n} ", theme.TEXT_STYLE),
                Span(bar, theme.OK_STYLE if frac_ok else theme.WARN_STYLE),
            ]))
        return out

    def _body_lines(self, width: int = 100) -> list[Line]:
        from .chrome import plan_row

        w = int(width)
        lines: list[Line] = [section_line("POLICY")]
        lines += self._policy_rows(w)
        lines.append(Line([]))
        lines.append(section_line("TASK"))
        lines += self._task_rows(w)
        lines.append(Line([]))
        lines.append(section_line("RUN"))
        lines += self._backend_camera_rows(w)
        lines += self._pacing_rows(w)
        lines += self._tail_rows(w)
        lines.append(Line([]))
        focused = self._cur() == "start"
        if self._host_alive() is False:
            plan_span = Span("⚠ host not reachable — Start host first (menu 1)",
                             theme.WARN_STYLE)
        else:
            plan_span = Span(self._plan(w),
                             theme.HIGHLIGHT_MUTED_STYLE if focused else theme.MUTED_STYLE)
        lines.append(plan_row("Run", [plan_span], focused=focused))
        lines.extend(self._scoreboard_lines(w))
        return lines

    def _focused_hint(self) -> str:
        field = self._cur()
        if field == "duration":
            return "0 = the saved yaml default · ←→ ±5 · ⏎ type a number"
        hint = self._common_hint(field)
        if hint:
            return hint
        if self._host_alive() is False:
            return "preflight will stop the launch while the host is down"
        return (f"device: {_device_note(self._policy, self.ctx.gpu_name)} · "
                "s starts · c collects corrections")

    # ── value formatting (also the summary surface the tests read) ────────────
    def _policy_display_path(self) -> str:
        from pathlib import Path as _Path

        if not self._policy:
            return ""
        try:
            return str(_Path(self._policy).relative_to(self._root))
        except ValueError:
            return collapse_home(self._policy)

    def _value(self, field: str, *, width: int = 60) -> str:
        if field == "policy":
            disp = self._policy_display_path() or collapse_home(self._policy) or "(none found)"
            suffix = "  default" if self._policy and self._policy == self._default_abs else ""
            return _clip_middle(disp, max(8, width - len(suffix))) + suffix
        if field == "base":
            return _clip_end(self._task_choices.name, width)
        if field == "task":
            t = (self._task_text or "(saved default)").splitlines()[0]
            return _clip_end(t, width)
        if field == "backend":
            return theme.choice(self._backend)
        if field in ("exec", "steps", "flow", "duration"):
            nf = self._num(field)
            disp = (nf.editor.value + "█") if field == self._cur() else nf.display()
            return _clip_end(disp, width)
        if field == "cameras":
            mode, note = self._cam_resolved()
            if self._cam_mode == "auto":
                return _clip_end(f"{theme.choice('auto')} · {mode} · {note}", width)
            return _clip_end(f"{theme.choice(mode)} · {note}", width)
        if field == "display":
            return theme.choice("on") if self._show else theme.choice("off")
        if field == "extra":
            return _clip_end(self._extra_text or "(none)", width)
        return ""

    def _summary_value(self, field: str) -> str:
        if field == "policy":
            disp = self._policy_display_path() or collapse_home(self._policy) or "(none found)"
            if self._policy and self._policy == self._default_abs:
                disp += "  default"
            return disp
        if field == "base":
            n = len(self._task_choices.tasks())
            return f"{self._task_choices.name} · {n} task string(s) · {self._base_source}"
        if field == "task":
            return (self._task_text or "(saved default)").replace("\n", " ")
        if field == "backend":
            if self._backend == "rtc":
                return f"rtc · action horizon {self._exec.display()}"
            return f"sync · action steps {self._steps.display()}"
        if field == "flow":
            return self._flow.display()
        if field == "cameras":
            mode, note = self._cam_resolved()
            origin = "auto-detected" if self._cam_mode == "auto" else "forced"
            return f"{mode} · {note} · {origin}"
        if field == "duration":
            return self._dur.display()
        if field == "display":
            return "on · live Rerun view" if self._show else "off · lower CPU"
        if field == "extra":
            return self._extra_text or "(none)"
        return ""


def run_headless(ctx, extra: list[str]) -> int:  # noqa: ANN001
    """No-TTY direct run of the eval action: resolve + validate the checkpoint, then front
    scripts/eval.sh directly (no app loop) with env defaults (INFERENCE / EXECUTION_HORIZON
    / DISPLAY_DATA). Emits the SAME flags the screen's _start computes from these defaults so
    headless and interactive front the script identically; --task is NOT passed (the config
    default applies) and --duration is 0 (the script omits --duration when 0). The script is
    the SOLE argv source; this path assembles no lerobot tokens."""
    root = _policy_root_path(ctx.cfg)
    policy, _note = resolve_eval_policy(ctx.cfg["POLICY_PATH"], root)
    if not policy:
        print(f"✗ no policy found: POLICY_PATH is auto, and no checkpoint exists under '{root}'.")
        return 1
    err = _checkpoint_error(policy)
    if err is not None:
        print(f"✗ {err}")
        return 1
    show = str(ctx.cfg["DISPLAY_DATA"]).lower() in ("1", "true", "yes", "on")
    # Same auto camera-slots resolution as the form's default mode; action-steps has no
    # env knob, so headless always passes 0 (the script omits the token = checkpoint
    # default), exactly like an untouched form.
    cam_slots, _note = detect_cam_slots(policy, cfg_get("rollout.rename_map", doc=ctx.doc) or {})
    argv = [
        "bash", str(EVAL_SCRIPT),
        "--policy", policy, "--backend", ctx.cfg["INFERENCE"],
        "--exec-horizon", str(ctx.cfg["EXECUTION_HORIZON"]),
        "--action-steps", "0", "--flow-steps", "0", "--cam-slots", cam_slots,
        "--duration", "0", "--display", "on" if show else "off",
        "--gpu", ctx.gpu_name, *extra,
    ]
    return runner.headless_run(argv)


__all__ = [
    "CAM_MODES", "EvalScreen", "HEADLESS_HOOK", "cam_map_conflicts", "cam_pairs",
    "detect_cam_detail", "detect_cam_slots", "resolve_base_dataset", "resolve_eval_policy",
    "run_headless", "shared_state", "training_dataset_name", "training_rename_map",
]
