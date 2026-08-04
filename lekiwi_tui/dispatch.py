"""dispatch.py — the lekiwi action dispatcher injected into the framework App.

The framework :class:`~lekiwi_tui.framework.app.App` is action-agnostic: it calls an
injected ``run_action(id, extra) -> awaitable[int]`` whenever a screen returns
:class:`RunAction`. :class:`Dispatcher` is that hook. It mirrors the Textual app's
``fire_action``:

  * **screen / stream / host** kind → lazily import ``screens.<mod>:<Class>`` and PUSH it
    onto the App stack (the screen then drives its own form / stream / suspend). A missing
    screen module (a WF-B screen not landed yet) degrades to a "not ported yet" toast,
    exactly like the original's ImportError fallback.
  * **suspend** kind → the app-level replay/view actions. They build the argv that fronts a
    ``scripts/*.sh`` launcher and hand the TTY to it via ``runner.suspend_run``.

Screen authoring convention (for the per-screen agents): a screen is constructed as
``ScreenClass(app, ctx)`` and MUST NOT use ``app`` during ``__init__`` (the root screen is
built before the App exists and has ``app`` injected just before ``run()``). Read state from
``ctx`` (``ctx.cfg`` / ``ctx.doc`` / ``ctx.gpu_name``). Complex async flows (a modal then a
suspend) live in a screen's ``RunAction`` handler here or as an ``async def`` the screen
returns to via ``RunAction``; ``handle_key`` itself stays sync and returns an Action.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from . import ROOT
from .app_registry import resolve_action
from .datasets import dataset_episodes, dataset_present, dataset_repo_id, record_root
from .framework import runner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .context import Context
    from .framework.app import App

# Absolute paths to the carried-over launcher scripts this dispatcher fronts.
SCRIPTS = ROOT / "scripts"


async def pick_dataset(app: "App", doc, extra: list[str] | None = None,
                       *, title: str) -> tuple[str, str] | None:
    """Pick which dataset to act on, then resolve it to ``(repo_id, root)``.

    Shows a DatasetPicker over the datasets dir (the parent of the configured record
    root) with a ``Custom path…`` row; the configured dataset is pre-selected so ⏎
    keeps today's default. repo_id is reconstructed as ``<ns>/<name>`` from the picked
    root's basename + the configured namespace (root drives loading; repo_id is the
    identity/label). In headless / direct mode (no live terminal) fall back to the
    configured dataset. Returns None if the user cancels.

    Module-level (not a Dispatcher method) so screens — replay/view flows AND the
    dataset editor's ``d`` key — share the one picker."""
    default_root = record_root(doc, extra or [])
    default_repo = dataset_repo_id(doc)
    ns = default_repo.rsplit("/", 1)[0] if "/" in default_repo else "local"

    if app is None or app.terminal is None:
        return default_repo, default_root

    from .framework.modals import PromptModalState
    from .widgets.pickers import CUSTOM, DatasetPicker

    chosen = await app.run_modal(DatasetPicker(
        str(Path(default_root).parent), default_root=default_root, title=title))
    if chosen is None:
        return None
    if chosen == CUSTOM:
        ans = await app.run_modal(PromptModalState(
            "Dataset folder (local path)", value=default_root,
            hint="⏎ apply · esc cancel"))
        if ans is None or not ans:
            return None
        root = os.path.expanduser(ans).rstrip("/")
    else:
        root = chosen
    name = Path(root).name
    return f"{ns}/{name}", root


class Dispatcher:
    """Resolves + runs a lekiwi action. Bound to the App after construction."""

    def __init__(self, ctx: "Context") -> None:
        self.ctx = ctx
        self.app: "App | None" = None

    def bind(self, app: "App") -> None:
        """Attach the live App (it and the dispatcher reference each other)."""
        self.app = app

    # The hook the App calls for a RunAction. Signature: (id, extra) -> awaitable[int].
    async def run_action(self, action_id: str, extra: "Sequence[str]" = ()) -> int:
        app = self.app
        assert app is not None, "Dispatcher.run_action called before bind(app)"
        action = resolve_action(action_id)
        if action is None:
            app.notify(f"unknown action: {action_id!r}", "error")
            return 1
        extra = list(extra)

        if action.kind in ("screen", "stream", "host"):
            return self._push_screen(app, action)
        if action.kind == "suspend":
            # host-kill is now a streaming SCREEN (screens/host_kill.py), reached via the
            # "host" kind above; only replay/view remain app-level suspend handlers.
            handler = {
                "replay": self._replay,
                "view": self._view,
            }.get(action.id)
            if handler is None:
                app.notify(f"no suspend handler for {action.id!r}", "error")
                return 1
            return await handler(app, extra)
        app.notify(f"unhandled kind {action.kind!r}", "error")
        return 1

    # ── screen / stream / host: lazy-import + push ────────────────────────────
    def _push_screen(self, app: "App", action) -> int:
        """Lazily import ``screens.<mod>:<Class>`` and push it. Degrade to a toast if the
        screen module is absent (the WF-B screens not landed yet) — the original's
        ImportError fallback."""
        module_path, _, cls_name = action.handler.partition(":")
        try:
            module = importlib.import_module(module_path)
            screen_cls = getattr(module, cls_name)
        except (ImportError, AttributeError):
            app.notify(f"{action.label} not ported yet.", "warn")
            return 0
        app.push(screen_cls(app, self.ctx))
        return 0

    # ── app-level suspend handlers ────────────────────────────────────────────
    async def _pick_dataset(self, app: "App", extra: list[str], *, title: str) -> tuple[str, str] | None:
        return await pick_dataset(app, self.ctx.doc, extra, title=title)

    async def _ask_episode(self, app: "App", *, title: str, repo_id: str, root: str) -> str | None:
        """Pick an episode via the rich EpisodeScreen (dataset identity + episode COUNT +
        valid range + range-checked entry), the original's EpisodeScreen. In headless /
        direct mode (no live terminal) fall back to "0". Returns None if the user aborts."""
        if app.terminal is None:
            return "0"
        from .screens.episode import EpisodeScreen

        # dataset_episodes() returns the count as a STRING ("24") or "?" when the metadata
        # is missing/unparseable; EpisodeScreen wants the int count (None = unknown), so
        # convert here. Skipping this left a str in the screen's `episodes >= 0` guard ->
        # "'>=' not supported between instances of 'str' and 'int'" on every replay/view.
        raw = dataset_episodes(root)
        episodes = int(raw) if raw.isdigit() else None
        return await app.run_modal(EpisodeScreen(
            app, self.ctx, title=title, repo_id=repo_id, root=str(root),
            episodes=episodes))

    async def _replay(self, app: "App", extra: list[str]) -> int:
        """do_replay: pick a dataset, then an episode (rich picker), then front
        scripts/replay.sh (dry-run safe).

        The chosen dataset overrides the replay.sh slice via passthrough
        ``--dataset.repo_id=`` / ``--dataset.root=`` — the SAME config_path-then-CLI-override
        mechanism record.sh uses (draccus applies the slice, then these win). They go before
        ``*extra`` so an explicit user override in extra still takes precedence."""
        picked = await self._pick_dataset(app, extra, title="Replay - choose dataset")
        if picked is None:
            return 0
        repo_id, root = picked
        ep = await self._ask_episode(app, title="Replay episode", repo_id=repo_id, root=root)
        if ep is None:
            return 0
        argv = [
            "bash", str(SCRIPTS / "replay.sh"), "--episode", ep,
            f"--dataset.repo_id={repo_id}", f"--dataset.root={root}", *extra,
        ]
        return runner.suspend_run(app, argv)

    async def _view(self, app: "App", extra: list[str]) -> int:
        """do_view: pick a dataset, gate that it is present, pick an episode (rich picker),
        then front view.sh with the chosen --repo-id / --root (view reads off disk, no slice)."""
        picked = await self._pick_dataset(app, extra, title="View - choose dataset")
        if picked is None:
            return 0
        repo_id, root = picked
        if not dataset_present(root):
            app.notify(
                f"No dataset found at {root}; record episodes first or pass --dataset.root=…",
                "error",
            )
            return 1
        ep = await self._ask_episode(app, title="View episode", repo_id=repo_id, root=root)
        if ep is None:
            return 0
        argv = [
            "bash", str(SCRIPTS / "view.sh"),
            "--repo-id", repo_id, "--root", str(root), "--episode-index", ep, *extra,
        ]
        return runner.suspend_run(app, argv)

__all__ = ["Dispatcher", "SCRIPTS", "pick_dataset"]
